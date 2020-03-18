import math
import os
import string
from functools import partial, reduce
from operator import eq, mul

import hypothesis
import hypothesis.extra.numpy as np_st
import hypothesis.strategies as st
import numpy as np
import openmdao.api as om
import pymop
import pytest
import scipy as sp
from deap.tools import uniform_reference_points

from openmdao_omt import (
    Nsga3Driver,
    VariableType,
    add_design_var,
    case_dataset,
    pareto_subset,
)
from openmdao_omt.testing import (
    NoiseComponent,
    PassthroughComponent,
    is_assertion_error,
)

ONES = np.ones((2,))
almost_equal = partial(np.allclose, rtol=1e-2, atol=1e-2)

vec_len = np.vectorize(len)

var_name_st = st.text(alphabet=string.ascii_letters, min_size=1)
var_type_st = st.sampled_from(VariableType)


@st.composite
def variable_st(draw):
    type_ = draw(var_type_st)
    shape = draw(np_st.array_shapes())

    if type_.bounded:
        dtype, dtype_st, eps = (
            (np.int, st.integers, 1) if type_.discrete else (np.float, st.floats, 1e-6)
        )

        lower = draw(
            np_st.arrays(
                shape=shape,
                dtype=dtype,
                elements=dtype_st(max_value=1e9, min_value=-1e9),
            )
        )
        upper = lower + draw(
            np_st.arrays(
                shape=shape,
                dtype=dtype,
                elements=dtype_st(min_value=eps, max_value=1e3),
            )
        )

        output = {"lower": lower, "upper": upper}

    else:
        output = {
            "values": draw(
                np_st.arrays(
                    shape=shape,
                    dtype=object,
                    elements=st.sets(
                        st.floats(allow_nan=False)
                        if type_.ordered
                        else st.text(alphabet=string.ascii_letters),
                        min_size=1,
                        max_size=10,
                    ),
                )
            )
        }

    return {"type": type_, "shape": shape, **output}


# @hypothesis.reproduce_failure("4.36.1", b"AXicY2RkYGAEYiQAAABVAAU=")
@hypothesis.settings(deadline=10000, max_examples=20, print_blob=True)
@hypothesis.given(
    variables=st.lists(
        st.tuples(var_name_st, variable_st()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    )
)
def test_variable_mixing(recording_path, variables):
    prob = om.Problem()

    for name, var in variables:
        group = prob.model.add_subsystem(name, om.Group())
        indeps = group.add_subsystem("indeps", om.IndepVarComp())
        if var["type"] is VariableType.CONTINUOUS:
            indeps.add_output("x", shape=var["shape"])
        else:
            indeps.add_discrete_output("x", None, shape=var["shape"])

        add_design_var(group, "indeps.x", **var)

    noise = prob.model.add_subsystem("noise", NoiseComponent())
    noise.add_objective("y")

    recorder = om.SqliteRecorder(recording_path)
    prob.driver = Nsga3Driver(
        generation_count=100, reference_partitions=2, random_seed=0
    )
    prob.model.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    cases = case_dataset(om.CaseReader(recording_path))

    expected_value_coverage = 0.66

    for name, var in variables:
        values = cases[f"{name}.indeps.x"].values
        if var["type"].bounded:
            upper = var["upper"]
            lower = var["lower"]

            assert np.all((lower <= values) & (values <= upper))
            assert np.all(
                np.ptp(values, axis=0) / (upper - lower) >= expected_value_coverage
            )
        else:
            unique_values = np.apply_along_axis(set, axis=0, arr=values)
            assert np.all(unique_values <= var["values"])

            assert np.all(
                vec_len(unique_values)
                >= np.ceil(vec_len(var["values"]) * expected_value_coverage)
            )


class PymopComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("problem", types=pymop.Problem)
        # self.options.declare("discrete_input", types=bool, default=False)

    def setup(self):
        problem = self.options["problem"]
        self.add_input("var", shape=(problem.n_var,))
        self.add_output("obj", shape=(problem.n_obj,))
        if problem.n_constr:
            self.add_output("con", shape=(problem.n_constr,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        out = self.options["problem"].evaluate(
            inputs["var"], return_values_of=["F", "G"], return_as_dictionary=True
        )
        outputs["obj"] = out["F"]
        if self.options["problem"].n_constr:
            outputs["con"] = out["G"]


class PymopGroup(om.Group):
    def initialize(self):
        self.options.declare("problem", types=pymop.Problem)

    def setup(self):
        problem = self.options["problem"]
        self.add_subsystem(
            "indeps", om.IndepVarComp("var", shape=(problem.n_var,)), promotes=["*"]
        )
        self.add_subsystem(
            "problem", PymopComponent(problem=self.options["problem"]), promotes=["*"]
        )
        add_design_var(
            self, "var", shape=(problem.n_var,), lower=problem.xl, upper=problem.xu
        )
        self.add_objective("obj")
        if problem.n_constr:
            self.add_constraint("con", upper=0.0)


def test_unconstrained_dtlz1(recording_path):
    recorder = om.SqliteRecorder(recording_path)

    pymop_problem = pymop.DTLZ1(n_var=3, n_obj=3)

    prob = om.Problem()
    prob.model = PymopGroup(problem=pymop_problem)
    prob.model.add_recorder(recorder)
    prob.driver = Nsga3Driver(generation_count=500, random_seed=0)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    cases = case_dataset(om.CaseReader(recording_path))
    pareto_cases = pareto_subset(cases)

    distance_function = "euclidean"
    ref_dirs = uniform_reference_points(pymop_problem.n_obj, p=4)
    ideal_pareto_front = pymop_problem.pareto_front(ref_dirs)
    min_pareto_point_distance = sp.spatial.distance.pdist(
        ideal_pareto_front, distance_function
    ).min()

    distances = sp.spatial.distance.cdist(
        pareto_cases["problem.obj"].values, ideal_pareto_front, distance_function
    )
    distances_to_ideal = np.min(distances, axis=0)

    assert distances_to_ideal.max() <= min_pareto_point_distance * 0.75
