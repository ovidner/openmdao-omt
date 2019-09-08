import itertools

import numpy as np
import openmdao.api as om
import pytest

from openmdao_omt.data_management import case_dataset, pareto_subset


def nans(shape):
    return np.ones(shape) * np.nan


@pytest.fixture
def recording_path(tmpdir):
    return tmpdir.join("recording.sql")


@pytest.mark.parametrize("weights", itertools.product((1.0, -1.0), repeat=3))
def test_pareto_dataset(recording_path, weights):
    var_shape = (3,)
    prob = om.Problem()
    prob.model.add_subsystem("indeps", om.IndepVarComp("x", nans(var_shape)))
    prob.model.add_subsystem(
        "passthrough", om.ExecComp("y=x", x=nans(var_shape), y=nans(var_shape))
    )
    prob.model.connect("indeps.x", "passthrough.x")

    prob.model.add_design_var(
        "indeps.x", lower=np.zeros(var_shape), upper=np.ones(var_shape)
    )
    prob.model.add_objective("passthrough.y", scaler=np.array(weights))

    prob.driver = om.DOEDriver(
        om.ListGenerator(
            [
                [("indeps.x", np.array(x))]
                for x in [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ]
        )
    )

    recorder = om.SqliteRecorder(recording_path)
    prob.model.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    ds = case_dataset(om.CaseReader(recording_path))
    assert len(ds["iter"]) == 7

    pareto_ds = pareto_subset(ds)
    assert pareto_ds is not ds
    assert len(pareto_ds["iter"]) == 3
    expected_pareto_set = -np.eye(3) * np.array(weights)

    assert np.all(np.isin(expected_pareto_set, pareto_ds["passthrough.y"]))
