import dataclasses
import enum
from collections import OrderedDict
from itertools import chain

import numpy as np
import pandas as pd
import pygmo
import xarray as xr
from openmdao.core.driver import Driver
from openmdao.core.problem import Problem
from openmdao.core.system import System
from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.solvers.solver import Solver

DESIGN_ID = "design"


def merge_dicts(dicts):
    result = OrderedDict()
    for d in dicts:
        result.update(d)
    return result


def _is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto-efficient points.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    # Next index in the is_efficient array to search for
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Removes dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def is_pareto_efficient(costs):
    ixs = np.argsort(
        ((costs - costs.mean(axis=0)) / (costs.std(axis=0) + 1e-7)).sum(axis=1)
    )
    costs = costs[ixs]
    is_efficient = _is_pareto_efficient(costs)
    is_efficient[ixs] = is_efficient.copy()
    return is_efficient


def constraint_dominated_ranks(costs, constraint_violations):
    # This consumes LOADS of memory for large arrays.
    # TODO: optimize
    feasibility = np.isclose(constraint_violations, 0)
    i_feasible_j_infeasible = feasibility[:, None] > feasibility
    i_and_j_feasible = feasibility[:, None] & feasibility
    i_less_infeasible_than_j = constraint_violations[:, None] < constraint_violations
    i_dominates_js_objectives = np.all(costs[:, None] < costs, axis=-1) & np.any(
        costs[:, None] <= costs, axis=-1
    )
    i_dominates_j = (
        i_feasible_j_infeasible
        | (~i_and_j_feasible & i_less_infeasible_than_j)
        | (i_and_j_feasible & i_dominates_js_objectives)
    )
    remaining = np.arange(len(costs))
    fronts = np.empty(len(costs), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(i_dominates_j[remaining[:, None], remaining], axis=0)
        fronts[remaining[~dominated]] = frontier_index

        remaining = remaining[dominated]
        frontier_index += 1
    return fronts


class VariableRole(enum.Enum):
    INTERMEDIATE = enum.auto()
    DESIGN = enum.auto()
    CONSTRAINT = enum.auto()
    OBJECTIVE = enum.auto()
    META = enum.auto()


@dataclasses.dataclass
class VariableScaling:
    multiplier: np.ndarray
    offset: np.ndarray


def variable_role(case_reader, variable_name):
    try:
        variable_meta = case_reader.problem_metadata["variables"][variable_name]
    except KeyError:
        return VariableRole.INTERMEDIATE

    try:
        if variable_meta["type"] == "obj":
            return VariableRole.OBJECTIVE
        elif variable_meta["type"] == "con":
            return VariableRole.CONSTRAINT
    except KeyError:
        return VariableRole.DESIGN


def case_dataset(case_reader, use_promoted_name=True):
    cases = case_reader.get_cases("root", recurse=True)

    ds = xr.Dataset(
        {
            "timestamp": (
                [DESIGN_ID],
                [pd.Timestamp.fromtimestamp(c.timestamp) for c in cases],
                {"role": VariableRole.META},
            )
        },
        coords={DESIGN_ID: [c.name for c in cases]},
    )
    ds[DESIGN_ID].attrs["role"] = VariableRole.META

    reference_case = cases[0]
    for name in reference_case.outputs.absolute_names():
        meta = reference_case._abs2meta
        problem_meta = case_reader.problem_metadata["variables"].get(name, {})
        discrete = "shape" not in meta[name]
        raw_values = [c.outputs[name] for c in cases]
        if discrete:
            shape = ()
            values = np.array(raw_values, dtype="O")
        else:
            shape = meta[name].get(
                "shape", getattr(reference_case.outputs[name], "shape", (1,))
            )
            values = np.stack(raw_values)

        ds[name] = (
            (DESIGN_ID, *(f"{name}_{dim}" for dim in range(len(shape)))),
            values,
            {
                "role": variable_role(case_reader, name),
                "units": meta.get("units", None),
                "scaling": VariableScaling(
                    multiplier=np.broadcast_to(
                        problem_meta.get("scaler") or 1.0, shape
                    ),
                    offset=np.broadcast_to(problem_meta.get("adder") or 0.0, shape),
                ),
            },
        )

    return ds


def design_space(ds):
    return ds.filter_by_attrs(type=lambda x: x and "desvar" in x)


def objective_space(ds, scale=False):
    objectives = ds.filter_by_attrs(type=lambda x: x and "objective" in x)
    if not scale:
        return objectives

    scaler_ds = xr.Dataset(
        {
            name: var.attrs["type"]["objective"]["scaler"] or 1.0
            for (name, var) in objectives.items()
        }
    )
    adder_ds = xr.Dataset(
        {
            name: var.attrs["type"]["objective"]["adder"] or 0.0
            for (name, var) in objectives.items()
        }
    )

    return objectives * scaler_ds + adder_ds


def constraint_space(ds):
    return ds.filter_by_attrs(type=lambda x: x and "constraint" in x)


def pareto_subset(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    if len(ds[DESIGN_ID]) < 1:
        raise ValueError("Supplied dataset has no designs.")
    scaled_objectives = objective_space(ds, scale=True).to_array()
    pareto_mask = xr.DataArray(
        is_pareto_efficient(scaled_objectives.T.values), dims=[DESIGN_ID]
    )

    return ds.where(pareto_mask, drop=True)


def feasible_subset(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    constraints = constraint_space(ds)
    eq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is not None
    )
    ineq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is None
    )

    lower_bound_ds = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["lower"]
            for (name, var) in ineq_constraints.items()
        }
    )
    upper_bound_ds = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["upper"]
            for (name, var) in ineq_constraints.items()
        }
    )

    ineq_feasibility_per_constraint = xr.ufuncs.logical_and(
        ineq_constraints >= lower_bound_ds, ineq_constraints <= upper_bound_ds
    ).to_array()

    # Applies all() on all dimensions except DESIGN_ID
    ineq_feasibility_per_design = ineq_feasibility_per_constraint.groupby(
        DESIGN_ID
    ).all(...)

    return ds.where(ineq_feasibility_per_design, drop=True)


def epsilonify(da: xr.DataArray, eps=np.finfo(float).eps) -> xr.DataArray:
    da = da.copy()
    da[da.isin([0.0])] = eps
    return da


def hv_ref_point(ds, offset_ratio=0.001):
    scaled_objectives = objective_space(ds, scale=True)

    nadir_point = scaled_objectives.max()
    ref_point = nadir_point + abs(nadir_point) * offset_ratio

    return ref_point.to_array()


def hypervolume(ds, ref_point=None):
    scaled_objectives = objective_space(ds, scale=True)

    hv = pygmo.hypervolume(scaled_objectives.to_array().T)

    if ref_point is None:
        ref_point = hv.refpoint()

    return xr.DataArray(
        hv.compute(ref_point), name="hypervolume", attrs={"units": None}
    )


def constraint_violations(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    eq_constraints_ds = ds.filter_by_attrs(
        type=lambda x: x and "constraint" in x and x["constraint"]["equals"] is not None
    )
    ineq_constraints_ds = ds.filter_by_attrs(
        type=lambda x: x and "constraint" in x and x["constraint"]["equals"] is None
    )

    lower_bound_da = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["lower"]
            for (name, var) in ineq_constraints_ds.items()
        }
    ).to_array()
    lower_bound_da_eps = epsilonify(lower_bound_da)
    upper_bound_da = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["upper"]
            for (name, var) in ineq_constraints_ds.items()
        }
    ).to_array()
    upper_bound_da_eps = epsilonify(upper_bound_da)

    ineq_cv_per_constraint = xr.ufuncs.fabs(
        xr.ufuncs.fmax(upper_bound_da, ineq_constraints_ds) / upper_bound_da_eps - 1
    ) + xr.ufuncs.fabs(
        xr.ufuncs.fmin(lower_bound_da, ineq_constraints_ds) / lower_bound_da_eps - 1
    )

    # Applies sum() on all dimensions except DESIGN_ID
    ineq_feasibility_per_design = (
        ineq_cv_per_constraint.to_array().groupby(DESIGN_ID).sum(...)
    )

    # Arranges the array in the same order as the input
    return ineq_feasibility_per_design.sel({DESIGN_ID: ds[DESIGN_ID]})


def wrapper_array(inp):
    if isinstance(inp, np.ndarray) and np.ndim(inp) == 0:
        inp = inp.item()

    a = np.empty((1,), dtype="O")
    a[0] = inp
    return a


def annotate_ds_with_constraint_violations(ds):
    cv = constraint_violations(ds)
    return ds.merge({"constraint_violation": cv})


def annotate_ds_with_rank(ds):
    scaled_objectives = objective_space(ds, scale=True).to_array()

    ranks = xr.DataArray(
        constraint_dominated_ranks(
            scaled_objectives.T.values, ds["constraint_violation"].values
        ),
        dims=[DESIGN_ID],
    )
    return ds.merge({"rank": ranks})


def generate_abs2meta(recording_requester):
    meta = {}
    ##### START ADAPTATION FROM SqliteRecorder #####
    driver = None

    # grab the system
    if isinstance(recording_requester, Driver):
        system = recording_requester._problem().model
        driver = recording_requester
    elif isinstance(recording_requester, System):
        system = recording_requester
    elif isinstance(recording_requester, Problem):
        system = recording_requester.model
        driver = recording_requester.driver
    elif isinstance(recording_requester, Solver):
        system = recording_requester._system()
    else:
        raise ValueError(
            "Driver encountered a recording_requester it cannot handle"
            ": {0}".format(recording_requester)
        )

    states = system._list_states_allprocs()

    if driver is None:
        desvars = system.get_design_vars(True, get_sizes=False)
        responses = system.get_responses(True, get_sizes=False)
        objectives = OrderedDict()
        constraints = OrderedDict()
        for name, data in responses.items():
            if data["type"] == "con":
                constraints[name] = data
            else:
                objectives[name] = data
    else:
        desvars = driver._designvars
        constraints = driver._cons
        objectives = driver._objs
        responses = driver._responses

    inputs = (
        system._var_allprocs_abs_names["input"]
        + system._var_allprocs_abs_names_discrete["input"]
    )

    outputs = (
        system._var_allprocs_abs_names["output"]
        + system._var_allprocs_abs_names_discrete["output"]
    )

    full_var_set = [
        (outputs, "output"),
        (desvars, "desvar"),
        (responses, "response"),
        (objectives, "objective"),
        (constraints, "constraint"),
    ]

    # # merge current abs2prom and prom2abs with this system's version
    # self._abs2prom["input"].update(system._var_abs2prom["input"])
    # self._abs2prom["output"].update(system._var_abs2prom["output"])
    # for v, abs_names in system._var_allprocs_prom2abs_list["input"].items():
    #     if v not in self._prom2abs["input"]:
    #         self._prom2abs["input"][v] = abs_names
    #     else:
    #         self._prom2abs["input"][v] = list(
    #             set(chain(self._prom2abs["input"][v], abs_names))
    #         )

    # # for outputs, there can be only one abs name per promoted name
    # for v, abs_names in system._var_allprocs_prom2abs_list["output"].items():
    #     self._prom2abs["output"][v] = abs_names

    # absolute pathname to metadata mappings for continuous & discrete variables
    # discrete mapping is sub-keyed on 'output' & 'input'
    real_meta = system._var_allprocs_abs2meta
    disc_meta = system._var_allprocs_discrete

    for var_set, var_type in full_var_set:
        for name in var_set:
            if name not in meta:
                try:
                    meta[name] = real_meta[name].copy()
                    meta[name]["discrete"] = False
                except KeyError:
                    meta[name] = disc_meta["output"][name].copy()
                    meta[name]["discrete"] = True
                meta[name]["type"] = {}
                meta[name]["explicit"] = name not in states
                # self._abs2meta[name]["tags"] = list(self._abs2meta[name].get("tags", []))

            if var_type not in meta[name]["type"]:
                try:
                    var_type_meta = var_set[name]
                except TypeError:
                    var_type_meta = {}
                meta[name]["type"][var_type] = var_type_meta

    for name in inputs:
        try:
            meta[name] = real_meta[name].copy()
            meta[name]["discrete"] = False
        except KeyError:
            meta[name] = disc_meta["input"][name].copy()
            meta[name]["discrete"] = True
        meta[name]["type"] = {"input": {}}
        meta[name]["explicit"] = True
        # self._abs2meta[name]["tags"] = list(self._abs2meta[name].get("tags", []))

    ##### END ADAPTATION FROM SqliteRecorder #####

    return meta


class DatasetRecorder(CaseRecorder):
    def __init__(self, record_viewer_data=False):
        if record_viewer_data:
            raise NotImplementedError(
                "This recorder does not support recording of metadata for viewing."
            )
        super().__init__(record_viewer_data=record_viewer_data)
        self.datasets = {}
        # self._abs2prom = {"input": {}, "output": {}}
        # self._prom2abs = {"input": {}, "output": {}}
        self._abs2meta = {}

    def startup(self, recording_requester):
        super().startup(recording_requester)
        # ds = xr.Dataset(data_vars={"counter": xr.DataArray(), "timestamp": xr.DataArray()}, coords={"name": xr.DataArray()})
        self.datasets[recording_requester] = []

        self._abs2meta.update(generate_abs2meta(recording_requester))

    def record_iteration_driver(self, recording_requester, data, metadata):
        timestamp = pd.Timestamp.fromtimestamp(metadata["timestamp"])

        all_vars = dict(chain(data["input"].items(), data["output"].items()))

        # hvplot borks of MultiIndex :((
        # design_idx = pd.MultiIndex.from_tuples(
        #     [(metadata["name"], 0, self._counter - 1, self._iteration_coordinate)],
        #     names=("driver", "rank", "counter", "name"),
        # )
        design_idx = [self._iteration_coordinate]

        def make_data_vars():
            for (name, value) in all_vars.items():
                meta = self._abs2meta[name]
                val = np.atleast_1d(value).copy()
                extra_dims = []
                if val.size > 1:
                    idx = pd.MultiIndex.from_tuples(np.ndindex(val.shape))
                    extra_dims = [(f"{name}_dim", idx)]
                    val = val.reshape((1, -1))

                yield (
                    name,
                    xr.DataArray(
                        data=val,
                        name=name,
                        attrs=meta,
                        coords=[(DESIGN_ID, design_idx), *extra_dims],
                    ),
                )

        meta_vars = {
            key: xr.DataArray([item], dims=[DESIGN_ID])
            for (key, item) in metadata.items()
            if key not in ["name", "success", "timestamp", "msg"]
        }
        data_vars = dict(make_data_vars())

        ds = xr.Dataset(
            data_vars={
                "timestamp": xr.DataArray([timestamp], dims=[DESIGN_ID]),
                "success": xr.DataArray([bool(metadata["success"])], dims=[DESIGN_ID]),
                "msg": xr.DataArray([metadata["msg"]], dims=[DESIGN_ID]),
                **meta_vars,
                **data_vars,
            },
        )

        self.datasets[recording_requester].append(ds)

    def record_iteration_problem(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of problems."
        )

    def record_iteration_solver(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of solvers."
        )

    def record_iteration_system(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of systems."
        )

    def record_derivatives_driver(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of derivatives."
        )

    def record_metadata_solver(self, recording_requester):
        pass

    def record_metadata_system(self, recording_requester):
        pass

    def record_viewer_data(self, model_viewer_data):
        pass

    def assemble_dataset(self, recording_requester):
        return xr.concat(self.datasets[recording_requester], dim=DESIGN_ID).squeeze()
