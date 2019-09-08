import dataclasses
import enum
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

DESIGN_ID = "iter"


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


def case_dataset(case_reader):
    top_case_ids = case_reader.list_cases(recurse=True, flat=False).keys()
    case_tree = merge_dicts(
        case_reader.get_case(case_id, recurse=True) for case_id in top_case_ids
    )

    reference_case = next(iter(case_tree.keys()))
    io_kwargs = dict(
        values=False, units=True, shape=True, hierarchical=True, out_stream=None
    )
    variables = dict(
        sorted(reference_case.list_outputs(**io_kwargs), key=lambda x: x[0])
    )
    cases = [x for x in case_reader.get_cases("root")]

    ds = xr.Dataset(
        {
            "timestamp": (
                [DESIGN_ID],
                [pd.Timestamp.fromtimestamp(c.timestamp) for c in cases],
                {"role": VariableRole.META},
            )
        },
        coords={DESIGN_ID: [c.iteration_coordinate for c in cases]},
    )
    ds[DESIGN_ID].attrs["role"] = VariableRole.META

    for name, meta in variables.items():
        problem_meta = case_reader.problem_metadata["variables"].get(name, {})
        ds[name] = (
            (DESIGN_ID, *list("d" * len(meta["shape"]))),
            [c.outputs[name] for c in cases],
            {
                "role": variable_role(case_reader, name),
                "units": meta["units"],
                "scaling": VariableScaling(
                    multiplier=np.broadcast_to(
                        problem_meta.get("scaler") or 1.0, meta["shape"]
                    ),
                    offset=np.broadcast_to(
                        problem_meta.get("adder") or 0.0, meta["shape"]
                    ),
                ),
            },
        )

    return ds


def pareto_subset(ds):
    objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    scaling_array = np.array(
        [
            (o.attrs["scaling"].multiplier, o.attrs["scaling"].offset)
            for o in objectives.values()
        ]
    )
    scaled_objectives = objectives * scaling_array[:, 0] + scaling_array[:, 1]
    stacked_objectives = scaled_objectives.to_stacked_array("objs", [DESIGN_ID])
    pareto_mask = xr.DataArray(
        is_pareto_efficient(stacked_objectives.values), dims=[DESIGN_ID]
    )

    return ds.where(pareto_mask, drop=True)
