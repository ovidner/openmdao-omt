import dataclasses
import enum

import numpy as np

import functools


def listify(fn=None, wrapper=list):
    """
    From https://github.com/shazow/unstdlib.py/blob/master/unstdlib/standard/list_.py#L149

    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example::

        >>> @listify
        ... def get_lengths(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths(["spam", "eggs"])
        [4, 4]
        >>>
        >>> @listify(wrapper=tuple)
        ... def get_lengths_tuple(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths_tuple(["foo", "bar"])
        (3, 3)
    """
    def listify_return(fn):
        @functools.wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))
        return listify_helper
    if fn is None:
        return listify_return
    return listify_return(fn)


@dataclasses.dataclass(frozen=True)
class VariableProperties:
    discrete: bool
    bounded: bool
    ordered: bool


class VariableType(VariableProperties, enum.Enum):
    CONTINUOUS = (False, True, True)
    INTEGER = (True, True, True)
    ORDINAL = (True, False, True)
    NOMINAL = (True, False, False)


def add_design_var(
    sys,
    name,
    *args,
    type=VariableType.CONTINUOUS,
    values=None,
    shape=(1,),
    **kwargs,
):
    if type.bounded and values:
        kwargs["lower"] = values[0]
        kwargs["upper"] = values[1]
        values = None

    if type.bounded and type.discrete:
        lower_int = kwargs.pop("lower", None)
        upper_int = kwargs.pop("upper", None)

    sys.add_design_var(name, *args, **kwargs)

    if sys._static_mode:
        design_vars = sys._static_design_vars
    else:
        design_vars = sys._design_vars

    # FIXME: Hacky McHackface
    if type.bounded and type.discrete:
        design_vars[name]["lower"] = np.broadcast_to(lower_int, shape)
        design_vars[name]["upper"] = np.broadcast_to(upper_int, shape)

    design_vars[name]["type"] = type
    design_vars[name]["values"] = values
    design_vars[name]["shape"] = shape
