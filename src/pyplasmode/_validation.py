"""Shared validation at public array and configuration boundaries."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def integer(value: object, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def identities(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> None:
    if (not values and not allow_empty) or any(
        not isinstance(value, str) or not value or value.strip() != value for value in values
    ):
        raise ValueError(f"{name} must contain non-empty normalized strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def indices(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative integer indices")
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the supported index range")
    return np.array(array, dtype=np.int64, copy=True)
