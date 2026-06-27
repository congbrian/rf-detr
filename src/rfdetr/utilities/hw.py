# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``(height, width)`` helpers for ``resolution`` and ``positional_encoding_size``.

Height-first everywhere (same order as ``predict(shape=...)``). Ints coerce to square pairs (``560`` → ``(560,
560)``).
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import SupportsIndex, TypeAlias, cast

Hw: TypeAlias = tuple[int, int]


def normalize_hw(value: object, *, field: str) -> Hw:
    """Coerce a square int or ``(height, width)`` pair to :data:`Hw`.

    Sequence elements may be integer-like (``numpy``/``torch`` scalars) via
    :func:`operator.index`; ``bool`` is rejected.

    Args:
        value: Square int or ``(height, width)`` pair to coerce.
        field: Config field name for error messages.

    Raises:
        ValueError: Unsupported shape or non-positive dimensions.
        TypeError: Non-integer sequence elements.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer or (height, width) pair, got {value!r}.")

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field} must be positive, got {value}.")
        return (value, value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                f"{field} must be a positive integer or (height, width) pair, got sequence of length {len(value)}.",
            )
        height = _index_dim(value[0], field=field, dim_name="height")
        width = _index_dim(value[1], field=field, dim_name="width")
        if height <= 0 or width <= 0:
            raise ValueError(f"{field} must contain positive integers for height and width, got ({height}, {width}).")
        return (height, width)

    raise ValueError(f"{field} must be a positive integer or (height, width) pair, got {type(value).__name__}.")


def pe_from_resolution(resolution: Hw, patch_size: int) -> Hw:
    """Map pixel ``resolution`` to patch-grid ``(h, w)`` (square ``patch_size``)."""
    height, width = resolution
    return (height // patch_size, width // patch_size)


def pe_tracks_resolution(pe: Hw, resolution: Hw, patch_size: int) -> bool:
    """True when ``pe`` equals :func:`pe_from_resolution` for ``resolution``."""
    return pe == pe_from_resolution(resolution, patch_size)


def _index_dim(value: object, *, field: str, dim_name: str) -> int:
    """Coerce one height/width element via :func:`operator.index`."""
    if isinstance(value, bool):
        raise ValueError(f"{field} {dim_name} must be an integer, got bool.")
    try:
        dim = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(
            f"{field} {dim_name} must be an integer, got {type(value).__name__}.",
        ) from error
    return dim
