# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``(height, width)`` helpers for ``resolution`` and ``positional_encoding_size``.

Height-first everywhere (same order as ``predict(shape=...)``). User input may be a square int (``560`` → ``(560,
560)``) or an explicit ``(height, width)`` pair; stored config values are always :data:`Hw` tuples.

``positional_encoding_size`` uses the same :data:`Hw` shape but counts **patch cells**, not pixels — see
:func:`pe_from_resolution`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import SupportsIndex, TypeAlias, cast

Hw: TypeAlias = tuple[int, int]


def expand_square_hw(value: object, *, field: str) -> Hw:
    """Expand a positive square side length to height-first ``(n, n)``.

    Args:
        value: Side length from config input (must be a positive int; bool rejected).
        field: Config field name used in error messages.

    Raises:
        ValueError: Non-integer, bool, or non-positive input.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    if not isinstance(value, int):
        raise ValueError(f"{field} must be a positive integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}.")
    return (value, value)


def parse_hw_pair(value: object, *, field: str) -> Hw:
    """Validate a length-2 sequence as height-first :data:`Hw` without square expansion.

    Args:
        value: ``(height, width)`` sequence; elements coerced via ``operator.index``.
        field: Config field name used in error messages.

    Raises:
        ValueError: Wrong length, non-positive dimensions, or unsupported type.
        TypeError: Non-integer sequence elements.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{field} must be a (height, width) pair, got {type(value).__name__}.",
        )
    if len(value) != 2:
        raise ValueError(
            f"{field} must be a (height, width) pair, got sequence of length {len(value)}.",
        )
    height = _index_dim(value[0], field=field, dim_name="height")
    width = _index_dim(value[1], field=field, dim_name="width")
    if height <= 0 or width <= 0:
        raise ValueError(f"{field} must contain positive integers for height and width, got ({height}, {width}).")
    return (height, width)


def coerce_hw_input(value: object, *, field: str) -> Hw:
    """Coerce config input to canonical :data:`Hw`.

    Accepts a square int or an explicit ``(height, width)`` pair.

    Args:
        value: User-supplied ``resolution`` or ``positional_encoding_size``.
        field: Config field name used in error messages.

    Raises:
        ValueError: Unsupported input shape or non-positive dimensions.
        TypeError: Non-integer sequence elements.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer or (height, width) pair, got {value!r}.")
    if isinstance(value, int):
        return expand_square_hw(value, field=field)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return parse_hw_pair(value, field=field)
    raise ValueError(f"{field} must be a positive integer or (height, width) pair, got {type(value).__name__}.")


def pe_from_resolution(resolution: Hw, patch_size: int) -> Hw:
    """Derive ``positional_encoding_size`` patch grid from pixel ``resolution``.

    ``resolution`` is in pixels; the result counts patch cells in the same height-first order. Each dimension uses
    integer floor division by ``patch_size``.

    Args:
        resolution: Input size in pixels.
        patch_size: ViT patch size (square).

    Returns:
        Patch-grid ``(h, w)`` for positional encodings.

    Examples:
        >>> pe_from_resolution((720, 960), patch_size=12)
        (60, 80)
    """
    height, width = resolution
    return (height // patch_size, width // patch_size)


def pe_tracks_resolution(pe: Hw, resolution: Hw, patch_size: int) -> bool:
    """Return whether ``pe`` matches the formula-derived grid for ``resolution``.

    Compares ``pe`` to :func:`pe_from_resolution`. Used by
    :meth:`~rfdetr.config.ModelConfig._sync_pe_with_resolution` to decide whether factory ``positional_encoding_size``
    should follow a custom ``resolution`` at construction. Returns ``False`` when the variant's factory PE was fixed for
    a published checkpoint (e.g. ``RFDETRBaseConfig`` ``(37, 37)`` at ``(560, 560)`` with ``patch_size=14``; formula
    would yield ``(40, 40)``).

    Args:
        pe: Positional-encoding patch grid to check.
        resolution: Pixel resolution to compare against.
        patch_size: ViT patch size (square).

    Returns:
        ``True`` when ``pe`` equals :func:`pe_from_resolution` for ``resolution``.

    Examples:
        >>> pe_tracks_resolution((44, 44), (704, 704), patch_size=16)
        True
        >>> pe_tracks_resolution((37, 37), (560, 560), patch_size=14)
        False
    """
    return pe == pe_from_resolution(resolution, patch_size)


def _index_dim(value: object, *, field: str, dim_name: str) -> int:
    """Coerce one height/width element via ``operator.index``.

    Raises:
        ValueError: Bool input.
        TypeError: Non-integer element.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} {dim_name} must be an integer, got bool.")
    try:
        dim = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(
            f"{field} {dim_name} must be an integer, got {type(value).__name__}.",
        ) from error
    return dim
