# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.hw."""

import pytest

from rfdetr.utilities.hw import (
    coerce_hw_input,
    expand_square_hw,
    parse_hw_pair,
    pe_from_resolution,
    pe_tracks_resolution,
)


def test_expand_square_hw() -> None:
    assert expand_square_hw(560, field="resolution") == (560, 560)


def test_expand_square_hw_rejects_bool() -> None:
    with pytest.raises(ValueError, match="resolution"):
        expand_square_hw(True, field="resolution")


def test_parse_hw_pair_rectangular() -> None:
    assert parse_hw_pair((720, 960), field="resolution") == (720, 960)


def test_parse_hw_pair_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="length"):
        parse_hw_pair((640,), field="resolution")


def test_coerce_hw_input_int_and_pair() -> None:
    assert coerce_hw_input(560, field="resolution") == (560, 560)
    assert coerce_hw_input((720, 960), field="resolution") == (720, 960)


def test_pe_from_resolution_rectangular() -> None:
    assert pe_from_resolution((720, 960), patch_size=12) == (60, 80)


def test_pe_tracks_resolution_at_factory_defaults() -> None:
    assert pe_tracks_resolution((44, 44), (704, 704), patch_size=16) is True  # RFDETRLarge defaults
    assert pe_tracks_resolution((37, 37), (560, 560), patch_size=14) is False  # RFDETRBase defaults


def test_pe_tracks_resolution_when_out_of_sync() -> None:
    assert pe_tracks_resolution((44, 44), (640, 640), patch_size=16) is False
