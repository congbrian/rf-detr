# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for transformer utilities, MS deformable attention core, and MSDeformAttn module."""

import io

import numpy as np
import pytest
import torch

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import Transformer, gen_encoder_output_proposals, gen_sineembed_for_position


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


_MSDeformInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]


def _build_ms_deform_inputs(
    bsz: int = 1,
    n_heads: int = 2,
    head_dim: int = 4,
    len_q: int = 3,
    npts: int = 1,
    levels: list[tuple[int, int]] | None = None,
) -> _MSDeformInputs:
    """Build minimal valid inputs for ms_deform_attn_core_pytorch.

    Args:
        bsz: Batch size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        len_q: Number of query elements.
        npts: Number of sampling points per level.
        levels: List of (H, W) int pairs; defaults to [(4, 4), (2, 2)].

    Returns:
        Tuple of (value, spatial_shapes_tensor, sampling_locations,
                  attention_weights, spatial_shapes_hw).
    """
    if levels is None:
        levels = [(4, 4), (2, 2)]
    nlvl = len(levels)

    total_hw = sum(ht * wd for ht, wd in levels)
    spatial_shapes_tensor = torch.tensor(levels, dtype=torch.long)
    value = torch.randn(bsz, n_heads, head_dim, total_hw)
    # sampling_locations: (bsz, len_q, n_heads, nlvl, npts, 2) in [0, 1]
    sampling_locations = torch.rand(bsz, len_q, n_heads, nlvl, npts, 2)
    # attention_weights: (bsz, len_q, n_heads, nlvl * npts)
    attention_weights = torch.softmax(torch.randn(bsz, len_q, n_heads, nlvl * npts), dim=-1)

    return value, spatial_shapes_tensor, sampling_locations, attention_weights, levels


def test_gen_encoder_output_proposals_passes_ij_indexing_to_meshgrid(monkeypatch) -> None:
    """`gen_encoder_output_proposals` should call `torch.meshgrid` with explicit ij indexing."""
    original_meshgrid = torch.meshgrid
    call_count = 0

    def _meshgrid_with_indexing_assertion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("indexing") != "ij":
            raise AssertionError("torch.meshgrid must be called with indexing='ij'")
        return original_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_with_indexing_assertion)

    memory = torch.randn(1, 4, 8)
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        spatial_shapes=spatial_shapes,
    )

    assert call_count == 1


def test_gen_sineembed_for_position_keeps_box_dimensions_in_sin_cos_order() -> None:
    """4D box positional embeddings must use the pretrained sin/cos order for all dimensions."""
    pos_tensor = torch.tensor([[[0.125, 0.25, 0.5, 0.75]]], dtype=torch.float32)
    dim = 4
    scale = 2 * torch.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    expected_parts = []
    for coord_idx in (1, 0, 2, 3):
        coord = pos_tensor[:, :, coord_idx] * scale
        encoded = coord[:, :, None] / dim_t
        expected_parts.append(torch.stack((encoded[:, :, 0::2].sin(), encoded[:, :, 1::2].cos()), dim=3).flatten(2))
    expected = torch.cat(expected_parts, dim=2)

    actual = gen_sineembed_for_position(pos_tensor, dim=dim)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_gen_encoder_output_proposals_rejects_non_square_ij_indexing(monkeypatch) -> None:
    """Wrong meshgrid indexing (xy vs ij) produces different proposals for non-square spatial shapes."""
    original_meshgrid = torch.meshgrid

    def _meshgrid_wrong_indexing(*args, **kwargs):
        kwargs["indexing"] = "xy"
        return original_meshgrid(*args, **kwargs)

    # Use non-square spatial shapes so that ij vs xy indexing produces observably different outputs.
    memory = torch.randn(1, 8, 8)
    spatial_shapes = torch.tensor([[2, 4]], dtype=torch.long)

    correct_memory, correct_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_wrong_indexing)

    wrong_memory, wrong_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    assert not torch.allclose(correct_proposals, wrong_proposals), (
        "xy indexing must produce different proposals than ij indexing for non-square spatial shapes"
    )


def test_gen_encoder_output_proposals_accepts_int_tuple_spatial_shapes() -> None:
    """`gen_encoder_output_proposals` must accept `spatial_shapes` as a tensor of int pairs."""
    batch = 2
    ht, wd = 4, 4
    memory = torch.randn(batch, ht * wd, 8)
    spatial_shapes = torch.tensor([[ht, wd]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(memory, spatial_shapes=spatial_shapes)

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, ht * wd, 4)


def test_gen_encoder_output_proposals_accepts_python_int_pair_spatial_shapes() -> None:
    """`gen_encoder_output_proposals` must accept `spatial_shapes` as `list[tuple[int, int]]` with no padding mask.

    Regression: `Transformer.forward` passes Python int pairs derived from `src.shape`, so the
    export-driven call path uses `list[tuple[int, int]]` rather than a tensor.
    """
    batch, ht, wd, dim = 2, 4, 4, 8
    memory = torch.randn(batch, ht * wd, dim)
    spatial_shapes = [(ht, wd)]  # Python int pairs, as produced by Transformer.forward()

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
    )

    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (batch, ht * wd, 4)


class TestMSDeformAttnCorePytorch:
    """Tests for ms_deform_attn_core_pytorch with Python int pair spatial shapes.

    Regression suite for torch.export.export compatibility: iterating over a spatial_shapes tensor yields FakeTensor
    scalars during FakeTensor tracing, which cannot be used as Python int split/view sizes.  The function now accepts an
    optional ``value_spatial_shapes_hw`` list of Python int pairs that bypasses tensor iteration.
    """

    @pytest.fixture
    def make_inputs(self) -> _MSDeformInputs:
        """Default two-level inputs: levels=[(4, 4), (2, 2)]."""
        return _build_ms_deform_inputs()

    @pytest.fixture
    def single_level_inputs(self) -> _MSDeformInputs:
        """Single-level inputs: levels=[(8, 8)]."""
        return _build_ms_deform_inputs(levels=[(8, 8)])

    def test_with_tensor_spatial_shapes(self, make_inputs: _MSDeformInputs) -> None:
        """Baseline: passing only the tensor spatial_shapes still works."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, _ = make_inputs

        output = ms_deform_attn_core_pytorch(value, spatial_shapes_tensor, sampling_locations, attention_weights)

        bsz, n_heads, head_dim, _ = value.shape
        len_q = sampling_locations.shape[1]
        assert output.shape == (bsz, len_q, n_heads * head_dim)

    def test_with_python_int_pair_spatial_shapes(self, make_inputs: _MSDeformInputs) -> None:
        """Regression: value_spatial_shapes_hw list of Python int pairs must be accepted.

        This is the torch.export.export-compatible code path: tensor scalar values (from iterating over a FakeTensor)
        cannot be used as split/view sizes, so the caller passes explicit Python int pairs via value_spatial_shapes_hw
        instead.
        """
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        bsz, n_heads, head_dim, _ = value.shape
        len_q = sampling_locations.shape[1]
        assert output.shape == (bsz, len_q, n_heads * head_dim)

    def test_tensor_and_hw_paths_produce_identical_outputs(self, make_inputs: _MSDeformInputs) -> None:
        """Python int pair path and tensor iteration path must produce the same result."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = make_inputs

        out_tensor_path = ms_deform_attn_core_pytorch(
            value, spatial_shapes_tensor, sampling_locations, attention_weights
        )
        out_hw_path = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        torch.testing.assert_close(out_tensor_path, out_hw_path)

    def test_single_level(self, single_level_inputs: _MSDeformInputs) -> None:
        """Single-level case with Python int pair path must not crash."""
        value, spatial_shapes_tensor, sampling_locations, attention_weights, levels = single_level_inputs

        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes_tensor,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        assert output.shape[0] == 1


class TestMSDeformAttnModule:
    """Tests for MSDeformAttn.forward covering the export-compatibility changes.

    Validates the module-level parameter threading and export-mode assert guard introduced in the torch.export.export
    compatibility fix.
    """

    _d_model = 32
    _n_heads = 4
    _n_levels = 2
    _n_points = 1
    _hw_pairs: list[tuple[int, int]] = [(4, 4), (2, 2)]

    def _make_module_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """Build minimal valid inputs for MSDeformAttn.forward.

        Returns:
            Tuple of (query, reference_points, input_flatten,
                      input_spatial_shapes, input_level_start_index, hw_pairs).
        """
        hw_pairs = self._hw_pairs
        total_len = sum(ht * wd for ht, wd in hw_pairs)
        bsz, len_q = 1, 3

        query = torch.randn(bsz, len_q, self._d_model)
        reference_points = torch.rand(bsz, len_q, self._n_levels, 2)
        input_flatten = torch.randn(bsz, total_len, self._d_model)
        input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        # Cumulative start index per level: [0, H0*W0]
        starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(self._n_levels)]
        input_level_start_index = torch.tensor(starts, dtype=torch.long)

        return query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, hw_pairs

    def test_forward_without_hw_param_backward_compat(self) -> None:
        """MSDeformAttn.forward without hw param produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, _ = self._make_module_inputs()

        output = module(query, ref_pts, input_flatten, spatial_shapes, level_start_index)

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_forward_with_hw_param_produces_correct_shape(self) -> None:
        """MSDeformAttn.forward with input_spatial_shapes_hw produces correct output shape."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_export_mode_forward_with_hw_param(self) -> None:
        """MSDeformAttn.forward in export mode with hw param must not raise."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        module.export()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs()

        output = module(
            query, ref_pts, input_flatten, spatial_shapes, level_start_index, input_spatial_shapes_hw=hw_pairs
        )

        bsz, len_q, _ = query.shape
        assert output.shape == (bsz, len_q, self._d_model)

    def test_export_flag_set_after_export_call(self) -> None:
        """Calling .export() must set _export=True, enabling the torch._assert guard path."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        assert not module._export

        module.export()

        assert module._export

    def test_export_mode_requires_spatial_shapes_hw(self) -> None:
        """Export mode without ``input_spatial_shapes_hw`` must raise a clear RuntimeError."""
        module = MSDeformAttn(
            d_model=self._d_model, n_levels=self._n_levels, n_heads=self._n_heads, n_points=self._n_points
        )
        module.export()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, _ = self._make_module_inputs()

        with pytest.raises(RuntimeError, match="input_spatial_shapes_hw"):
            module(query, ref_pts, input_flatten, spatial_shapes, level_start_index)


class TestMSDeformAttnRank5ExportPath:
    """Phase 1 CoreML / torch.export guards: export mode must keep sampling_locations ≤ rank 5.

    CoreML's MIL backend rejects rank-6 tensors. The eager path still uses rank-6
    ``(B, Q, heads, levels, points, 2)``; export mode merges ``(levels, points)`` into a
    single axis so every intermediate is ≤ rank 5, and the core must accept both layouts
    with identical numerics.
    """

    _d_model = 32
    _n_heads = 4
    _n_levels = 2
    _n_points = 2
    _hw_pairs: list[tuple[int, int]] = [(4, 4), (2, 2)]

    def _make_module_inputs(
        self,
        *,
        reference_last_dim: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """Build MSDeformAttn inputs with 2- or 4-dim reference points.

        Args:
            reference_last_dim: ``2`` for points or ``4`` for boxes (xywh).

        Returns:
            Tuple of (query, reference_points, input_flatten, spatial_shapes,
            level_start_index, hw_pairs).
        """
        if reference_last_dim not in (2, 4):
            raise ValueError(f"reference_last_dim must be 2 or 4, got {reference_last_dim!r}")
        hw_pairs = self._hw_pairs
        total_len = sum(ht * wd for ht, wd in hw_pairs)
        bsz, len_q = 1, 3

        query = torch.randn(bsz, len_q, self._d_model)
        reference_points = torch.rand(bsz, len_q, self._n_levels, reference_last_dim)
        input_flatten = torch.randn(bsz, total_len, self._d_model)
        input_spatial_shapes = torch.tensor(hw_pairs, dtype=torch.long)
        starts = [sum(ht * wd for ht, wd in hw_pairs[:idx]) for idx in range(self._n_levels)]
        input_level_start_index = torch.tensor(starts, dtype=torch.long)
        return query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, hw_pairs

    @pytest.mark.parametrize(
        "reference_last_dim",
        [
            pytest.param(2, id="ref_points"),
            pytest.param(4, id="ref_boxes"),
        ],
    )
    def test_export_mode_passes_rank5_sampling_locations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        reference_last_dim: int,
    ) -> None:
        """Export-mode forward must call the core with sampling_locations.ndim <= 5."""
        from rfdetr.models.ops.modules import ms_deform_attn as ms_deform_attn_mod

        captured_ndims: list[int] = []
        real_core = ms_deform_attn_core_pytorch

        def _capturing_core(
            value: torch.Tensor,
            value_spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor,
            attention_weights: torch.Tensor,
            value_spatial_shapes_hw: list[tuple[int, int]] | None = None,
        ) -> torch.Tensor:
            captured_ndims.append(sampling_locations.ndim)
            return real_core(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
                value_spatial_shapes_hw=value_spatial_shapes_hw,
            )

        monkeypatch.setattr(ms_deform_attn_mod, "ms_deform_attn_core_pytorch", _capturing_core)

        module = MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )
        module.export()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs(
            reference_last_dim=reference_last_dim
        )

        module(
            query,
            ref_pts,
            input_flatten,
            spatial_shapes,
            level_start_index,
            input_spatial_shapes_hw=hw_pairs,
        )

        assert captured_ndims, "core was not called"
        assert all(ndim <= 5 for ndim in captured_ndims), (
            f"export mode must keep sampling_locations rank <= 5 for CoreML, got ndims={captured_ndims}"
        )

    @pytest.mark.parametrize(
        "reference_last_dim",
        [
            pytest.param(2, id="ref_points"),
            pytest.param(4, id="ref_boxes"),
        ],
    )
    def test_export_mode_matches_eager_numerically(self, reference_last_dim: int) -> None:
        """Export-mode (rank-5) output must match eager (rank-6) for the same weights and inputs."""
        module = MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )
        module.eval()
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs(
            reference_last_dim=reference_last_dim
        )

        with torch.no_grad():
            eager_out = module(
                query,
                ref_pts,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )
            module.export()
            export_out = module(
                query,
                ref_pts,
                input_flatten,
                spatial_shapes,
                level_start_index,
                input_spatial_shapes_hw=hw_pairs,
            )

        torch.testing.assert_close(export_out, eager_out, rtol=1e-5, atol=1e-5)

    def test_core_rank5_sampling_locations_match_rank6(self) -> None:
        """ms_deform_attn_core_pytorch must accept merged rank-5 locations with rank-6 parity."""
        levels: list[tuple[int, int]] = [(4, 4), (2, 2)]
        bsz, n_heads, head_dim, len_q, npts = 1, 2, 4, 3, 2
        nlvl = len(levels)
        total_hw = sum(ht * wd for ht, wd in levels)

        value = torch.randn(bsz, n_heads, head_dim, total_hw)
        spatial_shapes = torch.tensor(levels, dtype=torch.long)
        sampling_locations_rank6 = torch.rand(bsz, len_q, n_heads, nlvl, npts, 2)
        # Merge (levels, points) in the same order as the export-mode view: levels slow, points fast.
        sampling_locations_rank5 = sampling_locations_rank6.flatten(3, 4)
        attention_weights = torch.softmax(torch.randn(bsz, len_q, n_heads, nlvl * npts), dim=-1)

        out_rank6 = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes,
            sampling_locations_rank6,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )
        out_rank5 = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes,
            sampling_locations_rank5,
            attention_weights,
            value_spatial_shapes_hw=levels,
        )

        torch.testing.assert_close(out_rank5, out_rank6, rtol=1e-5, atol=1e-5)

    def test_eager_mode_still_uses_rank6_sampling_locations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eager path should keep the historical rank-6 layout (train/infer unchanged)."""
        from rfdetr.models.ops.modules import ms_deform_attn as ms_deform_attn_mod

        captured_ndims: list[int] = []
        real_core = ms_deform_attn_core_pytorch

        def _capturing_core(
            value: torch.Tensor,
            value_spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor,
            attention_weights: torch.Tensor,
            value_spatial_shapes_hw: list[tuple[int, int]] | None = None,
        ) -> torch.Tensor:
            captured_ndims.append(sampling_locations.ndim)
            return real_core(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
                value_spatial_shapes_hw=value_spatial_shapes_hw,
            )

        monkeypatch.setattr(ms_deform_attn_mod, "ms_deform_attn_core_pytorch", _capturing_core)

        module = MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )
        query, ref_pts, input_flatten, spatial_shapes, level_start_index, hw_pairs = self._make_module_inputs(
            reference_last_dim=2
        )

        module(
            query,
            ref_pts,
            input_flatten,
            spatial_shapes,
            level_start_index,
            input_spatial_shapes_hw=hw_pairs,
        )

        assert captured_ndims == [6], f"eager path should use rank-6 sampling_locations, got {captured_ndims}"


class TestTransformerSpatialShapesHwThreading:
    """Phase 1: Transformer must thread Python (H, W) pairs into MSDeformAttn.

    ``torch.export`` / CoreML need concrete Python ints for split/view sizes inside
    deformable attention. The Transformer already builds ``spatial_shapes_hw`` for
    proposals; these tests require that list to reach ``MSDeformAttn.forward`` via the
    decoder, and that ``torch.compiler.is_exporting()`` builds ``spatial_shapes`` from
    those pairs instead of ``torch._shape_as_tensor``.
    """

    _d_model = 16
    _num_queries = 6
    _hw_pairs: list[tuple[int, int]] = [(4, 4), (2, 2)]

    def _build_transformer(self) -> Transformer:
        """Build a tiny eval-mode Transformer with two feature levels."""
        return Transformer(
            d_model=self._d_model,
            num_queries=self._num_queries,
            num_decoder_layers=1,
            sa_nhead=4,
            ca_nhead=4,
            num_feature_levels=2,
            dec_n_points=1,
            return_intermediate_dec=True,
            lite_refpoint_refine=True,
            use_grouppose_keypoints=False,
            two_stage=False,
        ).eval()

    def _build_inputs(
        self,
        batch_size: int = 1,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Build multi-scale inputs matching ``_hw_pairs``.

        Args:
            batch_size: Mini-batch size.

        Returns:
            ``srcs``, ``masks``, ``pos_embeds``, ``refpoint_embed``, ``query_feat``.
        """
        srcs = [
            torch.randn(batch_size, self._d_model, height, width) for height, width in self._hw_pairs
        ]
        masks = [torch.zeros(batch_size, height, width, dtype=torch.bool) for height, width in self._hw_pairs]
        pos_embeds = [
            torch.randn(batch_size, self._d_model, height, width) for height, width in self._hw_pairs
        ]
        refpoint_embed = torch.rand(self._num_queries, 4)
        query_feat = torch.randn(self._num_queries, self._d_model)
        return srcs, masks, pos_embeds, refpoint_embed, query_feat

    def test_forwards_spatial_shapes_hw_to_ms_deform_attn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Transformer.forward must pass ``input_spatial_shapes_hw`` into every MSDeformAttn call."""
        transformer = self._build_transformer()
        captured_hw: list[list[tuple[int, int]] | None] = []
        real_forward = MSDeformAttn.forward

        def _capturing_forward(
            self: MSDeformAttn,
            query: torch.Tensor,
            reference_points: torch.Tensor,
            input_flatten: torch.Tensor,
            input_spatial_shapes: torch.Tensor,
            input_level_start_index: torch.Tensor,
            input_padding_mask: torch.Tensor | None = None,
            input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
        ) -> torch.Tensor:
            captured_hw.append(input_spatial_shapes_hw)
            return real_forward(
                self,
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask,
                input_spatial_shapes_hw,
            )

        monkeypatch.setattr(MSDeformAttn, "forward", _capturing_forward)

        srcs, masks, pos_embeds, refpoint_embed, query_feat = self._build_inputs()
        with torch.no_grad():
            transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)

        assert captured_hw, "MSDeformAttn.forward was not called"
        for hw in captured_hw:
            assert hw == self._hw_pairs, f"expected spatial_shapes_hw={self._hw_pairs!r}, got {hw!r}"

    def test_spatial_shapes_uses_hw_pairs_under_torch_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under ``torch.compiler.is_exporting()``, spatial_shapes must come from Python hw pairs."""
        transformer = self._build_transformer()
        as_tensor_calls: list[object] = []
        shape_as_tensor_calls: list[object] = []

        real_as_tensor = torch.as_tensor
        real_shape_as_tensor = torch._shape_as_tensor

        def _tracking_as_tensor(*args: object, **kwargs: object) -> torch.Tensor:
            as_tensor_calls.append(args[0] if args else None)
            return real_as_tensor(*args, **kwargs)

        def _tracking_shape_as_tensor(*args: object, **kwargs: object) -> torch.Tensor:
            shape_as_tensor_calls.append(args[0] if args else None)
            return real_shape_as_tensor(*args, **kwargs)

        monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
        monkeypatch.setattr(torch, "as_tensor", _tracking_as_tensor)
        monkeypatch.setattr(torch, "_shape_as_tensor", _tracking_shape_as_tensor)

        srcs, masks, pos_embeds, refpoint_embed, query_feat = self._build_inputs()
        with torch.no_grad():
            transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)

        assert self._hw_pairs in as_tensor_calls, (
            f"expected torch.as_tensor({self._hw_pairs!r}) under is_exporting(), got calls={as_tensor_calls!r}"
        )
        assert shape_as_tensor_calls == [], (
            f"_shape_as_tensor must not run under is_exporting(), got {shape_as_tensor_calls!r}"
        )

    def test_eager_path_still_uses_shape_as_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eager / ONNX-TensorRT path must keep ``torch._shape_as_tensor`` (symbolic H/W)."""
        transformer = self._build_transformer()
        shape_as_tensor_calls: list[object] = []
        real_shape_as_tensor = torch._shape_as_tensor

        def _tracking_shape_as_tensor(*args: object, **kwargs: object) -> torch.Tensor:
            shape_as_tensor_calls.append(args[0] if args else None)
            return real_shape_as_tensor(*args, **kwargs)

        monkeypatch.setattr(torch.compiler, "is_exporting", lambda: False)
        monkeypatch.setattr(torch, "_shape_as_tensor", _tracking_shape_as_tensor)

        srcs, masks, pos_embeds, refpoint_embed, query_feat = self._build_inputs()
        with torch.no_grad():
            transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)

        assert len(shape_as_tensor_calls) == len(self._hw_pairs), (
            f"eager path should call _shape_as_tensor once per level, got {len(shape_as_tensor_calls)}"
        )


class TestGenEncoderOutputProposalsDynamicBatch:
    """Regression tests for dynamic batch support in gen_encoder_output_proposals.

    Ensures that the ONNX-symbolic refactoring (PR #950 / issue #949) does not bake a fixed batch dimension into
    proposals and that output shapes are correct for varying batch sizes.
    """

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_output_shape_invariant_across_batch_sizes(self, batch_size: int) -> None:
        """Output shapes must scale correctly with batch size, with no baked constants.

        Args:
            batch_size: Number of images in the batch.
        """
        ht, wd, dim = 4, 4, 8
        memory = torch.randn(batch_size, ht * wd, dim)
        spatial_shapes = [(ht, wd)]

        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )

        assert output_memory.shape == (batch_size, ht * wd, dim)
        assert output_proposals.shape == (batch_size, ht * wd, 4)

    def test_proposals_semantically_equivalent_across_batch_sizes(self) -> None:
        """Proposals for batch=1 and batch=4 must be identical per image.

        Regression: if batch_size were baked as a constant, repeating the same image
        N times would produce different proposals for each copy.
        """
        ht, wd, dim = 4, 4, 8
        memory_single = torch.randn(1, ht * wd, dim)
        memory_multi = memory_single.expand(4, -1, -1).contiguous()
        spatial_shapes = [(ht, wd)]

        _, proposals_single = gen_encoder_output_proposals(
            memory_single, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )
        _, proposals_multi = gen_encoder_output_proposals(
            memory_multi, memory_padding_mask=None, spatial_shapes=spatial_shapes
        )

        torch.testing.assert_close(proposals_single.expand(4, -1, -1), proposals_multi)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape_invariant_with_padding_mask(self, batch_size: int) -> None:
        """Output shapes must be correct when memory_padding_mask is provided with varying batch sizes.

        Regression for PR #950 / issue #949: the masked branch used .reshape(-1, h, w, 1) to infer the batch dimension
        dynamically; this test verifies the branch handles varying batch sizes without error.

        Args:
            batch_size: Number of images in the batch.
        """
        ht, wd, dim = 4, 4, 8
        total_hw = ht * wd
        memory = torch.randn(batch_size, total_hw, dim)
        # Mask shape: (batch, sum_hw) — True means padding (invalid position)
        memory_padding_mask = torch.zeros(batch_size, total_hw, dtype=torch.bool)
        spatial_shapes = [(ht, wd)]

        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, memory_padding_mask=memory_padding_mask, spatial_shapes=spatial_shapes
        )

        assert output_memory.shape == (batch_size, total_hw, dim)
        assert output_proposals.shape == (batch_size, total_hw, 4)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_onnx_export_with_dynamic_batch_axis(self, batch_size: int) -> None:
        """ONNX export with dynamic batch axis must run inference for batch sizes other than the trace batch.

        Regression for issue #949: exporting with a fixed trace batch baked `Reshape([8,...])` as a constant ONNX node,
        causing TRT engines to fail at inference for any batch != 8. Skipped when onnx or onnxruntime is not installed.
        """
        pytest.importorskip("onnx")
        onnxruntime = pytest.importorskip("onnxruntime")

        ht, wd, dim = 4, 4, 8
        spatial_shapes_list = [(ht, wd)]

        class _ProposalModule(torch.nn.Module):
            """Thin wrapper to export gen_encoder_output_proposals via torch.onnx."""

            def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Forward pass delegating to gen_encoder_output_proposals."""
                return gen_encoder_output_proposals(
                    memory, memory_padding_mask=None, spatial_shapes=spatial_shapes_list
                )

        module = _ProposalModule()
        trace_memory = torch.randn(2, ht * wd, dim)

        buf = io.BytesIO()
        torch.onnx.export(
            module,
            (trace_memory,),
            buf,
            input_names=["memory"],
            output_names=["output_memory", "output_proposals"],
            dynamic_axes={"memory": {0: "batch"}},
            opset_version=17,
        )
        buf.seek(0)
        onnx_bytes = buf.read()

        session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        memory_np = np.random.randn(batch_size, ht * wd, dim).astype(np.float32)
        out_memory, out_proposals = session.run(None, {"memory": memory_np})
        assert out_memory.shape == (batch_size, ht * wd, dim), f"wrong memory shape for batch={batch_size}"
        assert out_proposals.shape == (batch_size, ht * wd, 4), f"wrong proposals shape for batch={batch_size}"


def test_ms_deform_attn_core_pytorch_export_compatible() -> None:
    """torch.export.export must succeed on a module using ms_deform_attn_core_pytorch with hw param.

    Regression test for the FakeTensor tracing failure: iterating over spatial_shapes and using the scalar elements as
    split/view sizes fails during torch.export.export because FakeTensor data is not allocated. Passing
    value_spatial_shapes_hw (concrete Python ints from a module attribute) bypasses the tensor iteration entirely.
    """
    levels: list[tuple[int, int]] = [(4, 4), (2, 2)]
    bsz, n_heads, head_dim = 1, 2, 4
    total_hw = sum(ht * wd for ht, wd in levels)
    len_q, nlvl, npts = 3, len(levels), 1

    class _MinimalDeformAttn(torch.nn.Module):
        """Minimal wrapper to test torch.export.export on the hw-param code path."""

        def __init__(self, hw: list[tuple[int, int]]) -> None:
            super().__init__()
            self.hw = hw

        def forward(
            self,
            value: torch.Tensor,
            spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor,
            attention_weights: torch.Tensor,
        ) -> torch.Tensor:
            """Forward using concrete Python int pairs for export compatibility."""
            return ms_deform_attn_core_pytorch(
                value,
                spatial_shapes,
                sampling_locations,
                attention_weights,
                value_spatial_shapes_hw=self.hw,
            )

    value = torch.randn(bsz, n_heads, head_dim, total_hw)
    spatial_shapes = torch.tensor(levels, dtype=torch.long)
    sampling_locations = torch.rand(bsz, len_q, n_heads, nlvl, npts, 2)
    attention_weights = torch.softmax(torch.randn(bsz, len_q, n_heads, nlvl * npts), dim=-1)

    module = _MinimalDeformAttn(hw=levels)

    exported = torch.export.export(module, args=(value, spatial_shapes, sampling_locations, attention_weights))
    assert exported is not None
