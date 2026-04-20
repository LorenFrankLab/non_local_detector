"""Tests for the auto-select tiling heuristic used by the clusterless KDE path."""

from __future__ import annotations

from unittest import mock

import jax
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    _AUTO_TILE_DEFAULT_BUDGET_BYTES,
    _AUTO_TILE_INTERMEDIATE_COPIES,
    _AUTO_TILE_MEMORY_BUDGET_FRACTION,
    _AUTO_TILE_MIN_ENC_TILE,
    _default_memory_budget_bytes,
    auto_select_tile_sizes,
)


class TestAutoSelectReturnShape:
    """The helper returns a dict with the expected keys and value types."""

    def test_keys_are_block_size_and_enc_tile_size(self):
        out = auto_select_tile_sizes(
            n_enc=1000, n_dec=500, n_pos=100, memory_budget_bytes=int(1e9)
        )
        assert set(out) == {"block_size", "enc_tile_size"}
        assert isinstance(out["block_size"], int)
        assert out["enc_tile_size"] is None or isinstance(out["enc_tile_size"], int)


class TestAutoSelectBlockSize:
    """``block_size`` responds to budget, bounded by n_dec and floored at 1."""

    def test_block_size_never_exceeds_n_dec(self):
        # Generous budget would permit a huge block_size, but n_dec caps.
        out = auto_select_tile_sizes(
            n_enc=100,
            n_dec=64,
            n_pos=100,
            memory_budget_bytes=int(1e12),
        )
        assert out["block_size"] <= 64
        assert out["block_size"] == 64  # cap is binding

    def test_block_size_shrinks_with_tight_budget(self):
        # Tight budget: 12 KB. mark-kernel per-block = 3 * n_enc * block_size * 4.
        # For n_enc = 10000, budget = 12000 → block_size = 12000 // (3*10000*4)
        # = 0 → floor(1).
        out = auto_select_tile_sizes(
            n_enc=10000,
            n_dec=5000,
            n_pos=100,
            memory_budget_bytes=12_000,
        )
        assert out["block_size"] == 1

    def test_block_size_scales_linearly_with_budget(self):
        # Doubling the budget should double block_size (modulo the n_dec cap
        # and integer-division rounding).  Pick budgets that divide exactly
        # to avoid ±1 ULP flakiness from floor-division.
        n_enc = 10_000
        divisor = 3 * n_enc * 4  # budget // divisor == block_size
        small = auto_select_tile_sizes(
            n_enc=n_enc,
            n_dec=10_000,
            n_pos=100,
            memory_budget_bytes=divisor * 100,
        )
        large = auto_select_tile_sizes(
            n_enc=n_enc,
            n_dec=10_000,
            n_pos=100,
            memory_budget_bytes=divisor * 200,
        )
        assert small["block_size"] == 100
        assert large["block_size"] == 200
        assert large["block_size"] == 2 * small["block_size"]

    def test_block_size_floor_at_one(self):
        # Pathological: zero-byte budget.  Must not return 0 (which would
        # break dynamic_slice(block_size)).
        out = auto_select_tile_sizes(
            n_enc=10_000_000, n_dec=5000, n_pos=100, memory_budget_bytes=1
        )
        assert out["block_size"] == 1

    def test_block_size_formula(self):
        # budget // (3 * n_enc * 4); check an exact integer case.
        n_enc = 1000
        budget = 3 * n_enc * 4 * 2048  # allows block_size = 2048
        out = auto_select_tile_sizes(
            n_enc=n_enc,
            n_dec=10_000,
            n_pos=100,
            memory_budget_bytes=budget,
        )
        assert out["block_size"] == 2048


class TestAutoSelectEncTileSize:
    """``enc_tile_size`` is None when the full kernel fits, otherwise ≥ 256."""

    def test_none_when_position_kernel_fits(self):
        # Trigger: n_enc × n_pos × 4 × 3 > budget?
        # 10000 * 100 * 4 * 3 = 12 MB < 100 MB budget → no tiling.
        out = auto_select_tile_sizes(
            n_enc=10000, n_dec=5000, n_pos=100, memory_budget_bytes=int(1e8)
        )
        assert out["enc_tile_size"] is None

    def test_set_when_position_kernel_overflows_budget(self):
        # 1e6 * 100 * 4 * 3 = 1.2 GB > 100 MB budget → tiling required.
        out = auto_select_tile_sizes(
            n_enc=1_000_000,
            n_dec=5000,
            n_pos=100,
            memory_budget_bytes=int(1e8),
        )
        assert out["enc_tile_size"] is not None
        assert out["enc_tile_size"] >= _AUTO_TILE_MIN_ENC_TILE

    def test_enc_tile_size_floor_at_min(self):
        # Tiny budget → enc_tile_size formula gives ~0 but the floor holds.
        out = auto_select_tile_sizes(
            n_enc=1_000_000,
            n_dec=5000,
            n_pos=100,
            memory_budget_bytes=1,
        )
        assert out["enc_tile_size"] is not None
        assert out["enc_tile_size"] == _AUTO_TILE_MIN_ENC_TILE

    def test_enc_tile_size_formula(self):
        # When not at the floor: enc_tile = budget // (3 * n_pos * 4).
        # Pick n_pos = 100, budget = 3 * 100 * 4 * 4096 = 4.915 MB → 4096.
        # Trigger: need n_enc * n_pos * 4 * 3 > budget → n_enc > 4096.
        n_pos = 100
        budget = 3 * n_pos * 4 * 4096
        out = auto_select_tile_sizes(
            n_enc=20_000, n_dec=5000, n_pos=n_pos, memory_budget_bytes=budget
        )
        assert out["enc_tile_size"] == 4096

    def test_trigger_consistent_with_block_size_factor(self):
        """The enc_tile trigger uses the same 3x intermediate factor as block_size.

        Regression guard against the asymmetry the code-review flagged: if
        the trigger only checks ``n_enc * n_pos * 4 > budget``, a workload
        where the position kernel alone just barely fits but the 3-tensor
        working set overflows would silently go OOM at runtime.  With the
        symmetric trigger, such a workload triggers enc_tile_size.
        """
        # Sized so the raw kernel fits but the 3-tensor set doesn't:
        #   raw = n_enc * n_pos * 4 = 500 MB (< 1 GB budget)
        #   3x  = 1.5 GB (> 1 GB budget) → must tile
        n_enc, n_pos = 1_250_000, 100
        budget = int(1e9)
        assert n_enc * n_pos * 4 < budget  # raw kernel fits
        assert n_enc * n_pos * 4 * 3 > budget  # working set does not
        out = auto_select_tile_sizes(
            n_enc=n_enc, n_dec=1000, n_pos=n_pos, memory_budget_bytes=budget
        )
        assert out["enc_tile_size"] is not None


class TestAutoSelectEdgeCases:
    """Degenerate inputs still produce valid (static-traceable) values."""

    def test_n_enc_one(self):
        out = auto_select_tile_sizes(
            n_enc=1, n_dec=100, n_pos=50, memory_budget_bytes=int(1e8)
        )
        assert out["block_size"] == 100  # bounded by n_dec
        assert out["enc_tile_size"] is None

    def test_n_dec_one(self):
        out = auto_select_tile_sizes(
            n_enc=1000, n_dec=1, n_pos=50, memory_budget_bytes=int(1e8)
        )
        assert out["block_size"] == 1  # bounded by n_dec

    def test_negative_inputs_clamped_to_one(self):
        """Defense-in-depth: non-positive counts are treated as 1."""
        out = auto_select_tile_sizes(
            n_enc=0, n_dec=-5, n_pos=0, memory_budget_bytes=int(1e8)
        )
        assert out["block_size"] == 1


class TestAutoSelectDefaultBudget:
    """Default budget discovery falls back gracefully when JAX can't report."""

    def test_uses_gpu_bytes_limit_when_available(self):
        fake_device = mock.Mock()
        fake_device.memory_stats.return_value = {"bytes_limit": int(8e9)}
        with mock.patch.object(jax, "devices", return_value=[fake_device]):
            budget = _default_memory_budget_bytes()
        expected = int(8e9 * _AUTO_TILE_MEMORY_BUDGET_FRACTION)
        assert budget == expected

    def test_falls_back_when_memory_stats_missing(self):
        """Older JAX / non-GPU backends may not expose memory_stats."""
        fake_device = mock.Mock()
        fake_device.memory_stats.side_effect = AttributeError(
            "memory_stats not supported on this device"
        )
        with mock.patch.object(jax, "devices", return_value=[fake_device]):
            budget = _default_memory_budget_bytes()
        assert budget == _AUTO_TILE_DEFAULT_BUDGET_BYTES

    def test_falls_back_when_memory_stats_returns_empty(self):
        fake_device = mock.Mock()
        fake_device.memory_stats.return_value = {}
        with mock.patch.object(jax, "devices", return_value=[fake_device]):
            budget = _default_memory_budget_bytes()
        assert budget == _AUTO_TILE_DEFAULT_BUDGET_BYTES

    def test_falls_back_when_bytes_limit_is_zero(self):
        """Some CPU-only JAX builds report 0 for bytes_limit."""
        fake_device = mock.Mock()
        fake_device.memory_stats.return_value = {"bytes_limit": 0}
        with mock.patch.object(jax, "devices", return_value=[fake_device]):
            budget = _default_memory_budget_bytes()
        assert budget == _AUTO_TILE_DEFAULT_BUDGET_BYTES


class TestAutoSelectProductionScale:
    """On a typical production workload, the heuristic picks reasonable values."""

    def test_64_tetrode_workload_on_8gb_gpu(self):
        """8 GB GPU × 25% = 2 GB. Per-electrode size: 20k enc × 10k dec × 200 pos."""
        out = auto_select_tile_sizes(
            n_enc=20_000,
            n_dec=10_000,
            n_pos=200,
            memory_budget_bytes=int(2e9),
        )
        # Block should be large enough to amortize GEMM overhead but
        # not so large that it blows memory.
        assert 100 < out["block_size"] <= 10_000
        # Position kernel is 20k * 200 * 4 = 16 MB, well under 2 GB.
        assert out["enc_tile_size"] is None

    def test_80gb_a100_doesnt_blow_up(self):
        """On an A100 with 20 GB usable (25% of 80 GB), block_size is capped by n_dec."""
        out = auto_select_tile_sizes(
            n_enc=20_000,
            n_dec=10_000,
            n_pos=200,
            memory_budget_bytes=int(20e9),
        )
        assert out["block_size"] == 10_000
        assert out["enc_tile_size"] is None


class TestAutoSelectConstants:
    """The constants exposed from the module are the ones the heuristic uses."""

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("_AUTO_TILE_MEMORY_BUDGET_FRACTION", float),
            ("_AUTO_TILE_DEFAULT_BUDGET_BYTES", int),
            ("_AUTO_TILE_MIN_ENC_TILE", int),
            ("_AUTO_TILE_INTERMEDIATE_COPIES", int),
        ],
    )
    def test_constant_types(self, name: str, expected_type: type):
        import non_local_detector.likelihoods.clusterless_kde_log as mod

        assert isinstance(getattr(mod, name), expected_type)

    def test_memory_budget_fraction_is_sensible(self):
        assert 0.0 < _AUTO_TILE_MEMORY_BUDGET_FRACTION <= 1.0

    def test_intermediate_copies_positive(self):
        assert _AUTO_TILE_INTERMEDIATE_COPIES >= 1

    def test_min_enc_tile_at_least_one(self):
        assert _AUTO_TILE_MIN_ENC_TILE >= 1
