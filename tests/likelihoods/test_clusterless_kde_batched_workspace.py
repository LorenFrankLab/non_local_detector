"""The batched clusterless KDE passes keep their compiled workspace within the kernel budget."""

import jax
import jax.numpy as jnp
import pytest

from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.common import (
    _KDE_BLOCK_KERNEL_BYTES,
    _KDE_SAMPLE_TILE,
    padded_block_count,
    padded_tile_count,
)
from tests.conftest import precision_mode

pytestmark = pytest.mark.unit

N_FEATURES = 4
# A block size this large leaves the kernel budget to bound the blocks.
BLOCK_SIZE = 10_000_000


def sds(*shape, dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape, dtype)


def tiles_and_terms(samples, n_position_dims):
    """Shapes of every electrode's tiles and kernel settings."""
    per_electrode = [max(1, -(-n // _KDE_SAMPLE_TILE)) for n in samples]
    n_tiles = padded_tile_count(sum(per_electrode))
    n_electrodes = len(samples)
    tiles = clusterless_kde._EncodingTiles(
        sds(n_tiles, _KDE_SAMPLE_TILE, n_position_dims),
        sds(n_tiles, _KDE_SAMPLE_TILE, N_FEATURES),
        sds(n_tiles, _KDE_SAMPLE_TILE),
        sds(n_tiles, dtype=jnp.int32),
        sds(n_electrodes + 1, dtype=jnp.int32),
    )
    terms = clusterless_kde._ElectrodeTerms(
        sds(n_electrodes, N_FEATURES),
        sds(n_electrodes, N_FEATURES, dtype=bool),
        sds(n_electrodes),
        sds(n_electrodes),
    )
    return tiles, terms, per_electrode


def spike_blocks(spikes, rows, n_position_dims=None):
    n_blocks = padded_block_count(sum(-(-n // rows) for n in spikes))
    local = {}
    if n_position_dims is not None:
        local = {
            "positions": sds(n_blocks, rows, n_position_dims),
            "log_occupancy": sds(n_blocks, rows),
        }
    return clusterless_kde._SpikeBlocks(
        sds(n_blocks, rows, N_FEATURES),
        sds(n_blocks, rows),
        sds(n_blocks, rows, dtype=jnp.int32),
        sds(len(spikes) + 1, dtype=jnp.int32),
        **local,
    )


@pytest.mark.parametrize(
    ("spikes", "samples", "n_bins", "n_position_dims", "n_rows"),
    [
        ((100_000,) * 2, (200_000,) * 2, 2_500, 2, 50_000),
        ((2**18,) * 8, (1_000,) * 8, 300, 1, 20_000),
        ((20_000,) * 2, (5_000,) * 2, 40_000, 2, 2_000),
    ],
    ids=["2-d", "1-d", "40,000 bins"],
)
def test_nonlocal_pass_workspace_stays_within_the_kernel_budget(
    spikes, samples, n_bins, n_position_dims, n_rows
):
    """Sized as predict sizes it, the non-local pass holds the widest
    electrode's position factors and at most the kernel budget for one
    GEMM's buffers and its group of tiles; sized for a budget four times
    larger, it exceeds that. On 40,000 bins a group of 8 tiles alone would
    exceed it."""
    tiles, terms, per_electrode = tiles_and_terms(samples, n_position_dims)
    widest = max(per_electrode)
    rows, group_blocks, group_tiles = clusterless_kde._nonlocal_sizes(
        spikes, per_electrode, BLOCK_SIZE, n_bins, 4, widest
    )
    with precision_mode(False):
        compiled = clusterless_kde._add_log_mark_intensities.lower(
            sds(n_rows, n_bins),
            spike_blocks(spikes, rows),
            tiles,
            terms,
            sds(len(spikes)),
            sds(n_position_dims),
            sds(n_bins, n_position_dims),
            sds(n_bins),
            widest_tiles=widest,
            group_blocks=group_blocks,
            group_tiles=group_tiles,
            pos_tile_size=None,
            indices_are_sorted=True,
        ).compile()
    workspace = compiled.memory_analysis().temp_size_in_bytes
    factors = widest * _KDE_SAMPLE_TILE * n_bins * 4
    assert workspace <= factors + _KDE_BLOCK_KERNEL_BYTES, (
        f"{workspace / 2**20:.1f} MiB, factors {factors / 2**20:.1f} MiB, "
        f"{rows} rows, {group_tiles} tiles per group"
    )


def test_local_pass_workspace_stays_within_the_kernel_budget():
    """Sized as predict sizes it, the Local pass, whose groups of blocks of
    the most spikes the budget allows meet up to 2,048 samples at once, stays
    within the kernel budget for two electrodes of 2**18 spikes and 2,000,000
    samples each. The spikes fill whole blocks at that size and at four times
    it, which exceeds the budget."""
    spikes, samples, n_position_dims, n_rows = (
        (2**18,) * 2,
        (2_000_000,) * 2,
        2,
        20_000,
    )
    tiles, terms, per_electrode = tiles_and_terms(samples, n_position_dims)
    rows, group_blocks, group_tiles = clusterless_kde._local_sizes(
        spikes, per_electrode, BLOCK_SIZE, max(per_electrode)
    )
    with precision_mode(False):
        compiled = clusterless_kde._add_local_terms.lower(
            sds(n_rows),
            spike_blocks(spikes, rows, n_position_dims),
            tiles,
            terms,
            sds(len(spikes)),
            sds(n_position_dims),
            group_blocks=group_blocks,
            group_tiles=group_tiles,
            indices_are_sorted=True,
        ).compile()
    workspace = compiled.memory_analysis().temp_size_in_bytes
    assert workspace <= _KDE_BLOCK_KERNEL_BYTES, (
        f"{workspace / 2**20:.1f} MiB, {rows} rows"
    )
