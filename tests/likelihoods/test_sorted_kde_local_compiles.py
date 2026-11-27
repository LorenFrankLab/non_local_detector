"""The sorted-spikes KDE Local likelihood compiles a few times per decode."""

from collections import Counter
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import non_local_detector.likelihoods.common.kernel_density as kernel_density
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import sorted_spikes_kde
from non_local_detector.likelihoods.common import (
    _KDE_BLOCK_KERNEL_BYTES,
    EPS,
    KDEModel,
    kde_log_density,
    poisson,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from tests.conftest import precision_mode
from tests.likelihoods.conftest import count_compilations

pytestmark = pytest.mark.unit

DT = 0.01


def recording(n_units, duration=60.0):
    """A 2-D arena sweep whose occupancy covers only x < 60 cm.

    Rows far beyond x = 60 cm are deep occupancy tails, where the log-space
    evaluation runs. Every unit has a different spike count, spread over
    more than seven octaves, and fires within 15 cm of its own center.
    """
    rng = np.random.default_rng(n_units)
    time = np.arange(0.0, duration, DT)
    position = np.column_stack(
        (50 + 45 * np.sin(time / 3.1), 50 + 45 * np.cos(time / 4.7))
    )
    environment = Environment(
        place_bin_size=5.0, position_range=((0.0, 100.0), (0.0, 100.0))
    ).fit_place_grid(position, infer_track_interior=False)
    weights = (position[:, 0] < 60.0).astype(float)
    counts = np.unique(np.geomspace(17, 3_000, n_units).astype(int))
    assert counts.size == n_units
    encoded = np.flatnonzero(weights > 0)
    centers = position[rng.choice(encoded, n_units)]
    spike_times = []
    for center, n in zip(centers, counts, strict=True):
        in_field = time[np.linalg.norm(position - center, axis=1) < 15.0]
        spike_times.append(np.sort(rng.choice(in_field, n) + rng.uniform(0, DT, n)))
    return environment, time, position, spike_times, weights


def fit(n_units, duration=60.0, **kwargs):
    environment, time, position, spike_times, weights = recording(n_units, duration)
    model = fit_sorted_spikes_kde_encoding_model(
        time,
        position,
        spike_times,
        environment,
        weights=weights,
        disable_progress_bar=True,
        **kwargs,
    )
    return model, time, position, spike_times


def predict_local(model, time, position, spike_times, row_slice=None):
    return predict_sorted_spikes_kde_log_likelihood(
        time,
        position,
        spike_times,
        time_edges=np.arange(len(time) + 1) * DT,
        is_local=True,
        row_slice=row_slice,
        **model,
    )


def local_kernel_compiles(compiled):
    """The Local likelihood's compiled kernels, without the occupancy density."""
    return [
        name
        for name in compiled
        if "kde_log_densities" in name or "_local_kde_log_likelihood" in name
    ]


def test_local_kernels_compile_independent_of_unit_count():
    counts = []
    for n_units in (8, 40):
        model, time, position, spike_times = fit(n_units, position_std=4.0)
        jax.clear_caches()
        with count_compilations() as compiled:
            predict_local(model, time, position, spike_times)
        counts.append(len(local_kernel_compiles(compiled)))
    # The tiled density pass, its log-space repair and the Poisson terms.
    assert counts[0] == counts[1] <= 3, counts


def test_local_chunks_share_compiled_kernels():
    """Twelve detector-like chunks of 500 rows and a ragged last chunk of
    200, with and without occupancy-tail rows, reuse a few kernels."""
    model, time, position, spike_times = fit(40, duration=62.0, position_std=4.0)
    centers = (np.arange(len(time)) + 0.5) * DT
    log_occupancy = np.asarray(
        kde_log_density(
            model["occupancy_model"],
            np.column_stack([np.interp(centers, time, p) for p in position.T]),
        )
    )
    bounds = [*range(0, len(time), 500), len(time)]
    chunks = [slice(a, b) for a, b in pairwise(bounds)]
    assert chunks[-1].stop - chunks[-1].start == 200
    has_tail = [(log_occupancy[chunk] < np.log(EPS) - 10).any() for chunk in chunks]
    no_tail = [(log_occupancy[chunk] > np.log(EPS) + 10).all() for chunk in chunks]
    assert any(has_tail), has_tail
    assert any(no_tail), no_tail

    jax.clear_caches()
    with count_compilations() as compiled:
        for chunk in chunks:
            predict_local(model, time, position, spike_times, chunk)
    names = Counter(local_kernel_compiles(compiled))
    # The density pass and the Poisson terms compile once per row size, the
    # chunks' and the ragged chunk's, whatever the tail rows. Before, the
    # Local kernel compiled once per padded number of tail rows: 7 times.
    assert names["jit(_tiled_scaled_kde_log_densities)"] == 2, names
    assert names["jit(_local_kde_log_likelihood)"] == 2, names
    # The log-space repair compiles once per power-of-two flagged-set size.
    assert names["jit(_paired_log_space_kde_log_densities)"] <= 6, names


def test_units_never_mix_in_the_batched_local_pass(monkeypatch):
    """Each unit's counts are scored against its own density and mean rate:
    the full Local likelihood equals the sum of single-unit likelihoods.

    The default ``block_size=10_000`` evaluates blocks of 20,000 rows in
    two blocks of points, and the 24 tiles in two steps; the last block is
    ragged and padded. The units' fields, rates and
    spike counts all differ, so pairing a unit's counts with another's
    density or rate would change the result by orders of magnitude more
    than rounding.
    """
    monkeypatch.setattr(poisson, "COUNT_BLOCK_BYTES", 25_000 * 6 * 8)
    sizes = []
    tiled_rows = kernel_density._tiled_kde_rows

    def record_sizes(block_size, points, n_tiles, *args):
        sizes.append(
            (points.shape[0], n_tiles, *tiled_rows(block_size, points, n_tiles, *args))
        )
        return sizes[-1][2:]

    repairs = []
    repair = kernel_density._paired_log_space_kde_log_densities

    def record_repair(*args, **kwargs):
        repairs.append(kwargs["n_models"])
        return repair(*args, **kwargs)

    monkeypatch.setattr(kernel_density, "_tiled_kde_rows", record_sizes)
    monkeypatch.setattr(
        kernel_density, "_paired_log_space_kde_log_densities", record_repair
    )
    model, time, position, spike_times = fit(6, duration=450.0, block_size=10_000)
    sizes.clear()
    full = predict_local(model, time, position, spike_times)
    # Without the point-block query (tiled_kde_point_rows), which sizes
    # many points.
    sizes = [size for size in sizes if size[0] < np.iinfo(np.int32).max]
    assert [n_points for n_points, *_ in sizes] == [20_000, 20_000, 5_120]
    for n_points, n_tiles, rows, batch_size, _ in sizes[:2]:
        assert rows < n_points
        assert batch_size < n_tiles
    assert 6 in repairs

    expected = jnp.zeros_like(full)
    for unit in range(6):
        single = dict(model)
        single["marginal_models"] = model["marginal_models"][unit : unit + 1]
        single["mean_rates"] = model["mean_rates"][unit : unit + 1]
        single["place_fields"] = model["place_fields"][unit : unit + 1]
        expected = expected + predict_local(
            single, time, position, spike_times[unit : unit + 1]
        )
    np.testing.assert_allclose(full, expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("n_dims", [1, 2])
def test_local_workspace_stays_within_the_kernel_budget(monkeypatch, n_dims, x64):
    """Each Local block of 180 units of 14 to 42,000 samples
    (``block_size=10_000``), at the sizes a prediction calls its kernels with:
    every ``(n_units, n_rows)`` array takes at most a 64th of the kernel
    budget, the density pass's workspace, arguments and outputs together fit
    the budget, and the Poisson terms hold at most five such arrays. A 4x
    looser block, row padding or density-pass block exceeds them."""
    unit_bytes = _KDE_BLOCK_KERNEL_BYTES // 64
    calls = {}
    module_of = {
        "_tiled_scaled_kde_log_densities": kernel_density,
        "_local_kde_log_likelihood": sorted_spikes_kde,
    }

    def capture(name, result):
        original = getattr(module_of[name], name)

        def record(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return result(*args, **kwargs)

        monkeypatch.setattr(module_of[name], name, record)
        return original

    with precision_mode(x64):
        rng = np.random.default_rng(0)
        sizes = np.geomspace(14, 42_000, 180).astype(int)
        models = [
            KDEModel(std=3.0, block_size=10_000).fit(
                jnp.asarray(rng.uniform(0, 100, (size, n_dims)))
            )
            for size in sizes
        ]
        itemsize = jnp.zeros(()).dtype.itemsize
        # Several full blocks and a ragged one, even with 4x larger blocks.
        n_rows = 9 * unit_bytes // (len(models) * itemsize) + 100
        edges = np.arange(n_rows + 1) * 0.002
        scaled = capture(
            "_tiled_scaled_kde_log_densities",
            lambda points, *args, **kwargs: (
                jnp.zeros((len(models), points.shape[0])),
                jnp.zeros((len(models), points.shape[0]), bool),
            ),
        )
        local = capture(
            "_local_kde_log_likelihood",
            lambda log_marginals, *args: jnp.zeros(log_marginals.shape[1]),
        )
        environment = Environment(
            place_bin_size=5.0, position_range=((0.0, 100.0),) * n_dims
        ).fit_place_grid(np.array([[0.0] * n_dims, [100.0] * n_dims]))
        n_bins = environment.place_bin_centers_.shape[0]
        predict_sorted_spikes_kde_log_likelihood(
            np.array([0.0, edges[-1]]),
            np.array([[10.0] * n_dims, [90.0] * n_dims]),
            [np.sort(rng.uniform(0, edges[-1], 20)) for _ in models],
            environment=environment,
            marginal_models=models,
            occupancy_model=models[-1],
            occupancy=None,
            mean_rates=np.full(len(models), 2.0),
            place_fields=np.ones((len(models), n_bins)),
            no_spike_part_log_likelihood=np.ones(n_bins),
            is_track_interior=np.ones(n_bins, bool),
            disable_progress_bar=True,
            is_local=True,
            time_edges=edges,
        )
        assert len(calls["_tiled_scaled_kde_log_densities"]) >= 3

        def distinct(name):
            """One call per argument shape."""
            by_shape = {
                str(jax.tree_util.tree_map(jnp.shape, args)): (args, kwargs)
                for args, kwargs in calls[name]
            }
            return list(by_shape.values())

        def footprint(function, args, kwargs):
            analysis = function.lower(*args, **kwargs).compile().memory_analysis()
            return (
                analysis.temp_size_in_bytes
                + analysis.argument_size_in_bytes
                + analysis.output_size_in_bytes
            )

        for args, kwargs in distinct("_tiled_scaled_kde_log_densities"):
            assert len(models) * args[0].shape[0] * itemsize <= unit_bytes
            assert footprint(scaled, args, kwargs) <= _KDE_BLOCK_KERNEL_BYTES
        for args, kwargs in distinct("_local_kde_log_likelihood"):
            assert footprint(local, args, kwargs) <= 5 * unit_bytes
