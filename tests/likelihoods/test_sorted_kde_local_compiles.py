"""The sorted-spikes KDE Local likelihood compiles a few times per decode."""

import re
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

    With the default ``block_size=10_000`` and a spike-count block of
    24,960 rows, the density pass evaluates blocks of 24,960 rows in three
    blocks of 8,320 points, and the 24 tiles in two steps; the last block
    (20,040 rows) is padded to three point blocks. The units' fields, rates and
    spike counts all differ, so pairing a unit's counts with another's
    density or rate would change the result by orders of magnitude more
    than rounding.
    """
    monkeypatch.setattr(poisson, "COUNT_BLOCK_BYTES", 25_000 * 6 * 8)
    sizes = []
    scaled_pass = kernel_density._tiled_scaled_kde_log_densities

    def record_sizes(points, tile_samples, *args, rows, batch_size):
        sizes.append((points.shape[0], tile_samples.shape[0], rows, batch_size))
        return scaled_pass(
            points, tile_samples, *args, rows=rows, batch_size=batch_size
        )

    repairs = []
    repair = kernel_density._paired_log_space_kde_log_densities

    def record_repair(*args, **kwargs):
        repairs.append(kwargs["n_models"])
        return repair(*args, **kwargs)

    monkeypatch.setattr(kernel_density, "_tiled_scaled_kde_log_densities", record_sizes)
    monkeypatch.setattr(
        kernel_density, "_paired_log_space_kde_log_densities", record_repair
    )
    model, time, position, spike_times = fit(6, duration=450.0, block_size=10_000)
    sizes.clear()
    full = predict_local(model, time, position, spike_times)
    assert [n_points for n_points, *_ in sizes] == [24_960, 24_960]
    for _, n_tiles, rows, batch_size in sizes:
        assert rows == 8_320
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


@pytest.mark.parametrize("block_size", [300, 10_000])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("n_dims", [1, 2])
def test_local_workspace_stays_within_the_kernel_budget(
    monkeypatch, n_dims, x64, block_size
):
    """Each Local block of 180 units of 14 to 42,000 samples, at the sizes a
    prediction calls its kernels with: the density pass's and the Poisson
    terms' workspace, arguments and outputs each fit the kernel budget, and
    a full block is at most the spike-count block. At ``block_size=300`` in
    float32 the spike-count block binds (23,296 rows, less than one row per
    point block); elsewhere the budget does. A 4x looser count of the pass's (unit, row) arrays, or a 4x
    larger spike-count block, exceeds the budget."""
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
            KDEModel(std=3.0, block_size=block_size).fit(
                jnp.asarray(rng.uniform(0, 100, (size, n_dims)))
            )
            for size in sizes
        ]
        # Two spike-count blocks and a ragged one.
        count_rows = poisson._count_block_rows(len(models))
        n_rows = 2 * count_rows + 100
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
            occupancy_model=models[0],
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
            assert args[0].shape[0] <= count_rows
            assert footprint(scaled, args, kwargs) <= _KDE_BLOCK_KERNEL_BYTES
        for args, kwargs in distinct("_local_kde_log_likelihood"):
            assert footprint(local, args, kwargs) <= _KDE_BLOCK_KERNEL_BYTES
        if block_size == 300 and not x64:
            # Whole point blocks lose fewer rows than there are blocks.
            largest = max(
                args[0].shape[0] for args, _ in calls["_tiled_scaled_kde_log_densities"]
            )
            assert largest > count_rows - count_rows // 200


def test_bracketing_keys_match_a_search_for_each_point():
    """Both searches (per key for models with fewer keys than points, per
    point otherwise) find the keys a direct search of each point finds,
    with ties, repeated points, infinite keys and NaN or infinite points."""
    rng = np.random.default_rng(3)
    points = np.round(rng.uniform(-5, 5, 200), 1).astype(np.float32)
    points[[3, 50]] = np.nan
    points[7], points[8] = np.inf, -np.inf
    samples = [
        np.round(rng.uniform(-6, 6, (n, 1)), 1).astype(np.float32)
        for n in (0, 1, 30, 199, 200, 1_000)
    ]
    weights = [np.where(rng.uniform(size=len(s)) < 0.1, 0.0, 1.0) for s in samples]
    keys = kernel_density._sorted_model_keys(samples, weights)
    left, right = kernel_density._bracketing_keys(points, keys)
    for model, model_keys in enumerate(keys):
        if model_keys.size == 0:
            assert np.isinf(left[model]).all()
            assert np.isinf(right[model]).all()
            continue
        index = np.searchsorted(model_keys, points)
        last = model_keys.size - 1
        np.testing.assert_array_equal(
            left[model], model_keys[np.clip(index - 1, 0, last)]
        )
        np.testing.assert_array_equal(right[model], model_keys[np.clip(index, 0, last)])


def lowered_op_count(function, args, kwargs):
    """Operations in a jitted function's lowered program, whatever its shapes."""
    return len(re.findall(r"stablehlo\.\w+", function.lower(*args, **kwargs).as_text()))


def test_local_kernel_graphs_do_not_grow_with_unit_count(monkeypatch):
    """The Local kernels lower to the same program for 8 and 40 units of
    spike counts spread over seven octaves. Before, the Local kernel held an
    evaluation loop per padded spike count, so its graph grew with them."""
    names = {
        "_tiled_scaled_kde_log_densities": kernel_density,
        "_local_kde_log_likelihood": sorted_spikes_kde,
    }
    op_counts = []
    for n_units in (8, 40):
        model, time, position, spike_times = fit(n_units, position_std=4.0)
        calls = {}
        with monkeypatch.context() as patch:
            for name, module in names.items():
                original = getattr(module, name)

                def record(
                    *args, _name=name, _original=original, _calls=calls, **kwargs
                ):
                    _calls.setdefault(_name, (_original, args, kwargs))
                    return _original(*args, **kwargs)

                patch.setattr(module, name, record)
            predict_local(model, time, position, spike_times)
        op_counts.append(
            {name: lowered_op_count(*call) for name, call in calls.items()}
        )
    assert op_counts[0] == op_counts[1], op_counts


def flags_from_scaled_pass(monkeypatch, models, points, log_floor):
    """The flags the scaled pass returns, before any log-space repair."""
    flags = []
    original = kernel_density._tiled_scaled_kde_log_densities

    def record(*args, **kwargs):
        values, needs_log_space = original(*args, **kwargs)
        flags.append(np.asarray(needs_log_space))
        return values, needs_log_space

    with monkeypatch.context() as patch:
        patch.setattr(kernel_density, "_tiled_scaled_kde_log_densities", record)
        kernel_density.kde_log_density_rows(models, points, log_floor=log_floor)
    (needs_log_space,) = flags
    return needs_log_space


def test_log_floor_unflags_only_points_whose_density_is_below_it(monkeypatch):
    """A flagged point is left unflagged only where its true density (from
    the single-model log-space evaluation) is below its floor. In 2-D the
    scaled pass shifts by the kernel's peak, so points about 12 bandwidths
    or more from all of a model's samples are flagged (1-D shifts by the
    nearest sample, and rarely flags). Points 10 to 15 bandwidths away put
    true densities just below the bound, within its factor of four; with
    floors from 3 below to 3 above each true log density, every unflagged
    point's floor is above it, and some are unflagged. NaN and ``-inf``
    floors leave the flags unchanged."""
    rng = np.random.default_rng(11)
    models = [
        KDEModel(std=1.0, block_size=10_000).fit(
            jnp.asarray(rng.normal(0.0, 0.05, (size, 2)), jnp.float32)
        )
        for size in (1, 3, 20)
    ]
    radius = rng.uniform(10.0, 15.0, 2_000)
    angle = rng.uniform(0.0, 2 * np.pi, radius.size)
    points = jnp.asarray(
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle))), jnp.float32
    )
    no_floor = flags_from_scaled_pass(monkeypatch, models, points, None)
    assert no_floor.any()
    n_unflagged = 0
    for unit, model in enumerate(models):
        true = np.asarray(kde_log_density(model, points), np.float64)
        margin = rng.uniform(-3.0, 3.0, true.size)
        unit_terms = np.full(len(models), np.inf)
        unit_terms[unit] = 0.0
        flags = flags_from_scaled_pass(
            monkeypatch, models, points, (unit_terms, true + margin)
        )
        unflagged = no_floor[unit] & ~flags[unit]
        assert (margin[unflagged] > 0).all()
        n_unflagged += unflagged.sum()
        for floor in (np.nan, -np.inf):
            flags = flags_from_scaled_pass(
                monkeypatch, models, points, (unit_terms, np.full(true.size, floor))
            )
            np.testing.assert_array_equal(flags[unit], no_floor[unit])
    assert n_unflagged > 0


def test_log_floor_skips_repairs_without_changing_the_local_likelihood(monkeypatch):
    """On a 2-D recording with deep occupancy tails, the Local likelihood is
    bit-identical with and without the rate floor, which removes some but
    not all of the log-space repairs."""
    model, time, position, spike_times = fit(40, position_std=4.0)
    rows_with_floor = sorted_spikes_kde.kde_log_density_rows
    items = []
    repair = kernel_density._paired_log_space_kde_log_densities

    def record_repair(*args, **kwargs):
        items.append(int(np.asarray(args[2])[-1]))
        return repair(*args, **kwargs)

    monkeypatch.setattr(
        kernel_density, "_paired_log_space_kde_log_densities", record_repair
    )
    with_floor = np.asarray(predict_local(model, time, position, spike_times))
    items_with_floor = sum(items)
    items.clear()
    monkeypatch.setattr(
        sorted_spikes_kde,
        "kde_log_density_rows",
        lambda *args, log_floor=None, **kwargs: rows_with_floor(*args, **kwargs),
    )
    without_floor = np.asarray(predict_local(model, time, position, spike_times))
    items_without_floor = sum(items)
    np.testing.assert_array_equal(with_floor, without_floor)
    assert 0 < items_with_floor < items_without_floor, (
        items_with_floor,
        items_without_floor,
    )


def test_log_floor_leaves_every_local_rate_bit_identical():
    """The rates the Local kernel forms from the floored densities equal,
    bit for bit, those from fully repaired densities, at every unit and row
    of the same 2-D recording: a skipped density's rate is the clip either
    way. (The likelihood check above cannot see rates far below 1e-6 Hz.)"""
    model, _, position, _ = fit(40, position_std=4.0)
    points = jnp.asarray(position, jnp.float32)
    log_occupancy = np.asarray(kde_log_density(model["occupancy_model"], points))
    unit_floor = sorted_spikes_kde._unit_log_density_floor(model["mean_rates"])
    rates = []
    for log_floor in (None, (unit_floor, log_occupancy)):
        log_marginals = kernel_density.kde_log_density_rows(
            model["marginal_models"], points, log_floor=log_floor
        )
        rates.append(
            np.asarray(
                sorted_spikes_kde._kde_rates(
                    jnp.asarray(model["mean_rates"])[:, None],
                    jnp.asarray(log_marginals),
                    jnp.asarray(log_occupancy)[None, :],
                )
            )
        )
    np.testing.assert_array_equal(rates[0], rates[1])
    # Not vacuous: the clip is active at some units and rows.
    assert (rates[0] == sorted_spikes_kde._RATE_FLOOR_HZ).mean() > 0.01
