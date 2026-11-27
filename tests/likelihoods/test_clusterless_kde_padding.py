"""Padded, compiled log-space clusterless KDE kernels match the per-electrode formulation."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

import non_local_detector.likelihoods.common.kernel_density as kernel_density
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import (
    _log_joint_from_log_marginal,
    block_estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    log_kde_distance,
    mark_reference_scale,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    KDEModel,
    as_std_array,
    block_log_kde,
    get_position_at_time,
    padded_occupancy_count,
    padded_sample_count,
    padded_spike_count,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
)
from non_local_detector.likelihoods.mark_reference import (
    background_log_intensity,
)
from non_local_detector.time_edges import decode_bin_centers
from tests.conftest import precision_mode
from tests.likelihoods.conftest import (
    count_compilations,
    log_duration_per_spike,
    tolerance,
    training_mark_references,
)

pytestmark = pytest.mark.unit

# float32 bound for the padded kernels against the per-electrode formulation.
PER_ELECTRODE_FLOAT32 = {"rtol": 1e-5, "atol": 1e-4}


def fit_encoding(position_time, spike_times, spike_waveform_features, **params):
    """KDE encoding model plus the mark reference of its in-range training spikes."""
    encoding = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        **params,
    )
    reference = training_mark_references(
        position_time,
        spike_times,
        spike_waveform_features,
        mark_reference_scale(**params),
        params.get("weights"),
        params.get("position"),
    )
    return {**encoding, "mark_reference": reference}


def make_recording(x64, waveform_dims=(2, 4, 4)):
    """Electrodes with different encoding counts and mark dimensions."""
    rng = np.random.default_rng(31)
    env = Environment(place_bin_size=4.0, position_range=((0.0, 40.0), (0.0, 40.0)))
    position_time = np.arange(0, 30.0, 1 / 30)
    position = np.column_stack(
        (20 + 15 * np.sin(position_time / 3), 20 + 15 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    rates = (3.0, 12.0, 25.0)
    encoding_times = [np.sort(rng.uniform(0, 30, int(30 * r))) for r in rates]
    encoding_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(encoding_times, waveform_dims, strict=True)
    ]
    decode_times = [np.sort(rng.uniform(0, 4, int(4 * r))) for r in rates]
    decode_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(decode_times, waveform_dims, strict=True)
    ]
    edges = np.arange(0, 4.0 + 1e-9, 0.002)
    with precision_mode(x64):
        encoding = fit_encoding(
            position_time=position_time,
            position=position,
            spike_times=encoding_times,
            spike_waveform_features=encoding_marks,
            environment=env,
            position_std=3.0,
            waveform_std=24.0,
            block_size=16,
            disable_progress_bar=True,
        )
    return env, position_time, position, encoding, decode_times, decode_marks, edges


@pytest.fixture(scope="module", params=[False, True], ids=["float32", "float64"])
def recording(request):
    return request.param, make_recording(request.param)


def predict(recording, is_local, row_slice):
    _env, position_time, position, encoding, times, marks, edges = recording
    return predict_clusterless_kde_log_likelihood(
        position_time,
        position,
        times,
        marks,
        time_edges=edges,
        **encoding,
        is_local=is_local,
        row_slice=row_slice,
    )


def reference(recording, is_local, row_slice):
    """The per-electrode formulation from the public building blocks."""
    env, position_time, position, encoding, times, marks, edges = recording
    start, stop = row_slice.start, row_slice.stop
    durations = jnp.asarray(np.diff(edges)[start:stop])
    block_size = encoding["block_size"]
    position_std = jnp.asarray(encoding["position_std"])
    weights = encoding["encoding_weights"] or [None] * len(times)
    electrodes = zip(
        encoding["encoding_spike_waveform_features"],
        encoding["encoding_positions"],
        weights,
        encoding["mean_rates"],
        encoding["gpi_models"],
        encoding["mark_reference"],
        marks,
        times,
        strict=True,
    )

    def background(reference, dec_marks, selection):
        # log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS) + log q(mark) per spike.
        return jnp.asarray(
            background_log_intensity(
                reference, np.asarray(select_spike_rows(dec_marks, selection)), LOG_EPS
            )
        )

    if not is_local:
        bins = env.place_bin_centers_[env.is_track_interior_.ravel()]
        total = -durations[:, None] * encoding["summed_ground_process_intensity"]
        for (
            enc_marks,
            enc_pos,
            w,
            rate,
            _,
            reference,
            dec_marks,
            dec_times,
        ) in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            total = total + sum_spikes_into_rows(
                block_estimate_log_joint_mark_intensity(
                    select_spike_rows(dec_marks, selection),
                    enc_marks,
                    as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    encoding["occupancy"],
                    rate * RATE_REFERENCE_SECONDS,
                    log_kde_distance(bins, enc_pos, std=position_std),
                    block_size=block_size,
                    encoding_weights=w,
                    log_background=background(reference, dec_marks, selection),
                ),
                selection,
            )
    else:
        occupancy_model = encoding["occupancy_model"]
        positions = get_position_at_time(
            position_time, position, decode_bin_centers(edges, start, stop), env
        )
        occupancy = occupancy_model.predict(positions)
        total, expected = jnp.zeros(stop - start), jnp.zeros(stop - start)
        for (
            enc_marks,
            enc_pos,
            w,
            rate,
            gpi,
            reference,
            dec_marks,
            dec_times,
        ) in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            log_background = background(reference, dec_marks, selection)
            spike_positions = get_position_at_time(
                position_time, position, select_spike_rows(dec_times, selection), env
            )
            log_marginal = block_log_kde(
                jnp.concatenate(
                    (spike_positions, select_spike_rows(dec_marks, selection)), axis=1
                ),
                jnp.concatenate((enc_pos, enc_marks), axis=1),
                jnp.concatenate(
                    (
                        position_std,
                        as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    )
                ),
                block_size,
                w,
            )
            contribution = jnp.logaddexp(
                _log_joint_from_log_marginal(
                    log_marginal[None, :],
                    rate * RATE_REFERENCE_SECONDS,
                    occupancy_model.predict(spike_positions),
                )[0],
                log_background,
            )
            # No encoding mass means zero intensity, irrespective of a manually
            # supplied positive mean rate: the background explains the spike.
            if enc_pos.shape[0] == 0 or (w is not None and np.sum(w) == 0.0):
                contribution = log_background
            total = total + sum_spikes_into_rows(contribution, selection)
            expected = expected + rate * jnp.where(
                occupancy > 0,
                gpi.predict(positions) / jnp.where(occupancy > 0, occupancy, 1),
                0,
            )
        # Every electrode adds its background rate to the expected count.
        n_electrodes = len(encoding["mean_rates"])
        total = (total - durations * (expected + n_electrodes * RATE_EPS_HZ))[:, None]
    return (
        total
        + log_duration_per_spike(times, edges, row_slice, RATE_REFERENCE_SECONDS)[
            :, None
        ]
    )


@pytest.fixture(scope="module")
def full_range_references(recording):
    """``reference`` over every decoding row, per path; read-only.

    The formulation scores each row on its own, so its rows ``start:stop``
    are what ``reference`` computes for that range (bitwise, on these
    recordings). Slicing one full-range result computes it once per precision
    and path instead of once per row range, and keeps the expected rows
    independent of the row-range selection that ``predict`` is tested on.
    """
    x64, recording = recording
    edges = recording[-1]
    with precision_mode(x64):
        return {
            is_local: reference(recording, is_local, slice(0, len(edges) - 1))
            for is_local in (False, True)
        }


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("rows", [(0, 2000), (300, 333), (1990, 2000), (5, 6)])
def test_padded_kernels_match_the_per_electrode_formulation(
    recording, full_range_references, is_local, rows
):
    x64, recording = recording
    with precision_mode(x64):
        row_slice = slice(*rows)
        actual = predict(recording, is_local, row_slice)
        expected = full_range_references[is_local][row_slice]
        assert actual.shape == expected.shape
        assert actual.dtype == (jnp.float64 if x64 else jnp.float32)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(
            actual, expected, **tolerance(x64, PER_ELECTRODE_FLOAT32)
        )


@pytest.mark.parametrize("is_local", [False, True])
def test_degenerate_electrodes_match_the_per_electrode_formulation(is_local):
    """All-zero encoding weights, a zero mean rate and no encoding spikes."""
    env, position_time, position, encoding, times, marks, edges = make_recording(False)
    features = list(encoding["encoding_spike_waveform_features"])
    positions = list(encoding["encoding_positions"])
    weights = [
        np.ones(len(f)) if w is None else np.asarray(w)
        for f, w in zip(
            features,
            encoding["encoding_weights"] or [None] * len(features),
            strict=True,
        )
    ]
    weights[0] = np.zeros_like(weights[0])
    mean_rates = np.array(encoding["mean_rates"], dtype=float)
    mean_rates[1] = 0.0
    features[2], positions[2], weights[2] = (
        features[2][:0],
        positions[2][:0],
        weights[2][:0],
    )
    encoding = {
        **encoding,
        "encoding_spike_waveform_features": features,
        "encoding_positions": positions,
        "encoding_weights": weights,
        "mean_rates": jnp.asarray(mean_rates),
    }
    recording = (env, position_time, position, encoding, times, marks, edges)
    actual = predict(recording, is_local, slice(0, 2000))
    expected = reference(recording, is_local, slice(0, 2000))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, **PER_ELECTRODE_FLOAT32)


def make_isolation_recording():
    """Electrodes with alike marks whose encoding spikes lie in different places.

    Electrode 0 (two features) fires only left of x = 15 cm and electrode 1
    (four) only right of x = 25 cm, so a spike reduced against the other's
    samples would move its intensity across the arena. Electrode 2 fires
    everywhere with 600 samples (three tiles) and electrode 3 (one feature)
    has 257, one more than a tile. Decoding blocks of 16 spikes split
    electrode 1's 300 spikes over many blocks and GEMM groups.
    """
    rng = np.random.default_rng(37)
    env = Environment(place_bin_size=4.0, position_range=((0.0, 40.0), (0.0, 40.0)))
    position_time = np.arange(0, 30.0, 1 / 30)
    position = np.column_stack(
        (20 + 18 * np.sin(position_time / 2), 20 + 15 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    fires = [
        position[:, 0] < 15.0,
        position[:, 0] > 25.0,
        np.ones(len(position_time), bool),
        np.ones(len(position_time), bool),
    ]
    counts, n_features = (150, 400, 600, 257), (2, 4, 4, 1)
    encoding_times = [
        np.sort(rng.choice(position_time[where], n))
        for where, n in zip(fires, counts, strict=True)
    ]
    encoding_marks = [
        rng.normal(0, 10, (len(t), d))
        for t, d in zip(encoding_times, n_features, strict=True)
    ]
    decode_counts = (30, 300, 5, 70)
    decode_times = [np.sort(rng.uniform(0, 4, n)) for n in decode_counts]
    decode_marks = [
        rng.normal(0, 10, (n, d))
        for n, d in zip(decode_counts, n_features, strict=True)
    ]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=encoding_marks,
        environment=env,
        position_std=3.0,
        waveform_std=10.0,
        block_size=16,
        disable_progress_bar=True,
    )
    edges = np.arange(0, 4.0 + 1e-9, 0.002)
    return env, position_time, position, encoding, decode_times, decode_marks, edges


def electrode_subset(recording, electrodes):
    """The recording with only ``electrodes``, keeping every other model term."""
    env, position_time, position, encoding, times, marks, edges = recording
    per_electrode = (
        "encoding_spike_waveform_features",
        "encoding_positions",
        "encoding_weights",
        "gpi_models",
        "mark_reference",
        "mark_dimensions",
    )
    subset = {key: [encoding[key][e] for e in electrodes] for key in per_electrode}
    subset["mean_rates"] = np.asarray(encoding["mean_rates"])[electrodes]
    return (
        env,
        position_time,
        position,
        {**encoding, **subset},
        [times[e] for e in electrodes],
        [marks[e] for e in electrodes],
        edges,
    )


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("pos_tile_size", [None, 7])
def test_batched_electrodes_never_mix(is_local, pos_tile_size, reduction_path):
    """One pass over every electrode gives each spike its own electrode's
    intensity: its samples, weights, feature count, bandwidths and rate.
    Rows are summed by CPU's sequential scatter and by the off-CPU
    segmented scan.

    Electrode 0 alone and electrode 1 alone peak at opposite ends of the
    arena, so a spike reduced against the other's samples would differ by
    far more than float32 rounding from the per-electrode formulation.
    """
    recording = make_isolation_recording()
    recording = (
        *recording[:3],
        {**recording[3], "pos_tile_size": pos_tile_size},
        *recording[4:],
    )
    rows = slice(0, len(recording[-1]) - 1)
    actual = predict(recording, is_local, rows)
    expected = reference(recording, is_local, rows)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, **PER_ELECTRODE_FLOAT32)
    if not is_local:
        bins = recording[0].place_bin_centers_
        left, right = (
            np.asarray(reference(electrode_subset(recording, [e]), False, rows))
            for e in (0, 1)
        )
        assert bins[np.argmax(left.sum(axis=0)), 0] < 20.0
        assert bins[np.argmax(right.sum(axis=0)), 0] > 20.0


@pytest.mark.parametrize("is_local", [False, True])
def test_marks_with_another_feature_count_than_the_encoding_are_rejected(is_local):
    """Marks are padded to the most features of any electrode, so a decoding
    mark with fewer features than its encoding marks must be refused rather
    than padded; without a mark reference nothing else checks it."""
    env, position_time, position, encoding, times, marks, edges = (
        make_isolation_recording()
    )
    references = list(encoding["mark_reference"])
    references[1] = None
    marks = list(marks)
    marks[1] = marks[1][:, :2]
    recording = (
        env,
        position_time,
        position,
        {**encoding, "mark_reference": references},
        times,
        marks,
        edges,
    )
    with pytest.raises(ValidationError, match="electrode 1"):
        predict(recording, is_local, slice(0, 100))


def test_high_dimensional_marks_use_the_logsumexp_path():
    """Above eight waveform dimensions the kernels take the logsumexp branch."""
    recording = make_recording(False, waveform_dims=(9, 9, 10))
    for is_local in (False, True):
        actual = predict(recording, is_local, slice(0, 2000))
        expected = reference(recording, is_local, slice(0, 2000))
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, **PER_ELECTRODE_FLOAT32)


def test_ragged_final_encoding_tile_counts_each_sample_once():
    """Encoding tiles hold at most 2048 rows; when the padded count is not a
    multiple of that, the final tile overlaps the previous one and its
    repeated rows must not add to the intensity."""
    rng = np.random.default_rng(35)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 20.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    n_encoding = 2300
    tile_rows = clusterless_kde._LOG_REDUCTION_ENCODING_ROWS
    assert padded_sample_count(n_encoding) > tile_rows
    assert padded_sample_count(n_encoding) % tile_rows
    encoding_times = [np.sort(rng.uniform(0, 20, n_encoding))]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[rng.normal(0, 30, (n_encoding, 4))],
        environment=env,
        position_std=3.0,
        waveform_std=24.0,
        disable_progress_bar=True,
    )
    decode_times = [np.sort(rng.uniform(0, 2, 40))]
    recording = (
        env,
        position_time,
        position,
        encoding,
        decode_times,
        [rng.normal(0, 30, (40, 4))],
        np.arange(0, 2.0 + 1e-9, 0.002),
    )
    actual = predict(recording, False, slice(0, 1000))
    expected = reference(recording, False, slice(0, 1000))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, **PER_ELECTRODE_FLOAT32)


def test_electrodes_with_nearby_sizes_share_compiled_kernels():
    rng = np.random.default_rng(32)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 20.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    # 200 and 210 encoding spikes share one padded size; 40 and 50 decoding
    # spikes share another.
    encoding_times = [np.sort(rng.uniform(0, 20, n)) for n in (200, 210)]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[
            rng.normal(0, 30, (len(t), 4)) for t in encoding_times
        ],
        environment=env,
        disable_progress_bar=True,
    )
    assert padded_sample_count(200) == padded_sample_count(210)
    assert padded_spike_count(40, 100) == padded_spike_count(50, 100)
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (40, 50)]
    recording = (
        env,
        position_time,
        position,
        encoding,
        decode_times,
        [rng.normal(0, 30, (len(t), 4)) for t in decode_times],
        np.arange(0, 2.0 + 1e-9, 0.002),
    )
    kernels = (
        clusterless_kde._add_log_mark_intensities,
        clusterless_kde._add_local_terms,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
        predict(recording, is_local, None)
    assert [kernel._cache_size() for kernel in kernels] == [1, 1]


@pytest.mark.parametrize(
    "policy",
    [{"enc_tile_size": 64}, {"enc_tile_size": 64, "use_streaming": True}],
    ids=["tiled", "streaming"],
)
def test_tiled_electrodes_and_chunks_with_nearby_sizes_share_compiled_kernels(
    policy,
):
    """Encoding tiles and streaming pad like the default path, so electrodes
    and row chunks with nearby sizes reuse one compiled reduction."""
    rng = np.random.default_rng(34)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 20.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 20, n)) for n in (200, 210)]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[
            rng.normal(0, 30, (len(t), 4)) for t in encoding_times
        ],
        environment=env,
        disable_progress_bar=True,
        **policy,
    )
    assert padded_sample_count(200) == padded_sample_count(210)
    # Two row chunks per electrode; every chunk holds between 17 and 32
    # decoding spikes, which share one padded count.
    decode_times = [
        np.sort(np.concatenate([rng.uniform(0, 1, n0), rng.uniform(1, 2, n1)]))
        for n0, n1 in ((20, 30), (25, 18))
    ]
    assert padded_spike_count(17, 10_000) == padded_spike_count(32, 10_000)
    edges = np.arange(0, 2.0 + 1e-9, 0.002)
    recording = (
        env,
        position_time,
        position,
        {**encoding, "block_size": 10_000},
        decode_times,
        [rng.normal(0, 30, (len(t), 4)) for t in decode_times],
        edges,
    )
    kernel = clusterless_kde.estimate_log_joint_mark_intensity
    kernel.clear_cache()
    chunks = [
        predict(recording, False, rows) for rows in (slice(0, 500), slice(500, 1000))
    ]
    assert kernel._cache_size() == 1
    whole = predict(recording, False, slice(0, 1000))
    np.testing.assert_allclose(np.concatenate(chunks), whole, rtol=1e-5, atol=1e-4)


def test_occupancy_refit_with_one_fewer_sample_reuses_the_local_kernel():
    """EM refits the occupancy from decode bins, one sample fewer than the
    position samples of the first fit; the local kernel must not recompile."""
    rng = np.random.default_rng(33)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 20.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 20, 200))]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[rng.normal(0, 30, (200, 4))],
        environment=env,
        disable_progress_bar=True,
    )
    decode_times = [np.sort(rng.uniform(0, 2, 40))]
    recording = (
        env,
        position_time,
        position,
        encoding,
        decode_times,
        [rng.normal(0, 30, (40, 4))],
        np.arange(0, 2.0 + 1e-9, 0.002),
    )
    occupancy_model = encoding["occupancy_model"]
    n_samples = occupancy_model.samples_.shape[0]
    assert padded_occupancy_count(n_samples) == padded_occupancy_count(n_samples - 1)
    refit = KDEModel(
        std=occupancy_model.std, block_size=occupancy_model.block_size
    ).fit(occupancy_model.samples_[:-1], weights=occupancy_model.weights_[:-1])
    kernel = clusterless_kde._add_local_terms
    kernel.clear_cache()
    first = predict(recording, True, None)
    second = predict(
        (*recording[:3], {**encoding, "occupancy_model": refit}, *recording[4:]),
        True,
        None,
    )
    assert kernel._cache_size() == 1
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))


def make_translated_recording(offset, far_spike_time=None):
    """A 2-D recording whose arena starts ``offset`` cm from the origin.

    The animal circles in the first 60 cm of a 200 x 60 cm arena. With
    ``far_spike_time``, a decoding spike is added on the first electrode at
    that position sample, where the decoding position is moved 130 cm (37
    position bandwidths) from every occupancy sample.
    """
    rng = np.random.default_rng(36)
    env = Environment(
        place_bin_size=6.0,
        position_range=((offset, offset + 200.0), (offset, offset + 60.0)),
    )
    position_time = np.arange(0, 25.0, 1 / 30)
    position = offset + np.column_stack(
        (30 + 20 * np.sin(position_time / 3), 30 + 20 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 25, n)) for n in (150, 260)]
    encoding = fit_encoding(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[
            rng.normal(0, 30, (len(t), 4)) for t in encoding_times
        ],
        environment=env,
        disable_progress_bar=True,
    )
    # 70 and 100 decoding spikes, evaluated together padded to 256 rows.
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (70, 100)]
    decode_marks = [rng.normal(0, 30, (len(t), 4)) for t in decode_times]
    decode_position = position
    if far_spike_time is not None:
        decode_position = position.copy()
        decode_position[far_spike_time] = offset + np.array([180.0, 30.0])
        decode_times[0] = np.sort(
            np.append(decode_times[0], position_time[far_spike_time])
        )
        decode_marks[0] = np.concatenate([decode_marks[0], rng.normal(0, 30, (1, 4))])
    return (
        env,
        position_time,
        decode_position,
        encoding,
        decode_times,
        decode_marks,
        np.arange(0, 2.0 + 1e-9, 0.002),
    )


@pytest.fixture
def local_occupancy(monkeypatch):
    """Each Local occupancy pass at the decoding spikes, one for all electrodes.

    Records ``(points, log_density, repaired_rows)``: the rows of ``points``
    that the pass re-evaluated in log space.
    """
    passes = []
    repaired = None
    true_density = clusterless_kde.true_kde_log_density
    repair = kernel_density._repair_in_log_space

    def recording_density(points, *args, **kwargs):
        nonlocal repaired
        repaired = np.zeros(0, int)
        try:
            values = true_density(points, *args, **kwargs)
            passes.append((np.asarray(points), np.asarray(values), repaired))
        finally:
            repaired = None
        return values

    def recording_repair(values, needs_log_space, points, *args):
        nonlocal repaired
        if repaired is not None:
            repaired = np.flatnonzero(needs_log_space)
        return repair(values, needs_log_space, points, *args)

    monkeypatch.setattr(clusterless_kde, "true_kde_log_density", recording_density)
    monkeypatch.setattr(kernel_density, "_repair_in_log_space", recording_repair)
    return passes


def test_padding_rows_far_from_the_origin_are_not_repaired(local_occupancy):
    """Padding rows of an arena far from the origin cost no log-space pass.

    The arena 105 cm from the origin decodes with the shapes of the arena at
    the origin, so it compiles nothing that the first decode did not.
    """
    origin = make_translated_recording(0.0)
    translated = make_translated_recording(105.0)
    predict(origin, True, None)
    local_occupancy.clear()
    with count_compilations() as compiled:
        actual = predict(translated, True, None)
    assert [rows.size for _, _, rows in local_occupancy] == [0]
    assert compiled == []
    expected = reference(translated, True, slice(0, len(translated[-1]) - 1))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, **PER_ELECTRODE_FLOAT32)


def test_decoding_spike_far_from_all_occupancy_is_repaired(local_occupancy):
    """A real spike whose occupancy density underflows is re-evaluated in log space.

    So are spikes whose interpolated position lies toward it, but no padding
    row.
    """
    far_spike_time = 30  # 1 s, inside the 2 s decoding window
    recording = make_translated_recording(105.0, far_spike_time=far_spike_time)
    actual = predict(recording, True, None)
    far_position = recording[2][far_spike_time]
    n_real = sum(len(times) for times in recording[4])
    ((points, values, rows),) = local_occupancy
    assert np.all(rows < n_real)
    (far_row,) = np.flatnonzero(np.all(points == far_position, axis=1))
    assert far_row in rows

    occupancy_model = recording[3]["occupancy_model"]
    samples = np.asarray(occupancy_model.samples_, float)
    weights = np.asarray(occupancy_model.weights_, float)
    std = np.asarray(as_std_array(occupancy_model.std, 2), float)
    log_kernel = -0.5 * np.sum(((far_position - samples) / std) ** 2, axis=1) - np.sum(
        np.log(np.sqrt(2.0 * np.pi) * std)
    )
    expected = logsumexp(log_kernel, b=weights) - np.log(weights.sum())
    assert expected < -500.0
    np.testing.assert_allclose(values[far_row], expected, rtol=1e-6)
    assert np.all(np.isfinite(actual))
