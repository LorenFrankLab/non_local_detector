"""Compressed encoding dictionaries for the clusterless KDE.

A compressed fit replaces each electrode's encoding spikes with a smaller
weighted dictionary. The mean rates and exposure still come from every spike,
and the ground process comes from the dictionary, so the expected count stays
the integral of the compressed marked intensity over marks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    log_kde_distance,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import RATE_REFERENCE_SECONDS
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.time_edges import time_edges_from_centers

POSITION_STD = 4.0
WAVEFORM_STD = 2.0
# A finite background close enough to the kernel peaks that every spike stays
# on the scaled GEMM rather than the exact repair; its exp is subtracted.
LOG_BACKGROUND = -60.0
SAMPLING_FREQUENCY = 100.0


def _simulate_session(n_mark_dims=2, duration=400.0, seed=0, rare_rate=0.0):
    """One electrode on a 0-100 cm track: two units with distinct marks and fields.

    With ``rare_rate`` > 0 (Hz) a third, rare unit fires only near 95 cm with
    its own marks.
    """
    rng = np.random.default_rng(seed)
    position_time = np.arange(0.0, duration, 1.0 / SAMPLING_FREQUENCY)
    position = 50.0 + 45.0 * np.sin(2 * np.pi * position_time / 20.0)
    dt = 1.0 / SAMPLING_FREQUENCY
    centers, peaks = [25.0, 70.0], [30.0, 30.0]
    mark_means = [np.full(n_mark_dims, 10.0), np.full(n_mark_dims, 20.0)]
    if rare_rate > 0:
        centers.append(95.0)
        peaks.append(rare_rate)
        mark_means.append(np.full(n_mark_dims, 40.0))
    spike_times, marks = [], []
    for center, peak, mark_mean in zip(centers, peaks, mark_means, strict=True):
        rate = peak * np.exp(-0.5 * ((position - center) / 6.0) ** 2)
        fired = rng.random(position_time.size) < rate * dt
        times = position_time[fired] + rng.uniform(-dt / 2, dt / 2, fired.sum())
        spike_times.append(times)
        marks.append(mark_mean + rng.normal(0.0, 1.5, (fired.sum(), n_mark_dims)))
    spike_times = np.concatenate(spike_times)
    order = np.argsort(spike_times)
    spike_times = np.clip(spike_times[order], 0.0, duration - dt)
    return {
        "position_time": position_time,
        "position": position[:, None],
        "spike_times": [spike_times],
        "spike_waveform_features": [np.concatenate(marks)[order]],
    }


@pytest.fixture(scope="module")
def track_environment():
    env = Environment(
        environment_name="track", place_bin_size=2.0, position_range=((0.0, 100.0),)
    )
    return env.fit_place_grid(
        position=np.linspace(0.0, 100.0, 51)[:, None], infer_track_interior=False
    )


@pytest.fixture(scope="module")
def session():
    return _simulate_session()


def _fit(session, environment, **params):
    return fit_clusterless_kde_encoding_model(
        **session,
        environment=environment,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        block_size=1000,
        disable_progress_bar=True,
        **params,
    )


def _predict(encoding, session, environment, is_local=False):
    time_edges = np.arange(0.0, 60.0 + 1e-9, 0.02)
    in_window = [times < time_edges[-1] for times in session["spike_times"]]
    return np.asarray(
        predict_clusterless_kde_log_likelihood(
            position_time=session["position_time"],
            position=session["position"],
            spike_times=[
                t[k] for t, k in zip(session["spike_times"], in_window, strict=True)
            ],
            spike_waveform_features=[
                f[k]
                for f, k in zip(
                    session["spike_waveform_features"], in_window, strict=True
                )
            ],
            time_edges=time_edges,
            is_local=is_local,
            mark_reference=fit_mark_references(
                session["spike_waveform_features"], kernel_scale=WAVEFORM_STD
            ),
            **{k: v for k, v in encoding.items() if k != "block_size"},
            block_size=1000,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_at_least_as_large_as_the_spikes_is_the_full_kde(
    session, track_environment, method
):
    full = _fit(session, track_environment)
    n_spikes = len(session["spike_times"][0])
    compressed = _fit(
        session,
        track_environment,
        encoding_compression=method,
        n_encoding_kernels=n_spikes + 10,
    )
    for is_local in (False, True):
        np.testing.assert_allclose(
            _predict(compressed, session, track_environment, is_local),
            _predict(full, session, track_environment, is_local),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_keeps_rates_exposure_and_total_weight(
    session, track_environment, method
):
    full = _fit(session, track_environment)
    compressed = _fit(
        session,
        track_environment,
        encoding_compression=method,
        n_encoding_kernels=200,
    )
    assert compressed["encoding_positions"][0].shape == (200, 1)
    assert compressed["encoding_spike_waveform_features"][0].shape == (200, 2)
    np.testing.assert_allclose(compressed["mean_rates"], full["mean_rates"])
    assert compressed["encoding_exposure_seconds"] == full["encoding_exposure_seconds"]
    # Uniform inverse-probability weights keep the total exactly; Pareto
    # sampling keeps it in expectation. The KDE divides by the total anyway.
    np.testing.assert_allclose(
        np.sum(compressed["encoding_weights"][0]),
        np.sum(full["encoding_weights"][0]),
        rtol=1e-6 if method == "uniform" else 0.05,
    )
    np.testing.assert_allclose(compressed["occupancy"], full["occupancy"])


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_zero_weight_spikes_never_enter_the_dictionary(
    session, track_environment, method
):
    position_time = session["position_time"]
    weights = np.where(position_time < 200.0, 1.0, 0.0)
    compressed = _fit(
        session,
        track_environment,
        weights=weights,
        encoding_compression=method,
        n_encoding_kernels=100,
    )
    assert np.all(np.asarray(compressed["encoding_weights"][0]) > 0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_is_reproducible_from_its_seed(session, track_environment, method):
    fits = [
        _fit(
            session,
            track_environment,
            encoding_compression=method,
            n_encoding_kernels=150,
            compression_seed=seed,
        )
        for seed in (3, 3, 4)
    ]
    first, repeat, other = (np.asarray(f["encoding_positions"][0]) for f in fits)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


@pytest.mark.unit
def test_expected_count_is_the_marked_intensity_integrated_over_marks(
    track_environment,
):
    """For every method, sum_m lambda(x, m) dm equals the ground intensity."""
    session = _simulate_session(n_mark_dims=1, duration=200.0)
    mark_grid = np.linspace(-10.0, 45.0, 2201)
    for params in (
        {},
        {"encoding_compression": "uniform", "n_encoding_kernels": 150},
        {"encoding_compression": "stratified", "n_encoding_kernels": 150},
    ):
        encoding = _fit(session, track_environment, **params)
        bins = track_environment.place_bin_centers_
        log_joint = estimate_log_joint_mark_intensity(
            jnp.asarray(mark_grid[:, None]),
            jnp.asarray(encoding["encoding_spike_waveform_features"][0]),
            jnp.asarray([WAVEFORM_STD]),
            encoding["occupancy"],
            encoding["mean_rates"][0] * RATE_REFERENCE_SECONDS,
            log_kde_distance(
                jnp.asarray(bins),
                jnp.asarray(encoding["encoding_positions"][0]),
                jnp.asarray([POSITION_STD]),
            ),
            encoding_weights=jnp.asarray(encoding["encoding_weights"][0]),
            log_occupancy=encoding["log_occupancy"],
            log_background=jnp.full(mark_grid.size, LOG_BACKGROUND),
        )
        integrated = np.trapezoid(
            np.exp(np.asarray(log_joint, float)) - np.exp(LOG_BACKGROUND),
            mark_grid,
            axis=0,
        )
        np.testing.assert_allclose(
            integrated / RATE_REFERENCE_SECONDS,
            encoding["summed_ground_process_intensity"],
            rtol=1e-3,
            err_msg=str(params),
        )


@pytest.mark.unit
def test_stratified_dictionary_keeps_rare_units_in_rare_places(track_environment):
    session = _simulate_session(duration=400.0, rare_rate=0.5)
    marks = session["spike_waveform_features"][0]
    is_rare = marks[:, 0] > 30.0
    assert 10 <= is_rare.sum() <= 80

    def rare_kept(method, seed):
        encoding = _fit(
            session,
            track_environment,
            encoding_compression=method,
            n_encoding_kernels=300,
            compression_seed=seed,
        )
        return int(
            np.sum(
                np.asarray(encoding["encoding_spike_waveform_features"][0])[:, 0] > 30.0
            )
        )

    # Inclusion probabilities fall as the inverse square root of the stratum
    # size, so spikes in small strata are kept more often than uniformly.
    stratified = [rare_kept("stratified", seed) for seed in range(10)]
    uniform = [rare_kept("uniform", seed) for seed in range(10)]
    assert min(stratified) >= 1
    assert np.mean(stratified) > 2 * np.mean(uniform)


@pytest.mark.unit
def test_stratified_weights_are_unbiased_over_seeds(session, track_environment):
    """Averaged over seeds, the weighted dictionary reproduces the spike histogram."""
    full = _fit(session, track_environment)
    edges = np.linspace(0.0, 100.0, 11)
    target = np.histogram(
        np.asarray(full["encoding_positions"][0])[:, 0],
        edges,
        weights=np.asarray(full["encoding_weights"][0]),
    )[0]
    estimates = []
    for seed in range(40):
        encoding = _fit(
            session,
            track_environment,
            encoding_compression="stratified",
            n_encoding_kernels=300,
            compression_seed=seed,
        )
        estimates.append(
            np.histogram(
                np.asarray(encoding["encoding_positions"][0])[:, 0],
                edges,
                weights=np.asarray(encoding["encoding_weights"][0]),
            )[0]
        )
    # atol: one spike of Monte Carlo slack for nearly empty bins.
    np.testing.assert_allclose(np.mean(estimates, axis=0), target, rtol=0.05, atol=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "params",
    [
        {"encoding_compression": "kmeans", "n_encoding_kernels": 10},
        {"encoding_compression": "uniform"},
        {"encoding_compression": "stratified", "n_encoding_kernels": 0},
        {"encoding_compression": "uniform", "n_encoding_kernels": 2.5},
    ],
)
def test_invalid_compression_settings_raise(session, track_environment, params):
    with pytest.raises(ValidationError):
        _fit(session, track_environment, **params)


@pytest.mark.integration
@pytest.mark.parametrize(
    "compression",
    [
        {"encoding_compression": "stratified", "n_encoding_kernels": 200},
    ],
)
def test_detector_fits_and_decodes_with_a_compressed_kde(session, compression):
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2.0, position_range=((0.0, 100.0),)),
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": POSITION_STD,
            "waveform_std": WAVEFORM_STD,
            "block_size": 1000,
            **compression,
        },
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    detector.fit(
        session["position_time"],
        session["position"],
        session["spike_times"],
        session["spike_waveform_features"],
    )
    encoding = next(iter(detector.encoding_model_.values()))
    n_kernels = encoding["encoding_positions"][0].shape[0]
    assert n_kernels == compression.get("n_encoding_kernels", n_kernels)
    assert n_kernels < len(session["spike_times"][0])
    is_decoded = session["position_time"] < 30.0
    in_window = session["spike_times"][0] < 30.0
    results = detector.predict(
        [session["spike_times"][0][in_window]],
        [session["spike_waveform_features"][0][in_window]],
        time_edges=time_edges_from_centers(session["position_time"][is_decoded]),
        position=session["position"][is_decoded],
        position_time=session["position_time"][is_decoded],
    )
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))


@pytest.mark.unit
@pytest.mark.parametrize(
    "compression",
    [
        {"encoding_compression": "uniform", "n_encoding_kernels": 100},
        {"encoding_compression": "stratified", "n_encoding_kernels": 100},
    ],
)
def test_electrode_without_encoding_spikes_fits_and_predicts(
    session, track_environment, compression
):
    with_empty = {
        **session,
        "spike_times": [session["spike_times"][0], np.array([])],
        "spike_waveform_features": [
            session["spike_waveform_features"][0],
            np.zeros((0, 2)),
        ],
    }
    encoding = _fit(with_empty, track_environment, **compression)
    assert encoding["encoding_positions"][1].shape[0] == 0
    for is_local in (False, True):
        assert np.all(
            np.isfinite(_predict(encoding, with_empty, track_environment, is_local))
        )
