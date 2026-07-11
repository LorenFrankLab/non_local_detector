"""Registry integration + end-to-end smoke test for ``clusterless_diffusion``.

Confirms the algorithm is resolvable through the public
``_CLUSTERLESS_ALGORITHMS`` registry (used by ``ClusterlessDecoder`` /
``NonLocalClusterlessDetector`` via ``clusterless_algorithm="clusterless_diffusion"``)
and that a full fit/predict cycle on tiny simulated data yields a finite,
normalized posterior. This is a smoke test, not a perf/accuracy test -- the
simulated grid and spike counts are kept deliberately small.
"""

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.clusterless_diffusion import (
    _effective_block,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.models.decoder import ClusterlessDecoder
from non_local_detector.models.non_local_model import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

POSITION_STD = 3.0
WAVEFORM_STD = 24.0
NORMALIZATION_ATOL = 1e-6
SAVE_LOAD_RTOL = 1e-5
SAVE_LOAD_ATOL = 1e-10
DEFAULT_MEMORY_BUDGET = 536_870_912


def _toy_environment(upper: float, place_bin_size: float = 1.0) -> Environment:
    """A tiny 1D environment, fitted, for direct fit/predict-function tests."""
    env = Environment(position_range=[(0.0, upper)], place_bin_size=place_bin_size)
    return env.fit_place_grid(
        position=np.linspace(0.0, upper, 200)[:, None], infer_track_interior=True
    )


def _toy_electrode_data(
    rng: np.random.Generator,
    position_time: np.ndarray,
    n_electrodes: int = 2,
    n_spikes: int = 40,
    n_features: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Random spike times (within ``position_time``'s span) and marks per electrode."""
    spike_times = []
    spike_waveform_features = []
    for _ in range(n_electrodes):
        spike_times.append(
            np.sort(rng.uniform(position_time[0], position_time[-1], size=n_spikes))
        )
        spike_waveform_features.append(
            rng.normal(scale=WAVEFORM_STD, size=(n_spikes, n_features))
        )
    return spike_times, spike_waveform_features


def test_registry_and_end_to_end() -> None:
    """``clusterless_diffusion`` is registered and runs end-to-end."""
    # --- registry ---
    assert "clusterless_diffusion" in _CLUSTERLESS_ALGORITHMS
    assert _CLUSTERLESS_ALGORITHMS["clusterless_diffusion"] == (
        fit_clusterless_diffusion_encoding_model,
        predict_clusterless_diffusion_log_likelihood,
    )

    # --- tiny simulated data (2 tetrodes, 1 place field each, ~5s of data) ---
    sim = make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 20, 10),
        track_height=20.0,
        running_speed=15,
        sampling_frequency=500,
        n_runs=2,
        seed=0,
    )

    n_encode = int(0.7 * len(sim.position_time))
    encode_position_time = sim.position_time[:n_encode]
    encode_position = sim.position[:n_encode]
    encode_spike_times = [st[st <= encode_position_time[-1]] for st in sim.spike_times]
    encode_spike_waveform_features = [
        swf[st <= encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]

    algorithm_params = {
        "position_std": POSITION_STD,
        "waveform_std": WAVEFORM_STD,
        "disable_progress_bar": True,
    }

    # --- ClusterlessDecoder ---
    decoder = ClusterlessDecoder(
        environments=sim.environment,
        clusterless_algorithm="clusterless_diffusion",
        clusterless_algorithm_params=algorithm_params,
    )
    decoder.fit(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
    )

    test_position_time = sim.position_time[n_encode:]
    test_spike_times = [st[st > encode_position_time[-1]] for st in sim.spike_times]
    test_spike_waveform_features = [
        swf[st > encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]
    test_edges = np.linspace(test_position_time[0], test_position_time[-1], 11)

    decoder_results = decoder.predict(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
    )

    decoder_posterior = decoder_results.acausal_posterior.values
    assert np.all(np.isfinite(decoder_posterior))
    decoder_posterior_sum = np.sum(decoder_posterior, axis=1)
    np.testing.assert_allclose(
        decoder_posterior_sum, 1.0, atol=NORMALIZATION_ATOL, rtol=0.0
    )

    # --- NonLocalClusterlessDetector ---
    detector = NonLocalClusterlessDetector(
        environments=sim.environment,
        clusterless_algorithm="clusterless_diffusion",
        clusterless_algorithm_params=algorithm_params,
    )
    detector.fit(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
    )

    detector_results = detector.predict(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
    )

    detector_posterior = detector_results.acausal_posterior.values
    assert np.all(np.isfinite(detector_posterior))
    detector_posterior_sum = np.sum(detector_posterior, axis=1)
    np.testing.assert_allclose(
        detector_posterior_sum, 1.0, atol=NORMALIZATION_ATOL, rtol=0.0
    )


def test_refit_requires_encoding_refit() -> None:
    """A grid refit that genuinely changes the grid shape drops the device cache;
    the (grid-bound) encoding model must be re-fit -- not merely re-predicted --
    against the rebuilt grid, and a fresh predict then works (finite, right shape
    matching the NEW interior-bin count) with no stale-``Q`` shape error.

    Unlike ``_toy_environment``, the ``Environment`` here is constructed WITHOUT a
    fixed ``position_range``: with a pinned ``position_range``, a second
    ``fit_place_grid`` call produces a bit-identical grid regardless of the
    ``position`` argument passed to it, which would make the shape assertions
    below vacuous (old and new bases would be numerically identical). Letting
    ``fit_place_grid`` infer the grid extent from ``position`` means a wider
    second ``position`` array genuinely enlarges the grid.
    """
    rng = np.random.default_rng(0)

    # --- fit + one predict on the original grid: populates the device cache ---
    env = Environment(place_bin_size=1.0)
    position_time = np.linspace(0.0, 10.0, 200)
    position = position_time[:, None]
    env.fit_place_grid(position=position, infer_track_interior=True)
    n_interior_original = int(env.is_track_interior_.ravel().sum())
    spike_times, spike_waveform_features = _toy_electrode_data(rng, position_time)

    encoding_model = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        environment=env,
        position_std=POSITION_STD,
        disable_progress_bar=True,
    )
    assert hasattr(env, "_diffusion_device_basis_")

    decode_time = np.linspace(position_time[0], position_time[-1], 6)
    log_likelihood = predict_clusterless_diffusion_log_likelihood(
        decode_time,
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        **encoding_model,
    )
    assert np.all(np.isfinite(np.asarray(log_likelihood)))
    assert hasattr(env, "_diffusion_device_basis_")

    # --- rebuild the grid with a wider position span (no fixed position_range to
    # pin it, so the grid genuinely changes shape): device cache must be dropped ---
    new_position_time = np.linspace(0.0, 20.0, 300)
    new_position = new_position_time[:, None]
    env.fit_place_grid(position=new_position, infer_track_interior=True)
    n_interior_new = int(env.is_track_interior_.ravel().sum())
    assert n_interior_new != n_interior_original
    assert not hasattr(env, "_diffusion_device_basis_")

    # --- the old encoding model is grid-bound and stale; re-fit against the
    # rebuilt grid rather than reusing it ---
    new_spike_times, new_spike_waveform_features = _toy_electrode_data(
        rng, new_position_time
    )
    new_encoding_model = fit_clusterless_diffusion_encoding_model(
        new_position_time,
        new_position,
        new_spike_times,
        new_spike_waveform_features,
        environment=env,
        position_std=POSITION_STD,
        disable_progress_bar=True,
    )
    assert hasattr(env, "_diffusion_device_basis_")

    new_decode_time = np.linspace(new_position_time[0], new_position_time[-1], 6)
    new_log_likelihood = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            new_decode_time,
            new_position_time,
            new_position,
            new_spike_times,
            new_spike_waveform_features,
            **new_encoding_model,
        )
    )
    assert np.all(np.isfinite(new_log_likelihood))
    # Genuinely discriminating (unlike the pre-fix version, where the grid was
    # bit-identical before/after "refit" so this shape always matched): the
    # predict output's column count is tied to the device basis's row count,
    # which is n_interior for whatever grid it was built from. Since
    # n_interior_original (10) != n_interior_new (20) here, any stale-basis bug
    # -- e.g. get_device_basis returning the pre-refit eigenvectors, or predict
    # being called with the un-refit ``encoding_model`` against the rebuilt
    # environment -- would yield a 10-column result (or a shape-mismatch crash
    # inside predict, since occupancy/other encoding arrays are also sized to
    # n_interior_original), not 20. This assertion would then fail instead of
    # passing vacuously.
    assert new_log_likelihood.shape == (len(new_decode_time), n_interior_new)


def test_save_load_predict_parity(tmp_path) -> None:
    """``save_model`` -> ``load_model`` must not raise a JAX-``Device`` pickling
    error (the Environment's transient device cache is excluded from the pickle),
    and predict on the reloaded detector must match pre-save predict -- the device
    cache is lazily rebuilt on the first post-load predict."""
    sim = make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 20, 10),
        track_height=20.0,
        running_speed=15,
        sampling_frequency=500,
        n_runs=2,
        seed=0,
    )

    n_encode = int(0.7 * len(sim.position_time))
    encode_position_time = sim.position_time[:n_encode]
    encode_position = sim.position[:n_encode]
    encode_spike_times = [st[st <= encode_position_time[-1]] for st in sim.spike_times]
    encode_spike_waveform_features = [
        swf[st <= encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]

    algorithm_params = {
        "position_std": POSITION_STD,
        "waveform_std": WAVEFORM_STD,
        "disable_progress_bar": True,
    }

    detector = NonLocalClusterlessDetector(
        environments=sim.environment,
        clusterless_algorithm="clusterless_diffusion",
        clusterless_algorithm_params=algorithm_params,
    )
    detector.fit(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
    )

    test_position_time = sim.position_time[n_encode:]
    test_spike_times = [st[st > encode_position_time[-1]] for st in sim.spike_times]
    test_spike_waveform_features = [
        swf[st > encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]
    test_edges = np.linspace(test_position_time[0], test_position_time[-1], 11)

    results_before = detector.predict(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
    )
    posterior_before = results_before.acausal_posterior.values
    assert np.all(np.isfinite(posterior_before))

    model_path = str(tmp_path / "m.pkl")
    detector.save_model(model_path)  # must not raise on a JAX Device
    loaded_detector = NonLocalClusterlessDetector.load_model(model_path)

    results_after = loaded_detector.predict(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
    )
    posterior_after = results_after.acausal_posterior.values
    assert np.all(np.isfinite(posterior_after))

    np.testing.assert_allclose(
        posterior_after, posterior_before, rtol=SAVE_LOAD_RTOL, atol=SAVE_LOAD_ATOL
    )


def test_memory_budget_override_survives_handoff() -> None:
    """A non-default ``memory_budget`` passed at fit is stored in the encoding dict
    (not silently lost at the fit -> predict handoff) and actually drives the
    resolved ``effective_block`` used by predict."""
    rng = np.random.default_rng(1)
    env = _toy_environment(upper=10.0)
    position_time = np.linspace(0.0, 10.0, 200)
    position = position_time[:, None]
    spike_times, spike_waveform_features = _toy_electrode_data(rng, position_time)

    non_default_memory_budget = 1_000_000
    encoding_model = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        environment=env,
        position_std=POSITION_STD,
        memory_budget=non_default_memory_budget,
        disable_progress_bar=True,
    )

    assert encoding_model["memory_budget"] == non_default_memory_budget
    assert encoding_model["block_size"] == 10_000  # default, also carried in the dict

    n_enc = int(encoding_model["encoding_marks"][0].shape[0])
    n_bins = int(encoding_model["occupancy"].shape[0])
    resolved_rank = int(encoding_model["resolved_rank"])
    block_size = int(encoding_model["block_size"])

    overridden_block = _effective_block(
        non_default_memory_budget, n_enc, resolved_rank, n_bins, block_size
    )
    default_block = _effective_block(
        DEFAULT_MEMORY_BUDGET, n_enc, resolved_rank, n_bins, block_size
    )
    assert overridden_block != default_block
