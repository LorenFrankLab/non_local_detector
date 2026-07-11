"""Registry integration + end-to-end smoke test for ``clusterless_diffusion``.

Confirms the algorithm is resolvable through the public
``_CLUSTERLESS_ALGORITHMS`` registry (used by ``ClusterlessDecoder`` /
``NonLocalClusterlessDetector`` via ``clusterless_algorithm="clusterless_diffusion"``)
and that a full fit/predict cycle on tiny simulated data yields a finite,
normalized posterior. This is a smoke test, not a perf/accuracy test -- the
simulated grid and spike counts are kept deliberately small.
"""

import numpy as np

from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.clusterless_diffusion import (
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.models.decoder import ClusterlessDecoder
from non_local_detector.models.non_local_model import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

POSITION_STD = 3.0
WAVEFORM_STD = 24.0
NORMALIZATION_ATOL = 1e-6


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
