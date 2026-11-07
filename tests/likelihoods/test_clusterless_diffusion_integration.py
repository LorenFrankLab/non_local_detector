"""``clusterless_diffusion`` across environment refits and saving."""

import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_diffusion import (
    fit_clusterless_diffusion_encoding_model,
    mark_reference_scale,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.models.non_local_model import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

pytestmark = pytest.mark.integration

POSITION_STD = 3.0
WAVEFORM_STD = 24.0
SAVE_LOAD_RTOL = 1e-5
SAVE_LOAD_ATOL = 1e-10


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


def test_refit_requires_encoding_refit() -> None:
    """A grid refit that genuinely changes the grid shape drops the device cache;
    the (grid-bound) encoding model must be re-fit -- not merely re-predicted --
    against the rebuilt grid, and a fresh predict then works (finite, right shape
    matching the NEW interior-bin count) with no stale-``Q`` shape error.

    The ``Environment`` here is constructed WITHOUT a
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
    assert env.get_cached("diffusion_device_basis") is not None

    decode_time = np.linspace(position_time[0], position_time[-1], 6)
    log_likelihood = predict_clusterless_diffusion_log_likelihood(
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        time_edges=decode_time,
        mark_reference=fit_mark_references(
            spike_waveform_features, mark_reference_scale()
        ),
        **encoding_model,
    )
    assert np.all(np.isfinite(np.asarray(log_likelihood)))
    assert env.get_cached("diffusion_device_basis") is not None

    # --- rebuild the grid with a wider position span (no fixed position_range to
    # pin it, so the grid genuinely changes shape): device cache must be dropped ---
    new_position_time = np.linspace(0.0, 20.0, 300)
    new_position = new_position_time[:, None]
    env.fit_place_grid(position=new_position, infer_track_interior=True)
    n_interior_new = int(env.is_track_interior_.ravel().sum())
    assert n_interior_new != n_interior_original
    assert env.get_cached("diffusion_device_basis") is None

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
    assert env.get_cached("diffusion_device_basis") is not None

    new_decode_time = np.linspace(new_position_time[0], new_position_time[-1], 6)
    new_log_likelihood = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            new_position_time,
            new_position,
            new_spike_times,
            new_spike_waveform_features,
            time_edges=time_edges_from_centers(new_decode_time),
            mark_reference=fit_mark_references(
                new_spike_waveform_features, mark_reference_scale()
            ),
            **new_encoding_model,
        )
    )
    assert np.all(np.isfinite(new_log_likelihood))
    # The predict output's column count is tied to the device basis's row
    # count, which is n_interior for whatever grid it was built from. Since
    # the refit changes the grid (n_interior_original (10) != n_interior_new
    # (20)), any stale-basis bug -- e.g. get_device_basis returning the
    # pre-refit eigenvectors, or predict being called with the un-refit
    # ``encoding_model`` against the rebuilt environment -- would yield a
    # 10-column result (or a shape-mismatch crash inside predict, since
    # occupancy/other encoding arrays are also sized to n_interior_original),
    # not 20.
    assert new_log_likelihood.shape == (len(new_decode_time), n_interior_new)


def test_stale_encoding_model_after_refit_raises() -> None:
    """Predicting with an encoding model whose environment was refit to a
    DIFFERENT grid of the SAME interior-bin count must raise, not silently
    misattribute the likelihood to the old bins.

    This is the dangerous case the shape checks in the diffusion matmul cannot
    catch: fitting on interior bins 8..18 (11 bins) then refitting the same env to
    bins 9..19 (also 11 bins, different physical bins) leaves the stale encoding
    model's node_order/occupancy/bin-indices bound to the old layout while the
    device basis is rebuilt from the new graph -- finite output, wrong bins. The
    fit-time grid record turns that into a clear ValidationError.
    """
    env = Environment(position_range=[(0.0, 20.0)], place_bin_size=1.0)
    env = env.fit_place_grid(
        np.linspace(8.0, 18.0, 200)[:, None], infer_track_interior=True
    )
    n_before = int(env.is_track_interior_.ravel().sum())

    position_time = np.linspace(0.0, 1.0, 200)
    position = np.linspace(8.0, 18.0, 200)[:, None]
    spike_times = [np.array([0.3, 0.6, 0.9])]
    spike_waveform_features = [
        np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.3]], dtype=np.float32)
    ]
    encoding_model = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        environment=env,
        position_std=3.0,
        disable_progress_bar=True,
    )

    # Refit the SAME env object to a shifted interior with the same bin count.
    env.fit_place_grid(np.linspace(9.0, 19.0, 200)[:, None], infer_track_interior=True)
    assert int(env.is_track_interior_.ravel().sum()) == n_before  # same count

    decode_time = np.linspace(0.0, 1.0, 6)
    with pytest.raises(ValidationError, match="different environment grid"):
        predict_clusterless_diffusion_log_likelihood(
            position_time,
            position,
            [np.array([0.4, 0.7])],
            [np.array([[0.1, 0.0], [0.2, -0.5]], dtype=np.float32)],
            time_edges=decode_time,
            mark_reference=fit_mark_references(
                spike_waveform_features, mark_reference_scale()
            ),
            **encoding_model,
        )


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
    # Ten equal decode bins over the held-out span set the detector's bin rate.
    decode_sampling_frequency = 10 / (
        sim.position_time[-1] - sim.position_time[n_encode]
    )
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
        sampling_frequency=decode_sampling_frequency,
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
        time_edges=test_edges,
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
        time_edges=test_edges,
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
