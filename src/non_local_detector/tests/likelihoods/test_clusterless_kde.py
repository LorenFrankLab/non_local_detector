import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    DEFAULT_WAVEFORM_STD,
    select_spikes_in_rows,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.models.base import _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS


@pytest.mark.parametrize(
    "function",
    [
        clusterless_kde.fit_clusterless_kde_encoding_model,
        clusterless_kde.predict_clusterless_kde_log_likelihood,
        clusterless_kde.compute_local_log_likelihood,
        clusterless_kde.block_estimate_log_joint_mark_intensity,
    ],
)
def test_block_size_default_is_the_detectors(function):
    """Direct calls block spikes as a detector does: the old 100 was 3-4x slower."""
    default = inspect.signature(function).parameters["block_size"].default
    assert default == _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS["block_size"]


def test_select_spikes_in_rows_right_edge_last_bin():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    spikes = np.array([0.0, 0.5, 2.0, 3.0])
    selection = select_spikes_in_rows(spikes, 0, 3, time_edges=edges)
    # bins: [0,1), [1,2), [2,3]; right-edge 3.0 -> last bin index 2
    assert spikes[selection.indexer].tolist() == spikes.tolist()
    assert selection.bin_ind.tolist() == [0, 0, 2, 2]


def test_fit_and_predict_clusterless_kde_minimal(simple_1d_environment):
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]

    # one electrode: few encoding spikes with 2D waveform features
    enc_spike_times = jnp.array([2.0, 5.0, 7.5])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    mark_reference = fit_mark_references([enc_feats], kernel_scale=1.0)

    for key in (
        "occupancy",
        "occupancy_model",
        "gpi_models",
        "encoding_spike_waveform_features",
        "encoding_positions",
        "mean_rates",
        "summed_ground_process_intensity",
    ):
        assert key in encoding

    # decoding: 2 spikes within time edges with features near encoding cloud
    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [jnp.array([2.1, 5.2])]
    dec_feats = [jnp.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    ll_nonlocal = predict_clusterless_kde_log_likelihood(
        time_edges=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        **encoding,
        mark_reference=mark_reference,
        is_local=False,
    )
    # shape: (n_bins, n_interior_bins), one row per decode bin
    assert ll_nonlocal.shape[0] == t_edges.shape[0] - 1
    assert ll_nonlocal.ndim == 2 and ll_nonlocal.shape[1] > 0
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    ll_local = predict_clusterless_kde_log_likelihood(
        time_edges=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        **encoding,
        mark_reference=mark_reference,
        is_local=True,
    )
    assert ll_local.shape == (t_edges.shape[0] - 1, 1)
    assert jnp.all(jnp.isfinite(ll_local))


def test_select_spikes_in_rows_unsorted_and_interior_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    spikes = np.array(
        [2.0, 0.0, 1.0, 1.0, 0.5]
    )  # unsorted, includes interior edges 1.0
    selection = select_spikes_in_rows(spikes, 0, 3, time_edges=edges)
    # bins: [0,1), [1,2), [2,3]; interior edge 1.0 -> bin 1 (right side)
    # Unsorted spikes keep their original order in the selection.
    assert spikes[selection.indexer].tolist() == spikes.tolist()
    assert selection.bin_ind.tolist() == [2, 0, 1, 1, 0]
    assert not selection.indices_are_sorted


def test_fit_clusterless_kde_raises_without_place_grid():
    env = Environment(environment_name="nofit")
    t_pos = jnp.linspace(0.0, 10.0, 11)
    pos = jnp.linspace(0.0, 10.0, 11)[:, None]
    with pytest.raises(ValueError, match="fit_place_grid"):
        fit_clusterless_kde_encoding_model(
            position_time=t_pos,
            position=pos,
            spike_times=[jnp.array([2.0])],
            spike_waveform_features=[jnp.array([[0.0, 1.0]])],
            environment=env,
        )


def test_clusterless_kde_varying_electrode_feature_counts(simple_1d_environment):
    """Test that electrodes with different numbers of waveform features work correctly.

    A scalar waveform_std must not be expanded from the first electrode's
    feature count, which fails when other electrodes have different numbers of
    features (e.g., 3 vs 4 features per tetrode).
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]

    # Electrode 0: 4 waveform features (typical tetrode)
    enc_spike_times_0 = jnp.array([2.0, 5.0, 7.5])
    enc_feats_0 = jnp.array(
        [[0.0, 0.0, 0.1, 0.2], [1.0, -1.0, 0.5, -0.5], [0.5, 0.5, 0.3, 0.1]],
        dtype=float,
    )

    # Electrode 1: 3 waveform features (e.g., one bad channel)
    enc_spike_times_1 = jnp.array([3.0, 6.0, 8.0])
    enc_feats_1 = jnp.array(
        [[0.2, -0.1, 0.3], [0.8, 0.4, -0.2], [-0.3, 0.6, 0.1]], dtype=float
    )

    # Electrode 2: 4 waveform features again
    enc_spike_times_2 = jnp.array([1.0, 4.0])
    enc_feats_2 = jnp.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.1, 0.2]], dtype=float)

    # Fit with scalar waveform_std - this should work for all electrodes
    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times_0, enc_spike_times_1, enc_spike_times_2],
        spike_waveform_features=[enc_feats_0, enc_feats_1, enc_feats_2],
        environment=env,
        position_std=np.sqrt(1.0),
        waveform_std=24.0,  # Scalar - should expand per-electrode
        block_size=8,
        disable_progress_bar=True,
    )
    mark_reference = fit_mark_references(
        [enc_feats_0, enc_feats_1, enc_feats_2], kernel_scale=24.0
    )

    # Verify encoding succeeded
    assert len(encoding["encoding_spike_waveform_features"]) == 3
    assert encoding["encoding_spike_waveform_features"][0].shape[1] == 4
    assert encoding["encoding_spike_waveform_features"][1].shape[1] == 3
    assert encoding["encoding_spike_waveform_features"][2].shape[1] == 4

    # Decoding spikes with matching feature counts
    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [
        jnp.array([2.1, 5.2]),  # electrode 0
        jnp.array([3.1]),  # electrode 1
        jnp.array([1.1, 4.1]),  # electrode 2
    ]
    dec_feats = [
        jnp.array([[0.1, 0.05, 0.15, 0.25], [1.1, -0.9, 0.4, -0.4]], dtype=float),
        jnp.array([[0.25, -0.05, 0.35]], dtype=float),
        jnp.array([[0.15, 0.25, 0.35, 0.45], [-0.05, -0.15, 0.15, 0.25]], dtype=float),
    ]

    # Non-local prediction should work
    ll_nonlocal = predict_clusterless_kde_log_likelihood(
        time_edges=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        **encoding,
        mark_reference=mark_reference,
        is_local=False,
    )

    assert ll_nonlocal.shape[0] == t_edges.shape[0] - 1
    assert ll_nonlocal.ndim == 2 and ll_nonlocal.shape[1] > 0
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    # Local prediction should also work
    ll_local = predict_clusterless_kde_log_likelihood(
        time_edges=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        **encoding,
        mark_reference=mark_reference,
        is_local=True,
    )

    assert ll_local.shape == (t_edges.shape[0] - 1, 1)
    assert jnp.all(jnp.isfinite(ll_local))


def test_fit_clusterless_kde_rejects_nonfinite_features(simple_1d_environment):
    """A non-finite waveform feature (a corrupt spike) is rejected at fit,
    consistent with the GMM fit which validates its feature matrix."""
    from non_local_detector.exceptions import ValidationError

    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    enc_spike_times = jnp.array([2.0, 5.0, 7.5])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, jnp.nan], [0.5, 0.5]], dtype=float)

    with pytest.raises(ValidationError, match="finite"):
        fit_clusterless_kde_encoding_model(
            position_time=t_pos,
            position=pos,
            spike_times=[enc_spike_times],
            spike_waveform_features=[enc_feats],
            environment=env,
            position_std=np.sqrt(1.0),
            waveform_std=1.0,
            block_size=8,
            disable_progress_bar=True,
        )


def test_fit_clusterless_kde_ignores_nonfinite_features_out_of_window(
    simple_1d_environment,
):
    """A non-finite feature on an out-of-window spike must not abort the fit.

    Finiteness is validated after clipping spikes to the encoding interval, so a
    NaN feature on a spike that is discarded anyway (outside the window) does not
    reject the otherwise-valid encoding data."""
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    # Two in-window spikes with finite features, plus one out-of-window spike
    # (t=20 > position_time[-1]=10) whose NaN feature is clipped out.
    enc_spike_times = jnp.array([2.0, 7.5, 20.0])
    enc_feats = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, jnp.nan]], dtype=float)

    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    # Only the two in-window spikes survive, and they are all finite.
    bounded = np.asarray(encoding_model["encoding_spike_waveform_features"][0])
    assert bounded.shape[0] == 2
    assert np.all(np.isfinite(bounded))


@pytest.mark.parametrize("name", ["enc_tile_size", "pos_tile_size"])
@pytest.mark.parametrize("value", [0, -3, 2.5, True])
def test_invalid_tile_sizes_raise_at_fit_and_predict(name, value):
    time = np.linspace(0.0, 1.0, 101)
    position = np.linspace(0.0, 10.0, 101)[:, None]
    environment = Environment(place_bin_size=2.0).fit_place_grid(position)
    spikes, marks = [np.array([0.2, 0.4, 0.6])], [np.array([[1.0], [2.0], [3.0]])]
    with pytest.raises(ValidationError, match=name):
        fit_clusterless_kde_encoding_model(
            time, position, spikes, marks, environment, **{name: value}
        )
    model = fit_clusterless_kde_encoding_model(
        time, position, spikes, marks, environment
    )
    model[name] = value
    with pytest.raises(ValidationError, match=name):
        predict_clusterless_kde_log_likelihood(
            time,
            position,
            spikes,
            marks,
            time_edges=np.linspace(0, 1, 6),
            mark_reference=fit_mark_references(marks, DEFAULT_WAVEFORM_STD),
            **model,
        )


@pytest.fixture(scope="module")
def one_feature_kde():
    """A 1-D track with one electrode and one waveform feature."""
    rng = np.random.default_rng(31)
    position_time = np.arange(0.0, 100.0, 0.02)
    position = (5.0 + 4.0 * np.sin(position_time / 3.0))[:, None]
    environment = Environment(place_bin_size=0.5, position_range=((0.0, 10.0),))
    environment.fit_place_grid(position, infer_track_interior=False)
    spike_times = [np.sort(rng.uniform(0.0, 100.0, 300))]
    spike_marks = [rng.normal(0.0, 2.0, (300, 1))]
    encoding = fit_clusterless_kde_encoding_model(
        position_time,
        position,
        spike_times,
        spike_marks,
        environment,
        position_std=1.0,
        waveform_std=1.0,
        disable_progress_bar=True,
    )
    return encoding, fit_mark_references(spike_marks, 1.0)


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "tiling",
    [{}, {"pos_tile_size": 7}, {"enc_tile_size": 64, "use_streaming": True}],
    ids=["untiled", "position-tiled", "streamed"],
)
def test_kde_event_intensity_integrates_to_compensator(
    one_feature_kde, is_local, tiling
):
    """Integrated over marks, the scored intensity is the expected count."""
    encoding, mark_reference = one_feature_kde
    marks = np.linspace(-20.0, 20.0, 801)
    edges = np.arange(len(marks) + 1) * 0.01
    kwargs = dict(
        position_time=np.array([edges[0], edges[-1]]),
        position=np.full((2, 1), 5.0),
        time_edges=edges,
        is_local=is_local,
        mark_reference=mark_reference,
        **{**encoding, **tiling},
    )
    empty = predict_clusterless_kde_log_likelihood(
        spike_times=[np.empty(0)], spike_waveform_features=[np.empty((0, 1))], **kwargs
    )
    events = predict_clusterless_kde_log_likelihood(
        spike_times=[edges[:-1] + 0.005],
        spike_waveform_features=[marks[:, None]],
        **kwargs,
    )
    intensity = np.exp(np.asarray(events) - np.asarray(empty)) / np.diff(edges)[:, None]
    integrated = trapezoid(intensity, marks, axis=0)
    ground = -np.asarray(empty)[0] / np.diff(edges)[0]
    np.testing.assert_allclose(integrated, ground, rtol=1e-4, atol=1e-6)
