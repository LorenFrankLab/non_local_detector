"""Test numerical stability of log-space local KDE likelihood computation.

This test verifies that computing local likelihood entirely in log-space
prevents underflow issues that can occur when multiplying many small Gaussian
values in linear space before taking the log.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    _estimate_with_enc_chunking,
    compute_local_log_likelihood,
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    RATE_REFERENCE_SECONDS,
    block_log_kde,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    fit_mark_references,
)
from non_local_detector.tests.likelihoods.conftest import realistic_log_background

pytestmark = pytest.mark.unit


@pytest.fixture
def extreme_feature_data():
    """Synthetic encoding spikes and decoding spikes with extreme waveform features.

    Decoding marks lie about 50 units (about 17 mark bandwidths) from every
    encoding mark in each of 8 dimensions, so each spike's joint kernel is
    around exp(-1000): zero in float32 linear space, finite in log space. The
    decoding spikes are kept out of the encoding fit.
    """
    rng = np.random.default_rng(123)

    # Time and position
    n_time_pos = 500
    position_time = np.linspace(0, 5, n_time_pos)
    position = np.linspace(0, 50, n_time_pos)[:, None]  # 1D position

    n_electrodes = 2
    enc_spike_times, enc_features, dec_spike_times, dec_features = [], [], [], []
    for _i in range(n_electrodes):
        # Encoding spikes (training data) - features centered around 0
        n_enc_spikes = 40
        enc_spike_times.append(
            np.sort(rng.uniform(position_time[0], position_time[-1], n_enc_spikes))
        )
        enc_features.append(rng.standard_normal((n_enc_spikes, 8)) * 5.0)

        # Decoding spikes (test data) - features shifted far from encoding data
        n_dec_spikes = 20
        dec_spike_times.append(np.sort(rng.uniform(2.0, 3.0, n_dec_spikes)))
        dec_features.append(rng.standard_normal((n_dec_spikes, 8)) * 5.0 + 50.0)

    return {
        "position_time": position_time,
        "position": position,
        "enc_spike_times": enc_spike_times,
        "enc_spike_waveform_features": enc_features,
        "dec_spike_times": dec_spike_times,
        "dec_spike_waveform_features": dec_features,
        "dec_time_range": (2.0, 3.0),  # Time range of the decoding spikes
    }


@pytest.fixture
def simple_1d_environment():
    """Create a simple 1D environment for testing."""
    env = Environment(
        environment_name="line", place_bin_size=5.0, position_range=((0.0, 50.0),)
    )
    # Fit with dummy position
    dummy_pos = np.linspace(0.0, 50.0, 11)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    return env


def _local_ll(
    data, env, enc_model, time_edges, spike_times, spike_features, mark_reference
):
    """Local log likelihood, shape (n_time,)."""
    return np.asarray(
        compute_local_log_likelihood(
            time_edges=time_edges,
            position_time=data["position_time"],
            position=data["position"],
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            occupancy_model=enc_model["occupancy_model"],
            gpi_models=enc_model["gpi_models"],
            encoding_spike_waveform_features=enc_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=enc_model["encoding_positions"],
            environment=env,
            mean_rates=jnp.array(enc_model["mean_rates"]),
            position_std=enc_model["position_std"],
            waveform_std=enc_model["waveform_std"],
            mark_reference=mark_reference,
            block_size=50,
            disable_progress_bar=True,
        )
    )[:, 0]


def test_local_likelihood_log_space_prevents_underflow(
    extreme_feature_data, simple_1d_environment
):
    """Spike terms whose joint kernel underflows float32 match a float64 reference.

    Each decoding spike adds ``logaddexp(log intensity, log background)`` to its
    time bin, where the intensity is ``rate * joint KDE / occupancy`` at the
    animal's position. Here every joint kernel is around exp(-1000). A mark
    reference centred far from every mark puts the background near exp(-3900),
    so each spike term is its log intensity: a kernel lost to underflow would
    leave the background, thousands of nats away from the reference.
    """
    data = extreme_feature_data
    env = simple_1d_environment
    position_std, waveform_std = 6.0, 3.0
    enc_model = fit_clusterless_kde_encoding_model(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["enc_spike_times"],
        spike_waveform_features=data["enc_spike_waveform_features"],
        environment=env,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=50,
        disable_progress_bar=True,
    )
    n_electrodes, n_features = len(data["dec_spike_times"]), 8
    far_reference = [
        MarkReference(np.full(n_features, -1e30), np.full(n_features, 1e-30))
    ] * n_electrodes
    dec_start, dec_end = data["dec_time_range"]
    time_edges = time_edges_from_centers(np.linspace(dec_start, dec_end, 10))

    with_spikes = _local_ll(
        data,
        env,
        enc_model,
        time_edges,
        data["dec_spike_times"],
        data["dec_spike_waveform_features"],
        far_reference,
    )
    baseline = _local_ll(
        data,
        env,
        enc_model,
        time_edges,
        [np.empty(0)] * n_electrodes,
        [np.empty((0, n_features))] * n_electrodes,
        far_reference,
    )

    # float64 reference for the summed spike terms of each time bin.
    occupancy_model = enc_model["occupancy_model"]
    occupancy_samples = np.asarray(occupancy_model.samples_, np.float64)[:, 0]
    occupancy_weights = np.asarray(occupancy_model.weights_, np.float64)
    occupancy_std = float(np.ravel(occupancy_model.std)[0])
    expected = np.zeros(time_edges.shape[0] - 1)
    for electrode in range(n_electrodes):
        spike_times = data["dec_spike_times"][electrode]
        marks = data["dec_spike_waveform_features"][electrode]
        spike_position = np.interp(
            spike_times, data["position_time"], data["position"][:, 0]
        )
        log_occupancy = logsumexp(
            norm.logpdf(spike_position[:, None], occupancy_samples[None], occupancy_std)
            + np.log(occupancy_weights),
            axis=1,
        ) - np.log(occupancy_weights.sum())
        encoding_position = np.asarray(
            enc_model["encoding_positions"][electrode], np.float64
        )[:, 0]
        encoding_marks = np.asarray(
            enc_model["encoding_spike_waveform_features"][electrode], np.float64
        )
        log_joint_kernel = norm.logpdf(
            spike_position[:, None], encoding_position[None], position_std
        ) + norm.logpdf(marks[:, None, :], encoding_marks[None], waveform_std).sum(-1)
        assert np.all(log_joint_kernel < -800.0)  # zero in float32 linear space
        log_intensity = (
            np.log(enc_model["mean_rates"][electrode] * RATE_REFERENCE_SECONDS)
            + logsumexp(log_joint_kernel, axis=1)
            - np.log(encoding_position.shape[0])
            - log_occupancy
        )
        log_background = LOG_EPS + far_reference[electrode].log_density(marks)
        assert np.all(log_background < log_intensity - 1000.0)
        rows = np.searchsorted(time_edges, spike_times, side="right") - 1
        np.add.at(expected, rows, np.logaddexp(log_intensity, log_background))

    assert np.all(np.isfinite(with_spikes))
    np.testing.assert_allclose(with_spikes - baseline, expected, rtol=1e-5)


def test_local_likelihood_log_space_moderate_features(
    extreme_feature_data, simple_1d_environment
):
    """Test that log-space local likelihood works correctly with moderate features.

    This is a sanity check that the log-space implementation doesn't break
    normal cases where linear-space would have worked fine.
    """
    data = extreme_feature_data
    env = simple_1d_environment

    # Modify data to have moderate features (not extreme)
    rng = np.random.default_rng(0)
    moderate_spike_features = [
        rng.standard_normal((len(times), 8)) * 5.0  # No shift, moderate scale
        for times in data["enc_spike_times"]
    ]

    # Fit encoding model
    enc_model = fit_clusterless_kde_encoding_model(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["enc_spike_times"],
        spike_waveform_features=moderate_spike_features,
        environment=env,
        position_std=6.0,
        waveform_std=10.0,  # Larger std → less risk of underflow
        block_size=50,
        disable_progress_bar=True,
    )

    # Decode on time window
    time = np.linspace(1.0, 2.0, 10)

    # Compute local likelihood
    ll_local = compute_local_log_likelihood(
        time_edges=time_edges_from_centers(time),
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["enc_spike_times"],
        spike_waveform_features=moderate_spike_features,
        occupancy_model=enc_model["occupancy_model"],
        gpi_models=enc_model["gpi_models"],
        encoding_spike_waveform_features=enc_model["encoding_spike_waveform_features"],
        encoding_positions=enc_model["encoding_positions"],
        environment=env,
        mean_rates=jnp.array(enc_model["mean_rates"]),
        position_std=enc_model["position_std"],
        waveform_std=enc_model["waveform_std"],
        mark_reference=fit_mark_references(moderate_spike_features, kernel_scale=10.0),
        block_size=50,
        disable_progress_bar=True,
    )

    # All values should be finite and reasonable
    assert np.all(np.isfinite(ll_local)), "Local likelihood should be finite"
    assert np.all(ll_local < 0), "Log-likelihood should be negative"
    assert ll_local.shape == (len(time), 1), "Shape should be correct"


def test_block_log_kde_matches_float64_reference_far_from_samples():
    """block_log_kde stays finite and accurate where every kernel underflows.

    Evaluation points sit about 100 units (100 bandwidths) from every sample in
    10 dimensions, so each kernel is around exp(-50,000), far below the
    smallest float32 or float64 number; the log density must still match a
    float64 log-sum-exp evaluation.
    """
    rng = np.random.default_rng(456)
    n_eval, n_samples, n_dims = 10, 20, 10
    eval_points = rng.standard_normal((n_eval, n_dims)) * 5.0 + 100.0
    samples = rng.standard_normal((n_samples, n_dims)) * 5.0
    std = np.ones(n_dims)

    log_density = np.asarray(block_log_kde(eval_points, samples, std, block_size=5))

    expected = logsumexp(
        norm.logpdf(eval_points[:, None, :], samples[None], std).sum(-1), axis=1
    ) - np.log(n_samples)
    assert np.all(expected < -1000.0)
    assert np.all(np.isfinite(log_density))
    np.testing.assert_allclose(log_density, expected, rtol=1e-5)


def test_scaled_gemm_marginal_propagates_nan():
    """A NaN marginal on the scaled-GEMM path propagates, not hidden.

    ``jnp.where(marginal > 0, ..., -inf)`` would launder a NaN marginal (from
    non-finite inputs) to ``-inf`` (zero intensity), since ``NaN > 0`` is
    ``False``, contradicting the module's propagate-NaN contract. It must
    surface as NaN, also after the background is added.
    """
    encoding_features = jnp.array([[0.0], [jnp.nan]])  # (n_enc=2, 1); a NaN mark
    log_position_distance = jnp.array([[-0.1, -0.2], [-0.3, -0.4]])  # (2, n_pos=2)
    log_w = jnp.log(jnp.array([0.5, 0.5]))  # (2,)
    occupancy = jnp.array([1.0, 1.0])  # (n_pos=2,)
    log_background = realistic_log_background(jnp.zeros((1, 1)))

    log_joint = _estimate_with_enc_chunking(
        jnp.zeros((1, 1)),
        encoding_features,
        jnp.ones(1),
        occupancy,
        1.0,
        log_position_distance,
        log_w,
        2,
        None,
        log_background,
    )
    out = jnp.logaddexp(log_joint, log_background[:, None])
    assert bool(jnp.any(jnp.isnan(out))), (
        "A NaN marginal must propagate to core.py's diagnostics, not be "
        "silently replaced by the background."
    )


@pytest.mark.parametrize("enc_tile_size", [2, 4])
def test_scaled_gemm_all_zero_weights_score_the_background_not_nan(enc_tile_size):
    """All-zero encoding weights are a zero intensity, never NaN.

    When every encoding spike is de-weighted (``log_w == -inf``) a compensation
    computing ``-inf - -inf`` would produce NaN. That is a genuine "no effective
    encoding data" case, not a broken input: the intensity is zero (``-inf``)
    and each spike's term is its background, with one encoding tile or several.
    """
    n_enc, n_dec, n_features, n_pos = 4, 2, 2, 3
    rng = np.random.default_rng(0)
    log_position_distance = jnp.asarray(-np.abs(rng.standard_normal((n_enc, n_pos))))
    log_w = jnp.full((n_enc,), -jnp.inf)  # every encoding spike de-weighted
    occupancy = jnp.ones((n_pos,))
    dec_feats = jnp.asarray(rng.standard_normal((n_dec, n_features)))
    enc_feats = jnp.asarray(rng.standard_normal((n_enc, n_features)))
    log_background = realistic_log_background(dec_feats)
    expected = np.broadcast_to(np.asarray(log_background)[:, None], (n_dec, n_pos))

    log_joint = np.asarray(
        _estimate_with_enc_chunking(
            dec_feats,
            enc_feats,
            jnp.ones((n_features,)),
            occupancy,
            1.0,
            log_position_distance,
            log_w,
            enc_tile_size,
            None,
            log_background,
        )
    )
    assert np.all(np.isneginf(log_joint)), (
        "All-zero weights must give zero intensity, not NaN."
    )
    np.testing.assert_array_equal(
        np.logaddexp(log_joint, np.asarray(log_background)[:, None]), expected
    )


def test_log_fit_validates_features_after_windowing(simple_1d_environment):
    """clusterless_kde fit: finiteness is checked on in-window spikes only.

    A NaN feature on a spike outside the encoding interval is clipped out before
    the fit, so it must not abort fitting; a NaN on an in-window spike still
    raises."""
    from non_local_detector.exceptions import ValidationError

    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 50.0, 101)[:, None]

    # Out-of-window NaN (t=20 > position_time[-1]=10) is discarded -> fit succeeds.
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
    bounded = np.asarray(encoding_model["encoding_spike_waveform_features"][0])
    assert bounded.shape[0] == 2
    assert np.all(np.isfinite(bounded))

    # In-window NaN still aborts the fit.
    bad_feats = jnp.array([[0.0, 0.0], [1.0, jnp.nan]], dtype=float)
    with pytest.raises(ValidationError, match="finite"):
        fit_clusterless_kde_encoding_model(
            position_time=t_pos,
            position=pos,
            spike_times=[jnp.array([2.0, 7.5])],
            spike_waveform_features=[bad_feats],
            environment=env,
            position_std=np.sqrt(1.0),
            waveform_std=1.0,
            block_size=8,
            disable_progress_bar=True,
        )
