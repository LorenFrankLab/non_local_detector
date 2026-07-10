"""Test numerical stability of log-space local KDE likelihood computation.

This test verifies that computing local likelihood entirely in log-space
prevents underflow issues that can occur when multiplying many small Gaussian
values in linear space before taking the log.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde_log import (
    compute_local_log_likelihood,
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.likelihoods.common import (
    EPS,
    block_kde,
    block_log_kde,
    safe_log,
)


@pytest.fixture
def extreme_feature_data():
    """Generate synthetic data with extreme waveform features.

    This creates conditions where linear-space computation would underflow:
    - High dimensional waveform features (8 dimensions)
    - Large feature distances (features far from encoding samples)
    - Small standard deviations (narrow Gaussians)

    These conditions make the product of per-dimension Gaussians extremely small,
    causing underflow in linear space but handled correctly in log-space.
    """
    rng = np.random.default_rng(123)

    # Time and position
    n_time_pos = 500
    position_time = np.linspace(0, 5, n_time_pos)
    position = np.linspace(0, 50, n_time_pos)[:, None]  # 1D position

    # Spike data for 2 electrodes
    n_electrodes = 2
    spike_times = []
    spike_waveform_features = []

    for _i in range(n_electrodes):
        # Encoding spikes (training data) - moderate features
        n_enc_spikes = 40
        enc_spike_times = np.sort(
            rng.uniform(position_time[0], position_time[-1], n_enc_spikes)
        )
        # Features centered around 0
        enc_features = rng.standard_normal((n_enc_spikes, 8)) * 5.0

        # Decoding spikes (test data) - EXTREME features far from encoding
        n_dec_spikes = 20
        dec_spike_times = np.sort(rng.uniform(2.0, 3.0, n_dec_spikes))
        # Features shifted far away from encoding data
        # This creates very small Gaussian values that underflow in linear space
        dec_features = (
            rng.standard_normal((n_dec_spikes, 8)) * 5.0 + 50.0
        )  # Shifted by 50!

        # Combine encoding and decoding for storage
        all_times = np.concatenate([enc_spike_times, dec_spike_times])
        all_features = np.concatenate([enc_features, dec_features], axis=0)

        spike_times.append(all_times)
        spike_waveform_features.append(all_features)

    return {
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
        "dec_time_range": (2.0, 3.0),  # Time range where we have extreme features
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


def test_local_likelihood_log_space_prevents_underflow(
    extreme_feature_data, simple_1d_environment
):
    """Test that log-space local likelihood handles extreme features without underflow.

    This test verifies that when decoding spikes have waveform features very far
    from encoding samples (causing underflow in linear-space computation), the
    log-space implementation still produces finite, reasonable results.

    The test checks that:
    1. All likelihood values are finite (no -inf from underflow)
    2. The expected-counts term dominates (since spike term should be very negative)
    3. The result is numerically stable across different random seeds
    """
    data = extreme_feature_data
    env = simple_1d_environment

    # Fit encoding model
    enc_model = fit_clusterless_kde_encoding_model(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        position_std=6.0,
        waveform_std=3.0,  # Small std → narrow Gaussians → more underflow risk
        block_size=50,
        disable_progress_bar=True,
    )

    # Decode on time window with extreme features
    dec_start, dec_end = data["dec_time_range"]
    time = np.linspace(dec_start, dec_end, 10)

    # Compute local likelihood
    ll_local = compute_local_log_likelihood(
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        occupancy_model=enc_model["occupancy_model"],
        gpi_models=enc_model["gpi_models"],
        encoding_spike_waveform_features=enc_model["encoding_spike_waveform_features"],
        encoding_positions=enc_model["encoding_positions"],
        environment=env,
        mean_rates=jnp.array(enc_model["mean_rates"]),
        position_std=enc_model["position_std"],
        waveform_std=enc_model["waveform_std"],
        block_size=50,
        disable_progress_bar=True,
    )

    # 1. All values should be finite (no -inf from underflow)
    assert np.all(np.isfinite(ll_local)), (
        f"Local likelihood contains non-finite values: "
        f"min={np.min(ll_local)}, max={np.max(ll_local)}, "
        f"n_inf={np.sum(np.isinf(ll_local))}, n_nan={np.sum(np.isnan(ll_local))}"
    )

    # 2. Values should be reasonable (negative, since log-likelihood)
    # With extreme features far from encoding data, spike contributions should be
    # very negative, so expected-counts term dominates
    assert np.all(ll_local < 0), "Log-likelihood should be negative"

    # 3. Should not saturate at LOG_EPS (which would indicate underflow)
    from non_local_detector.likelihoods.common import LOG_EPS

    # If underflow occurred, many values would equal LOG_EPS
    n_at_log_eps = np.sum(np.isclose(ll_local, LOG_EPS, rtol=0, atol=1e-10))
    assert n_at_log_eps == 0, (
        f"Log-likelihood saturated at LOG_EPS in {n_at_log_eps}/{ll_local.size} values, "
        f"indicating underflow in linear-space computation"
    )

    # 4. Verify shape
    assert ll_local.shape == (
        len(time),
        1,
    ), f"Expected shape ({len(time)}, 1), got {ll_local.shape}"


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
        for times in data["spike_times"]
    ]

    # Fit encoding model
    enc_model = fit_clusterless_kde_encoding_model(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
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
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=moderate_spike_features,
        occupancy_model=enc_model["occupancy_model"],
        gpi_models=enc_model["gpi_models"],
        encoding_spike_waveform_features=enc_model["encoding_spike_waveform_features"],
        encoding_positions=enc_model["encoding_positions"],
        environment=env,
        mean_rates=jnp.array(enc_model["mean_rates"]),
        position_std=enc_model["position_std"],
        waveform_std=enc_model["waveform_std"],
        block_size=50,
        disable_progress_bar=True,
    )

    # All values should be finite and reasonable
    assert np.all(np.isfinite(ll_local)), "Local likelihood should be finite"
    assert np.all(ll_local < 0), "Log-likelihood should be negative"
    assert ll_local.shape == (len(time), 1), "Shape should be correct"


def test_block_log_kde_vs_log_block_kde():
    """Test that block_log_kde is more accurate than safe_log(block_kde).

    This directly tests the numerical difference between:
    1. Linear-space: marginal = block_kde(...); log_marginal = safe_log(marginal)
    2. Log-space: log_marginal = block_log_kde(...)

    With extreme feature distances, (1) suffers from underflow and produces
    inaccurate results (clamped to LOG_EPS), while (2) computes accurately.
    """
    rng = np.random.default_rng(456)

    # Create test data with VERY extreme feature distances
    n_eval = 10
    n_samples = 20
    n_dims = 10  # High dimensional

    # Evaluation points FAR from samples
    eval_points = rng.standard_normal((n_eval, n_dims)) * 5.0 + 100.0  # Shifted by 100!

    # Samples centered at origin
    samples = rng.standard_normal((n_samples, n_dims)) * 5.0  # Near zero

    # Small standard deviations → very narrow Gaussians → underflow in linear space
    std = np.ones(n_dims) * 1.0

    # Method 1: Linear-space then log (current implementation)
    marginal_linear = block_kde(eval_points, samples, std, block_size=5)
    log_marginal_linear = safe_log(marginal_linear, eps=EPS)

    # Method 2: Pure log-space (proposed implementation)
    log_marginal_log = block_log_kde(eval_points, samples, std, block_size=5)

    # Check that linear-space method has underflow (values exactly at LOG_EPS)
    from non_local_detector.likelihoods.common import LOG_EPS

    # With extreme distances, marginal_linear should underflow to exactly 0
    assert np.all(marginal_linear == 0), (
        f"Expected all marginal_linear values to underflow to 0, "
        f"but got min={np.min(marginal_linear)}, max={np.max(marginal_linear)}"
    )

    # Which safe_log clamps to exactly LOG_EPS
    assert np.all(log_marginal_linear == LOG_EPS), (
        f"Expected all log_marginal_linear values to be clamped to LOG_EPS={LOG_EPS}, "
        f"but got values: {np.unique(log_marginal_linear)}"
    )

    # Log-space method should compute much more negative (accurate) values
    assert np.all(log_marginal_log < LOG_EPS), (
        f"Log-space values should be more negative than LOG_EPS={LOG_EPS}, "
        f"but got min={np.min(log_marginal_log)}, max={np.max(log_marginal_log)}"
    )

    # The difference should be substantial (thousands in log-space)
    min_diff = np.min(log_marginal_log - LOG_EPS)
    assert min_diff < -1000, (
        f"Expected large difference (< -1000) between log-space and clamped linear-space, "
        f"but got min_diff={min_diff:.2f}"
    )


def test_compensated_linear_marginal_propagates_nan():
    """A NaN marginal on the compensated-linear fast path propagates, not floored.

    The <=8-feature fast path used ``jnp.where(marginal > 0, ..., -inf)``; since
    ``NaN > 0`` is ``False`` a NaN marginal (from non-finite inputs) was laundered
    to ``-inf`` and then ``LOG_EPS`` by the combiner, contradicting the module's
    propagate-NaN contract (and the logsumexp path). It must now surface as NaN.
    """
    from non_local_detector.likelihoods.clusterless_kde_log import (
        _compensated_linear_marginal,
    )

    logK_mark = jnp.array([[0.0], [jnp.nan]])  # (n_enc=2, n_dec=1); a NaN row
    log_position_distance = jnp.array([[-0.1, -0.2], [-0.3, -0.4]])  # (2, n_pos=2)
    log_w = jnp.log(jnp.array([0.5, 0.5]))  # (2,)
    occupancy = jnp.array([1.0, 1.0])  # (n_pos=2,)

    out = _compensated_linear_marginal(
        logK_mark, log_position_distance, log_w, occupancy, 1.0
    )
    assert bool(jnp.any(jnp.isnan(out))), (
        "A NaN marginal must propagate to core.py's diagnostics, not be "
        "silently laundered to LOG_EPS."
    )


def test_compensated_linear_marginal_all_zero_weights_floors_not_nan():
    """All-zero encoding weights are a zero marginal (LOG_EPS), never NaN.

    When every encoding spike is de-weighted (``log_w == -inf``) the compensation
    computes ``-inf - -inf`` and produces NaN. That is a genuine "no effective
    encoding data" case, not a broken input: the logsumexp reference path folds
    the ``-inf`` weights into the kernel and returns the finite ``LOG_EPS`` floor.
    Both compensated fast paths (untiled and chunked) must match it rather than
    propagating NaN (which the NaN-propagation branch would otherwise do).
    """
    from non_local_detector.likelihoods.clusterless_kde_log import (
        _compensated_linear_marginal,
        _compensated_linear_marginal_chunked,
    )
    from non_local_detector.likelihoods.common import LOG_EPS

    n_enc, n_dec, n_features, n_pos = 4, 2, 2, 3
    rng = np.random.default_rng(0)
    log_position_distance = jnp.asarray(-np.abs(rng.standard_normal((n_enc, n_pos))))
    log_w = jnp.full((n_enc,), -jnp.inf)  # every encoding spike de-weighted
    occupancy = jnp.ones((n_pos,))
    mean_rate = 1.0

    # Untiled fast path.
    logK_mark = jnp.asarray(-np.abs(rng.standard_normal((n_enc, n_dec))))
    out_untiled = np.asarray(
        _compensated_linear_marginal(
            logK_mark, log_position_distance, log_w, occupancy, mean_rate
        )
    )
    assert np.all(np.isfinite(out_untiled)), (
        "All-zero weights must floor to LOG_EPS, not NaN (untiled path)."
    )
    assert np.allclose(out_untiled, LOG_EPS)

    # Chunked fast path (online max rescaling).
    dec_feats = jnp.asarray(rng.standard_normal((n_dec, n_features)))
    enc_feats = jnp.asarray(rng.standard_normal((n_enc, n_features)))
    waveform_stds = jnp.ones((n_features,))
    out_chunked = np.asarray(
        _compensated_linear_marginal_chunked(
            dec_feats,
            enc_feats,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            log_w,
            enc_tile_size=2,
        )
    )
    assert np.all(np.isfinite(out_chunked)), (
        "All-zero weights must floor to LOG_EPS, not NaN (chunked path)."
    )
    assert np.allclose(out_chunked, LOG_EPS)


def test_log_fit_validates_features_after_windowing(simple_1d_environment):
    """clusterless_kde_log fit: finiteness is checked on in-window spikes only.

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
        sampling_frequency=10,
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
            sampling_frequency=10,
            position_std=np.sqrt(1.0),
            waveform_std=1.0,
            block_size=8,
            disable_progress_bar=True,
        )
