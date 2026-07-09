"""Per-sample weights must flow through the clusterless encoding fits.

These guard the EM M-step, which re-fits the encoding model with the Local-state
posterior as ``weights`` (models/base.py). If the fit ignores ``weights`` the
posterior-weighted refit is silently uniform.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)


def _predict_nonlocal(env, encoding, t_pos, pos, dec_spike_times, dec_feats, t_edges):
    return np.asarray(
        predict_clusterless_kde_log_likelihood(
            time=t_edges,
            position_time=t_pos,
            position=pos,
            spike_times=dec_spike_times,
            spike_waveform_features=dec_feats,
            occupancy=encoding["occupancy"],
            occupancy_model=encoding["occupancy_model"],
            gpi_models=encoding["gpi_models"],
            encoding_spike_waveform_features=encoding[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding["encoding_positions"],
            encoding_weights=encoding["encoding_weights"],
            environment=env,
            mean_rates=jnp.asarray(encoding["mean_rates"]),
            summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
            position_std=jnp.asarray(encoding["position_std"]),
            waveform_std=jnp.asarray(encoding["waveform_std"]),
            is_local=False,
            block_size=16,
            disable_progress_bar=True,
        )
    )


def _fit(env, t_pos, pos, spike_times, feats, weights=None):
    return fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=feats,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=16,
        disable_progress_bar=True,
    )


def test_weights_change_the_clusterless_kde_occupancy(simple_1d_environment):
    """Non-uniform weights must change the fitted occupancy (else weights ignored)."""
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes = [jnp.array([2.0, 5.0, 7.5])]
    feats = [jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    uniform = _fit(env, t_pos, pos, spikes, feats, weights=None)
    # Up-weight the left half of the track so occupancy shifts left.
    w = np.where(np.asarray(t_pos) < 5.0, 3.0, 0.5)
    weighted = _fit(env, t_pos, pos, spikes, feats, weights=w)

    assert not np.allclose(
        np.asarray(uniform["occupancy"]), np.asarray(weighted["occupancy"])
    ), "weights had no effect on occupancy -- they are being ignored"


def test_binary_weights_match_subset_fit_clusterless_kde(simple_1d_environment):
    """Binary weights (1 on a position block, 0 elsewhere) reproduce a subset fit.

    Weight-1 covers position times in [3, 7]; encoding spikes at 4/5/6 fall inside
    (weight 1) and spikes at 1/9 fall outside (weight 0). The full weighted fit must
    give the same decode likelihood as fitting on only the in-block data.
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)  # dt = 0.1
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]

    all_spike_times = jnp.array([1.0, 4.0, 5.0, 6.0, 9.0])
    all_feats = jnp.array(
        [[2.0, 2.0], [0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [-2.0, -2.0]], dtype=float
    )
    in_block = np.array([False, True, True, True, False])

    # Binary weights: 1.0 on position samples with 3.0 <= t <= 7.0, else 0.0.
    t_np = np.asarray(t_pos)
    weights = np.where((t_np >= 3.0) & (t_np <= 7.0), 1.0, 0.0)
    block = (t_np >= 3.0) & (t_np <= 7.0)

    weighted = _fit(env, t_pos, pos, [all_spike_times], [all_feats], weights=weights)
    subset = _fit(
        env,
        t_pos[block],
        pos[block],
        [all_spike_times[in_block]],
        [all_feats[in_block]],
        weights=None,
    )

    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [jnp.array([4.2, 5.6])]
    dec_feats = [jnp.array([[0.1, 0.05], [0.9, -0.8]], dtype=float)]

    ll_weighted = _predict_nonlocal(
        env, weighted, t_pos, pos, dec_spike_times, dec_feats, t_edges
    )
    ll_subset = _predict_nonlocal(
        env, subset, t_pos[block], pos[block], dec_spike_times, dec_feats, t_edges
    )

    assert np.allclose(ll_weighted, ll_subset, rtol=1e-5, atol=1e-6), (
        f"binary-weighted fit != subset fit; max|diff|="
        f"{np.abs(ll_weighted - ll_subset).max():.3e}"
    )


def test_weights_change_the_clusterless_gmm_occupancy(simple_1d_environment):
    """Non-uniform weights must change the fitted GMM occupancy (else ignored).

    A weaker check than the KDE binary-subset test: the GMM's EM / k-means init is
    not exactly reproducible under weight-0 points, so we only assert that weights
    have an effect (the module previously hard-coded ``weights = None``).
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    spikes = [jnp.array([2.0, 3.0, 5.0, 6.0, 7.5])]
    feats = [
        jnp.array(
            [[0.0, 0.0], [0.2, 0.1], [1.0, -1.0], [-0.3, 0.4], [0.5, 0.5]], dtype=float
        )
    ]
    kw = {
        "spike_times": spikes,
        "spike_waveform_features": feats,
        "environment": env,
        "gmm_components_occupancy": 4,
        "gmm_components_gpi": 2,
        "gmm_components_joint": 2,
        "gmm_random_state": 0,
        "disable_progress_bar": True,
    }
    uniform = fit_clusterless_gmm_encoding_model(
        position_time=t_pos, position=pos, weights=None, **kw
    )
    w = np.where(np.asarray(t_pos) < 5.0, 3.0, 0.5)
    weighted = fit_clusterless_gmm_encoding_model(
        position_time=t_pos, position=pos, weights=w, **kw
    )
    assert not np.allclose(
        np.asarray(uniform["log_occupancy"]), np.asarray(weighted["log_occupancy"])
    ), "weights had no effect on the GMM occupancy -- they are being ignored"


def test_predict_preserves_old_positional_api(simple_1d_environment):
    """Old positional predict calls must still bind is_local/block_size correctly.

    Guards against re-introducing the API break where ``encoding_weights`` was
    inserted mid-signature: a positional ``is_local=True`` (16th positional arg)
    must bind to ``is_local``, not to ``encoding_weights`` (which would raise
    ``TypeError: 'bool' object is not iterable``).
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    enc = _fit(
        env,
        t_pos,
        pos,
        [jnp.array([2.0, 5.0, 7.5])],
        [jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)],
    )
    t_edges = jnp.linspace(0.0, 10.0, 6)
    # Positional call through is_local (16th positional arg), as pre-change callers did.
    ll = predict_clusterless_kde_log_likelihood(
        t_edges,
        t_pos,
        pos,
        [jnp.array([4.2, 5.6])],
        [jnp.array([[0.1, 0.05], [0.9, -0.8]], dtype=float)],
        enc["occupancy"],
        enc["occupancy_model"],
        enc["gpi_models"],
        enc["encoding_spike_waveform_features"],
        enc["encoding_positions"],
        env,
        jnp.asarray(enc["mean_rates"]),
        enc["summed_ground_process_intensity"],
        jnp.asarray(enc["position_std"]),
        jnp.asarray(enc["waveform_std"]),
        True,  # is_local, positional
    )
    assert np.asarray(ll).shape == (t_edges.shape[0], 1)  # local -> (n_time, 1)


def _fit_gmm(env, t_pos, pos, spikes, feats, weights):
    return fit_clusterless_gmm_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        spike_waveform_features=feats,
        environment=env,
        weights=weights,
        gmm_components_occupancy=4,
        gmm_components_gpi=2,
        gmm_components_joint=2,
        gmm_random_state=0,
        disable_progress_bar=True,
    )


def test_all_zero_weights_are_finite_clusterless_gmm(simple_1d_environment):
    """All-zero weights must warn and fall back to an unweighted fit, not crash.

    An all-zero ``sample_weight`` makes the GMM EM M-step divide by 0 and raise
    ``RuntimeError: Fitting failed``. A fully-masked encoding group (e.g. a state
    the EM posterior never assigns) must instead warn and produce a finite
    (unweighted) encoding model so the fit degrades gracefully.
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    spikes = [jnp.array([2.0, 3.0, 5.0, 6.0, 7.5])]
    feats = [
        jnp.array(
            [[0.0, 0.0], [0.2, 0.1], [1.0, -1.0], [-0.3, 0.4], [0.5, 0.5]], dtype=float
        )
    ]

    with pytest.warns(UserWarning, match="sum to 0"):
        enc = _fit_gmm(env, t_pos, pos, spikes, feats, weights=np.zeros(201))

    assert np.all(np.isfinite(np.asarray(enc["log_occupancy"])))
    assert np.all(np.isfinite(np.asarray(enc["mean_rates"])))
    assert np.all(np.isfinite(np.asarray(enc["summed_ground_process_intensity"])))


def test_per_electrode_zero_weights_are_finite_clusterless_gmm(simple_1d_environment):
    """An electrode whose spikes all fall in the zero-weight region must not crash.

    Global weight is positive (left half weighted, right half 0), so there is no
    global warning, but the second electrode's spikes all fall in the zero-weight
    right half -> its per-spike weights are all 0. That all-zero per-electrode
    ``sample_weight`` previously crashed the GPI / joint GMM fit; it must fall back
    to an unweighted per-electrode fit instead.
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    # Left electrode fires in the weighted half; right electrode fires in the zero half.
    spikes = [jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([6.0, 7.0, 8.0, 9.0])]
    feats = [
        jnp.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.3], [0.4, -0.2]], dtype=float),
        jnp.array([[1.0, -1.0], [0.9, -0.8], [1.1, -1.2], [0.8, -0.9]], dtype=float),
    ]
    w = np.where(np.asarray(t_pos) < 5.0, 1.0, 0.0)

    enc = _fit_gmm(env, t_pos, pos, spikes, feats, weights=w)

    mean_rates = np.asarray(enc["mean_rates"])
    assert np.all(np.isfinite(mean_rates))
    assert np.all(np.isfinite(np.asarray(enc["log_occupancy"])))
    assert np.all(np.isfinite(np.asarray(enc["summed_ground_process_intensity"])))
    # The zero-weight electrode contributes ~0 rate; the weighted one is larger.
    assert mean_rates[1] < mean_rates[0]


def test_all_zero_weights_are_finite_clusterless_kde(simple_1d_environment):
    """All-zero weights: the KDE fit stays finite with zero mean rate (no NaN/crash)."""
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes = [jnp.array([2.0, 5.0, 7.5])]
    feats = [jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    enc = _fit(env, t_pos, pos, spikes, feats, weights=np.zeros(101))

    assert np.all(np.isfinite(np.asarray(enc["occupancy"])))
    assert np.all(np.isfinite(np.asarray(enc["mean_rates"])))
    assert np.allclose(np.asarray(enc["mean_rates"]), 0.0)
    assert np.all(np.isfinite(np.asarray(enc["summed_ground_process_intensity"])))


def test_none_weights_match_uniform_clusterless_gmm(simple_1d_environment):
    """``weights=None`` (unweighted path) matches explicit uniform weights.

    ``None`` takes the historical byte-identical unweighted GMM path; explicit
    all-ones weights go through the weighted EM, which differs only by a float32
    reduction-order (both express "every sample counts equally"). They must agree
    to floating-point tolerance -- a large gap would mean ``None`` no longer
    reproduces the pre-weights fit.
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    spikes = [jnp.array([2.0, 3.0, 5.0, 6.0, 7.5])]
    feats = [
        jnp.array(
            [[0.0, 0.0], [0.2, 0.1], [1.0, -1.0], [-0.3, 0.4], [0.5, 0.5]], dtype=float
        )
    ]

    none_enc = _fit_gmm(env, t_pos, pos, spikes, feats, weights=None)
    ones_enc = _fit_gmm(env, t_pos, pos, spikes, feats, weights=np.ones(201))

    assert np.allclose(
        np.asarray(none_enc["log_occupancy"]),
        np.asarray(ones_enc["log_occupancy"]),
        atol=1e-4,
        rtol=1e-4,
    )
