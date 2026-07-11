"""Regression test for issue #32.

A zero-encoding-spike electrode (a unit that never fired during training) has an
empty mark-density KDE. When it fires during *decoding*, the log-space
``is_local=True`` path evaluated ``log(density)`` over an empty sample set:
``logsumexp(empty) - logsumexp(empty) = -inf - -inf = NaN``, which the HMM filter
then propagated into the Local-state posterior. The probability-space path is
NaN-free (``mean_rate = 0`` floors the whole term to ``LOG_EPS``). Both paths must
agree.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_kde, clusterless_kde_log
from non_local_detector.likelihoods.common import LOG_EPS, block_log_kde, log_kde


@pytest.mark.unit
def test_log_kde_empty_samples_floors_to_log_eps():
    """The empty-KDE NaN source (#32) is guarded at the ``log_kde`` level.

    With no samples, ``log_den = logsumexp(empty) = -inf`` and a naive
    ``log_num - log_den`` is ``-inf - -inf = NaN``. ``log_kde`` instead returns
    the finite ``LOG_EPS`` floor (via its ``isneginf(log_den)`` guard), so a
    zero-encoding-spike electrode contributes a floor, not a NaN.
    """
    eval_points = jnp.asarray(
        np.random.default_rng(0).standard_normal((5, 3)), dtype=jnp.float32
    )
    empty_samples = jnp.zeros((0, 3))
    std = jnp.ones(3)

    direct = np.asarray(log_kde(eval_points, empty_samples, std, jnp.zeros((0,))))
    blocked = np.asarray(
        block_log_kde(eval_points=eval_points, samples=empty_samples, std=std)
    )
    for out in (direct, blocked):
        assert np.all(np.isfinite(out))
        assert_allclose(out, LOG_EPS)


def _zero_encoding_spike_scenario():
    """Two electrodes; electrode 1 has zero encoding spikes but fires at decode."""
    rng = np.random.default_rng(0)
    dt = 0.02
    n_time = 50
    time = np.arange(n_time) * dt
    t_end = float(time[-1])

    position_time = np.linspace(0.0, t_end, 100)
    position = np.linspace(0.0, 10.0, position_time.size)[:, None]

    n_features = 2
    # Electrode 0: normal encoding spikes.
    e0_enc_times = np.sort(rng.uniform(0.0, t_end, 40))
    e0_enc_feats = rng.standard_normal((e0_enc_times.size, n_features)).astype(
        np.float32
    )
    # Electrode 1: ZERO encoding spikes -> empty mark KDE.
    e1_enc_times = np.zeros((0,))
    e1_enc_feats = np.zeros((0, n_features), dtype=np.float32)

    # Both electrodes fire during decoding (electrode 1 triggers the empty KDE).
    e0_dec_times = np.sort(rng.uniform(0.0, t_end, 15))
    e0_dec_feats = rng.standard_normal((e0_dec_times.size, n_features)).astype(
        np.float32
    )
    e1_dec_times = np.sort(rng.uniform(0.0, t_end, 15))
    e1_dec_feats = rng.standard_normal((e1_dec_times.size, n_features)).astype(
        np.float32
    )

    env = Environment(position_range=[(0.0, 10.0)], place_bin_size=1.0)
    env = env.fit_place_grid(position=position, infer_track_interior=True)

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "enc_spike_times": [e0_enc_times, e1_enc_times],
        "enc_feats": [e0_enc_feats, e1_enc_feats],
        "dec_spike_times": [e0_dec_times, e1_dec_times],
        "dec_feats": [e0_dec_feats, e1_dec_feats],
        "env": env,
    }


def _fit_predict_local(module, s):
    """Fit ``module``'s encoding model and run its is_local=True prediction."""
    encoding = module.fit_clusterless_kde_encoding_model(
        position_time=jnp.asarray(s["position_time"]),
        position=jnp.asarray(s["position"]),
        spike_times=[jnp.asarray(t) for t in s["enc_spike_times"]],
        spike_waveform_features=[jnp.asarray(f) for f in s["enc_feats"]],
        environment=s["env"],
        sampling_frequency=50,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )
    return module.predict_clusterless_kde_log_likelihood(
        jnp.asarray(s["time"]),
        jnp.asarray(s["position_time"]),
        jnp.asarray(s["position"]),
        [jnp.asarray(t) for t in s["dec_spike_times"]],
        [jnp.asarray(f) for f in s["dec_feats"]],
        **encoding,
        is_local=True,
    )


@pytest.mark.unit
def test_kde_log_zero_encoding_spike_local_is_finite_and_matches_prob():
    """Zero-encoding-spike electrode must not NaN the log-space local path (#32).

    The empty mark KDE must contribute the ``LOG_EPS`` floor per decode spike
    (matching the probability-space path), not ``NaN``.
    """
    s = _zero_encoding_spike_scenario()

    ll_log = np.asarray(_fit_predict_local(clusterless_kde_log, s))
    ll_prob = np.asarray(_fit_predict_local(clusterless_kde, s))

    assert np.all(np.isfinite(ll_log)), (
        "zero-encoding-spike electrode produced NaN/Inf in the log-space "
        "is_local=True likelihood"
    )
    assert_allclose(
        ll_log,
        ll_prob,
        rtol=1e-3,
        atol=1e-3,
        err_msg="log-space local likelihood diverged from the probability-space path",
    )
