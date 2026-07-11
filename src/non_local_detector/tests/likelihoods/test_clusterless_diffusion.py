"""Density-correctness, weighted-EM, and degeneracy contracts for clusterless_diffusion.

These are the executable numerical contracts for the ``clusterless_diffusion``
likelihood (fit + non-local predict). The diffusion likelihood estimates the same
marked-point-process intensity as ``clusterless_kde`` but replaces the Gaussian
*position* kernel with the environment's graph heat kernel ``exp(-t L)``
(``t = position_std**2 / 2``); the *mark* kernel stays a Gaussian over waveform
features (via ``kde_distance``).

The joint mark density (spec sec 1/4) is

    p_e(x, m_j) = (H_t D_e)[x, j] / ( (sum_i w_i) * dV(x) )
    D_e[:, j]   = sum_i w_i * K_mark(m_j, m_i) * onehot(bin(x_i))

where ``H_t`` is the mass-conserving heat kernel, ``sum_i w_i`` is the weighted
encoding count, and ``dV(x) = bin_sizes`` the per-bin measure. There is NO
per-column normalization to unit integral, and the mark Gaussian normalizer lives
only in ``kde_distance``. Getting any of those constants wrong yields a
spatially-constant likelihood error that rank/argmax posterior tests cannot catch;
the absolute-value and mark-marginal tests below pin them.
"""

import warnings

import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from non_local_detector.environment import Environment  # noqa: E402
from non_local_detector.likelihoods.clusterless_diffusion import (  # noqa: E402
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import kde_distance  # noqa: E402
from non_local_detector.likelihoods.common import (  # noqa: E402
    EPS,
    LOG_EPS,
    get_position_at_time,
    get_spike_time_bin_ind,
    interpolate_weights_at_spike_times,
)
from non_local_detector.likelihoods.diffusion import (  # noqa: E402
    cached_eigenbasis,
    connected_component_labels,
    diffuse,
    environment_graph,
    get_device_basis,
    heat_kernel_apply,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (  # noqa: E402
    _full_to_local,
    _interior_bin_indices,
)

WAVEFORM_STD = 6.0
POSITION_STD = 1.5


def _make_1d_env(bin_size=1.0, lo=0.0, hi=8.0):
    """1D linear environment with every bin interior (connected graph)."""
    env = Environment(
        environment_name="line", place_bin_size=bin_size, position_range=((lo, hi),)
    )
    dummy = np.linspace(lo, hi, 41)[:, None]
    return env.fit_place_grid(position=dummy, infer_track_interior=False)


def _sim(seed=0, hi=8.0, n_pos=200, n_elec=2, n_features=2, weights=None):
    """Simulated 1D clusterless data: wandering position + random marks/spikes."""
    rng = np.random.default_rng(seed)
    dt = 0.02
    n_time = 40
    time = np.arange(n_time) * dt
    t_end = float(time[-1])
    position_time = np.linspace(0.0, t_end, n_pos)
    position = (np.sin(np.linspace(0.0, 3.0 * np.pi, n_pos)) * 0.5 + 0.5) * hi
    position = np.clip(position, 0.01, hi - 0.01)[:, None]
    enc_t = [np.sort(rng.uniform(0.0, t_end, 30)) for _ in range(n_elec)]
    enc_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in enc_t
    ]
    dec_t = [np.sort(rng.uniform(0.0, t_end, 12)) for _ in range(n_elec)]
    dec_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in dec_t
    ]
    env = _make_1d_env(1.0, 0.0, hi)
    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "enc_t": enc_t,
        "enc_f": enc_f,
        "dec_t": dec_t,
        "dec_f": dec_f,
        "env": env,
        "weights": weights,
    }


def _fit(s, **overrides):
    kwargs = {
        "sampling_frequency": 50,
        "position_std": POSITION_STD,
        "waveform_std": WAVEFORM_STD,
        "weights": s["weights"],
        "disable_progress_bar": True,
    }
    kwargs.update(overrides)
    return fit_clusterless_diffusion_encoding_model(
        s["position_time"],
        s["position"],
        [jnp.asarray(t) for t in s["enc_t"]],
        [jnp.asarray(f) for f in s["enc_f"]],
        s["env"],
        **kwargs,
    )


def _predict(s, enc, **overrides):
    encoding = dict(enc)
    encoding.update(overrides)
    return predict_clusterless_diffusion_log_likelihood(
        jnp.asarray(s["time"]),
        s["position_time"],
        s["position"],
        [jnp.asarray(t) for t in s["dec_t"]],
        [jnp.asarray(f) for f in s["dec_f"]],
        is_local=False,
        **encoding,
    )


# ----------------------------------------------------------------------------
# Task 3 — fit
# ----------------------------------------------------------------------------

FIT_KEYS = {
    "environment",
    "occupancy",
    "summed_ground_process_intensity",
    "encoding_bin_indices",
    "encoding_marks",
    "encoding_weights",
    "weight_total",
    "mean_rates",
    "resolved_rank",
    "node_order",
    "bin_sizes",
    "position_std",
    "waveform_std",
    "block_size",
    "memory_budget",
    "disable_progress_bar",
}


def test_fit_returns_finite_densities_and_keys():
    s = _sim(seed=0)
    enc = _fit(s)

    assert FIT_KEYS.issubset(enc.keys())

    occupancy = np.asarray(enc["occupancy"])
    gpi = np.asarray(enc["summed_ground_process_intensity"])
    assert np.all(np.isfinite(occupancy))
    assert np.all(occupancy >= 0.0)
    assert np.all(np.isfinite(gpi))
    assert np.all(gpi >= 0.0)

    assert isinstance(enc["resolved_rank"], int)
    assert enc["resolved_rank"] >= 1

    n_elec = len(s["enc_t"])
    for key in (
        "encoding_bin_indices",
        "encoding_marks",
        "encoding_weights",
        "weight_total",
        "mean_rates",
    ):
        assert len(enc[key]) == n_elec, key

    # densities live on interior bins (node_order length)
    n_interior = np.asarray(enc["node_order"]).shape[0]
    assert occupancy.shape == (n_interior,)
    assert gpi.shape == (n_interior,)


# ----------------------------------------------------------------------------
# Task 6 — density correctness
# ----------------------------------------------------------------------------


def test_mark_marginal_recovery():
    """Integral over space of p_e recovers the weighted mark marginal.

    sum_x p_e(x, m_j) * dV(x) == sum_i w_i K(m_j, m_i) / sum_i w_i, because the
    heat kernel conserves mass (sum_x (H D_e)[:, j] == sum_i w_i K) and
    p_e = (H D_e) / (sum_i w_i * dV). Guards the sum_i w_i / dV normalization
    (spec sec 1/4).
    """
    weights = 0.5 + np.linspace(0.0, 1.0, 200)  # smooth, strictly positive, non-uniform
    s = _sim(seed=1, weights=weights)
    enc = _fit(s)
    env = s["env"]

    e = 0
    enc_bins = jnp.asarray(enc["encoding_bin_indices"][e])
    enc_marks = jnp.asarray(enc["encoding_marks"][e])
    w = np.asarray(enc["encoding_weights"][e])
    w_total = float(enc["weight_total"][e])
    dV = np.asarray(enc["bin_sizes"])
    n_bins = dV.shape[0]

    rng = np.random.default_rng(7)
    dec_marks = jnp.asarray(
        rng.standard_normal((5, enc_marks.shape[1])).astype(np.float32)
    )

    wf_std = jnp.full(enc_marks.shape[1], WAVEFORM_STD)
    K = kde_distance(dec_marks, enc_marks, wf_std)  # (n_enc, n_dec)
    D = (
        jnp.zeros((n_bins, dec_marks.shape[0]))
        .at[enc_bins]
        .add(jnp.asarray(w)[:, None] * K)
    )
    Lam, Q, labels, n_components = get_device_basis(env, enc["resolved_rank"])
    P = np.asarray(
        heat_kernel_apply(
            Lam, Q, enc["position_std"], D, labels, n_components=n_components
        )
    )
    p_e = P / (w_total * dV[:, None])

    lhs = (p_e * dV[:, None]).sum(axis=0)  # sum_x p_e dV  -> (n_dec,)
    rhs = (w[:, None] * np.asarray(K)).sum(axis=0) / w_total  # sum_i w_i K / sum w_i

    assert np.allclose(lhs, rhs, rtol=1e-3, atol=1e-6), (
        f"mark-marginal mismatch: max|diff|={np.abs(lhs - rhs).max():.3e}"
    )


def test_absolute_log_likelihood_small_fixture():
    """Absolute log-intensity on a hand-computable 3-bin / 2-encoding fixture.

    Hand derivation (interior bins x0=0.5, x1=1.5, x2=2.5; dV=[1,1,1]):

      mark kernel (2 features, std s = WAVEFORM_STD, single Gaussian normalizer):
        K[i, j] = prod_d N(dec_m[j, d]; enc_m[i, d], s)
      encoding histogram, weighted, scattered to bins b_i with weight w_i:
        D[:, j] = sum_i w_i K[i, j] onehot(b_i)
        S[b]    = sum_i w_i onehot(b_i)               (mark-marginalized)
      heat kernel H = diffuse (mass-conserving; full rank -> H @ field):
        P = H @ D ,  Shat = H @ S ,  Ohat = H @ O
      densities (divide by weighted count * dV, NOT unit-integral):
        pi          = clip(Ohat / (sum_w_pos * dV), EPS)
        p_e[:, j]   = P[:, j] / (w_total * dV)
        p_gpi       = Shat / (w_total * dV)
        mean_rate   = w_total / sum_w_pos
      intensity and ground process (spec sec 3):
        lc[:, j]    = log(clip(mean_rate * p_e[:, j] / pi, EPS))
        summed_gpi  = clip(mean_rate * p_gpi / pi, EPS)
        ll[t, x]    = -summed_gpi[x] + sum_{j in bin t} lc[x, j]

    This pins every constant Z: the single mark normalizer, /w_total (not /n_enc,
    since weights are non-uniform), /dV, /pi, and *mean_rate.
    """
    env = _make_1d_env(1.0, 0.0, 3.0)
    graph, node_order, bin_sizes = environment_graph(env)
    n_bins = node_order.shape[0]
    assert n_bins == 3
    dV = np.asarray(bin_sizes)

    # position: pos(t) = 0.5 + 2.0 t over t in [0, 1] -> t=0 -> bin0, t=1 -> bin2
    n_pos = 101
    position_time = np.linspace(0.0, 1.0, n_pos)
    position = (0.5 + 2.0 * position_time)[:, None]
    # Non-uniform ramp whose endpoints (the encoding-spike weights) sum to 3.0 != n_enc
    # (2), so this test also pins /sum_w_i rather than /n_enc.
    weights = np.linspace(0.5, 2.5, n_pos)
    sum_w_pos = float(weights.sum())

    enc_times = np.array([0.0, 1.0])  # -> positions 0.5, 2.5 -> bins 0, 2
    enc_marks = np.array([[0.3, -0.4], [1.1, 0.7]], dtype=np.float32)
    dec_times = np.array([0.2, 0.7])
    dec_marks = np.array([[0.0, 0.0], [0.9, 0.5]], dtype=np.float32)
    time = np.array([0.0, 0.5, 1.0])  # 2 usable time bins: dec 0.2 -> 0, dec 0.7 -> 1

    enc = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        [jnp.asarray(enc_times)],
        [jnp.asarray(enc_marks)],
        env,
        sampling_frequency=50,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        weights=weights,
        heat_kernel_rank=n_bins,  # force full basis for exact hand math
        disable_progress_bar=True,
    )
    ll = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            position_time,
            position,
            [jnp.asarray(dec_times)],
            [jnp.asarray(dec_marks)],
            is_local=False,
            **enc,
        )
    )

    # ---- independent numpy hand computation ----
    eigvals, eigvecs = cached_eigenbasis(env, n_bins)  # full basis
    labels = connected_component_labels(graph)

    def s_gauss(x, m, sd):
        return np.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))

    # bins of encoding spikes via interpolation (independent of production internals)
    f2l = _full_to_local(node_order, env.is_track_interior_.ravel().shape[0])
    enc_pos = get_position_at_time(position_time, position, enc_times, env)
    enc_bins = _interior_bin_indices(env, enc_pos, f2l)
    assert list(enc_bins) == [0, 2]
    w_enc = interpolate_weights_at_spike_times(enc_times, position_time, weights)
    w_total = float(w_enc.sum())
    mean_rate = w_total / sum_w_pos

    n_dec = dec_marks.shape[0]
    K = np.ones((2, n_dec))  # (n_enc, n_dec), explicit single-normalizer mark kernel
    for i in range(2):
        for j in range(n_dec):
            K[i, j] = s_gauss(dec_marks[j, 0], enc_marks[i, 0], WAVEFORM_STD) * s_gauss(
                dec_marks[j, 1], enc_marks[i, 1], WAVEFORM_STD
            )

    D = np.zeros((n_bins, n_dec))
    S = np.zeros((n_bins, 1))
    for i, b in enumerate(enc_bins):
        D[b] += w_enc[i] * K[i]
        S[b, 0] += w_enc[i]

    occ_pos = get_position_at_time(position_time, position, position_time, env)
    occ_bins = _interior_bin_indices(env, occ_pos, f2l)
    occ_field = np.bincount(occ_bins, weights=weights, minlength=n_bins)[:, None]

    P = diffuse(eigvals, eigvecs, POSITION_STD, D, component_labels=labels)
    Shat = diffuse(eigvals, eigvecs, POSITION_STD, S, component_labels=labels)
    Ohat = diffuse(eigvals, eigvecs, POSITION_STD, occ_field, component_labels=labels)

    pi = np.clip(Ohat[:, 0] / (sum_w_pos * dV), EPS, None)
    p_e = P / (w_total * dV[:, None])
    p_gpi = Shat[:, 0] / (w_total * dV)
    lc = np.log(np.clip(mean_rate * p_e / pi[:, None], EPS, None))  # (n_bins, n_dec)
    summed_gpi = np.clip(mean_rate * p_gpi / pi, EPS, None)

    expected = -summed_gpi[None, :] * np.ones((len(time), 1))
    seg = get_spike_time_bin_ind(dec_times, time)  # [0, 1]
    for j, t_bin in enumerate(seg):
        expected[t_bin] += lc[:, j]

    assert ll.shape == (len(time), n_bins)
    assert np.allclose(ll, expected, rtol=1e-3, atol=1e-3), (
        f"absolute log-likelihood mismatch: max|diff|={np.abs(ll - expected).max():.3e}"
    )


def test_mass_invariance_across_ranks():
    """Integrated diffused mass sum_x (H D_e)[:, j] is rank-invariant (mass rescale).

    Not pointwise: the density legitimately changes with rank. Only the integrated
    mass must stay put -- the property that fails if heat_kernel_apply clips without
    the per-component mass-rescale.
    """
    s = _sim(seed=2, hi=12.0)
    env = s["env"]
    graph, node_order, bin_sizes = environment_graph(env)
    n_bins = node_order.shape[0]
    full_rank = n_bins
    half_rank = max(1, n_bins // 2)

    enc = _fit(s)
    e = 0
    enc_bins = jnp.asarray(enc["encoding_bin_indices"][e])
    enc_marks = jnp.asarray(enc["encoding_marks"][e])
    w = jnp.asarray(enc["encoding_weights"][e])

    rng = np.random.default_rng(9)
    dec_marks = jnp.asarray(
        rng.standard_normal((4, enc_marks.shape[1])).astype(np.float32)
    )
    K = kde_distance(dec_marks, enc_marks, jnp.full(enc_marks.shape[1], WAVEFORM_STD))
    D = jnp.zeros((n_bins, dec_marks.shape[0])).at[enc_bins].add(w[:, None] * K)

    masses = {}
    for rank in (full_rank, half_rank):
        Lam, Q, labels, n_components = get_device_basis(env, rank)
        P = heat_kernel_apply(
            Lam, Q, enc["position_std"], D, labels, n_components=n_components
        )
        masses[rank] = np.asarray(P.sum(axis=0))

    assert np.allclose(masses[full_rank], masses[half_rank], rtol=1e-4, atol=1e-8), (
        f"integrated mass changed with rank: "
        f"max|diff|={np.abs(masses[full_rank] - masses[half_rank]).max():.3e}"
    )
    # sanity: the mass equals the input column mass (sum_i w_i K)
    input_mass = np.asarray(D.sum(axis=0))
    assert np.allclose(masses[full_rank], input_mass, rtol=1e-4, atol=1e-8)


# ----------------------------------------------------------------------------
# Task 6 — weighted EM + degeneracy
# ----------------------------------------------------------------------------


def test_binary_weights_match_hard_subset():
    """A binary-weight fit equals a fit on the hard subset (zero-weight spikes dropped).

    Mirrors test_clusterless_weights: weight 1.0 on the position block [3, 7];
    encoding spikes at 4/5/6 fall inside (weight 1), 1/9 outside (weight 0). The
    weighted fit and the subset fit must decode to the same log-likelihood.
    """
    env = _make_1d_env(1.0, 0.0, 10.0)
    position_time = np.linspace(0.0, 10.0, 101)
    position = np.linspace(0.0, 10.0, 101)[:, None]

    all_spike_times = np.array([1.0, 4.0, 5.0, 6.0, 9.0])
    all_feats = np.array(
        [[2.0, 2.0], [0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [-2.0, -2.0]],
        dtype=np.float32,
    )
    in_block = np.array([False, True, True, True, False])

    t_np = position_time
    weights = np.where((t_np >= 3.0) & (t_np <= 7.0), 1.0, 0.0)
    block = (t_np >= 3.0) & (t_np <= 7.0)

    common = {
        "sampling_frequency": 10,
        "position_std": POSITION_STD,
        "waveform_std": WAVEFORM_STD,
        "disable_progress_bar": True,
    }
    weighted = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        [jnp.asarray(all_spike_times)],
        [jnp.asarray(all_feats)],
        env,
        weights=weights,
        **common,
    )
    subset = fit_clusterless_diffusion_encoding_model(
        position_time[block],
        position[block],
        [jnp.asarray(all_spike_times[in_block])],
        [jnp.asarray(all_feats[in_block])],
        env,
        weights=None,
        **common,
    )

    time = np.linspace(0.0, 10.0, 6)
    dec_spike_times = [jnp.asarray(np.array([4.2, 5.6]))]
    dec_feats = [jnp.asarray(np.array([[0.1, 0.05], [0.9, -0.8]], dtype=np.float32))]

    def predict(enc, pt, pos):
        return np.asarray(
            predict_clusterless_diffusion_log_likelihood(
                jnp.asarray(time),
                pt,
                pos,
                dec_spike_times,
                dec_feats,
                is_local=False,
                **enc,
            )
        )

    ll_weighted = predict(weighted, position_time, position)
    ll_subset = predict(subset, position_time[block], position[block])

    assert np.allclose(ll_weighted, ll_subset, rtol=1e-5, atol=1e-6), (
        f"binary-weighted fit != subset fit; max|diff|="
        f"{np.abs(ll_weighted - ll_subset).max():.3e}"
    )


def test_nonlocal_finite_and_zero_rate():
    """Non-local output is finite; a zero-weight electrode floors decode spikes to LOG_EPS."""
    env = _make_1d_env(1.0, 0.0, 10.0)
    position_time = np.linspace(0.0, 10.0, 201)
    position = np.linspace(0.0, 10.0, 201)[:, None]
    # left electrode fires in the weighted half, right electrode in the zero half
    enc_times = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([6.0, 7.0, 8.0, 9.0])]
    enc_feats = [
        np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.3], [0.4, -0.2]], dtype=np.float32),
        np.array(
            [[1.0, -1.0], [0.9, -0.8], [1.1, -1.2], [0.8, -0.9]], dtype=np.float32
        ),
    ]
    weights = np.where(position_time < 5.0, 1.0, 0.0)

    with pytest.warns(UserWarning, match="zero total encoding weight"):
        enc = fit_clusterless_diffusion_encoding_model(
            position_time,
            position,
            [jnp.asarray(t) for t in enc_times],
            [jnp.asarray(f) for f in enc_feats],
            env,
            weights=weights,
            sampling_frequency=20,
            position_std=POSITION_STD,
            waveform_std=WAVEFORM_STD,
            disable_progress_bar=True,
        )

    assert enc["weight_total"][1] == 0.0
    assert enc["mean_rates"][1] == 0.0

    # decode: only the zero-rate electrode has spikes, so its contribution is isolated
    time = np.linspace(0.0, 10.0, 6)
    dec_times = [np.array([]), np.array([2.5, 7.5])]
    dec_feats = [
        np.zeros((0, 2), dtype=np.float32),
        np.array([[1.0, -1.0], [0.9, -0.9]], dtype=np.float32),
    ]
    ll = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            position_time,
            position,
            [jnp.asarray(t) for t in dec_times],
            [jnp.asarray(f) for f in dec_feats],
            is_local=False,
            **enc,
        )
    )

    assert np.all(np.isfinite(ll))
    gpi = np.asarray(enc["summed_ground_process_intensity"])
    seg = get_spike_time_bin_ind(dec_times[1], time)
    for t_bin in seg:
        # only the zero-rate electrode contributed at this bin -> exactly LOG_EPS + (-gpi)
        assert np.allclose(ll[t_bin] + gpi, LOG_EPS, atol=1e-6), (
            f"zero-rate electrode did not floor to LOG_EPS at time bin {t_bin}"
        )


def test_block_parity():
    """A tiny effective block (memory_budget forced small) matches one large block."""
    s = _sim(seed=3, hi=10.0)
    enc = _fit(s)

    ll_big = np.asarray(
        _predict(s, enc)
    )  # default 512 MiB budget -> block == block_size
    ll_small = np.asarray(_predict(s, enc, memory_budget=1))  # -> effective_block == 1

    assert np.allclose(ll_big, ll_small, rtol=1e-5, atol=1e-6), (
        f"block parity failed: max|diff|={np.abs(ll_big - ll_small).max():.3e}"
    )


def test_zero_rate_fit_and_predict_finite_and_warn():
    """All-zero weights: both the occupancy and per-electrode warnings fire; predict
    stays finite and floors every decode spike to LOG_EPS."""
    env = _make_1d_env(1.0, 0.0, 6.0)
    position_time = np.linspace(0.0, 6.0, 101)
    position = np.linspace(0.0, 6.0, 101)[:, None]
    enc_times = [np.array([1.0, 2.5, 4.0])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=np.float32)]
    weights = np.zeros(101)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        enc = fit_clusterless_diffusion_encoding_model(
            position_time,
            position,
            [jnp.asarray(t) for t in enc_times],
            [jnp.asarray(f) for f in enc_feats],
            env,
            weights=weights,
            sampling_frequency=20,
            position_std=POSITION_STD,
            waveform_std=WAVEFORM_STD,
            disable_progress_bar=True,
        )
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert any("occupancy weights sum to 0" in m for m in messages), messages
    assert any("zero total encoding weight" in m for m in messages), messages

    assert np.all(np.isfinite(np.asarray(enc["occupancy"])))
    assert np.all(np.isfinite(np.asarray(enc["summed_ground_process_intensity"])))
    assert enc["weight_total"][0] == 0.0
    assert enc["mean_rates"][0] == 0.0

    time = np.array([0.0, 2.0, 4.0, 6.0])  # 3 usable bins
    dec_times = [np.array([1.0, 3.0])]  # -> time bins 0 and 1
    dec_feats = [np.array([[0.1, 0.1], [0.9, -0.8]], dtype=np.float32)]
    ll = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            position_time,
            position,
            [jnp.asarray(t) for t in dec_times],
            [jnp.asarray(f) for f in dec_feats],
            is_local=False,
            **enc,
        )
    )
    assert np.all(np.isfinite(ll))
    gpi = np.asarray(enc["summed_ground_process_intensity"])
    # all electrodes zero-rate -> gpi floored to EPS everywhere
    assert np.allclose(gpi, EPS)
    seg = get_spike_time_bin_ind(dec_times[0], time)
    for t_bin in seg:
        assert np.allclose(ll[t_bin], LOG_EPS - EPS, atol=1e-6)
