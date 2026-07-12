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

import time
import warnings

import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from non_local_detector.environment import Environment  # noqa: E402
from non_local_detector.exceptions import ValidationError  # noqa: E402
from non_local_detector.likelihoods import clusterless_kde  # noqa: E402
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
    n_connected_components,
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


def _disconnected_env():
    """Two-segment linearized track (edge_spacing > 0) -> a >= 2-component interior
    graph, so a truncated heat_kernel_rank can drop a component's null mode.

    Mirrors ``test_get_distances_to_interior_bins_gap_position_snaps_to_nearest_interior``
    in ``tests/environment/test_track_graph.py``.
    """
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(50.0, 0.0))
    g.add_node(2, pos=(60.0, 0.0))
    g.add_node(3, pos=(110.0, 0.0))
    g.add_edge(0, 1, distance=50.0, edge_id=0)
    g.add_edge(2, 3, distance=50.0, edge_id=1)

    env = Environment(
        environment_name="two-segment",
        place_bin_size=5.0,
        track_graph=g,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
    )
    return env.fit_place_grid(position_1d, infer_track_interior=True)


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


def _predict(s, enc, is_local=False, **overrides):
    encoding = dict(enc)
    encoding.update(overrides)
    return predict_clusterless_diffusion_log_likelihood(
        jnp.asarray(s["time"]),
        s["position_time"],
        s["position"],
        [jnp.asarray(t) for t in s["dec_t"]],
        [jnp.asarray(f) for f in s["dec_f"]],
        is_local=is_local,
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


def test_local_block_parity():
    """The local branch's per-decode-spike-block loop (``spike_bins[block]``,
    ``decode_features[block]``, ``seg[block]``) must give the same output whether
    it runs in one big block or many tiny ones. ``test_block_parity`` only
    exercises ``is_local=False`` (``_predict`` hardcodes non-local by default), so
    it never touches this slicing in the local branch. ``_sim``'s default 2
    electrodes x 12 decode spikes each, combined with ``memory_budget=1`` (forces
    ``effective_block == 1``, i.e. one spike per block), exercises many
    single-spike blocks per electrode against one big block."""
    s = _sim(seed=3, hi=10.0)
    enc = _fit(s)

    ll_big = np.asarray(_predict(s, enc, is_local=True))  # one block per electrode
    ll_small = np.asarray(
        _predict(s, enc, is_local=True, memory_budget=1)
    )  # one decode spike per block

    assert ll_big.shape == ll_small.shape == (s["time"].shape[0], 1)
    assert np.allclose(ll_big, ll_small, rtol=1e-5, atol=1e-6), (
        f"local block parity failed: max|diff|={np.abs(ll_big - ll_small).max():.3e}"
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


# ----------------------------------------------------------------------------
# Task 5 -- local predict path
# ----------------------------------------------------------------------------


def test_local_equals_nonlocal_per_spike():
    """Per-spike identity (spec sec 3, Local): local predict's contribution at a
    single decode spike must reconstruct the non-local predict's cell at that
    spike's ``(time_bin, animal_bin)``.

    Both paths build the identical diffused column ``P = heat_kernel_apply(D_e)``
    for a decode spike; non-local returns the whole column (all interior bins),
    local indexes just ``bin(x_a(t_j))``. On a fixture with exactly ONE decode
    spike (isolating one electrode's one column -- no other spike/electrode
    contributes to the compared cell) and an animal that is stationary ONLY
    within the single decode-time-bin window containing that spike (so the
    animal's bin is the same at the decode spike's exact time AND at that
    bin's grid point ``time[t_bin]``, which is all the identity needs), it
    holds exactly::

        ll_local[t_bin, 0] == ll_nonlocal[t_bin, animal_bin]

    because both sides equal ``-summed_gpi[animal_bin] + lc[animal_bin]`` for the
    same ``lc`` (same D_e, same heat_kernel_apply call, same electrode).

    Position is NOT held constant for the whole session: it ramps linearly
    across all 6 interior bins for t in [0, 2) (carrying the 3 encoding spikes
    through 3 distinct bins), then holds at ``x_c`` (bin 3) for t in [2, 5],
    covering both the decode spike (t=3.3) and its time bin's grid point
    (time[3]=3.0). This matters: if position were constant for the ENTIRE
    session, both the encoding spikes AND the occupancy samples would land in
    a single bin ``bin0``, so every bin's diffused value in the compared row
    would be ``H[bin, bin0] * (same per-electrode scalar)`` -- the ``H[bin,
    bin0]`` heat-kernel column cancels identically between ``p_e(bin, j)`` and
    ``occupancy(bin)`` in the ``lc`` ratio, making the row bin-INVARIANT by
    construction. Such a fixture cannot tell a correct animal-bin lookup from a
    wrong one: the assertion would pass even if the local branch indexed the
    wrong bin. The ``np.ptp`` guard below makes that degenerate case a hard
    failure rather than a silent false pass.
    """
    env = _make_1d_env(1.0, 0.0, 6.0)  # 6 interior bins, centers 0.5..5.5
    position_time = np.linspace(0.0, 5.0, 101)
    x_c = 3.2  # bin 3 ([3, 4))
    # Ramp through bins 0-5 for t < 2, then stationary at x_c (bin 3) for t >= 2.
    position = np.where(position_time <= 2.0, 0.5 + 2.5 * position_time, x_c)[:, None]

    # Encoding spikes during the ramp -> land in 3 distinct interior bins (1, 2, 4).
    enc_times = [np.array([0.3, 0.9, 1.5])]
    enc_feats = [np.array([[0.0, 0.0], [0.4, -0.3], [-0.2, 0.5]], dtype=np.float32)]

    enc = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        [jnp.asarray(t) for t in enc_times],
        [jnp.asarray(f) for f in enc_feats],
        env,
        sampling_frequency=20,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        weights=None,
        disable_progress_bar=True,
    )

    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Single decode spike, single electrode, inside the stationary window [2, 5].
    dec_times = [np.array([3.3])]
    dec_feats = [np.array([[0.3, -0.2]], dtype=np.float32)]

    ll_nonlocal = np.asarray(
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
    ll_local = np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            position_time,
            position,
            [jnp.asarray(t) for t in dec_times],
            [jnp.asarray(f) for f in dec_feats],
            is_local=True,
            **enc,
        )
    )

    assert ll_local.shape == (time.shape[0], 1)
    n_bins = np.asarray(enc["node_order"]).shape[0]
    assert ll_nonlocal.shape == (time.shape[0], n_bins)
    assert np.all(np.isfinite(ll_local))

    n_total_bins = env.is_track_interior_.ravel().shape[0]
    full_to_local = _full_to_local(np.asarray(enc["node_order"]), n_total_bins)
    animal_position = get_position_at_time(position_time, position, dec_times[0], env)
    animal_bin = int(_interior_bin_indices(env, animal_position, full_to_local)[0])
    t_bin = int(get_spike_time_bin_ind(dec_times[0], time)[0])
    assert animal_bin == 3  # bin containing x_c = 3.2

    # Regression guard: if a future fixture edit collapses this row back to
    # (near-)flat, the identity below would pass vacuously even for a WRONG
    # animal-bin index. Fail loudly instead of silently losing discrimination.
    row_ptp = float(np.ptp(ll_nonlocal[t_bin]))
    assert row_ptp > 0.5, (
        f"non-local row at t_bin={t_bin} is nearly flat (ptp={row_ptp:.3e} bits); "
        "fixture no longer discriminates the animal's bin"
    )

    assert np.allclose(
        ll_local[t_bin, 0], ll_nonlocal[t_bin, animal_bin], rtol=1e-5, atol=1e-6
    ), (
        f"local != nonlocal at the spike's (time_bin, animal_bin) cell: "
        f"local={ll_local[t_bin, 0]:.6f} nonlocal={ll_nonlocal[t_bin, animal_bin]:.6f}"
    )


def test_local_finite_and_zero_rate():
    """Local output is finite, shape (n_time, 1); a zero-rate electrode's decode
    spikes each contribute exactly LOG_EPS (isolated by giving only that
    electrode decode spikes, mirroring test_nonlocal_finite_and_zero_rate)."""
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
            is_local=True,
            **enc,
        )
    )

    assert ll.shape == (time.shape[0], 1)
    assert np.all(np.isfinite(ll))

    n_total_bins = env.is_track_interior_.ravel().shape[0]
    full_to_local = _full_to_local(np.asarray(enc["node_order"]), n_total_bins)
    interpolated_position = get_position_at_time(position_time, position, time, env)
    animal_bins = _interior_bin_indices(env, interpolated_position, full_to_local)
    gpi = np.asarray(enc["summed_ground_process_intensity"])
    local_gpi = gpi[animal_bins]

    seg = get_spike_time_bin_ind(dec_times[1], time)
    for t_bin in seg:
        # only the zero-rate electrode contributed at this bin -> exactly
        # LOG_EPS - local_gpi[t_bin]
        assert np.allclose(ll[t_bin, 0] + local_gpi[t_bin], LOG_EPS, atol=1e-6), (
            f"zero-rate electrode did not floor to LOG_EPS at time bin {t_bin}"
        )


# ----------------------------------------------------------------------------
# Task 7 -- Tier-1 input validation (fit + predict)
# ----------------------------------------------------------------------------


def test_validation_fit():
    """Every Tier-1 fit-time contract (spec sec 4) raises ValidationError, and the
    in-window / out-of-window encoding-feature split matches clusterless_kde."""
    s = _sim(seed=10)

    # heat_kernel_rank: float, bool, zero, negative.
    for bad_rank in (3.0, True, 0, -1):
        with pytest.raises(ValidationError):
            _fit(s, heat_kernel_rank=bad_rank)

    # heat_kernel_rank < n_components: a disconnected (>= 2 component) environment,
    # rank below that count must raise (surfaced via cached_eigenbasis ->
    # _require_rank_covers_components; not re-implemented here, just exercised).
    disconnected_env = _disconnected_env()
    n_components = n_connected_components(disconnected_env)
    assert n_components >= 2, "fixture must be graph-disconnected to test this path"
    with pytest.raises(ValidationError):
        fit_clusterless_diffusion_encoding_model(
            np.linspace(0.0, 1.0, 10),
            np.zeros((10, 1)),
            [],
            [],
            disconnected_env,
            heat_kernel_rank=1,
            disable_progress_bar=True,
        )

    # memory_budget: non-finite, zero, negative, non-int.
    for bad_budget in (float("inf"), 0, -1, 1.5):
        with pytest.raises(ValidationError):
            _fit(s, memory_budget=bad_budget)

    # block_size: non-positive.
    with pytest.raises(ValidationError):
        _fit(s, block_size=0)

    # position_std / waveform_std: non-positive OR non-finite (inf position_std makes
    # exp(-t*lambda) hit inf*0 = NaN on the null mode -> a non-finite model).
    for bad_std in (0.0, -1.0, np.inf):
        with pytest.raises(ValidationError):
            _fit(s, position_std=bad_std)
    for bad_std in (0.0, np.inf):
        with pytest.raises(ValidationError):
            _fit(s, waveform_std=bad_std)

    # waveform_std dimensionality: fit never evaluates the mark kernel, so an
    # ill-shaped bandwidth must be caught here (not later in kde_distance). The
    # fixture has 2-feature marks; an empty array and a length-3 vector are wrong.
    for bad_std in ([], [1.0, 2.0, 3.0]):
        with pytest.raises(ValidationError):
            _fit(s, waveform_std=bad_std)
    # A matching (n_features,) vector and a scalar are both accepted.
    _fit(s, waveform_std=[24.0, 24.0])
    _fit(s, waveform_std=24.0)

    # weights (via common.validate_weights): wrong length, non-finite, negative.
    n_pos = s["position"].shape[0]
    with pytest.raises(ValidationError):
        _fit(s, weights=np.ones(n_pos + 1))
    bad_weights = np.ones(n_pos)
    bad_weights[0] = np.nan
    with pytest.raises(ValidationError):
        _fit(s, weights=bad_weights)
    bad_weights = np.ones(n_pos)
    bad_weights[0] = -1.0
    with pytest.raises(ValidationError):
        _fit(s, weights=bad_weights)

    # non-finite position.
    bad_position = np.array(s["position"], copy=True)
    bad_position[0, 0] = np.nan
    s_bad_position = dict(s)
    s_bad_position["position"] = bad_position
    with pytest.raises(ValidationError):
        _fit(s_bad_position)

    # Non-finite IN-window encoding spike_waveform_features raises; a non-finite
    # feature on an OUT-of-window encoding spike does NOT raise.
    env = _make_1d_env(1.0, 0.0, 8.0)
    position_time = np.linspace(0.0, 4.0, 101)
    position = np.linspace(0.0, 8.0, 101)[:, None]

    with pytest.raises(ValidationError):
        fit_clusterless_diffusion_encoding_model(
            position_time,
            position,
            [jnp.asarray([2.0])],  # in-window: [position_time[0], position_time[-1]]
            [jnp.asarray([[np.nan, 0.0]], dtype=np.float32)],
            env,
            position_std=POSITION_STD,
            waveform_std=WAVEFORM_STD,
            disable_progress_bar=True,
        )

    # One in-window (finite) spike keeps the electrode non-zero-rate; the second
    # spike, with a non-finite feature, is clipped out by is_in_bounds and must not
    # be validated.
    enc_out_of_window = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        [jnp.asarray([2.0, 10.0])],
        [jnp.asarray([[0.1, 0.2], [np.nan, 0.0]], dtype=np.float32)],
        env,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        disable_progress_bar=True,
    )
    assert np.all(np.isfinite(np.asarray(enc_out_of_window["occupancy"])))
    assert np.all(
        np.isfinite(np.asarray(enc_out_of_window["summed_ground_process_intensity"]))
    )


def test_validation_predict():
    """Every Tier-1 predict-time contract (spec sec 4) raises ValidationError, and
    a non-finite feature on an out-of-window decode spike does NOT raise."""
    s = _sim(seed=11)
    enc = _fit(s)

    # non-finite time.
    bad_time = np.array(s["time"], copy=True)
    bad_time[0] = np.nan
    with pytest.raises(ValidationError):
        _predict(dict(s, time=bad_time), enc)

    # non-finite decoding position raises for the LOCAL path (which reads position);
    # the non-local path never reads position (see the positionless case below).
    bad_position = np.array(s["position"], copy=True)
    bad_position[0, 0] = np.nan
    with pytest.raises(ValidationError):
        _predict(dict(s, position=bad_position), enc, is_local=True)

    # Non-local decoding accepts position/position_time = None (the documented
    # positionless API, matching clusterless_kde): it must not raise or require them.
    ll_positionless = _predict(dict(s, position=None, position_time=None), enc)
    assert np.all(np.isfinite(np.asarray(ll_positionless)))

    # non-finite IN-window decode spike_waveform_features on a valid (non-zero-rate)
    # electrode raises.
    bad_feats = [np.array(f, copy=True) for f in s["dec_f"]]
    bad_feats[0][0, 0] = np.nan  # s["dec_t"][0][0] lies within [time[0], time[-1]]
    with pytest.raises(ValidationError):
        _predict(dict(s, dec_f=bad_feats), enc)

    # A non-finite feature on an OUT-of-window decode spike must not raise.
    time = s["time"]
    out_of_window_dec_t = [
        np.array([time[-1] + 10.0, time[-1] + 20.0]),
        s["dec_t"][1],
    ]
    out_of_window_dec_f = [
        np.array([[np.nan, 0.0], [0.0, np.nan]], dtype=np.float32),
        s["dec_f"][1],
    ]
    ll = _predict(dict(s, dec_t=out_of_window_dec_t, dec_f=out_of_window_dec_f), enc)
    assert np.all(np.isfinite(np.asarray(ll)))


def test_validation_predict_local():
    """LOCAL-branch mirror of ``test_validation_predict``'s decode-feature checks.

    The local path (``clusterless_diffusion.py`` ~523) duplicates the same
    in-window ``validate_finite(decode_features, ...)`` Tier-1 call as the
    non-local path (~640) by hand, rather than sharing it. Nothing in the
    committed suite calls ``predict_clusterless_diffusion_log_likelihood`` with
    ``is_local=True`` and a non-finite decode feature, so the local branch's
    copy of the check is unverified without this test.
    """
    s = _sim(seed=11)
    enc = _fit(s)

    # non-finite IN-window decode spike_waveform_features on a valid (non-zero-rate)
    # electrode raises.
    bad_feats = [np.array(f, copy=True) for f in s["dec_f"]]
    bad_feats[0][0, 0] = np.nan  # s["dec_t"][0][0] lies within [time[0], time[-1]]
    with pytest.raises(ValidationError):
        _predict(dict(s, dec_f=bad_feats), enc, is_local=True)

    # A non-finite feature on an OUT-of-window decode spike must not raise.
    time = s["time"]
    out_of_window_dec_t = [
        np.array([time[-1] + 10.0, time[-1] + 20.0]),
        s["dec_t"][1],
    ]
    out_of_window_dec_f = [
        np.array([[np.nan, 0.0], [0.0, np.nan]], dtype=np.float32),
        s["dec_f"][1],
    ]
    ll = _predict(
        dict(s, dec_t=out_of_window_dec_t, dec_f=out_of_window_dec_f),
        enc,
        is_local=True,
    )
    assert np.all(np.isfinite(np.asarray(ll)))


@pytest.mark.parametrize("is_local", [False, True])
def test_validation_predict_zero_rate_precedence(is_local):
    """Tier-1 decode-feature validation must run BEFORE the zero-rate ``continue``
    guard, in both predict branches.

    ``clusterless_diffusion.py`` slices ``decode_features`` and calls
    ``validate_finite`` on it, THEN checks ``if electrode_weight_total == 0:
    ... continue`` (local ~517-538, non-local ~634-655). If that order were
    ever reversed, a zero-rate electrode's non-finite in-window decode feature
    would be silently skipped (the ``continue`` fires before validation) and
    the spike would floor to ``LOG_EPS`` instead of raising. This test pins the
    current (correct) ordering: a zero-rate electrode with a non-finite
    in-window decode feature must raise ``ValidationError``, not floor.
    """
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
    assert enc["weight_total"][1] == 0.0  # electrode 1 is the zero-rate electrode

    time = np.linspace(0.0, 10.0, 6)
    # Only the zero-rate electrode (1) gets a decode spike, and it is IN-window
    # ([time[0], time[-1]] = [0, 10]) so it reaches validate_finite before the
    # zero-rate `continue`.
    dec_times = [np.array([]), np.array([7.5])]
    dec_feats = [
        np.zeros((0, 2), dtype=np.float32),
        np.array([[np.nan, -1.0]], dtype=np.float32),
    ]
    with pytest.raises(ValidationError):
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            position_time,
            position,
            [jnp.asarray(t) for t in dec_times],
            [jnp.asarray(f) for f in dec_feats],
            is_local=is_local,
            **enc,
        )


# ----------------------------------------------------------------------------
# Task 10 -- scientific validation: KDE agreement, geometry win (goal A),
# grid-independence, and non-uniform-dV density correctness.
#
# These validate the *feature*, not the constants: that clusterless_diffusion
# tracks clusterless_kde on barrier-free geometry (agreement), that its heat
# kernel refuses to cross an impassable barrier where the Euclidean KDE leaks
# (the whole point of the feature), that a physical position_std makes the
# decoded posterior grid-independent, and that the non-uniform-dV code path
# (which every uniform-grid test misses) integrates to a proper density.
# ----------------------------------------------------------------------------

# clusterless_kde's default waveform bandwidth; agreement/geometry match it so the
# two algorithms differ *only* in the position smoother (heat kernel vs Gaussian).
GEOM_WAVEFORM_STD = 24.0


def _fit_diffusion(
    position_time, position, enc_t, enc_f, env, position_std, sampling_frequency=100
):
    return fit_clusterless_diffusion_encoding_model(
        jnp.asarray(position_time),
        jnp.asarray(position),
        [jnp.asarray(t) for t in enc_t],
        [jnp.asarray(f) for f in enc_f],
        env,
        sampling_frequency=sampling_frequency,
        position_std=position_std,
        waveform_std=GEOM_WAVEFORM_STD,
        disable_progress_bar=True,
    )


def _fit_kde(
    position_time, position, enc_t, enc_f, env, position_std, sampling_frequency=100
):
    return clusterless_kde.fit_clusterless_kde_encoding_model(
        jnp.asarray(position_time),
        jnp.asarray(position),
        [jnp.asarray(t) for t in enc_t],
        [jnp.asarray(f) for f in enc_f],
        env,
        sampling_frequency=sampling_frequency,
        position_std=position_std,
        waveform_std=GEOM_WAVEFORM_STD,
        disable_progress_bar=True,
    )


def _predict_diffusion_nonlocal(time, position_time, position, dec_t, dec_f, enc):
    return np.asarray(
        predict_clusterless_diffusion_log_likelihood(
            jnp.asarray(time),
            jnp.asarray(position_time),
            jnp.asarray(position),
            [jnp.asarray(t) for t in dec_t],
            [jnp.asarray(f) for f in dec_f],
            is_local=False,
            **enc,
        )
    )


def _predict_kde_nonlocal(time, position_time, position, dec_t, dec_f, enc):
    return np.asarray(
        clusterless_kde.predict_clusterless_kde_log_likelihood(
            jnp.asarray(time),
            jnp.asarray(position_time),
            jnp.asarray(position),
            [jnp.asarray(t) for t in dec_t],
            [jnp.asarray(f) for f in dec_f],
            is_local=False,
            **enc,
        )
    )


def _clusterless_sim(
    position, n_time=40, n_elec=2, n_features=2, n_enc=40, n_dec=15, seed=0
):
    """Random clusterless encoding/decoding spikes over a given trajectory.

    Diffusion and KDE both consume the SAME spikes; only their position smoother
    differs, so per-time-bin rank agreement isolates the smoother.
    """
    rng = np.random.default_rng(seed)
    dt = 0.02
    time = np.arange(n_time) * dt
    t_end = float(time[-1])
    position_time = np.linspace(0.0, t_end, position.shape[0])
    enc_t = [np.sort(rng.uniform(0.0, t_end, n_enc)) for _ in range(n_elec)]
    enc_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in enc_t
    ]
    dec_t = [np.sort(rng.uniform(0.0, t_end, n_dec)) for _ in range(n_elec)]
    dec_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in dec_t
    ]
    return time, position_time, enc_t, enc_f, dec_t, dec_f


def _median_per_timebin_spearman(ll_a, ll_b):
    """Median over time bins of Spearman(ll_a[t], ll_b[t]); skips flat KDE rows."""
    rhos = [
        stats.spearmanr(ll_a[t], ll_b[t]).statistic
        for t in range(ll_a.shape[0])
        if np.ptp(ll_b[t]) > 0
    ]
    return float(np.nanmedian(rhos)), len(rhos)


def test_agreement_with_kde_simple_geometry():
    """On barrier-free geometry the decoded non-local posterior of
    clusterless_diffusion tracks clusterless_kde by per-time-bin rank correlation.

    Anchor for goal A (spec sec Testing, "Agreement"): with a matched waveform
    bandwidth the two algorithms share every input and differ only in the position
    smoother (graph heat kernel vs Euclidean Gaussian), so on a wall-less 1D track
    and a wall-less 2D open field the posteriors must rank-agree. Threshold is the
    documented floor (median per-time-bin Spearman > 0.6; the Task-0 spike measured
    ~1.000). Both diffusion and KDE non-local predict return interior bins in the
    same natural (``np.where(is_track_interior)``) order, so the rows compare
    bin-for-bin.
    """
    # --- 1D linear track (every bin interior) ---
    env_1d = Environment(
        environment_name="agree_line",
        place_bin_size=1.0,
        position_range=((0.0, 15.0),),
    ).fit_place_grid(np.linspace(0.0, 15.0, 61)[:, None], infer_track_interior=False)
    pos_1d = np.clip(
        (np.sin(np.linspace(0.0, 3.0 * np.pi, 250)) * 0.5 + 0.5) * 15.0, 0.1, 14.9
    )[:, None]
    time, pt, et, ef, dt_, df = _clusterless_sim(pos_1d, seed=0)
    enc_d = _fit_diffusion(pt, pos_1d, et, ef, env_1d, position_std=3.0)
    enc_k = _fit_kde(pt, pos_1d, et, ef, env_1d, position_std=3.0)
    ll_d = _predict_diffusion_nonlocal(time, pt, pos_1d, dt_, df, enc_d)
    ll_k = _predict_kde_nonlocal(time, pt, pos_1d, dt_, df, enc_k)
    assert ll_d.shape == ll_k.shape
    median_1d, n_1d = _median_per_timebin_spearman(ll_d, ll_k)
    print(
        f"[AGREE 1D] median per-timebin Spearman(diffusion, kde)={median_1d:.3f} "
        f"(n_timebins={n_1d}, n_bins={ll_d.shape[1]})"
    )
    assert median_1d > 0.6, (
        f"1D diffusion posterior does not track KDE: median Spearman={median_1d:.3f}"
    )

    # --- 2D open field (inferred interior, no barriers) ---
    rng = np.random.default_rng(1)
    env_2d = Environment(
        environment_name="agree_of",
        place_bin_size=2.0,
        position_range=((0.0, 20.0), (0.0, 20.0)),
    ).fit_place_grid(rng.uniform(1.0, 19.0, size=(4000, 2)), infer_track_interior=True)
    pos_2d = np.column_stack(
        [
            (np.sin(np.linspace(0.0, 3.0 * np.pi, 300)) * 0.5 + 0.5) * 18.0 + 1.0,
            (np.cos(np.linspace(0.0, 2.0 * np.pi, 300)) * 0.5 + 0.5) * 18.0 + 1.0,
        ]
    )
    time, pt, et, ef, dt_, df = _clusterless_sim(pos_2d, seed=2)
    enc_d = _fit_diffusion(pt, pos_2d, et, ef, env_2d, position_std=4.0)
    enc_k = _fit_kde(pt, pos_2d, et, ef, env_2d, position_std=4.0)
    ll_d = _predict_diffusion_nonlocal(time, pt, pos_2d, dt_, df, enc_d)
    ll_k = _predict_kde_nonlocal(time, pt, pos_2d, dt_, df, enc_k)
    assert ll_d.shape == ll_k.shape
    median_2d, n_2d = _median_per_timebin_spearman(ll_d, ll_k)
    print(
        f"[AGREE 2D] median per-timebin Spearman(diffusion, kde)={median_2d:.3f} "
        f"(n_timebins={n_2d}, n_bins={ll_d.shape[1]})"
    )
    assert median_2d > 0.6, (
        f"2D diffusion posterior does not track KDE: median Spearman={median_2d:.3f}"
    )


def _make_two_room_env(seed=0):
    """Two rooms x in [2, 16] and x in [24, 38], gap x in (16, 24) four bins wide:
    a disconnected (2-component) manifold graph -- a genuine impassable barrier.

    Mirrors ``make_two_room_env`` in ``test_sorted_spikes_diffusion.py``.
    """
    rng = np.random.default_rng(seed)
    left = rng.uniform([2.0, 2.0], [16.0, 38.0], size=(12000, 2))
    right = rng.uniform([24.0, 2.0], [38.0, 38.0], size=(12000, 2))
    position = np.vstack([left, right])
    rng.shuffle(position)
    env = Environment(
        environment_name="two_rooms",
        place_bin_size=2.0,
        position_range=((0.0, 40.0), (0.0, 40.0)),
    ).fit_place_grid(position, infer_track_interior=True)
    return env, position


def test_geometry_no_barrier_leak():
    """THE headline test (goal A): a decode spike's mark evidence must not cross an
    impassable barrier.

    Two disconnected rooms; ALL encoding spikes are in the LEFT room (a place cell
    near the barrier, hard-masked to zero for x > 16 so there are exactly zero
    right-room encoding spikes). For one decode spike whose mark matches the
    left-room encoding marks, the non-local posterior ``softmax(ll_row)`` over
    interior bins is measured for leaked mass in the RIGHT room.

    Diffusion's right room is a disconnected graph component with zero encoding
    spikes, so its joint mark density floors there (LOG_EPS) and the posterior
    assigns it ~no mass. The Euclidean KDE smooths across the physical gap, leaking
    a material fraction. This is the estimator-level proof that fit -> pixellate ->
    diffuse -> predict respects the barrier where KDE cannot.
    """
    env, position = _make_two_room_env()
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 2  # a genuine barrier

    sampling_frequency = 100
    time = np.arange(position.shape[0]) / sampling_frequency
    # Left-room place cell near the barrier (center x=14): Gaussian tuning hard-masked
    # to zero for x > 16, so there are exactly zero right-room encoding spikes.
    center = np.array([14.0, 20.0])
    rng = np.random.default_rng(2)
    rate = 50.0 * np.exp(-((position - center) ** 2).sum(axis=1) / (2 * 4.0**2))
    rate[position[:, 0] > 16.0] = 0.0
    spike_mask = rng.random(position.shape[0]) < rate / sampling_frequency
    assert (position[spike_mask][:, 0] > 16.0).sum() == 0  # zero right-room spikes
    enc_t = [time[spike_mask]]
    # Encoding marks centered at the origin; the decode spike's mark matches.
    enc_f = [rng.normal(0.0, 1.0, size=(int(spike_mask.sum()), 2)).astype(np.float32)]

    position_std = 8.0
    enc_d = _fit_diffusion(
        time, position, enc_t, enc_f, env, position_std, sampling_frequency
    )
    enc_k = _fit_kde(
        time, position, enc_t, enc_f, env, position_std, sampling_frequency
    )

    # One decode spike, mark = [0, 0] (matches the left-room encoding marks), in a
    # single usable time bin.
    decode_time = np.array([0.0, time[-1]])
    dec_t = [np.array([time[position.shape[0] // 2]])]
    dec_f = [np.array([[0.0, 0.0]], dtype=np.float32)]
    ll_d = _predict_diffusion_nonlocal(decode_time, time, position, dec_t, dec_f, enc_d)
    ll_k = _predict_kde_nonlocal(decode_time, time, position, dec_t, dec_f, enc_k)

    interior = env.is_track_interior_.ravel()
    interior_centers = env.place_bin_centers_[interior]
    right_room = interior_centers[:, 0] > 20.0
    seg = int(get_spike_time_bin_ind(dec_t[0], decode_time)[0])

    def _posterior_right_room_mass(ll_row):
        post = np.exp(ll_row - ll_row.max())
        post /= post.sum()
        return float(post[right_room].sum())

    diffusion_leak = _posterior_right_room_mass(ll_d[seg])
    kde_leak = _posterior_right_room_mass(ll_k[seg])
    print(
        f"[LEAK] diffusion_leak={diffusion_leak:.3e} kde_leak={kde_leak:.4f} "
        f"ratio={kde_leak / max(diffusion_leak, 1e-30):.2e} "
        f"(right_bins={int(right_room.sum())}/{int(interior.sum())}, "
        f"n_enc_spikes={int(spike_mask.sum())})"
    )

    # Diffusion cannot cross the disconnected component; KDE leaks a material fraction.
    assert diffusion_leak < 0.01, (
        f"diffusion leaked across barrier: {diffusion_leak:.3e}"
    )
    assert kde_leak > 0.05, (
        f"KDE did not leak enough to be discriminating: {kde_leak:.3e}"
    )
    assert kde_leak > 10 * diffusion_leak, (
        f"KDE must leak materially more than diffusion: "
        f"kde={kde_leak:.3e} diffusion={diffusion_leak:.3e}"
    )


def _make_open_field_env(name, place_bin_size, seed=99):
    rng = np.random.default_rng(seed)
    return Environment(
        environment_name=name,
        place_bin_size=place_bin_size,
        position_range=((0.0, 20.0), (0.0, 20.0)),
    ).fit_place_grid(rng.uniform(1.0, 19.0, size=(4000, 2)), infer_track_interior=True)


def _decode_diffusion_posterior(env, position, position_std, seed=5):
    time, pt, et, ef, dt_, df = _clusterless_sim(
        position, n_time=30, n_elec=3, n_enc=60, n_dec=25, seed=seed
    )
    enc = _fit_diffusion(pt, position, et, ef, env, position_std=position_std)
    ll = _predict_diffusion_nonlocal(time, pt, position, dt_, df, enc)
    interior = env.is_track_interior_.ravel()
    centers = env.place_bin_centers_[interior]
    post = np.exp(ll - ll.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    return post, centers


def test_grid_independence():
    """A physical ``position_std`` makes the decoded posterior stable across bin sizes.

    The SAME open-field data is decoded on a coarse (bin 4.0) and a fine (bin 2.0)
    grid at the SAME physical ``position_std``. Because the heat-kernel time
    ``t = position_std**2 / 2`` is in coordinate units (not bins), the decoded
    posterior as a function of PHYSICAL position must agree: the per-time-bin
    expected (center-of-mass) position tracks within a fraction of the coarse bin,
    and the coarse posterior resampled onto the fine grid rank-correlates with the
    fine posterior. (argmax is intentionally NOT used -- the weakly-informative
    per-spike posterior is multimodal, so argmax jumps between near-equal peaks; the
    center-of-mass and rank-correlation metrics are the robust physical comparisons.)
    If ``position_std`` were grid-relative, the coarse grid would smooth over 2x the
    physical distance of the fine grid and these would diverge.
    """
    position = np.column_stack(
        [
            (np.sin(np.linspace(0.0, 3.0 * np.pi, 300)) * 0.5 + 0.5) * 18.0 + 1.0,
            (np.cos(np.linspace(0.0, 2.0 * np.pi, 300)) * 0.5 + 0.5) * 18.0 + 1.0,
        ]
    )
    position_std = 6.0
    env_coarse = _make_open_field_env("grid_coarse", 4.0)
    env_fine = _make_open_field_env("grid_fine", 2.0)
    post_c, cen_c = _decode_diffusion_posterior(env_coarse, position, position_std)
    post_f, cen_f = _decode_diffusion_posterior(env_fine, position, position_std)

    # Decoded physical expected position per time bin agrees across grids.
    exp_c = post_c @ cen_c
    exp_f = post_f @ cen_f
    dist = np.linalg.norm(exp_c - exp_f, axis=1)

    # Resample the coarse posterior onto the fine centers (nearest coarse bin) and
    # rank-correlate the two posteriors per time bin (posterior SHAPE in physical space).
    nn = cKDTree(cen_c).query(cen_f)[1]
    rhos = [
        stats.spearmanr(post_c[t][nn], post_f[t]).statistic
        for t in range(post_f.shape[0])
        if np.ptp(post_c[t][nn]) > 0 and np.ptp(post_f[t]) > 0
    ]
    median_rho = float(np.nanmedian(rhos))
    print(
        f"[GRID] expected-pos dist median={np.median(dist):.3f} max={dist.max():.3f} "
        f"(coarse bin=4.0) | resampled per-timebin Spearman median={median_rho:.3f}"
    )

    # Sanity: the expected position genuinely moves, so the agreement is non-vacuous.
    assert np.ptp(exp_f[:, 0]) > 1.0 or np.ptp(exp_f[:, 1]) > 1.0

    assert np.median(dist) < 1.0, (  # a small fraction of the coarse bin size (4.0)
        f"expected position not grid-stable: median dist={np.median(dist):.3f}"
    )
    assert dist.max() < 2.0, f"worst-case grid drift too large: {dist.max():.3f}"
    assert median_rho > 0.85, (
        f"posterior shape not grid-stable: median Spearman={median_rho:.3f}"
    )


def _make_nonuniform_dv_env():
    """Linearized two-segment track with ``edge_spacing > 0`` -> gap bins (a
    2-component graph) AND genuinely non-uniform interior ``bin_sizes``: edge 0
    (length 20) splits into 4 bins of 5.0, edge 1 (length 22) into 5 bins of 4.4,
    so ``ptp(bin_sizes) = 0.6``. Every uniform-grid test in this file has
    ``bin_sizes == 1.0`` and so never exercises the ``dV`` path.
    """
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(20.0, 0.0))
    g.add_node(2, pos=(23.0, 0.0))
    g.add_node(3, pos=(45.0, 0.0))
    g.add_edge(0, 1, distance=20.0, edge_id=0)
    g.add_edge(2, 3, distance=22.0, edge_id=1)
    env = Environment(
        environment_name="nonuniform_dv",
        place_bin_size=5.0,
        track_graph=g,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 20.0, 60), np.linspace(23.0, 45.0, 60)]
    )
    return env.fit_place_grid(position_1d, infer_track_interior=True)


def test_nonuniform_dv_density_correctness():
    """Density correctness on a NON-UNIFORM ``dV`` grid (closes a coverage gap).

    The mark-marginal recovery ``Sum_x p_e(x, m_j) * dV(x) == Sum_i w_i K / Sum_i w_i``
    is the property that pins the ``dV`` normalization, but on a uniform grid
    (``dV == 1.0``) it is a no-op that would pass even if the code dropped ``dV``.
    Here ``dV`` is genuinely non-uniform (4.4 vs 5.0), so:

    * ``bin_sizes`` must be non-uniform (``ptp > 0``) AND equal the true geometric bin
      widths recomputed independently from ``place_bin_edges_`` -- a regression that
      dropped ``dV`` to 1.0 breaks this directly.
    * The reconstructed density integrated against the INDEPENDENT widths recovers
      the weighted mark marginal (heat-kernel mass conservation, per-component on
      this disconnected env). Integrating the density against those independent
      widths (not the ``dV`` used to build it) is what makes a dropped-``dV``
      regression fail rather than cancel.
    * Positive control: integrating against a uniform 1.0 measure does NOT recover
      the marginal, proving the non-uniform ``dV`` is load-bearing.
    """
    env = _make_nonuniform_dv_env()
    graph, node_order, bin_sizes = environment_graph(env)
    bin_sizes = np.asarray(bin_sizes)
    is_interior = env.is_track_interior_.ravel()
    # Independent geometric bin widths straight from the environment grid.
    independent_widths = np.diff(env.place_bin_edges_.ravel())[is_interior]

    assert np.ptp(bin_sizes) > 0.0, "fixture must have non-uniform dV"
    assert np.allclose(bin_sizes, independent_widths), (
        "bin_sizes must carry the true non-uniform geometry, not a dropped-to-1.0 dV"
    )
    print(
        f"[NONUNIF] bin_sizes ptp={np.ptp(bin_sizes):.3f} "
        f"min={bin_sizes.min():.3f} max={bin_sizes.max():.3f} "
        f"n_components={nx.number_connected_components(graph)}"
    )

    # Weighted encoding on the track (2D position on y = 0).
    n_pos = 200
    position_time = np.linspace(0.0, 1.0, n_pos)
    x = np.concatenate(
        [
            np.linspace(0.5, 19.5, n_pos // 2),
            np.linspace(23.5, 44.5, n_pos - n_pos // 2),
        ]
    )
    position = np.column_stack([x, np.zeros_like(x)])
    rng = np.random.default_rng(11)
    weights = 0.5 + np.linspace(
        0.0, 1.0, n_pos
    )  # smooth, strictly positive, non-uniform
    enc_t = np.sort(rng.uniform(0.0, 1.0, 40))
    enc_f = rng.standard_normal((enc_t.size, 2)).astype(np.float32)
    waveform_std = 6.0
    enc = fit_clusterless_diffusion_encoding_model(
        jnp.asarray(position_time),
        jnp.asarray(position),
        [jnp.asarray(enc_t)],
        [jnp.asarray(enc_f)],
        env,
        sampling_frequency=100,
        position_std=6.0,
        waveform_std=waveform_std,
        weights=weights,
        disable_progress_bar=True,
    )

    enc_bins = jnp.asarray(enc["encoding_bin_indices"][0])
    enc_marks = jnp.asarray(enc["encoding_marks"][0])
    w = np.asarray(enc["encoding_weights"][0])
    w_total = float(enc["weight_total"][0])
    dV = np.asarray(enc["bin_sizes"])
    n_bins = dV.shape[0]

    dec_marks = jnp.asarray(
        rng.standard_normal((5, enc_marks.shape[1])).astype(np.float32)
    )
    K = kde_distance(dec_marks, enc_marks, jnp.full(enc_marks.shape[1], waveform_std))
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
    rhs = (w[:, None] * np.asarray(K)).sum(axis=0) / w_total  # weighted mark marginal

    # Integrate the density against the INDEPENDENT widths. With correct dV this
    # recovers rhs; had dV regressed to 1.0, p_e would be ~4.7x larger and this fails.
    lhs = (p_e * independent_widths[:, None]).sum(axis=0)
    assert np.allclose(lhs, rhs, rtol=1e-3, atol=1e-6), (
        f"non-uniform-dV mark-marginal mismatch: max|diff|={np.abs(lhs - rhs).max():.3e}"
    )

    # Positive control: a uniform 1.0 measure must NOT recover the marginal here.
    lhs_uniform = p_e.sum(axis=0)
    assert not np.allclose(lhs_uniform, rhs, rtol=1e-3, atol=1e-6), (
        "uniform-measure integral spuriously matched: fixture is not exercising dV"
    )
    print(
        f"[NONUNIF] mark-marginal max|lhs-rhs|={np.abs(lhs - rhs).max():.3e} "
        f"| uniform-measure control max|diff|={np.abs(lhs_uniform - rhs).max():.3e}"
    )


# ----------------------------------------------------------------------------
# Task 11 -- non-gating speed benchmark (goal B). Folds in the Task 0 spike's
# large-grid perf probe, but drives the PRODUCTION fit/predict entry points
# instead of the spike's hand-rolled reimplementation.
# ----------------------------------------------------------------------------


def _time_predict_only(predict_fn, iters=5):
    """Warm up JIT (compile + device transfer, not timed), then time
    ``predict_fn`` over ``iters`` iterations, blocking on the device output
    each call so async dispatch doesn't hide work outside the timer."""
    jax.block_until_ready(predict_fn())
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(predict_fn())
    return (time.perf_counter() - start) / iters


@pytest.mark.slow
def test_benchmark_diffusion_vs_kde_large_grid():
    """Non-gating goal-B speed benchmark: production predict-only wall time,
    ``clusterless_diffusion`` vs ``clusterless_kde``, on a large 2D grid with
    many encoding/decode spikes.

    This is NOT a regression gate: JAX's lazy device-memory preallocation and
    kernel warm-up make wall-clock timing flaky run-to-run (see the module's
    Goal B docs -- the win grows with spike count but the constant factor is
    hardware/allocator dependent). The only assertion is that both outputs are
    finite; the measured times and speedup are printed on the ``[BENCH]`` line
    for manual inspection.
    """
    rng = np.random.default_rng(0)
    # Dense uniform 2D coverage + infer_track_interior=True fills the grid
    # interior (place_bin_size = 20/160 -> ~80x48 grid -> several thousand
    # filled interior bins), mirroring the retired Task 0 spike's perf fixture.
    n_bins_side = 160
    place_bin_size = 20.0 / n_bins_side
    n_pos = 20_000
    n_elec = 8
    n_features = 2
    n_enc = 8_000
    n_dec = 300
    n_time = 60

    dt = 0.02
    time_bins = np.arange(n_time) * dt
    t_end = float(time_bins[-1])
    position_time = np.linspace(0.0, t_end, n_pos)
    position = np.column_stack([rng.uniform(0, 10, n_pos), rng.uniform(-3, 3, n_pos)])
    env = Environment(
        environment_name="bench_grid",
        position_range=[(0, 10), (-3, 3)],
        place_bin_size=place_bin_size,
    ).fit_place_grid(position=position, infer_track_interior=True)
    n_interior = int(env.is_track_interior_.sum())
    assert n_interior > 2000, f"grid too small for a meaningful benchmark: {n_interior}"

    enc_t = [np.sort(rng.uniform(0.0, t_end, n_enc)) for _ in range(n_elec)]
    enc_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in enc_t
    ]
    dec_t = [np.sort(rng.uniform(0.0, t_end, n_dec)) for _ in range(n_elec)]
    dec_f = [
        rng.standard_normal((t.size, n_features)).astype(np.float32) for t in dec_t
    ]

    # Realistic domain-to-bandwidth ratio (domain is 10x6) -- a wide bandwidth
    # over-smooths and degenerates the auto-selected rank, producing a fake speedup.
    position_std = 2.0
    waveform_std = 24.0

    tm = jnp.asarray(time_bins)
    pt = jnp.asarray(position_time)
    po = jnp.asarray(position)
    dec_t_j = [jnp.asarray(t) for t in dec_t]
    dec_f_j = [jnp.asarray(f) for f in dec_f]

    # ---- KDE: fit ONCE, then time PREDICT-ONLY ----
    enc_kde = clusterless_kde.fit_clusterless_kde_encoding_model(
        pt,
        po,
        [jnp.asarray(t) for t in enc_t],
        [jnp.asarray(f) for f in enc_f],
        env,
        sampling_frequency=50,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=10_000,
        disable_progress_bar=True,
    )

    def kde_predict():
        return clusterless_kde.predict_clusterless_kde_log_likelihood(
            tm, pt, po, dec_t_j, dec_f_j, is_local=False, **enc_kde
        )

    ll_kde = np.asarray(kde_predict())
    dt_kde = _time_predict_only(kde_predict)

    # ---- diffusion: fit ONCE (builds + caches the bandwidth-aware low-rank
    # basis), then time PREDICT-ONLY -- apples-to-apples with the KDE timing above.
    enc_diff = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        [jnp.asarray(t) for t in enc_t],
        [jnp.asarray(f) for f in enc_f],
        env,
        sampling_frequency=50,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=10_000,
        disable_progress_bar=True,
    )

    def diffusion_predict():
        return predict_clusterless_diffusion_log_likelihood(
            tm, pt, po, dec_t_j, dec_f_j, is_local=False, **enc_diff
        )

    ll_diff = np.asarray(diffusion_predict())
    dt_diff = _time_predict_only(diffusion_predict)

    assert np.all(np.isfinite(ll_kde))
    assert np.all(np.isfinite(ll_diff))

    speedup = dt_kde / dt_diff if dt_diff > 0 else float("inf")
    print(
        f"[BENCH] {n_interior} interior bins | KDE predict: {dt_kde * 1e3:.1f} ms | "
        f"diffusion predict: {dt_diff * 1e3:.1f} ms | speedup: {speedup:.2f}x "
        f"(resolved_rank={enc_diff['resolved_rank']}, n_enc/elec={n_enc}, "
        f"n_dec/elec={n_dec}, n_elec={n_elec})"
    )
