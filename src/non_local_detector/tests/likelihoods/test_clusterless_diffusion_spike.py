"""SPIKE (Task 0): validate clusterless_diffusion non-local math + perf vs KDE.
Throwaway; folded into the production suite by the final task."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import kde_distance
from non_local_detector.likelihoods.common import LOG_EPS, get_spike_time_bin_ind
from non_local_detector.likelihoods.diffusion import cached_eigenbasis, environment_graph


def _sim(seed=0, n_bins_side=10, infer_track_interior=True, fill=False, n_pos=200):
    rng = np.random.default_rng(seed)
    dt = 0.02
    n_time = 60
    time = np.arange(n_time) * dt
    t_end = float(time[-1])
    pt = np.linspace(0.0, t_end, n_pos)
    if fill:
        # Dense uniform 2D coverage so infer_track_interior=True marks a large,
        # *filled* interior. This is how the timing test gets a >2000-bin 2D
        # diffusion domain. (infer_track_interior=False would make every bin
        # interior — but that path currently trips a pre-existing boundary bug in
        # make_nD_track_graph_from_environment; inferred interior leaves a
        # non-interior border, so no interior node touches the grid edge.)
        pos = np.column_stack([rng.uniform(0, 10, pt.size),
                               rng.uniform(-3, 3, pt.size)])
    else:
        pos = np.column_stack([np.linspace(0, 10, pt.size),
                               np.sin(np.linspace(0, 2 * np.pi, pt.size)) * 2])
    n_features = 2
    enc_t = [np.sort(rng.uniform(0, t_end, 60)) for _ in range(3)]
    enc_f = [rng.standard_normal((t.size, n_features)).astype(np.float32) for t in enc_t]
    dec_t = [np.sort(rng.uniform(0, t_end, 20)) for _ in range(3)]
    dec_f = [rng.standard_normal((t.size, n_features)).astype(np.float32) for t in dec_t]
    env = Environment(position_range=[(0, 10), (-3, 3)],
                      place_bin_size=20.0 / n_bins_side)
    env = env.fit_place_grid(position=pos, infer_track_interior=infer_track_interior)
    return dict(time=time, pt=pt, pos=pos, enc_t=enc_t, enc_f=enc_f,
                dec_t=dec_t, dec_f=dec_f, env=env)


def _diffusion_nonlocal_spike(s, position_std=6.0, waveform_std=24.0, low_rank=False):
    """Minimal non-local clusterless_diffusion, split into fit (run once) + a
    ``predict`` closure (the part goal B times).

    ``low_rank=True`` uses the **bandwidth-aware truncated** basis
    (``cached_heat_kernel_eigenbasis``) that goal B actually depends on;
    ``low_rank=False`` uses the full-rank basis (correctness reference).
    Returns the ``predict`` closure (call it to get a jnp ``(n_time, n_bins)``).
    """
    from non_local_detector.likelihoods.diffusion import cached_heat_kernel_eigenbasis
    from non_local_detector.likelihoods.sorted_spikes_diffusion import (
        _full_to_local, _interior_bin_indices)
    from non_local_detector.likelihoods.common import get_position_at_time, weighted_mean_rate
    env = s["env"]
    graph, node_order, _ = environment_graph(env)
    if low_rank:
        eigvals, eigvecs = cached_heat_kernel_eigenbasis(env, position_std)  # bandwidth-aware low rank
    else:
        eigvals, eigvecs = cached_eigenbasis(env, rank=None)                 # full rank
    eigvals = jnp.asarray(eigvals); Q = jnp.asarray(eigvecs)
    interior = env.is_track_interior_.ravel()
    n_bins = Q.shape[0]
    dV = 1.0  # uniform grid in the spike; exact ΔV = bin_sizes in production
    coeff = jnp.exp(-(position_std ** 2 / 2.0) * eigvals)

    def diffuse_cols(F):  # single-component clip+rescale-to-input-mass
        sm = Q @ (coeff[:, None] * (Q.T @ F))
        cl = jnp.clip(sm, 0.0, None)
        scale = jnp.where(cl.sum(0) > 0, F.sum(0) / jnp.where(cl.sum(0) > 0, cl.sum(0), 1.0), 0.0)
        return cl * scale

    f2l = _full_to_local(node_order, interior.shape[0])
    def bins_of(p):
        return _interior_bin_indices(env, p, f2l)

    # --- fit (run once; not timed) ---
    occ_pos = get_position_at_time(s["pt"], s["pos"], s["pt"], env)
    O = jnp.zeros(n_bins).at[jnp.asarray(bins_of(occ_pos))].add(1.0)
    w_pos = float(len(occ_pos))
    pi = jnp.clip(diffuse_cols(O[:, None])[:, 0] / (w_pos * dV), 1e-15, None)
    per_e, summed_gpi = [], jnp.zeros(n_bins)
    for e in range(len(s["enc_t"])):
        enc_bins = jnp.asarray(bins_of(get_position_at_time(s["pt"], s["pos"], s["enc_t"][e], env)))
        enc_marks = jnp.asarray(s["enc_f"][e]); n_enc = enc_marks.shape[0]
        mean_rate = weighted_mean_rate(np.ones(n_enc), w_pos)
        S = jnp.zeros(n_bins).at[enc_bins].add(1.0)
        summed_gpi = summed_gpi + mean_rate * diffuse_cols(S[:, None])[:, 0] / (n_enc * dV) / pi
        per_e.append((enc_bins, enc_marks, n_enc, mean_rate,
                      jnp.asarray(s["dec_f"][e]),
                      jnp.asarray(get_spike_time_bin_ind(s["dec_t"][e], s["time"]))))
    n_dec_time = s["time"].shape[0]

    def predict():  # decode-only — this is what goal B times
        ll = jnp.zeros((n_dec_time, n_bins)) - summed_gpi[None, :]
        for (enc_bins, enc_marks, n_enc, mean_rate, dm, seg) in per_e:
            K = kde_distance(dm, enc_marks, jnp.full(2, waveform_std))  # already exp'd — no jnp.exp
            D = jnp.zeros((n_bins, dm.shape[0])).at[enc_bins].add(K)    # scatter (uniform w=1)
            p_e = diffuse_cols(D) / (n_enc * dV)
            lc = jnp.log(jnp.clip(mean_rate * p_e / pi[:, None], jnp.exp(LOG_EPS), None))
            ll = ll + jnp.zeros((n_dec_time, n_bins)).at[seg].add(lc.T)
        return ll  # jnp device array

    return predict


@pytest.mark.slow
def test_spike_agreement_and_speed():
    s = _sim(seed=0)
    ll_diff = np.asarray(_diffusion_nonlocal_spike(s)())  # full-rank correctness reference; call the closure
    # KDE reference
    enc = clusterless_kde.fit_clusterless_kde_encoding_model(
        jnp.asarray(s["pt"]), jnp.asarray(s["pos"]),
        [jnp.asarray(t) for t in s["enc_t"]], [jnp.asarray(f) for f in s["enc_f"]],
        s["env"], sampling_frequency=50, position_std=6.0, waveform_std=24.0,
        block_size=100, disable_progress_bar=True)
    ll_kde = np.asarray(clusterless_kde.predict_clusterless_kde_log_likelihood(
        jnp.asarray(s["time"]), jnp.asarray(s["pt"]), jnp.asarray(s["pos"]),
        [jnp.asarray(t) for t in s["dec_t"]], [jnp.asarray(f) for f in s["dec_f"]],
        **enc, is_local=False))
    assert np.all(np.isfinite(ll_diff))
    # posterior agreement per time bin (rank correlation of the two log-likelihood rows)
    rhos = [stats.spearmanr(ll_diff[t], ll_kde[t]).statistic
            for t in range(ll_diff.shape[0]) if np.ptp(ll_kde[t]) > 0]
    median_rho = float(np.nanmedian(rhos))
    print(f"[SPIKE] median per-timebin Spearman(diffusion, kde) = {median_rho:.3f}")
    assert median_rho > 0.6, "diffusion posterior does not track KDE on simple geometry"


def _time(fn, iters=5):
    jax.block_until_ready(fn())            # warm compile, not timed
    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())        # block on the DEVICE output each iter
    return (time.perf_counter() - t0) / iters


@pytest.mark.slow
def test_spike_speed_large_grid():
    # Dense uniform 2D coverage + infer_track_interior=True fills the grid interior
    # (place_bin_size = 20/160 = 0.125 -> 80x48 grid -> ~3839 filled interior bins),
    # so Q genuinely has ~n_grid rows without tripping the make_nD_track_graph
    # boundary bug that infer_track_interior=False hits on 2D grids.
    s = _sim(seed=1, n_bins_side=160, infer_track_interior=True, fill=True, n_pos=20000)
    n_interior = int(s["env"].is_track_interior_.sum())   # the ACTUAL Q size / diffusion domain
    assert n_interior > 2000, f"interior too small for a meaningful perf gate: {n_interior}"

    # Realistic domain-to-bandwidth ratio: domain is 10x6, so use position_std=2.0
    # (NOT 6.0, which over-smooths and makes the auto rank degenerate -> fake speedup).
    POSITION_STD = 2.0
    dec_t = [jnp.asarray(t) for t in s["dec_t"]]
    dec_f = [jnp.asarray(f) for f in s["dec_f"]]
    tm, pt, po = jnp.asarray(s["time"]), jnp.asarray(s["pt"]), jnp.asarray(s["pos"])

    # KDE: fit ONCE, then time PREDICT-ONLY (returns a jnp device array).
    enc = clusterless_kde.fit_clusterless_kde_encoding_model(
        pt, po, [jnp.asarray(t) for t in s["enc_t"]], [jnp.asarray(f) for f in s["enc_f"]],
        s["env"], sampling_frequency=50, position_std=POSITION_STD, waveform_std=24.0,
        block_size=10_000, disable_progress_bar=True)
    kde_predict = lambda: clusterless_kde.predict_clusterless_kde_log_likelihood(
        tm, pt, po, dec_t, dec_f, **enc, is_local=False)
    dt_kde = _time(kde_predict)

    # Diffusion: fit ONCE with the BANDWIDTH-AWARE LOW-RANK basis (goal B) at the SAME
    # bandwidth, then time the PREDICT-ONLY closure — apples-to-apples with KDE predict.
    diff_predict = _diffusion_nonlocal_spike(s, position_std=POSITION_STD, low_rank=True)
    from non_local_detector.likelihoods.diffusion import cached_heat_kernel_eigenbasis
    _, Qk = cached_heat_kernel_eigenbasis(s["env"], POSITION_STD)
    dt_diff = _time(diff_predict)

    print(f"[SPIKE] {n_interior} interior bins | auto rank={Qk.shape[1]} | "
          f"KDE predict: {dt_kde*1e3:.1f} ms | diffusion predict (low-rank): "
          f"{dt_diff*1e3:.1f} ms | speedup: {dt_kde/dt_diff:.2f}x")
    # Informational gate: record all four numbers; decide goal B in review.
