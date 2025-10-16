import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import chunked_filter_smoother, smoother, viterbi
from non_local_detector.core import filter as hmm_filter
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)


@pytest.mark.parametrize("n_chunks", [2, 3, 5])
def test_filter_smoother_with_sorted_kde_nonlocal(n_chunks, simple_1d_environment):
    # Encoding data
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    spikes = [np.array([2.0, 5.0, 5.1]), np.array([1.5, 7.2])]
    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        environment=env,
        weights=np.ones_like(t_pos),
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # Decoding log-likelihoods (non-local across interior bins)
    t_edges = np.linspace(0.0, 10.0, 21)  # 20 time bins
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )
    assert ll.ndim == 2
    n_time, n_states = ll.shape

    # Simple near-identity transition (row-stochastic)
    tm = np.eye(n_states) * 0.9 + (np.ones((n_states, n_states)) / n_states) * 0.1
    tm = tm / tm.sum(axis=1, keepdims=True)

    init = np.ones((n_states,)) / n_states
    (marg_like, _), (filtered, predicted) = hmm_filter(
        jnp.asarray(init), jnp.asarray(tm), jnp.asarray(ll)
    )
    assert filtered.shape == (n_time, n_states)
    # Each row sums to ~1
    rowsum = filtered.sum(axis=1)
    assert np.allclose(np.array(rowsum), 1.0, rtol=1e-6, atol=1e-8)
    assert np.isfinite(np.array(marg_like))

    smoothed = smoother(jnp.asarray(tm), filtered)
    assert smoothed.shape == (n_time, n_states)
    rowsum_s = smoothed.sum(axis=1)
    assert np.allclose(np.array(rowsum_s), 1.0, rtol=1e-6, atol=1e-8)

    path = viterbi(jnp.asarray(init), jnp.asarray(tm), jnp.asarray(ll))
    assert path.shape == (n_time,)
    assert path.min() >= 0 and path.max() < n_states

    # Chunked equivalence: pass precomputed ll to avoid recomputation differences
    state_ind = np.arange(n_states)
    (
        acausal_post_chunked,
        acausal_state_probs_chunked,
        marg_like_chunked,
        causal_state_probs_chunked,
        predictive_state_probs_chunked,
        ll_cache,
        causal_post_chunked,
    ) = chunked_filter_smoother(
        time=t_edges,
        state_ind=state_ind,
        initial_distribution=init,
        transition_matrix=tm,
        log_likelihood_func=lambda *args, **kwargs: None,
        log_likelihood_args=(),
        is_missing=None,
        n_chunks=n_chunks,
        log_likelihoods=np.asarray(ll),
        cache_log_likelihoods=False,
    )

    # Compare to non-chunked results
    assert acausal_post_chunked.shape == smoothed.shape
    assert np.allclose(acausal_post_chunked, np.asarray(smoothed), rtol=1e-5, atol=1e-6)
    assert np.allclose(causal_post_chunked, np.asarray(filtered), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_chunks", [2, 4])
def test_chunked_equals_nonchunked_clusterless_kde_nonlocal(n_chunks, simple_1d_environment):
    # Encoding data
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    enc_times = [np.array([2.0, 5.0, 7.5])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]
    enc = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=enc_times,
        spike_waveform_features=enc_feats,
        environment=env,
        weights=np.ones_like(t_pos),
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    t_edges = np.linspace(0.0, 10.0, 21)
    dec_times = [np.array([2.1, 5.2])]
    dec_feats = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    ll = predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_times,
        spike_waveform_features=dec_feats,
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        encoding_spike_weights=enc["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc["position_std"]),
        waveform_std=jnp.asarray(enc["waveform_std"]),
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )

    n_time, n_states = ll.shape
    tm = np.eye(n_states) * 0.9 + (np.ones((n_states, n_states)) / n_states) * 0.1
    tm = tm / tm.sum(axis=1, keepdims=True)
    init = np.ones((n_states,)) / n_states

    # Non-chunked
    (marg_like, _), (filtered, _) = hmm_filter(
        jnp.asarray(init), jnp.asarray(tm), jnp.asarray(ll)
    )
    smoothed = smoother(jnp.asarray(tm), filtered)

    # Chunked
    state_ind = np.arange(n_states)
    (
        acausal_post_chunked,
        _,
        _,
        _,
        _,
        _,
        causal_post_chunked,
    ) = chunked_filter_smoother(
        time=t_edges,
        state_ind=state_ind,
        initial_distribution=init,
        transition_matrix=tm,
        log_likelihood_func=lambda *args, **kwargs: None,
        log_likelihood_args=(),
        is_missing=None,
        n_chunks=n_chunks,
        log_likelihoods=np.asarray(ll),
        cache_log_likelihoods=False,
    )

    assert np.allclose(acausal_post_chunked, np.asarray(smoothed), rtol=1e-5, atol=1e-6)
    assert np.allclose(causal_post_chunked, np.asarray(filtered), rtol=1e-5, atol=1e-6)
