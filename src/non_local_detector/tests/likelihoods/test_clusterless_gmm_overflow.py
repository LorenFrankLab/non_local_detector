"""Over-fitted clusterless GMM: its ground intensity overflows only where predicted.

A stress regression on a 2-D simulation with fifteen electrodes: fitted with
many components and almost no covariance regularization, the GMM's
``rate * p_gpi / p_occ`` exceeds the float32 range at some bins. The fitted
intensity and the non-local log likelihood must then be infinite exactly
there and correct elsewhere.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references

# The stress fit is integration-scale; marking it excludes it from fast
# feedback loops (``pytest -m "not integration"``).
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def comparison_data():
    """A 2-D trajectory with fifteen electrodes of 80-120 spikes each."""
    rng = np.random.default_rng(123)

    # Time parameters - longer duration for better statistics
    dt = 0.02  # 20 ms bins
    n_time = 100
    time_edges = np.arange(n_time + 1) * dt  # n_time bins

    # Position parameters - smoother trajectory
    position_time = np.linspace(0, time_edges[-1], 400)
    t_normalized = np.linspace(0, 1, len(position_time))
    position = np.column_stack(
        [
            10 * t_normalized,  # linear x motion
            2 * np.sin(4 * np.pi * t_normalized),  # sinusoidal y motion
        ]
    )

    # Spike parameters - enough electrodes for stable statistics
    n_electrodes = 15
    n_features = 4
    encoding_spike_times = []
    encoding_spike_features = []

    for elec_idx in range(n_electrodes):
        # Generate spatially modulated firing
        n_spikes = rng.integers(80, 120)
        times = np.sort(rng.uniform(time_edges[0], time_edges[-1], n_spikes))
        encoding_spike_times.append(times)

        # Features with electrode-specific bias
        features = rng.standard_normal((n_spikes, n_features)).astype(np.float32)
        features += elec_idx * 0.5  # electrode-specific offset
        encoding_spike_features.append(features)

    # Decoding spikes
    decoding_spike_times = [times[: len(times) // 3] for times in encoding_spike_times]
    decoding_spike_features = [
        feats[: len(feats) // 3] for feats in encoding_spike_features
    ]

    # Create environment
    environment = Environment(position_range=[(0, 10), (-3, 3)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time_edges": time_edges,
        "position_time": position_time,
        "position": position,
        "encoding_spike_times": encoding_spike_times,
        "encoding_spike_features": encoding_spike_features,
        "decoding_spike_times": decoding_spike_times,
        "decoding_spike_features": decoding_spike_features,
        "environment": environment,
    }


# With 16 occupancy / 32 joint components on ~100 spikes per electrode the
# fitted occupancy density reaches log p ~ -364 *inside* the track and the
# GPI/occupancy ratio overflows float32.
GMM_COMPONENTS_STRESS = (16, 32)


def _fit_gmm(data, gmm_components, gmm_reg_covar):
    """Fit the clusterless GMM and its training mark references.

    Returns
    -------
    gmm_encoding : dict
    mark_reference : list of MarkReference
    """
    encoding_spike_features = [
        jnp.asarray(sf) for sf in data["encoding_spike_features"]
    ]
    gmm_encoding = fit_clusterless_gmm_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=[jnp.asarray(st) for st in data["encoding_spike_times"]],
        spike_waveform_features=encoding_spike_features,
        environment=data["environment"],
        gmm_components_occupancy=gmm_components[0],
        gmm_components_joint=gmm_components[1],
        gmm_random_state=42,
        gmm_reg_covar=gmm_reg_covar,
        disable_progress_bar=True,
    )
    mark_reference = fit_mark_references(
        encoding_spike_features, mark_reference_scale(gmm_reg_covar=gmm_reg_covar)
    )
    return gmm_encoding, mark_reference


def test_overparameterized_gmm_overflow_locations(comparison_data):
    """Stress regression: an over-fitted GMM overflows only where predicted.

    With 16 occupancy / 32 joint components on ~100 spikes per electrode the
    fitted ratio ``rate * p_gpi / p_occ`` exceeds the float32 range at several
    bins (and the float64 range at one). The reference is the complete
    summed intensity accumulated in float64 from the fitted models' own log
    densities (rate included, aggregated over electrodes): the log likelihood
    must be ``-inf`` exactly where that sum exceeds the working dtype's
    maximum, finite and equal to the reference elsewhere, and never NaN or
    ``+inf``. At least one candidate per time bin must stay finite.
    """
    # Nearly unregularized, so the over-fitted components are sharp enough
    # to overflow.
    gmm_enc, mark_reference = _fit_gmm(
        comparison_data, gmm_components=GMM_COMPONENTS_STRESS, gmm_reg_covar=1e-6
    )
    bins = gmm_enc["interior_place_bin_centers"]
    log_occ = np.asarray(gmm_enc["log_occupancy"], dtype=np.float64)
    log_terms = np.stack(
        [
            np.log(np.float64(rate))
            + np.asarray(gpi.score_samples(bins), np.float64)
            - log_occ
            for gpi, rate in zip(
                gmm_enc["gpi_models"], gmm_enc["mean_rates"], strict=True
            )
            if gpi is not None
        ]
    )
    expected_intensity = np.exp(log_terms).sum(axis=0)  # (n_bins,) float64
    summed = np.asarray(gmm_enc["summed_ground_process_intensity"])
    working_max = np.finfo(summed.dtype).max
    overflow = expected_intensity > working_max
    clearly_finite = expected_intensity < 0.5 * working_max
    assert (overflow | clearly_finite).all(), "no bin may sit at the rounding boundary"
    if summed.dtype == np.float32:
        assert overflow.any(), "stress configuration must overflow float32 somewhere"

    summed = summed.astype(np.float64)
    assert np.all(np.isposinf(summed[overflow]))
    np.testing.assert_allclose(
        summed[clearly_finite], expected_intensity[clearly_finite], rtol=1e-4
    )

    data = comparison_data
    ll_gmm = np.asarray(
        predict_clusterless_gmm_log_likelihood(
            time_edges=jnp.asarray(data["time_edges"]),
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=[jnp.asarray(st) for st in data["decoding_spike_times"]],
            spike_waveform_features=[
                jnp.asarray(sf) for sf in data["decoding_spike_features"]
            ],
            **gmm_enc,
            mark_reference=mark_reference,
            is_local=False,
        )
    )
    assert not np.any(np.isnan(ll_gmm))
    assert not np.any(np.isposinf(ll_gmm))
    assert np.all(np.isneginf(ll_gmm[:, overflow]))
    assert np.all(np.isfinite(ll_gmm[:, clearly_finite]))
    assert np.all(np.isfinite(ll_gmm).any(axis=1)), (
        "every time bin needs a finite candidate"
    )
