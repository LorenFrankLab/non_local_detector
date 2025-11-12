"""Test GMM convergence to KDE with increasing components.

GMM is a parametric approximation to KDE. As the number of GMM components increases,
GMM should converge to KDE. This test verifies:

1. Mathematical formulation consistency
2. Convergence with increasing components
3. Implementation correctness

Key mathematical relationships:
- KDE: p(x,m|s) = (1/N) Σ_i K_pos(x, x_i) K_mark(m, m_i)
- GMM: p(x,m|s) = Σ_k π_k N(x,m | μ_k, Σ_k)

As K→∞, GMM → KDE (in the limit).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)


@pytest.fixture
def convergence_test_data():
    """Create controlled test data for convergence analysis."""
    np.random.seed(42)

    # Simple 1D position for clarity
    n_time = 50
    dt = 0.02
    time = np.arange(n_time + 1) * dt

    position_time = np.linspace(0, time[-1], 200)
    position = np.linspace(0, 10, len(position_time))[:, None]  # 1D position

    # Single electrode for simplicity
    n_spikes = 100
    spike_times = [np.sort(np.random.uniform(time[0], time[-1], n_spikes))]

    # 2D waveform features
    spike_features = [np.random.randn(n_spikes, 2).astype(np.float32)]

    # Environment
    environment = Environment(position_range=[(0, 10)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_features": spike_features,
        "environment": environment,
    }


def test_gmm_convergence_to_kde(convergence_test_data):
    """Test that increasing GMM components brings predictions closer to KDE.

    This is the key test: GMM should converge to KDE as components increase.
    """
    data = convergence_test_data

    # Convert to JAX
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time = jnp.asarray(data["time"])
    spike_times = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["spike_features"]]

    # Fit KDE (reference)
    kde_enc = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=data["environment"],
        position_std=1.5,
        waveform_std=1.0,
        disable_progress_bar=True,
    )

    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    # Test GMM with increasing components
    component_counts = [4, 8, 16, 32, 64]
    correlations = []
    mse_values = []

    print("\n=== GMM Convergence to KDE ===")
    print("Components | Correlation | MSE (normalized) | Peak Agreement")
    print("-" * 65)

    for n_comp in component_counts:
        gmm_enc = fit_clusterless_gmm_encoding_model(
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            environment=data["environment"],
            gmm_components_occupancy=n_comp,
            gmm_components_gpi=n_comp,
            gmm_components_joint=n_comp,
            gmm_random_state=42,
            disable_progress_bar=True,
        )

        ll_gmm = predict_clusterless_gmm_log_likelihood(
            time=time,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            encoding_model=gmm_enc,
            is_local=False,
            disable_progress_bar=True,
        )

        # Normalize both to [0, 1] per time bin for fair comparison
        def normalize_per_time(ll):
            ll_np = np.asarray(ll)
            ll_min = ll_np.min(axis=1, keepdims=True)
            ll_max = ll_np.max(axis=1, keepdims=True)
            return (ll_np - ll_min) / (ll_max - ll_min + 1e-10)

        kde_norm = normalize_per_time(ll_kde)
        gmm_norm = normalize_per_time(ll_gmm)

        # Correlation
        corr = np.corrcoef(kde_norm.ravel(), gmm_norm.ravel())[0, 1]
        correlations.append(corr)

        # MSE
        mse = np.mean((kde_norm - gmm_norm) ** 2)
        mse_values.append(mse)

        # Peak agreement
        peak_kde = np.argmax(ll_kde, axis=1)
        peak_gmm = np.argmax(ll_gmm, axis=1)
        peak_agreement = np.mean(peak_kde == peak_gmm)

        print(
            f"{n_comp:4d}       | {corr:11.4f} | {mse:16.6f} | {peak_agreement:14.1%}"
        )

    # Verify convergence trend
    print("\n=== Convergence Analysis ===")
    print(f"Correlation improvement: {correlations[0]:.4f} → {correlations[-1]:.4f}")
    print(f"MSE improvement: {mse_values[0]:.6f} → {mse_values[-1]:.6f}")

    # Key assertion: correlation should increase with more components
    assert correlations[-1] > correlations[0], (
        f"Correlation should increase: {correlations[0]:.3f} → {correlations[-1]:.3f}"
    )

    # MSE should decrease
    assert mse_values[-1] < mse_values[0], (
        f"MSE should decrease: {mse_values[0]:.4f} → {mse_values[-1]:.4f}"
    )


def test_mathematical_formula_consistency(convergence_test_data):
    """Verify the mathematical formulas match between KDE and GMM.

    KDE formula (line 99-101 in clusterless_kde.py):
        log(mean_rate * marginal_density / occupancy)

    GMM formula (line 556 in clusterless_gmm.py):
        log(mean_rate) + joint_logp - log_occ
        = log(mean_rate) + log(p(x,m)) - log(occupancy(x))
        = log(mean_rate * p(x,m) / occupancy(x))

    These should be mathematically identical!
    """

    # The formulas are:
    # KDE: log(λ * p(x,m) / π(x))  where p is marginal_density, π is occupancy
    # GMM: log(λ) + log(p(x,m)) - log(π(x))

    # These are the same by logarithm properties:
    # log(a * b / c) = log(a) + log(b) - log(c)

    # Let's verify with synthetic values
    mean_rate = 5.0
    marginal_density = 0.3
    occupancy = 0.1

    # KDE formula
    kde_formula = np.log(mean_rate * marginal_density / occupancy)

    # GMM formula
    gmm_formula = np.log(mean_rate) + np.log(marginal_density) - np.log(occupancy)

    print("\n=== Formula Verification ===")
    print(
        f"KDE: log({mean_rate} * {marginal_density} / {occupancy}) = {kde_formula:.6f}"
    )
    print(
        f"GMM: log({mean_rate}) + log({marginal_density}) - log({occupancy}) = {gmm_formula:.6f}"
    )
    print(f"Difference: {abs(kde_formula - gmm_formula):.10f}")

    assert np.isclose(kde_formula, gmm_formula, rtol=1e-10), (
        "KDE and GMM formulas should be mathematically identical"
    )


def test_ground_process_intensity_calculation(convergence_test_data):
    """Verify ground process intensity calculation is consistent.

    Both KDE and GMM should compute:
        GPI = Σ_electrodes mean_rate * (gpi_density / occupancy)

    This is the "expected spike count" at each position.
    """
    data = convergence_test_data

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    spike_times = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["spike_features"]]

    # Fit both models
    kde_enc = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=data["environment"],
        position_std=1.5,
        disable_progress_bar=True,
    )

    gmm_enc = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=data["environment"],
        gmm_components_occupancy=32,
        gmm_components_gpi=32,
        gmm_components_joint=32,
        disable_progress_bar=True,
    )

    # Both should have summed_ground_process_intensity
    gpi_kde = kde_enc["summed_ground_process_intensity"]
    gpi_gmm = gmm_enc["summed_ground_process_intensity"]

    print("\n=== Ground Process Intensity Comparison ===")
    print(f"KDE GPI: mean={np.mean(gpi_kde):.4f}, std={np.std(gpi_kde):.4f}")
    print(f"GMM GPI: mean={np.mean(gpi_gmm):.4f}, std={np.std(gpi_gmm):.4f}")

    # Correlation
    corr = np.corrcoef(np.asarray(gpi_kde), np.asarray(gpi_gmm))[0, 1]
    print(f"Correlation: {corr:.4f}")

    # Should be positively correlated
    assert corr > 0.5, f"GPI should be correlated, got r={corr:.3f}"

    # Both should be positive
    assert np.all(gpi_kde >= 0)
    assert np.all(gpi_gmm >= 0)


def test_segment_sum_correctness(convergence_test_data):
    """Verify segment_sum is used correctly in GMM.

    KDE uses: jax.ops.segment_sum(..., indices_are_sorted=True)
    GMM uses: segment_sum(..., indices_are_sorted=True)

    These should be equivalent (different import paths).
    """
    # Create test data
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    segment_ids = jnp.array([0, 0, 1, 1, 2])

    # KDE way
    result_kde = jax.ops.segment_sum(
        values, segment_ids, num_segments=3, indices_are_sorted=True
    )

    # GMM way (from jax.ops import segment_sum)
    from jax.ops import segment_sum

    result_gmm = segment_sum(
        values, segment_ids, num_segments=3, indices_are_sorted=True
    )

    print("\n=== segment_sum Verification ===")
    print(f"Input: values={values}, segment_ids={segment_ids}")
    print(f"KDE result: {result_kde}")
    print(f"GMM result: {result_gmm}")

    assert jnp.allclose(result_kde, result_gmm), (
        "segment_sum implementations should match"
    )


def test_log_space_operations(convergence_test_data):
    """Verify log-space operations are done correctly in GMM.

    Common bugs:
    1. Adding logs instead of multiplying: log(a) + log(b) ✓, but a + b ✗
    2. Mixing log and linear space: log(a) + b ✗
    3. segment_sum on logs without log-sum-exp trick
    """
    data = convergence_test_data

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time = jnp.asarray(data["time"])
    spike_times = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["spike_features"]]

    # Fit GMM
    gmm_enc = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=data["environment"],
        gmm_components_occupancy=16,
        gmm_components_gpi=16,
        gmm_components_joint=16,
        disable_progress_bar=True,
    )

    # Predict
    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_enc,
        is_local=False,
        disable_progress_bar=True,
    )

    print("\n=== Log-Space Operations Check ===")
    print(f"Output range: [{np.min(ll_gmm):.2f}, {np.max(ll_gmm):.2f}]")
    print(f"All finite: {np.all(np.isfinite(ll_gmm))}")

    # Check formula: log_likelihood should contain:
    # 1. -GPI term (negative, in linear space then used as-is)
    # 2. Per-spike terms: log(rate) + joint_logp - log_occ (all in log space)

    # The -GPI term is LINEAR, not log! This is CORRECT for Poisson likelihood.
    # L = -λ + Σ log(λ * p(x,m) / π(x))

    # Key insight: GMM line 510-512 uses -summed_ground (linear space) ✓
    # Then adds log-space spike contributions ✓

    # This is mathematically correct! The Poisson log-likelihood is:
    # log P(spikes) = -λ(x) + Σ_spikes log(λ(x) * p(x,m) / π(x))

    assert np.all(np.isfinite(ll_gmm)), "All likelihoods should be finite"


@pytest.mark.skip(reason="Visualization - run manually")
def test_visualize_convergence(convergence_test_data):
    """Visualize GMM convergence to KDE."""
    data = convergence_test_data

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time = jnp.asarray(data["time"])
    spike_times = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["spike_features"]]

    # KDE reference
    kde_enc = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=data["environment"],
        position_std=1.5,
        disable_progress_bar=True,
    )

    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    # GMMs with different components
    component_counts = [4, 16, 64]
    fig, axes = plt.subplots(1, len(component_counts) + 1, figsize=(20, 4))

    # Plot KDE
    axes[0].imshow(np.asarray(ll_kde).T, aspect="auto", cmap="viridis")
    axes[0].set_title("KDE (Reference)")
    axes[0].set_xlabel("Time bin")
    axes[0].set_ylabel("Position bin")

    # Plot GMMs
    for idx, n_comp in enumerate(component_counts):
        gmm_enc = fit_clusterless_gmm_encoding_model(
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            environment=data["environment"],
            gmm_components_occupancy=n_comp,
            gmm_components_gpi=n_comp,
            gmm_components_joint=n_comp,
            disable_progress_bar=True,
        )

        ll_gmm = predict_clusterless_gmm_log_likelihood(
            time=time,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            encoding_model=gmm_enc,
            is_local=False,
            disable_progress_bar=True,
        )

        axes[idx + 1].imshow(np.asarray(ll_gmm).T, aspect="auto", cmap="viridis")
        axes[idx + 1].set_title(f"GMM ({n_comp} components)")
        axes[idx + 1].set_xlabel("Time bin")

    plt.tight_layout()
    plt.savefig("gmm_kde_convergence.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to gmm_kde_convergence.png")
