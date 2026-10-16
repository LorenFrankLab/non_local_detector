"""Tests for GMM fitting convergence and basic clusterless likelihood behavior.

Numerical KDE/GMM equivalence is checked in test_kde_gmm_equivalence.py using
bandwidth-matched models. This module checks fitted ground-process intensities,
likelihood finiteness, and EM convergence diagnostics.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    mark_reference_scale as kde_mark_reference_scale,
)
from non_local_detector.likelihoods.gmm import GaussianMixtureModel
from non_local_detector.likelihoods.mark_reference import fit_mark_references


@pytest.fixture
def convergence_test_data():
    """Create controlled test data for convergence analysis."""
    rng = np.random.default_rng(42)

    # Simple 1D position for clarity
    n_time = 50
    dt = 0.02
    time_edges = np.arange(n_time + 1) * dt  # n_time bins

    position_time = np.linspace(0, time_edges[-1], 200)
    position = np.linspace(0, 10, len(position_time))[:, None]  # 1D position

    # Single electrode for simplicity
    n_spikes = 100
    spike_times = [np.sort(rng.uniform(time_edges[0], time_edges[-1], n_spikes))]

    # 2D waveform features
    spike_features = [rng.standard_normal((n_spikes, 2)).astype(np.float32)]

    # Environment
    environment = Environment(position_range=[(0, 10)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time_edges": time_edges,
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_features": spike_features,
        "environment": environment,
    }


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


def test_gmm_nonlocal_likelihood_is_finite(convergence_test_data):
    """A fitted GMM produces finite non-local likelihoods on this small dataset."""
    data = convergence_test_data

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time_edges = jnp.asarray(data["time_edges"])
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
        gmm_components_joint=16,
        disable_progress_bar=True,
    )

    # Predict
    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time_edges=time_edges,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        **gmm_enc,
        # Training spikes all lie within the position time range.
        mark_reference=fit_mark_references(spike_features, mark_reference_scale()),
        is_local=False,
    )

    print("\n=== Log-Space Operations Check ===")
    print(f"Output range: [{np.min(ll_gmm):.2f}, {np.max(ll_gmm):.2f}]")
    print(f"All finite: {np.all(np.isfinite(ll_gmm))}")

    assert np.all(np.isfinite(ll_gmm)), "All likelihoods should be finite"


@pytest.mark.skip(reason="Visualization - run manually")
def test_visualize_convergence(convergence_test_data):
    """Visualize GMM convergence to KDE."""
    data = convergence_test_data

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time_edges = jnp.asarray(data["time_edges"])
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
        time_edges=time_edges,
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
        mark_reference=fit_mark_references(
            spike_features,
            kde_mark_reference_scale(waveform_std=kde_enc["waveform_std"]),
        ),
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
            gmm_components_joint=n_comp,
            disable_progress_bar=True,
        )

        ll_gmm = predict_clusterless_gmm_log_likelihood(
            time_edges=time_edges,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_features,
            **gmm_enc,
            mark_reference=fit_mark_references(spike_features, mark_reference_scale()),
            is_local=False,
        )

        axes[idx + 1].imshow(np.asarray(ll_gmm).T, aspect="auto", cmap="viridis")
        axes[idx + 1].set_title(f"GMM ({n_comp} components)")
        axes[idx + 1].set_xlabel("Time bin")

    plt.tight_layout()
    plt.savefig("gmm_kde_convergence.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to gmm_kde_convergence.png")


def test_gmm_max_iter_warning():
    """``GaussianMixtureModel.fit`` should warn and set ``converged_=False``
    when EM hits ``max_iter`` before the lower-bound delta drops below ``tol``.
    """
    rng = np.random.default_rng(0)
    # Two well-separated Gaussians: a reasonable EM target that still needs
    # more than one iteration to drive the lower-bound delta below 1e-20.
    X = jnp.asarray(
        np.vstack(
            [
                rng.normal(loc=[0.0, 0.0], size=(100, 2)),
                rng.normal(loc=[5.0, 5.0], size=(100, 2)),
            ]
        ).astype(np.float32)
    )
    key = jax.random.PRNGKey(0)

    model = GaussianMixtureModel(n_components=3, max_iter=1, tol=1e-20)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.converged_ is False
    user_warnings = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert user_warnings, (
        f"Expected a UserWarning about non-convergence, got: "
        f"{[str(w.message) for w in recorded]}"
    )


def test_gmm_converged_true_on_well_separated_data():
    """A fit that reaches tolerance well before ``max_iter`` sets
    ``converged_=True`` and emits no non-convergence ``UserWarning``."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        np.vstack(
            [
                rng.normal(loc=[0.0, 0.0], scale=0.1, size=(200, 2)),
                rng.normal(loc=[10.0, 10.0], scale=0.1, size=(200, 2)),
            ]
        ).astype(np.float32)
    )
    key = jax.random.PRNGKey(0)

    model = GaussianMixtureModel(n_components=2, max_iter=200, tol=1e-3, n_init=1)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.converged_ is True
    assert model.n_iter_ < model.max_iter
    non_conv = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert non_conv == [], (
        f"Unexpected non-convergence warning on a converging fit: "
        f"{[str(w.message) for w in non_conv]}"
    )


def test_gmm_converged_true_at_max_iter_boundary():
    """``converged_`` must be True when the lower-bound delta drops below
    ``tol`` on the same iteration that hits ``max_iter``.

    The lower-bound delta is only finite from the second iteration onward
    (the first compares against an initial ``-inf``). Initializing at the
    solution makes the delta negligible by iteration 2, so with ``max_iter=2``
    the loop stops with ``final_i == max_iter`` AND ``delta <= tol``. The old
    ``n_iter < max_iter`` heuristic mislabeled this as non-converged.
    """
    rng = np.random.default_rng(0)
    c0 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(200, 2))
    c1 = rng.normal(loc=[10.0, 10.0], scale=0.1, size=(200, 2))
    X = jnp.asarray(np.vstack([c0, c1]).astype(np.float32))
    key = jax.random.PRNGKey(0)

    # Initialize at the (well-separated) cluster solution -- means, weights AND
    # covariances -- so the first M-step is essentially a no-op and the
    # lower-bound delta is negligible by the second iteration.
    means_init = jnp.asarray(
        np.vstack([c0.mean(axis=0), c1.mean(axis=0)]).astype(np.float32)
    )
    covariances_init = jnp.asarray(
        np.stack([np.cov(c0, rowvar=False), np.cov(c1, rowvar=False)]).astype(
            np.float32
        )
    )
    model = GaussianMixtureModel(
        n_components=2,
        max_iter=2,
        tol=1e-2,
        n_init=1,
        means_init=means_init,
        weights_init=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        covariances_init=covariances_init,
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.n_iter_ == 2, (
        "Test premise: EM should run exactly max_iter=2 iterations to exercise "
        f"the boundary; got n_iter_={model.n_iter_}"
    )
    assert model.converged_ is True, (
        "EM reached tol on the final allowed iteration but converged_ was False "
        "(max_iter off-by-one)"
    )
    non_conv = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert non_conv == [], (
        f"Unexpected non-convergence warning at the converged boundary: "
        f"{[str(w.message) for w in non_conv]}"
    )
