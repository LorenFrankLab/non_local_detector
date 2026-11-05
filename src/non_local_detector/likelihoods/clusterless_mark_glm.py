"""Clusterless mark-mixture GLM likelihood.

Each electrode's joint mark intensity is a sum over mark components,

    lambda_e(x, m) = sum_k lambda_ek(x) N(m; mu_ek, Sigma_ek),

where the components are a Gaussian mixture over the electrode's marks alone and
each ``lambda_ek(x)`` is a Poisson GLM rate map. The fit has two stages:

1.  **Marks.** A Dirichlet-process variational Bayes Gaussian mixture is fit to
    the electrode's training marks, with each spike counted by its encoding
    weight. The prior prunes components the marks do not support, so
    ``n_mark_components`` is an upper bound, not the number used.
2.  **Space.** Each spike is split between the components by its mark
    responsibilities, and each component's rate map is fit to those soft counts
    with occupancy as exposure. The maps use the sorted-spikes GLM's basis
    (cubic splines along the branches of a track graph, or a tensor spline over
    the environment), position-cell sums and log-rate prior (see
    ``fit_sorted_spikes_glm_encoding_model``).

The marks are fit without position, so the mixture cannot place components on
the positions of scattered spikes, and the rate maps follow the track's topology
rather than linear position. As in the other clusterless likelihoods, every
spike's intensity also includes the background ``beta * q(m)`` (see
``mark_reference``).

Limitations:

- The responsibilities do not depend on position. Where two units' marks
  overlap, a component's soft counts mix their place fields, so
  position-specific mark differences are attenuated.
- Weights count as spikes in the mark mixture's prior: a fit whose weights sum
  to few spikes (a rarely occupied state in EM) supports fewer components.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from sklearn.mixture import BayesianGaussianMixture  # type: ignore[import-untyped]

from non_local_detector._warnings import user_stacklevel
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import DataQualityWarning, ValidationError
from non_local_detector.likelihoods.clusterless_gmm import DEFAULT_GMM_REG_COVAR
from non_local_detector.likelihoods.clusterless_spikes import (
    ElectrodeSpikes,
    add_background_only,
    add_electrode_spike_terms,
)
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    SpikeTimeOrder,
    as_array,
    as_arrays,
    deterministic_row_add,
    deterministic_row_sum,
    environment_grid_fingerprint,
    get_position_at_time,
    log_bin_duration_evidence,
    pad_rows,
    padded_spike_count,
    prepare_decode_rows,
    require_position,
    select_spike_rows,
    snap_off_track_positions,
    spike_row_ids,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_positive_int,
    validate_spike_feature_pair,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.encoding_model import ClusterlessEncodingModel
from non_local_detector.likelihoods.gmm import estimate_log_gaussian_prob
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_rate,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    GLM_LOG_RATE_PRIOR_STD,
    TensorSplineBasis,
    TrackGraphBasis,
    fit_rate_maps,
    glm_rates,
    make_spline_predict_matrix,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
    requires_time_edges,
)

__all__ = [
    "ClusterlessMarkGLMEncodingModel",
    "fit_clusterless_mark_glm_encoding_model",
    "predict_clusterless_mark_glm_log_likelihood",
]


# Variational components supported by fewer weighted spikes than this are
# dropped after the fit, and the remaining weights renormalized. A component the
# marks do not use keeps about one spike of prior mass but almost no
# responsibility, at any number of spikes.
_MIN_COMPONENT_SPIKES = 1.0


@dataclass(frozen=True, eq=False)
class MarkMixture:
    """A fitted Gaussian mixture over one electrode's marks.

    Attributes
    ----------
    weights : np.ndarray, shape (n_components,)
        Mixture weights; they sum to 1.
    means : np.ndarray, shape (n_components, n_features)
    precisions_cholesky : np.ndarray, shape (n_components, n_features, n_features)
        Cholesky factors ``L`` of each component's precision, so that
        ``(m - mu) @ L`` is a standard normal (scikit-learn's convention).
    """

    weights: np.ndarray
    means: np.ndarray
    precisions_cholesky: np.ndarray

    @property
    def n_components(self) -> int:
        """Number of mixture components.

        Returns
        -------
        n_components : int
            Number of mixture components.
        """
        return len(self.weights)

    def log_component_density(self, marks: np.ndarray) -> np.ndarray:
        """``log N(m; mu_k, Sigma_k)`` per mark and component, in float64.

        Parameters
        ----------
        marks : np.ndarray, shape (n_spikes, n_features)
            Spike marks.

        Returns
        -------
        log_density : np.ndarray, shape (n_spikes, n_components)
            Log density of each mark under each component.
        """
        marks = np.asarray(marks, dtype=float)
        # (m - mu) @ L, subtracting first so a large common offset cancels.
        standardized = np.einsum(
            "nkd,kde->nke", marks[:, None] - self.means[None], self.precisions_cholesky
        )
        log_det = np.sum(
            np.log(np.diagonal(self.precisions_cholesky, axis1=1, axis2=2)), axis=1
        )
        return cast(
            np.ndarray,
            log_det[None]
            - 0.5 * marks.shape[1] * np.log(2.0 * np.pi)
            - 0.5 * np.sum(standardized**2, axis=2),
        )

    def responsibilities(self, marks: np.ndarray) -> np.ndarray:
        """Each component's posterior probability for each mark.

        Parameters
        ----------
        marks : np.ndarray, shape (n_spikes, n_features)
            Spike marks.

        Returns
        -------
        responsibilities : np.ndarray, shape (n_spikes, n_components)
            Rows sum to 1.
        """
        log_joint = self.log_component_density(marks) + np.log(self.weights)[None]
        return cast(
            np.ndarray,
            np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True)),
        )


class _WeightedBayesianGaussianMixture(BayesianGaussianMixture):  # type: ignore[misc]  # sklearn is untyped
    """Variational Bayes Gaussian mixture whose samples count by a weight.

    Scales each sample's responsibilities by its weight in the initialization
    and every M-step, so the sufficient statistics are weighted sums. The
    default mean and covariance priors are the weighted mean and covariance of
    the data (scikit-learn's defaults, weighted), and convergence uses the
    variational bound of the weighted data. With integer weights the fit is
    that of the replicated samples. Overrides private scikit-learn methods;
    ``fit_weighted`` raises unless each override ran, so an incompatible
    scikit-learn cannot silently ignore the weights.
    """

    def fit_weighted(
        self, X: np.ndarray, sample_weight: np.ndarray
    ) -> _WeightedBayesianGaussianMixture:
        """Fit to ``X`` with each sample counted by ``sample_weight``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples.
        sample_weight : np.ndarray, shape (n_samples,)
            Positive weight of each sample.

        Returns
        -------
        self : _WeightedBayesianGaussianMixture
            The fitted mixture.
        """
        self._sample_weight = np.asarray(sample_weight, dtype=float)
        self._n_check_parameters = self._n_initialize = 0
        self._n_m_step = self._n_lower_bound = 0
        self.fit(X)
        if 0 in (
            self._n_check_parameters,
            self._n_initialize,
            self._n_m_step,
            self._n_lower_bound,
        ):
            msg = (
                "scikit-learn's BayesianGaussianMixture internals changed: the "
                "sample weights were not applied. Please report this issue."
            )
            raise RuntimeError(msg)
        return self

    def _check_parameters(self, X: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """Weighted default mean and covariance priors."""
        self._n_check_parameters += 1
        super()._check_parameters(X, *args, **kwargs)
        weight = self._sample_weight
        if self.mean_prior is None:
            self.mean_prior_ = np.average(X, axis=0, weights=weight)
        if self.covariance_prior is None:
            n_features = X.shape[1]
            if weight.sum() ** 2 / np.sum(weight**2) < n_features + 1:
                # Too few effective spikes for a full-rank covariance (one spike
                # can carry nearly all the weight): the unbiased estimate is
                # singular or, at a zero denominator, infinite, and a component
                # left with no weight inherits it. Use the weighted spread plus
                # the regularization added to every component covariance.
                deviation = X - np.average(X, axis=0, weights=weight)
                covariance = (deviation * weight[:, None]).T @ deviation / (
                    weight.sum()
                ) + self.reg_covar * np.eye(n_features)
            else:
                covariance = np.atleast_2d(np.cov(X.T, aweights=weight))
            self.covariance_prior_ = {
                "full": covariance,
                "tied": covariance,
                "diag": np.diag(covariance),
                "spherical": np.diag(covariance).mean(),
            }[self.covariance_type]

    def _compute_lower_bound(self, log_resp: np.ndarray, log_prob_norm: float) -> float:
        """Variational bound with each sample's entropy term weighted."""
        self._n_lower_bound += 1
        # scikit-learn's bound has -sum_i sum_k r_ik log r_ik; the weighted
        # samples' bound has -sum_i w_i sum_k r_ik log r_ik. Its other terms
        # depend on the data only through the weighted M-step's statistics.
        entropy_correction = np.sum(
            (1.0 - self._sample_weight)[:, None] * np.exp(log_resp) * log_resp
        )
        return cast(
            float,
            super()._compute_lower_bound(log_resp, log_prob_norm) + entropy_correction,
        )

    def _initialize(
        self, X: np.ndarray, resp: np.ndarray, *args: Any, **kwargs: Any
    ) -> Any:
        """Weighted initial sufficient statistics."""
        self._n_initialize += 1
        return super()._initialize(
            X, resp * self._sample_weight[:, None], *args, **kwargs
        )

    def _m_step(
        self, X: np.ndarray, log_resp: np.ndarray, *args: Any, **kwargs: Any
    ) -> Any:
        """Weighted M-step."""
        self._n_m_step += 1
        return super()._m_step(
            X, log_resp + np.log(self._sample_weight)[:, None], *args, **kwargs
        )


def fit_mark_mixture(
    marks: np.ndarray,
    weights: np.ndarray,
    *,
    n_components: int = 32,
    weight_concentration_prior: float | None = None,
    reg_covar: float = DEFAULT_GMM_REG_COVAR,
    max_spikes: int = 20_000,
    max_iter: int = 500,
    tol: float = 1e-4,
    random_state: int = 0,
) -> MarkMixture:
    """Fit a Dirichlet-process variational Bayes mixture to weighted marks.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_features)
        Spike marks (waveform features).
    weights : np.ndarray, shape (n_spikes,)
        Nonnegative weight of each spike; zero-weight spikes are dropped. At
        least one must be positive.
    n_components : int, optional
        Upper bound on the number of components, by default 32.
    weight_concentration_prior : float, optional
        Dirichlet-process concentration; smaller values favor fewer
        components. By default None (scikit-learn's ``1 / n_components``).
    reg_covar : float, optional
        Added to each covariance diagonal, in squared mark units, by default
        ``DEFAULT_GMM_REG_COVAR``.
    max_spikes : int, optional
        At most this many positive-weight spikes, drawn at random with
        ``random_state``, are used for the fit, by default 20,000.
    max_iter : int, optional
        Maximum variational iterations, by default 500.
    tol : float, optional
        Convergence tolerance on the variational lower bound per weighted
        spike, in nats, by default 1e-4. scikit-learn's bound is a sum over
        spikes, so its tolerance is ``tol`` times the fit's total weight.
    random_state : int, optional
        Seed for the subsample and the k-means initialization, by default 0.

    Returns
    -------
    mixture : MarkMixture
        The components supported by at least one weighted spike (the most
        supported one if none is), with their weights renormalized. A single
        spike gives one component centred on its mark with covariance
        ``reg_covar`` times the identity, as the clusterless GMM fits it.
    """
    marks = np.asarray(marks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0.0
    marks, weights = marks[positive], weights[positive]
    if len(marks) > max_spikes:
        keep = np.sort(
            np.random.default_rng(random_state).choice(
                len(marks), max_spikes, replace=False
            )
        )
        marks, weights = marks[keep], weights[keep]
    if len(marks) == 1:
        n_features = marks.shape[1]
        return MarkMixture(
            weights=np.ones(1),
            means=marks,
            precisions_cholesky=np.eye(n_features)[None] / np.sqrt(reg_covar),
        )
    model = _WeightedBayesianGaussianMixture(
        n_components=min(n_components, len(marks)),
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol * weights.sum(),
        init_params="kmeans",
        random_state=random_state,
    ).fit_weighted(marks, weights)
    counts = weights @ model.predict_proba(marks)
    kept = counts >= min(_MIN_COMPONENT_SPIKES, counts.max())
    return MarkMixture(
        weights=model.weights_[kept] / model.weights_[kept].sum(),
        means=model.means_[kept],
        precisions_cholesky=model.precisions_cholesky_[kept],
    )


def mark_reference_scale(
    mark_reg_covar: float = DEFAULT_GMM_REG_COVAR, **_fit_params: object
) -> float:
    """Mark-kernel width the reference adds in quadrature to each feature's spread.

    Parameters
    ----------
    mark_reg_covar : float, optional
        The mark mixture's covariance regularization, a variance in squared
        feature units.
    **_fit_params
        Other fit parameters, ignored.

    Returns
    -------
    scale : float
        ``sqrt(mark_reg_covar)``.
    """
    return float(np.sqrt(mark_reg_covar))


class ClusterlessMarkGLMEncodingModel(ClusterlessEncodingModel):
    """The encoding model ``fit_clusterless_mark_glm_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_clusterless_mark_glm_log_likelihood``, documented there.
    """

    emission_design_info: TrackGraphBasis | TensorSplineBasis
    mark_mixtures: list[MarkMixture | None]
    rate_coefficients: list[np.ndarray | jax.Array | None]
    log_rate_maps: list[np.ndarray | None]
    summed_ground_process_intensity: np.ndarray
    spike_block_size: int


def fit_clusterless_mark_glm_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    n_mark_components: int = 32,
    mark_weight_concentration: float | None = None,
    mark_reg_covar: float = DEFAULT_GMM_REG_COVAR,
    mark_fit_max_spikes: int = 20_000,
    mark_max_iter: int = 500,
    mark_tol: float = 1e-4,
    mark_random_state: int = 0,
    emission_knot_spacing: float = np.sqrt(12.5) * 2,
    log_rate_prior_std: float = GLM_LOG_RATE_PRIOR_STD,
    spike_block_size: int = 1000,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessMarkGLMEncodingModel:
    """Fit the clusterless mark-mixture GLM encoding model.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Tracking timestamps in seconds.
    position : array_like, shape (n_time_position, n_position_dims)
        Animal position at each tracking sample.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Encoding spike times per electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Encoding spike marks per electrode.
    environment : Environment
        The fitted spatial environment.
    weights : array_like, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM), by
        default None (uniform). A spike counts by the weight interpolated at its
        time, in both the mark mixture and the rate maps.
    n_mark_components : int, optional
        Upper bound on each electrode's mark components, by default 32. The
        variational prior prunes the components the marks do not support.
    mark_weight_concentration : float, optional
        Dirichlet-process concentration of the mark mixture; smaller values
        favor fewer components. By default None (``1 / n_mark_components``).
    mark_reg_covar : float, optional
        Added to each mark covariance diagonal, in squared mark units, by
        default ``DEFAULT_GMM_REG_COVAR`` (1.0).
    mark_fit_max_spikes : int, optional
        At most this many spikes per electrode, drawn at random, fit the mark
        mixture, by default 20,000. Every spike contributes to the rate maps.
    mark_max_iter : int, optional
        Maximum variational iterations of the mark mixture, by default 500.
    mark_tol : float, optional
        Convergence tolerance of the mark mixture's variational bound per
        weighted spike, in nats, by default 1e-4.
    mark_random_state : int, optional
        Seed of the mark subsample and k-means initialization, by default 0.
    emission_knot_spacing : float, optional
        Spacing of the rate-map spline knots in position units, by default
        ``sqrt(12.5) * 2`` (about 7.07), as in the sorted-spikes GLM.
    log_rate_prior_std : float, optional
        Prior SD, in nats, of each component's log-rate deviation from its
        baseline across the interior bins, by default
        ``GLM_LOG_RATE_PRIOR_STD``.
    spike_block_size : int, optional
        Decode spikes scored per block at predict, by default 1000. Stored in
        the encoding model.
    disable_progress_bar : bool, optional
        Stored for prediction's progress bars, by default False.
    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds. Clips encoding support using the
        original interpolation basis. Uniform samples otherwise include endpoint
        half-cells; a singleton requires explicit bounds or a tracking interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
        Encoding support does not automatically mark decode bins missing.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support.

    Returns
    -------
    encoding_model : ClusterlessMarkGLMEncodingModel
        Keys ``algorithm`` (``"clusterless_mark_glm"``), ``environment`` and
        ``environment_grid``, and:

        emission_design_info : TrackGraphBasis or TensorSplineBasis
            The rate maps' spline basis.
        mark_mixtures : list of MarkMixture or None
            Per electrode; None for a zero-rate electrode (no positive-weight
            encoding spikes).
        rate_coefficients : list of np.ndarray or None
            Per electrode, shape (n_components, n_coefficients).
        log_rate_maps : list of np.ndarray or None
            Per electrode, shape (n_interior_bins, n_components): log rate in
            Hz of each component at the interior bin centers.
        summed_ground_process_intensity : jax.Array, shape (n_interior_bins,)
            Sum of every component's rate, in Hz.
        mean_rates : jax.Array, shape (n_electrodes,)
            Weighted mean firing rate per electrode, in Hz.
        mark_dimensions : list of int
        spike_block_size, disable_progress_bar
        rate_units : str
            ``"Hz"``.
        encoding_exposure_seconds : float
            Weighted encoding occupancy time in seconds.
    """
    validate_positive_int("spike_block_size", spike_block_size)
    position_time = as_array(position_time)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    weight_sum = float(exposure_weights.sum())
    warn_if_no_encoding_exposure(
        exposure_weights, "Clusterless mark GLM", clusterless=True
    )

    is_track_interior = environment.is_track_interior_flat_
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )
    encoding_positions = environment.decoding_position(position[support.indices])

    # Position of each tracking sample within the support (-1 outside it).
    support_position = np.full(len(position), -1)
    support_position[support.indices] = np.arange(len(support.indices))
    mark_mixtures: list[MarkMixture | None] = []
    mean_rates: list[float] = []
    mark_dimensions: list[int] = []
    spike_events: list[_SpikeEvent] = []
    for electrode, (electrode_times, electrode_marks) in enumerate(
        zip(spike_times, spike_waveform_features, strict=True)
    ):
        validate_spike_feature_pair(electrode_times, electrode_marks, electrode)
        electrode_times = np.asarray(electrode_times, dtype=float)
        electrode_marks = np.asarray(electrode_marks, dtype=float)
        mark_dimensions.append(int(electrode_marks.shape[1]))
        # Each spike's unit event is split between its bracketing samples by
        # their weights (as in ``EncodingSupport.event_counts``); its weight is
        # the sum of the two shares.
        spike_index, left, right, fraction = support.event_basis(electrode_times)
        # Only the spikes the support owns enter the fit, as in the KDE.
        validate_finite(electrode_marks[spike_index], "spike_waveform_features")
        left_share = (1.0 - fraction) * weights[left]
        right_share = fraction * weights[right]
        spike_weight = left_share + right_share
        positive = spike_weight > 0.0
        if not np.any(positive):
            # No effective spikes to fit: the electrode is zero-rate, and
            # predict explains each observed decode spike by the background
            # alone (as the KDE and GMM do).
            warnings.warn(
                f"Clusterless mark GLM: electrode {electrode} has no effective "
                "encoding spikes (empty or zero total weight); it is treated as "
                "zero-rate -- its observed decode spikes are explained by the "
                "background rate.",
                DataQualityWarning,
                stacklevel=user_stacklevel(),
            )
            mark_mixtures.append(None)
            mean_rates.append(0.0)
            continue
        mean_rates.append(weighted_mean_rate(spike_weight, weight_sum))
        marks = electrode_marks[spike_index[positive]]
        mixture = fit_mark_mixture(
            marks,
            spike_weight[positive],
            n_components=n_mark_components,
            weight_concentration_prior=mark_weight_concentration,
            reg_covar=mark_reg_covar,
            max_spikes=mark_fit_max_spikes,
            max_iter=mark_max_iter,
            tol=mark_tol,
            random_state=mark_random_state,
        )
        mark_mixtures.append(mixture)
        spike_events.append(
            (
                mixture,
                marks,
                support_position[left[positive]],
                support_position[right[positive]],
                left_share[positive],
                right_share[positive],
            )
        )

    emission_design_info, coefficients, interior_rates = fit_rate_maps(
        environment,
        environment.place_bin_edges_,
        interior_place_bin_centers,
        encoding_positions,
        _soft_cell_counts(spike_events),
        exposure_weights[support.indices],
        emission_knot_spacing=emission_knot_spacing,
        log_rate_prior_std=log_rate_prior_std,
        n_maps=sum(event[0].n_components for event in spike_events),
    )
    coefficients = np.asarray(coefficients)  # type: ignore[assignment]  # name reused for the host copy
    interior_rates = np.asarray(interior_rates)  # type: ignore[assignment]  # name reused for the host copy
    rate_coefficients: list[np.ndarray | jax.Array | None] = []
    log_rate_maps: list[np.ndarray | None] = []
    first = 0
    for mixture in mark_mixtures:  # type: ignore[assignment]  # name reused; elements may be None
        if mixture is None:
            rate_coefficients.append(None)  # type: ignore[unreachable]  # follows from the name reuse above
            log_rate_maps.append(None)
            continue
        maps = slice(first, first + mixture.n_components)
        rate_coefficients.append(coefficients[maps])
        log_rate_maps.append(np.log(interior_rates[:, maps]))
        first += mixture.n_components
    summed_ground_process_intensity = jnp.asarray(
        interior_rates.sum(axis=1)
        if interior_rates.size
        else np.zeros(len(interior_place_bin_centers))
    )

    return {
        "algorithm": "clusterless_mark_glm",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "emission_design_info": emission_design_info,
        "mark_mixtures": mark_mixtures,
        "rate_coefficients": rate_coefficients,
        "log_rate_maps": log_rate_maps,
        "summed_ground_process_intensity": np.asarray(summed_ground_process_intensity),
        "mean_rates": np.asarray(jnp.asarray(mean_rates)),
        "mark_dimensions": mark_dimensions,
        "spike_block_size": spike_block_size,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


# Per fitted electrode: mixture, marks, left/right support positions and shares.
_SpikeEvent: TypeAlias = tuple[
    MarkMixture, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _soft_cell_counts(
    spike_events: list[_SpikeEvent],
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Sum each spike's shares, split by its mark responsibilities, into cells.

    Parameters
    ----------
    spike_events : list of tuple
        Per fitted electrode: its ``MarkMixture``, the marks of its
        positive-weight spikes, shape (n_spikes, n_features), the support
        positions of each spike's left and right samples, shape (n_spikes,),
        and the weight shares of those samples, shape (n_spikes,).

    Returns
    -------
    counts : callable
        Takes each support sample's position-cell index, shape (n_samples,),
        and the number of cells, and returns the soft counts of every
        component, electrode by electrode, shape (n_components, n_cells).
        Only one electrode's responsibilities are held at a time.
    """

    def counts(cell: np.ndarray, n_cells: int) -> np.ndarray:
        rows = []
        for mixture, marks, left, right, left_share, right_share in spike_events:
            responsibilities = mixture.responsibilities(marks)
            n_components = mixture.n_components
            electrode_counts = np.zeros(n_cells * n_components)
            for sample, share in ((left, left_share), (right, right_share)):
                inside = sample >= 0
                index = (
                    cell[sample[inside]][:, None] * n_components
                    + np.arange(n_components)[None]
                )
                electrode_counts += np.bincount(
                    index.ravel(),
                    weights=(share[inside, None] * responsibilities[inside]).ravel(),
                    minlength=n_cells * n_components,
                )
            rows.append(electrode_counts.reshape(n_cells, n_components).T)
        return np.concatenate(rows) if rows else np.zeros((0, n_cells))

    return counts


def _component_padding(mark_mixtures: list[MarkMixture | None]) -> int:
    """Components every electrode is padded to, a power of two.

    At least the largest mixture, so electrodes and refits share few compiled
    kernels.
    """
    largest = max((m.n_components for m in mark_mixtures if m is not None), default=1)
    return 1 << (largest - 1).bit_length()


class _PaddedElectrode(NamedTuple):
    """One fitted electrode's mixture and rate terms, padded for shared kernels."""

    means: jax.Array
    precisions_cholesky: jax.Array
    is_component: jax.Array
    coefficients: jax.Array
    log_rates: np.ndarray | None
    n_components: int


def _padded_mixture(
    mixture: MarkMixture, n_components: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Means, precision factors and a validity mask padded to ``n_components``.

    Padding lets every electrode share one compiled kernel; padded components
    are masked out of the intensity.

    Parameters
    ----------
    mixture : MarkMixture
        Fitted mark mixture of one electrode.
    n_components : int
        Padded number of components, at least ``mixture.n_components``.

    Returns
    -------
    means : jax.Array, shape (n_components, n_features)
    precisions_cholesky : jax.Array, shape (n_components, n_features, n_features)
    is_component : jax.Array of bool, shape (n_components,)
        False for padding.
    """
    n_features = mixture.means.shape[1]
    n_padding = n_components - mixture.n_components
    means = np.concatenate([mixture.means, np.zeros((n_padding, n_features))])
    precisions_cholesky = np.concatenate(
        [mixture.precisions_cholesky, np.tile(np.eye(n_features), (n_padding, 1, 1))]
    )
    is_component = np.arange(n_components) < mixture.n_components
    return (
        jnp.asarray(means),
        jnp.asarray(precisions_cholesky),
        jnp.asarray(is_component),
    )


def _masked_log_density(
    features: jax.Array,
    means: jax.Array,
    precisions_cholesky: jax.Array,
    is_component: jax.Array,
) -> jax.Array:
    """``log N(m; mu_k, Sigma_k)``, shape (n_spikes, n_components).

    ``-inf`` for padded components, and for a mark that overflows float32 (an
    infinite feature gives NaN in the precision product): its density is zero
    under every component, as for a finite mark whose squared distance
    overflows.
    """
    log_density = estimate_log_gaussian_prob(
        features, means, precisions_cholesky, "full"
    )
    return cast(
        jax.Array,
        jnp.where(is_component[None] & ~jnp.isnan(log_density), log_density, -jnp.inf),
    )


@partial(
    jax.jit,
    static_argnames=("block_size", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_electrode_intensities(
    log_likelihood: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    means: jax.Array,
    precisions_cholesky: jax.Array,
    is_component: jax.Array,
    scaled_rates: jax.Array,
    log_rate_scale: jax.Array,
    log_background: jax.Array,
    *,
    block_size: int,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add one electrode's non-local log intensities to their rows.

    The intensity ``sum_k N_k(m) lambda_k(x)`` is a matrix product: with
    ``a(m) = max_k log N_k(m)`` and ``b(x) = max_k log lambda_k(x)``, it is
    ``exp(a + b) * (exp(log N - a) @ exp(log lambda - b))``. Each factor is at
    most 1; a component whose density factor underflows contributes less than
    ``exp(-87)`` of the largest density times the largest rate ratio, which is
    bounded by the rate floor.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    features : jax.Array, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jax.Array, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    means, precisions_cholesky, is_component : jax.Array
        The padded mark mixture (see ``_padded_mixture``).
    scaled_rates : jax.Array, shape (n_components, n_bins)
        ``exp(log lambda_k(x) - b(x))``; zero for padded components.
    log_rate_scale : jax.Array, shape (n_bins,)
        ``b(x)``.
    log_background : jax.Array, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    block_size : int
        Decoding spikes per block.
    indices_are_sorted : bool
        Whether ``row_ids`` are sorted in nondecreasing order.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_bins)
    """

    def add_block(number: int, log_likelihood: jax.Array) -> jax.Array:
        first = number * block_size
        block = jax.lax.dynamic_slice_in_dim(features, first, block_size)
        rows = jax.lax.dynamic_slice_in_dim(row_ids, first, block_size)
        background = jax.lax.dynamic_slice_in_dim(log_background, first, block_size)
        log_density = _masked_log_density(
            block, means, precisions_cholesky, is_component
        )
        # A mark beyond float32's range has -inf density under every component;
        # a zero scale then gives a -inf intensity, which leaves the background.
        log_scale = jnp.max(log_density, axis=1)
        log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
        mixed = jnp.matmul(
            jnp.exp(log_density - log_scale[:, None]),
            scaled_rates,
            precision=jax.lax.Precision.HIGHEST,
        )
        log_intensity = jnp.log(mixed) + log_scale[:, None] + log_rate_scale[None]
        return deterministic_row_add(
            log_likelihood,
            jnp.logaddexp(log_intensity, background[:, None]),
            rows,
            indices_are_sorted=indices_are_sorted,
        )

    return cast(
        jax.Array,
        jax.lax.fori_loop(
            0, features.shape[0] // block_size, add_block, log_likelihood
        ),
    )


@partial(jax.jit, static_argnames=("indices_are_sorted",), donate_argnums=0)
def _add_electrode_local_terms(
    log_likelihood: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    means: jax.Array,
    precisions_cholesky: jax.Array,
    is_component: jax.Array,
    spike_log_rates: jax.Array,
    log_background: jax.Array,
    *,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add one electrode's local spike terms to their rows.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows,)
        Running sum over electrodes; donated.
    features : jax.Array, shape (n_padded_spikes, n_features)
        Waveform features of each decoding spike.
    row_ids : jax.Array, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    means, precisions_cholesky, is_component : jax.Array
        The padded mark mixture (see ``_padded_mixture``).
    spike_log_rates : jax.Array, shape (n_padded_spikes, n_components)
        Log rate of each component at each spike's position, in Hz.
    log_background : jax.Array, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    indices_are_sorted : bool
        Whether ``row_ids`` are sorted in nondecreasing order.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows,)
    """
    log_density = _masked_log_density(
        features, means, precisions_cholesky, is_component
    )
    terms = jnp.logaddexp(
        jax.scipy.special.logsumexp(log_density + spike_log_rates, axis=1),
        log_background,
    )
    return log_likelihood + deterministic_row_sum(
        terms,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


def _device_marks(marks: np.ndarray, n_rows: int) -> jax.Array:
    """Return marks padded to ``n_rows`` on the device, in JAX's default dtype.

    A mark beyond float32's range overflows to inf in the cast; the
    likelihood treats it as background-only (see ``_add_electrode_intensities``),
    so the cast need not warn.
    """
    with np.errstate(over="ignore"):
        return jnp.asarray(pad_rows(marks, n_rows))


@requires_time_edges
def predict_clusterless_mark_glm_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    environment: Environment,
    emission_design_info: TrackGraphBasis | TensorSplineBasis,
    mark_mixtures: list[MarkMixture | None],
    rate_coefficients: list[np.ndarray | None],
    log_rate_maps: list[np.ndarray | None],
    summed_ground_process_intensity: np.ndarray,
    mean_rates: np.ndarray,
    is_local: bool = False,
    spike_block_size: int = 1000,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mark_dimensions: list[int],
    mark_reference: list[MarkReference | None],
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    environment_grid: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the (non-local or local) log likelihood of the mark-mixture GLM.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Tracking timestamps in seconds; read only for the local likelihood.
        May be None when ``is_local`` is False.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Animal position; read only for the local likelihood.
        May be None when ``is_local`` is False.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Decoding spike times per electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Decoding spike marks per electrode.
    environment : Environment
        The environment the model was fit on.
    emission_design_info : TrackGraphBasis or TensorSplineBasis
        The rate maps' spline basis, from the fit.
    mark_mixtures : list of MarkMixture or None
        Per electrode, from the fit; None for a zero-rate electrode.
    rate_coefficients : list of np.ndarray or None
        Per electrode, shape (n_components, n_coefficients), from the fit.
    log_rate_maps : list of np.ndarray or None
        Per electrode, shape (n_interior_bins, n_components), from the fit.
    summed_ground_process_intensity : np.ndarray, shape (n_interior_bins,)
        Summed component rates in Hz, from the fit.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Mean rate per electrode in Hz, from the fit (validated, not used).
    is_local : bool, optional
        If True, evaluate at the animal's position: the expected count at each
        bin's center and each spike's term at its own time. By default False
        (every interior bin).
    spike_block_size : int, optional
        Decode spikes scored per block, by default 1000.
    disable_progress_bar : bool, optional
        If True, hide the per-electrode progress bar. By default False.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` stay the FULL decoding edges.
    time_edges : np.ndarray, shape (n_bins + 1,), keyword-only
        Decoding bin edges.
    algorithm : str, optional
        The algorithm the fit recorded. Accepted so the encoding dictionary
        can be passed as keywords; unused.
    mark_dimensions : list of int, keyword-only
        Mark dimension per electrode, from the fit.
    mark_reference : list of MarkReference or None, keyword-only
        Reference mark density per electrode for the background intensity.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the fit; anything other than ``"Hz"`` raises.
    encoding_exposure_seconds : float, optional, keyword-only
        Recorded by the fit; unused.
    environment_grid : dict | None, optional
        The fit's record of the environment's grid; prediction raises if the
        environment has since been refit to another grid.
    _spike_time_order : SpikeTimeOrder, optional
        Internal ordering preparation shared across a detector's states and
        chunks; direct callers omit it.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_interior_bins) or (n_rows, 1)
        Non-local or local (``is_local``) log likelihood. ``n_rows`` is
        ``n_bins`` unless ``row_slice`` is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    validate_positive_int("spike_block_size", spike_block_size)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        mark_mixtures=mark_mixtures,
        mean_rates=mean_rates,
        mark_dimensions=mark_dimensions,
        mark_reference=mark_reference,
    )
    for electrode, (electrode_times, electrode_marks, expected) in enumerate(
        zip(spike_times, spike_waveform_features, mark_dimensions, strict=True)
    ):
        validate_spike_feature_pair(electrode_times, electrode_marks, electrode)
        if np.shape(electrode_marks)[1] != expected:
            msg = f"waveform feature dimension changed for electrode {electrode}"
            raise ValidationError(
                msg,
                expected=f"{expected} features per spike",
                got=f"{np.shape(electrode_marks)[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    n_rows = rows.stop - rows.start
    durations = jnp.asarray(rows.durations())
    if is_local:
        # Bound once here and read by add_electrode below, on this path only.
        local_position_time, local_position = require_position(
            position_time, position, "local clusterless mark GLM likelihood"
        )
        local_position_time = np.asarray(local_position_time)
        local_position = np.asarray(local_position)
        local_position = (
            local_position if local_position.ndim > 1 else local_position[:, None]
        )
        row_position = snap_off_track_positions(
            get_position_at_time(
                local_position_time,
                local_position,
                decode_bin_centers(rows.time_edges, rows.start, rows.stop),
                environment,
            ),
            environment,
        )
        row_design = jnp.asarray(
            make_spline_predict_matrix(emission_design_info, row_position)
        )
        expected_rate = jnp.zeros((n_rows,))
        log_likelihood = jnp.zeros((n_rows,))
    else:
        log_likelihood = -durations[:, None] * (
            jnp.asarray(summed_ground_process_intensity)[None]
            + background_rate(len(spike_times))
        )

    n_components = _component_padding(mark_mixtures)
    # Each fitted electrode's mixture and rate coefficients, padded to a fixed
    # component count so that electrodes share one compiled rate kernel;
    # padded components are masked out. None marks a zero-rate electrode.
    electrode_models: list[_PaddedElectrode | None] = []
    for mixture, coefficients, log_rates in zip(
        mark_mixtures, rate_coefficients, log_rate_maps, strict=True
    ):
        if mixture is None:
            electrode_models.append(None)
            continue
        means, precisions_cholesky, is_component = _padded_mixture(
            mixture, n_components
        )
        # A fitted mixture has its rate coefficients.
        padded_coefficients = jnp.asarray(
            pad_rows(cast(np.ndarray, coefficients), n_components)
        )
        if is_local:
            expected_rate = expected_rate + glm_rates(
                row_design, padded_coefficients
            ) @ is_component.astype(row_design.dtype)
        electrode_models.append(
            _PaddedElectrode(
                means,
                precisions_cholesky,
                is_component,
                padded_coefficients,
                log_rates,
                mixture.n_components,
            )
        )

    def add_electrode(
        log_likelihood: jax.Array,
        spikes: ElectrodeSpikes,
        model: _PaddedElectrode | None,
    ) -> jax.Array:
        if model is None:
            # A zero-rate electrode: each observed spike is explained by the
            # background alone, uniformly across position bins.
            return add_background_only(log_likelihood, spikes)
        if is_local:
            n_padded = padded_spike_count(spikes.n_spikes, spike_block_size)
            spike_position = snap_off_track_positions(
                get_position_at_time(
                    local_position_time,
                    local_position,
                    select_spike_rows(spikes.times, spikes.selection),
                    environment,
                ),
                environment,
            )
            spike_log_rates = jnp.log(
                glm_rates(
                    jnp.asarray(
                        pad_rows(
                            make_spline_predict_matrix(
                                emission_design_info, spike_position
                            ),
                            n_padded,
                        )
                    ),
                    model.coefficients,
                )
            )
            return cast(
                jax.Array,
                _add_electrode_local_terms(
                    log_likelihood,
                    _device_marks(spikes.marks, n_padded),
                    jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
                    model.means,
                    model.precisions_cholesky,
                    model.is_component,
                    spike_log_rates,
                    jnp.asarray(pad_rows(spikes.log_background, n_padded)),
                    indices_are_sorted=spikes.selection.indices_are_sorted,
                ),
            )
        n_padded = padded_spike_count(spikes.n_spikes, spike_block_size, fine=True)
        log_rates = jnp.asarray(model.log_rates)
        log_rate_scale = jnp.max(log_rates, axis=1)
        scaled_rates = jnp.pad(
            jnp.exp(log_rates - log_rate_scale[:, None]).T,
            ((0, n_components - model.n_components), (0, 0)),
        )
        return cast(
            jax.Array,
            _add_electrode_intensities(
                log_likelihood,
                _device_marks(spikes.marks, n_padded),
                jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
                model.means,
                model.precisions_cholesky,
                model.is_component,
                scaled_rates,
                log_rate_scale,
                jnp.asarray(pad_rows(spikes.log_background, n_padded)),
                block_size=min(spike_block_size, n_padded),
                indices_are_sorted=spikes.selection.indices_are_sorted,
            ),
        )

    log_likelihood = add_electrode_spike_terms(
        log_likelihood,
        spike_times,
        spike_waveform_features,
        mark_reference,
        electrode_models,
        add_electrode,
        rows=rows,
        log_background_floor=LOG_RATE_EPS_HZ,
        desc=f"{'Local' if is_local else 'Position-Bin'} Likelihood (mark GLM)",
        disable_progress_bar=disable_progress_bar,
    )

    duration_evidence = log_bin_duration_evidence(
        spike_times,
        rows.time_edges,
        row_slice,
        rows.spike_time_order,
        _time_grid=rows.time_grid,
    )
    if is_local:
        log_likelihood = log_likelihood - durations * (
            expected_rate + background_rate(len(spike_times))
        )
        return (log_likelihood + duration_evidence)[:, None]
    return log_likelihood + duration_evidence[:, None]
