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
- Off a track graph the rate maps use a patsy spline basis, which cannot be
  pickled, as for ``sorted_spikes_glm``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp  # type: ignore[import-untyped]
from sklearn.mixture import BayesianGaussianMixture  # type: ignore[import-untyped]
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.encoding_time import (
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_gmm import DEFAULT_GMM_REG_COVAR
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    _pad_rows,
    _padded_spike_count,
    _SpikeTimeOrder,
    decode_bin_centers,
    deterministic_row_add,
    deterministic_row_sum,
    environment_grid_fingerprint,
    get_position_at_time,
    log_bin_duration_evidence,
    resolve_row_slice,
    select_decode_marks,
    select_spike_rows,
    select_spikes_in_rows,
    snap_off_track_positions,
    spike_row_ids,
    sum_spikes_into_rows,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_spike_feature_pair,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.gmm import _estimate_log_gaussian_prob
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
    background_rate,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    GLM_LOG_RATE_PRIOR_STD,
    DesignInfo,
    SelectedColumnsDesign,
    TrackGraphBasis,
    _fit_rate_maps,
    _glm_rates,
    make_spline_predict_matrix,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)

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
        """Number of mixture components."""
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
        return (
            log_det[None]
            - 0.5 * marks.shape[1] * np.log(2.0 * np.pi)
            - 0.5 * np.sum(standardized**2, axis=2)
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
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


class _WeightedBayesianGaussianMixture(BayesianGaussianMixture):
    """Variational Bayes Gaussian mixture whose samples count by a weight.

    Scales each sample's responsibilities by its weight in the initialization
    and every M-step, so the sufficient statistics are weighted sums. The
    mean and covariance priors are scikit-learn's defaults (the data's mean and
    covariance), and convergence uses its unweighted lower bound. Overrides
    private scikit-learn methods; ``fit_weighted`` raises unless both ran, so
    an incompatible scikit-learn cannot silently ignore the weights.
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
        self._n_initialize = self._n_m_step = 0
        self.fit(X)
        if self._n_initialize == 0 or self._n_m_step == 0:
            raise RuntimeError(
                "scikit-learn's BayesianGaussianMixture internals changed: the "
                "sample weights were not applied. Please report this issue."
            )
        return self

    def _initialize(self, X, resp, *args, **kwargs):
        """Weighted initial sufficient statistics."""
        self._n_initialize += 1
        return super()._initialize(
            X, resp * self._sample_weight[:, None], *args, **kwargs
        )

    def _m_step(self, X, log_resp, *args, **kwargs):
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


def fit_clusterless_mark_glm_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    spike_waveform_features: list[np.ndarray],
    environment: Environment,
    weights: np.ndarray | None = None,
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
    encoding_time_range: np.ndarray | None = None,
    valid_position_intervals: np.ndarray | None = None,
    waveform_std: object = None,
    position_std: object = None,
    block_size: object = None,
    _encoding_support: object = None,
    **kwargs: object,
) -> dict:
    """Fit the clusterless mark-mixture GLM encoding model.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Tracking timestamps in seconds.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Animal position at each tracking sample.
    spike_times : list of np.ndarray, each shape (n_spikes,)
        Encoding spike times per electrode.
    spike_waveform_features : list of np.ndarray, each shape (n_spikes, n_features)
        Encoding spike marks per electrode.
    environment : Environment
        The fitted spatial environment.
    weights : np.ndarray, shape (n_time_position,), optional
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
    waveform_std, position_std, block_size : optional
        Ignored. A detector's default ``clusterless_algorithm_params`` hold
        these KDE parameters and are passed to every clusterless algorithm.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support.
    **kwargs
        Any other name raises a ``ValidationError`` that lists the valid
        parameters.

    Returns
    -------
    encoding_model : dict
        Keys ``environment`` and ``environment_grid``, and:

        emission_design_info : TrackGraphBasis, SelectedColumnsDesign or DesignInfo
            The rate maps' spline basis.
        mark_mixtures : list of MarkMixture or None
            Per electrode; None for a zero-rate electrode (no positive-weight
            encoding spikes).
        rate_coefficients : list of np.ndarray or None
            Per electrode, shape (n_components, n_coefficients).
        log_rate_maps : list of np.ndarray or None
            Per electrode, shape (n_interior_bins, n_components): log rate in
            Hz of each component at the interior bin centers.
        summed_ground_process_intensity : jnp.ndarray, shape (n_interior_bins,)
            Sum of every component's rate, in Hz.
        mean_rates : jnp.ndarray, shape (n_electrodes,)
            Weighted mean firing rate per electrode, in Hz.
        mark_dimensions : list of int
        spike_block_size, disable_progress_bar
        rate_units : str
            ``"Hz"``.
        encoding_exposure_seconds : float
            Weighted encoding occupancy time in seconds.
    """
    if kwargs:
        valid = [
            name
            for name, parameter in inspect.signature(
                fit_clusterless_mark_glm_encoding_model
            ).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not name.startswith("_")
        ]
        raise ValidationError(
            "unknown clusterless_mark_glm fit parameter(s): "
            f"{', '.join(sorted(kwargs))}",
            expected=f"parameters among: {', '.join(valid)}",
            got=", ".join(f"{name}={value!r}" for name, value in kwargs.items()),
            hint="Check the spelling. A detector passes its "
            "clusterless_algorithm_params to this fit.",
        )
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
    position = position if position.ndim > 1 else position[:, None]
    weight_sum = float(exposure_weights.sum())
    warn_if_no_encoding_exposure(
        exposure_weights, "Clusterless mark GLM", clusterless=True
    )

    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        is_track_interior = np.ones(len(environment.place_bin_centers_), dtype=bool)
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )
    encoding_positions = position[support.indices]
    if (
        environment.track_graph is not None
        and encoding_positions.shape[1] > 1
        and len(encoding_positions)
    ):
        encoding_positions = environment._linear_position(encoding_positions)[:, None]

    # Position of each tracking sample within the support (-1 outside it).
    support_position = np.full(len(position), -1)
    support_position[support.indices] = np.arange(len(support.indices))
    mark_mixtures: list[MarkMixture | None] = []
    mean_rates, mark_dimensions, spike_events = [], [], []
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
                UserWarning,
                stacklevel=2,
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

    emission_design_info, coefficients, interior_rates = _fit_rate_maps(
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
    coefficients = np.asarray(coefficients)
    interior_rates = np.asarray(interior_rates)
    rate_coefficients: list[np.ndarray | None] = []
    log_rate_maps: list[np.ndarray | None] = []
    first = 0
    for mixture in mark_mixtures:
        if mixture is None:
            rate_coefficients.append(None)
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
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "emission_design_info": emission_design_info,
        "mark_mixtures": mark_mixtures,
        "rate_coefficients": rate_coefficients,
        "log_rate_maps": log_rate_maps,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "mean_rates": jnp.asarray(mean_rates),
        "mark_dimensions": mark_dimensions,
        "spike_block_size": spike_block_size,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


def _soft_cell_counts(
    spike_events: list[tuple],
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


def _padded_mixture(
    mixture: MarkMixture, n_components: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Means, precision factors and a validity mask padded to ``n_components``.

    Padding lets every electrode share one compiled kernel; padded components
    are masked out of the intensity.

    Parameters
    ----------
    mixture : MarkMixture
    n_components : int
        Padded number of components, at least ``mixture.n_components``.

    Returns
    -------
    means : jnp.ndarray, shape (n_components, n_features)
    precisions_cholesky : jnp.ndarray, shape (n_components, n_features, n_features)
    is_component : jnp.ndarray of bool, shape (n_components,)
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
    features: jnp.ndarray,
    means: jnp.ndarray,
    precisions_cholesky: jnp.ndarray,
    is_component: jnp.ndarray,
) -> jnp.ndarray:
    """``log N(m; mu_k, Sigma_k)``, shape (n_spikes, n_components).

    ``-inf`` for padded components.
    """
    log_density = _estimate_log_gaussian_prob(
        features, means, precisions_cholesky, "full"
    )
    return jnp.where(is_component[None], log_density, -jnp.inf)


@partial(
    jax.jit,
    static_argnames=("block_size", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_electrode_intensities(
    log_likelihood: jnp.ndarray,
    features: jnp.ndarray,
    row_ids: jnp.ndarray,
    means: jnp.ndarray,
    precisions_cholesky: jnp.ndarray,
    is_component: jnp.ndarray,
    scaled_rates: jnp.ndarray,
    log_rate_scale: jnp.ndarray,
    log_background: jnp.ndarray,
    *,
    block_size: int,
    indices_are_sorted: bool,
) -> jnp.ndarray:
    """Add one electrode's non-local log intensities to their rows.

    The intensity ``sum_k N_k(m) lambda_k(x)`` is a matrix product: with
    ``a(m) = max_k log N_k(m)`` and ``b(x) = max_k log lambda_k(x)``, it is
    ``exp(a + b) * (exp(log N - a) @ exp(log lambda - b))``. Each factor is at
    most 1; a component whose density factor underflows contributes less than
    ``exp(-87)`` of the largest density times the largest rate ratio, which is
    bounded by the rate floor.

    Parameters
    ----------
    log_likelihood : jnp.ndarray, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    means, precisions_cholesky, is_component : jnp.ndarray
        The padded mark mixture (see ``_padded_mixture``).
    scaled_rates : jnp.ndarray, shape (n_components, n_bins)
        ``exp(log lambda_k(x) - b(x))``; zero for padded components.
    log_rate_scale : jnp.ndarray, shape (n_bins,)
        ``b(x)``.
    log_background : jnp.ndarray, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, n_bins)
    """

    def add_block(number: int, log_likelihood: jnp.ndarray) -> jnp.ndarray:
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

    return jax.lax.fori_loop(
        0, features.shape[0] // block_size, add_block, log_likelihood
    )


@partial(jax.jit, static_argnames=("indices_are_sorted",), donate_argnums=0)
def _add_electrode_local_terms(
    log_likelihood: jnp.ndarray,
    features: jnp.ndarray,
    row_ids: jnp.ndarray,
    means: jnp.ndarray,
    precisions_cholesky: jnp.ndarray,
    is_component: jnp.ndarray,
    spike_log_rates: jnp.ndarray,
    log_background: jnp.ndarray,
    *,
    indices_are_sorted: bool,
) -> jnp.ndarray:
    """Add one electrode's local spike terms to their rows.

    Parameters
    ----------
    log_likelihood : jnp.ndarray, shape (n_rows,)
        Running sum over electrodes; donated.
    features : jnp.ndarray, shape (n_padded_spikes, n_features)
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    means, precisions_cholesky, is_component : jnp.ndarray
        The padded mark mixture (see ``_padded_mixture``).
    spike_log_rates : jnp.ndarray, shape (n_padded_spikes, n_components)
        Log rate of each component at each spike's position, in Hz.
    log_background : jnp.ndarray, shape (n_padded_spikes,)

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows,)
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


@requires_time_edges
def predict_clusterless_mark_glm_log_likelihood(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    spike_waveform_features: list[np.ndarray],
    environment: Environment,
    emission_design_info: DesignInfo | TrackGraphBasis | SelectedColumnsDesign,
    mark_mixtures: list[MarkMixture | None],
    rate_coefficients: list[np.ndarray | None],
    log_rate_maps: list[np.ndarray | None],
    summed_ground_process_intensity: jnp.ndarray,
    mean_rates: jnp.ndarray,
    is_local: bool = False,
    spike_block_size: int = 1000,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    mark_dimensions: list[int],
    mark_reference: list[MarkReference | None],
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    environment_grid: dict | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Predict the (non-local or local) log likelihood of the mark-mixture GLM.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Tracking timestamps in seconds; read only for the local likelihood.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Animal position; read only for the local likelihood.
    spike_times : list of np.ndarray, each shape (n_spikes,)
        Decoding spike times per electrode.
    spike_waveform_features : list of np.ndarray, each shape (n_spikes, n_features)
        Decoding spike marks per electrode.
    environment : Environment
        The environment the model was fit on.
    emission_design_info : DesignInfo, TrackGraphBasis or SelectedColumnsDesign
        The rate maps' spline basis, from the fit.
    mark_mixtures : list of MarkMixture or None
        Per electrode, from the fit; None for a zero-rate electrode.
    rate_coefficients : list of np.ndarray or None
        Per electrode, shape (n_components, n_coefficients), from the fit.
    log_rate_maps : list of np.ndarray or None
        Per electrode, shape (n_interior_bins, n_components), from the fit.
    summed_ground_process_intensity : jnp.ndarray, shape (n_interior_bins,)
        Summed component rates in Hz, from the fit.
    mean_rates : jnp.ndarray, shape (n_electrodes,)
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
    _spike_time_order : _SpikeTimeOrder, optional
        Internal ordering preparation shared across a detector's states and
        chunks; direct callers omit it.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid; direct callers omit it.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, n_interior_bins) or (n_rows, 1)
        Non-local or local (``is_local``) log likelihood. ``n_rows`` is
        ``n_bins`` unless ``row_slice`` is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
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
            raise ValidationError(
                f"waveform feature dimension changed for electrode {electrode}",
                expected=f"{expected} features per spike",
                got=f"{np.shape(electrode_marks)[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_rows = row_stop - row_start
    durations = jnp.asarray(_time_grid.durations(row_start, row_stop))
    if is_local:
        position_time = np.asarray(position_time)
        position = np.asarray(position)
        position = position if position.ndim > 1 else position[:, None]
        row_position = snap_off_track_positions(
            get_position_at_time(
                position_time,
                position,
                decode_bin_centers(time_edges, row_start, row_stop),
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
    for electrode, (
        electrode_times,
        electrode_marks,
        mixture,
        coefficients,
        log_rates,
        reference,
    ) in enumerate(
        tqdm(
            zip(
                spike_times,
                spike_waveform_features,
                mark_mixtures,
                rate_coefficients,
                log_rate_maps,
                mark_reference,
                strict=True,
            ),
            desc=f"{'Local' if is_local else 'Non-Local'} Likelihood (mark GLM)",
            unit="electrode",
            disable=disable_progress_bar,
        )
    ):
        if mixture is not None:
            means, precisions_cholesky, is_component = _padded_mixture(
                mixture, n_components
            )
            # Components padded to a fixed count share one compiled rate
            # kernel; padded components are masked out.
            padded_coefficients = jnp.asarray(_pad_rows(coefficients, n_components))
            if is_local:
                expected_rate = expected_rate + _glm_rates(
                    row_design, padded_coefficients
                ) @ is_component.astype(row_design.dtype)
        selection = select_spikes_in_rows(
            electrode_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        n_spikes = selection.bin_ind.shape[0]
        if n_spikes == 0:
            continue
        marks = select_decode_marks(electrode_marks, selection, electrode)
        log_background = background_log_intensity(reference, marks, LOG_RATE_EPS_HZ)
        if mixture is None:
            # A zero-rate electrode: each observed spike is explained by the
            # background alone, uniformly across position bins.
            contribution = sum_spikes_into_rows(log_background, selection)
            log_likelihood = log_likelihood + (
                contribution if is_local else contribution[:, None]
            )
            continue
        if is_local:
            n_padded = _padded_spike_count(n_spikes, spike_block_size)
            spike_position = snap_off_track_positions(
                get_position_at_time(
                    position_time,
                    position,
                    select_spike_rows(electrode_times, selection),
                    environment,
                ),
                environment,
            )
            spike_log_rates = jnp.log(
                _glm_rates(
                    jnp.asarray(
                        _pad_rows(
                            make_spline_predict_matrix(
                                emission_design_info, spike_position
                            ),
                            n_padded,
                        )
                    ),
                    padded_coefficients,
                )
            )
            log_likelihood = _add_electrode_local_terms(
                log_likelihood,
                jnp.asarray(_pad_rows(marks, n_padded)),
                jnp.asarray(spike_row_ids(selection, n_padded)),
                means,
                precisions_cholesky,
                is_component,
                spike_log_rates,
                jnp.asarray(_pad_rows(log_background, n_padded)),
                indices_are_sorted=selection.indices_are_sorted,
            )
            continue
        n_padded = _padded_spike_count(n_spikes, spike_block_size, fine=True)
        log_rates = jnp.asarray(log_rates)
        log_rate_scale = jnp.max(log_rates, axis=1)
        scaled_rates = jnp.pad(
            jnp.exp(log_rates - log_rate_scale[:, None]).T,
            ((0, n_components - mixture.n_components), (0, 0)),
        )
        log_likelihood = _add_electrode_intensities(
            log_likelihood,
            jnp.asarray(_pad_rows(marks, n_padded)),
            jnp.asarray(spike_row_ids(selection, n_padded)),
            means,
            precisions_cholesky,
            is_component,
            scaled_rates,
            log_rate_scale,
            jnp.asarray(_pad_rows(log_background, n_padded)),
            block_size=min(spike_block_size, n_padded),
            indices_are_sorted=selection.indices_are_sorted,
        )

    duration_evidence = log_bin_duration_evidence(
        spike_times, time_edges, row_slice, _spike_time_order, _time_grid=_time_grid
    )
    if is_local:
        log_likelihood = log_likelihood - durations * (
            expected_rate + background_rate(len(spike_times))
        )
        return (log_likelihood + duration_evidence)[:, None]
    return log_likelihood + duration_evidence[:, None]
