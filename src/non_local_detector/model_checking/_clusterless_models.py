"""Fitted clusterless encoding models as marked point process intensities.

Every supported model has a joint mark intensity of the form

    lambda(x, m) = Lambda(x) * f(m | x),

where ``Lambda(x)`` is the ground intensity in Hz at position ``x`` and
``f(m | x)`` is a Gaussian mixture over marks:

- clusterless KDE (linear and log space): one component per encoding spike,
  centred on its mark, weighted by its encoding weight times the Gaussian
  position kernel at ``x``;
- clusterless diffusion: the same components, weighted by the graph heat kernel
  from the spike's bin to the bin containing ``x``;
- clusterless GMM: the joint mixture's components conditioned on ``x``.

The goodness-of-fit transforms only need the operations below, evaluated with
the decoder's own definitions. Time integrals for the MDCI transform use the
environment's interior place bins, as the non-local likelihood does.
"""

from typing import Any

import numpy as np
from scipy.special import logsumexp  # type: ignore[import-untyped]

from non_local_detector.likelihoods.common import KDEModel, kde_log_density
from non_local_detector.model_checking.conditional_marks import (
    GaussianMixtureConditionalCDF,
)

SUPPORTED_ALGORITHMS = (
    "clusterless_kde",
    "clusterless_gmm",
    "clusterless_diffusion",
)
_BATCH = 512


def electrode_model(encoding_model: dict, algorithm: str, electrode: int) -> Any:
    """The fitted intensity of one electrode, for a supported algorithm."""
    if algorithm == "clusterless_kde":
        return _KDEElectrode(encoding_model, electrode)
    if algorithm == "clusterless_gmm":
        return _GMMElectrode(encoding_model, electrode)
    if algorithm == "clusterless_diffusion":
        return _DiffusionElectrode(encoding_model, electrode)
    raise ValueError(
        f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got {algorithm!r}"
    )


def _gaussian_log_kernel(points, centers, std, block_elements=2**18):
    """``log N(points_i; centers_j, diag(std**2))``, shape (n_points, n_centers).

    Each coordinate is subtracted before it is squared, so a large common
    offset cannot cancel nearby differences. Rows are processed in blocks of
    about ``block_elements`` entries, updated in place, so the work stays in
    cache and no temporary is larger than a block.
    """
    points = np.asarray(points, float)
    centers = np.asarray(centers, float)
    std = np.broadcast_to(np.asarray(std, float), (points.shape[1],))
    scale = -0.5 / std**2
    result = np.empty((len(points), len(centers)))
    step = max(1, block_elements // max(len(centers), 1))
    work = np.empty((min(step, len(points)), len(centers)))
    for first in range(0, len(points), step):
        out = result[first : first + step]
        block = work[: len(out)]
        for dim in range(points.shape[1]):
            np.subtract.outer(
                points[first : first + step, dim], centers[:, dim], out=block
            )
            np.multiply(block, block, out=block)
            if dim == 0:
                np.multiply(block, scale[dim], out=out)
            else:
                block *= scale[dim]
                out += block
    result -= np.sum(np.log(np.sqrt(2 * np.pi) * std))
    return result


def _interior_bins(environment):
    """Interior bin centers and a full-grid to interior index map."""
    interior = (
        np.ones(len(environment.place_bin_centers_), dtype=bool)
        if environment.is_track_interior_ is None
        else np.asarray(environment.is_track_interior_).ravel()
    )
    full_to_local = np.full(interior.size, -1)
    full_to_local[np.flatnonzero(interior)] = np.arange(interior.sum())
    return np.asarray(environment.place_bin_centers_, float)[interior], full_to_local


def _require_bins(bins):
    if np.any(bins < 0):
        raise ValueError("positions must lie in interior bins of the environment")
    return bins


class _MarkKernelMixture:
    """Shared machinery for models whose components are encoding spikes.

    Subclasses provide ``_log_position_weights(bins_or_positions)`` giving the
    log position weight of every encoding spike, and the per-bin normalizer.
    Marks use Gaussian kernels of bandwidth ``waveform_std``.
    """

    marks: np.ndarray
    weights: np.ndarray
    waveform_std: np.ndarray
    n_bins: int

    def _mark_cdf(self):
        return GaussianMixtureConditionalCDF(
            self.weights,
            self.marks,
            np.broadcast_to(self.waveform_std**2, self.marks.shape),
            "diag",
            batch_size=128,
        )

    def _time_integrated_cdf(self, log_mass, marks, feature_order):
        """CDFs of the mark mixture with component weights ``w_j exp(log_mass_j)``."""
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights) + log_mass
        return _log_weighted_mixture_cdf(
            log_weights,
            self.marks,
            np.broadcast_to(self.waveform_std**2, self.marks.shape),
            "diag",
            marks,
            feature_order,
        )

    def _log_mark_kernel(self, marks):
        """``log w_j + log K_m(m_i - m_j)``, shape (n_encoding, n_marks)."""
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights)
        return (
            _gaussian_log_kernel(marks, self.marks, self.waveform_std).T
            + log_weights[:, None]
        )


class _KDEElectrode(_MarkKernelMixture):
    """Clusterless KDE: Gaussian position and mark kernels on encoding spikes."""

    def __init__(self, model: dict, electrode: int) -> None:
        self.environment = model["environment"]
        self.mean_rate = float(model["mean_rates"][electrode])
        self.occupancy_model: KDEModel = model["occupancy_model"]
        self.gpi_model: KDEModel = model["gpi_models"][electrode]
        self.positions = np.asarray(model["encoding_positions"][electrode], float)
        self.marks = np.asarray(
            model["encoding_spike_waveform_features"][electrode], float
        )
        weights = model["encoding_weights"][electrode]
        self.weights = (
            np.ones(len(self.marks)) if weights is None else np.asarray(weights, float)
        )
        if self.mean_rate <= 0 or not np.any(self.weights > 0):
            raise ValueError(f"electrode {electrode} has no fitted encoding spikes")
        n_position_dims = self.positions.shape[1]
        self.position_std = np.broadcast_to(
            np.asarray(model["position_std"], float), (n_position_dims,)
        )
        self.waveform_std = np.broadcast_to(
            np.asarray(model["waveform_std"], float), (self.marks.shape[1],)
        )
        self.bin_centers, self.full_to_local = _interior_bins(self.environment)
        self.n_bins = len(self.bin_centers)
        self.conditional_cdf = GaussianMixtureConditionalCDF.from_kde(
            self.marks,
            self.waveform_std,
            weights=self.weights,
            encoding_positions=self.positions,
            position_std=self.position_std,
        )
        self._log_occupancy_at_bins = np.asarray(
            kde_log_density(self.occupancy_model, self.bin_centers), float
        )

    def ground_intensity(self, positions):
        log_ratio = np.asarray(
            kde_log_density(self.gpi_model, positions), float
        ) - np.asarray(kde_log_density(self.occupancy_model, positions), float)
        return self.mean_rate * np.exp(log_ratio)

    def conditional_mark_cdf(self, positions, marks, feature_order):
        return self.conditional_cdf.transform(
            marks, feature_order=feature_order, given_positions=positions
        )

    def bin_indices(self, positions):
        return _require_bins(
            self.full_to_local[self.environment.get_bin_ind(positions)]
        )

    def _log_position_kernel_at_bins(self):
        return _gaussian_log_kernel(self.bin_centers, self.positions, self.position_std)

    def log_intensity_at_bins(self, marks):
        """``log lambda(b, m_i)``, shape (n_bins, n_marks)."""
        position = self._log_position_kernel_at_bins()  # (n_bins, n_encoding)
        offset = np.log(self.mean_rate) - np.log(self.weights.sum())
        return (
            _log_matmul_exp(position, self._log_mark_kernel(marks))
            + offset
            - self._log_occupancy_at_bins[:, None]
        )

    def time_integrated_mark_cdf(self, time_in_bins, marks, feature_order):
        """CDFs of ``Gamma(m) / integral(Gamma)``, ``Gamma(m) = int lambda dt``."""
        with np.errstate(divide="ignore"):
            log_time = np.log(time_in_bins)
        log_mass = logsumexp(
            (log_time - self._log_occupancy_at_bins)[:, None]
            + self._log_position_kernel_at_bins(),
            axis=0,
        )
        return self._time_integrated_cdf(log_mass, marks, feature_order)


class _DiffusionElectrode(_MarkKernelMixture):
    """Clusterless diffusion: heat-kernel position weights on encoding spikes.

    The model diffuses every source bin with one fixed, normalized spatial
    kernel, so its intensity is exactly this mixture of encoding-spike mark
    kernels weighted by the kernel's columns.
    """

    def __init__(self, model: dict, electrode: int) -> None:
        from non_local_detector.likelihoods.clusterless_diffusion import (
            _diffuse_field,
            _spatial_kernel,
        )

        self.environment = model["environment"]
        self.mean_rate = float(model["mean_rates"][electrode])
        self.weight_total = float(model["weight_total"][electrode])
        if self.mean_rate <= 0 or self.weight_total <= 0:
            raise ValueError(f"electrode {electrode} has no fitted encoding spikes")
        node_order = np.asarray(model["node_order"])
        self.n_bins = node_order.size
        self.full_to_local = np.full(len(self.environment.place_bin_centers_), -1)
        self.full_to_local[node_order] = np.arange(self.n_bins)
        self.marks = np.asarray(model["encoding_marks"][electrode], float)
        self.weights = np.asarray(model["encoding_weights"][electrode], float)
        self.waveform_std = np.broadcast_to(
            np.asarray(model["waveform_std"], float), (self.marks.shape[1],)
        )
        bins = np.asarray(model["encoding_bin_indices"][electrode])
        self.sources, self.source_index = np.unique(bins, return_inverse=True)
        # Encoding spikes grouped by source bin, for per-source reductions.
        self._by_source = np.argsort(self.source_index, kind="stable")
        self._sorted_source_index = self.source_index[self._by_source]
        self._source_starts = np.searchsorted(
            self._sorted_source_index, np.arange(self.sources.size)
        )
        kernel = _spatial_kernel(
            self.environment,
            int(model["resolved_rank"]),
            float(np.asarray(model["position_std"])),
        )
        with np.errstate(divide="ignore"):
            # (n_bins, n_sources)
            self.log_heat = np.log(np.asarray(kernel, float)[:, self.sources])
        bin_sizes = np.asarray(model["bin_sizes"], float)
        occupancy = np.asarray(model["occupancy"], float)
        spike_field = np.bincount(bins, weights=self.weights, minlength=self.n_bins)
        p_gpi = _diffuse_field(spike_field, kernel) / (self.weight_total * bin_sizes)
        self._ground_at_bins = self.mean_rate * p_gpi / occupancy
        # log of mean_rate / (sum w * dV(b) * occupancy(b)), per bin.
        self._log_bin_scale = (
            np.log(self.mean_rate)
            - np.log(self.weight_total)
            - np.log(bin_sizes * occupancy)
        )

    def bin_indices(self, positions):
        return _require_bins(
            self.full_to_local[self.environment.get_bin_ind(positions)]
        )

    def ground_intensity(self, positions):
        return self._ground_at_bins[self.bin_indices(positions)]

    def conditional_mark_cdf(self, positions, marks, feature_order):
        bins = self.bin_indices(positions)

        def log_weights(first, last):
            return self.log_heat[bins[first:last]][:, self.source_index]

        return self._mark_cdf().transform(
            marks, feature_order=feature_order, component_log_weights=log_weights
        )

    def log_intensity_at_bins(self, marks):
        """``log lambda(b, m_i)``, shape (n_bins, n_marks)."""
        # log sum_{j in source} w_j K_m(m_i - m_j), shape (n_sources, n_marks),
        # shifted per source so that no source's sum underflows.
        mark = self._log_mark_kernel(marks)[self._by_source]
        shift = np.maximum.reduceat(mark, self._source_starts, axis=0)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        by_source = np.add.reduceat(
            np.exp(mark - shift[self._sorted_source_index]),
            self._source_starts,
            axis=0,
        )
        with np.errstate(divide="ignore"):
            log_by_source = np.log(by_source) + shift
        return (
            _log_matmul_exp(self.log_heat, log_by_source) + self._log_bin_scale[:, None]
        )

    def time_integrated_mark_cdf(self, time_in_bins, marks, feature_order):
        with np.errstate(divide="ignore"):
            log_time = np.log(time_in_bins)
        log_mass = logsumexp(
            (log_time + self._log_bin_scale)[:, None] + self.log_heat, axis=0
        )[self.source_index]
        return self._time_integrated_cdf(log_mass, marks, feature_order)


def _full_covariances(model) -> np.ndarray:
    """A fitted GMM's covariances as ``(n_components, n_dims, n_dims)``."""
    covariances = np.asarray(model.covariances_, dtype=np.float64)
    n_components, n_dims = np.shape(model.means_)
    if model.covariance_type == "full":
        return covariances
    if model.covariance_type == "tied":
        return np.repeat(covariances[None], n_components, axis=0)
    if model.covariance_type == "diag":
        return np.stack([np.diag(c) for c in covariances])
    return np.stack([np.eye(n_dims) * c for c in covariances])


class _GMMElectrode:
    """Clusterless GMM: the ground-process mixture and the joint conditional.

    Fits store the joint mixture's position marginal as ``p_gpi``, so
    ``Lambda(x) = mu p_gpi(x) / p_occ(x)`` integrates the event intensity.
    The transforms use ``lambda(x, m) = mu p_joint(x, m) / p_occ(x)`` and
    the joint mixture's conditional mark density.
    """

    def __init__(self, model: dict, electrode: int) -> None:
        self.joint = model["joint_models"][electrode]
        if self.joint is None:
            raise ValueError(f"electrode {electrode} has no fitted encoding spikes")
        self.environment = model["environment"]
        self.mean_rate = float(model["mean_rates"][electrode])
        self.gpi = model["gpi_models"][electrode]
        self.occupancy = model["occupancy_model"]
        self.bin_centers, self.full_to_local = _interior_bins(self.environment)
        self.n_bins = len(self.bin_centers)
        p = self.n_position_dims = self.bin_centers.shape[1]
        self.conditional_cdf = GaussianMixtureConditionalCDF.from_gmm(
            self.joint, n_position_dims=p
        )
        weights = np.asarray(self.joint.weights_, float)
        keep = weights > 0
        self._weights = weights[keep]
        self._means = np.asarray(self.joint.means_, float)[keep]
        self._covariances = _full_covariances(self.joint)[keep]
        chol = np.linalg.cholesky(self._covariances[:, :p, :p])
        residual = self.bin_centers[:, None, :] - self._means[None, :, :p]
        whitened = np.linalg.solve(chol[None], residual[..., None])[..., 0]
        # log(pi_k N(b; position block of component k)), shape (n_bins, K).
        self._log_position_at_bins = (
            np.log(self._weights)[None]
            - 0.5 * np.sum(whitened**2, axis=2)
            - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)[None]
            - 0.5 * p * np.log(2 * np.pi)
        )
        self._log_rate_over_occupancy_at_bins = np.log(self.mean_rate) - self._score(
            self.occupancy, self.bin_centers
        )

    @staticmethod
    def _score(model, points):
        return np.asarray(model.score_samples(np.asarray(points)), float)

    def _log_ground(self, positions):
        return (
            np.log(self.mean_rate)
            + self._score(self.gpi, positions)
            - self._score(self.occupancy, positions)
        )

    def ground_intensity(self, positions):
        return np.exp(self._log_ground(positions))

    def conditional_mark_cdf(self, positions, marks, feature_order):
        return self.conditional_cdf.transform(
            marks, feature_order=feature_order, given_positions=positions
        )

    def bin_indices(self, positions):
        return _require_bins(
            self.full_to_local[self.environment.get_bin_ind(positions)]
        )

    def log_intensity_at_bins(self, marks):
        """``log mu + log p_joint(b, m_i) - log p_occ(b)``, (n_bins, n)."""
        result = np.empty((self.n_bins, len(marks)))
        for first in range(0, len(marks), _BATCH):
            block = marks[first : first + _BATCH]
            points = np.concatenate(
                (
                    np.repeat(self.bin_centers, len(block), axis=0),
                    np.tile(block, (self.n_bins, 1)),
                ),
                axis=1,
            )
            result[:, first : first + len(block)] = self._score(
                self.joint, points
            ).reshape(self.n_bins, len(block))
        return result + self._log_rate_over_occupancy_at_bins[:, None]

    def time_integrated_mark_cdf(self, time_in_bins, marks, feature_order):
        """CDFs of the mixture over (bin, component) that ``Gamma(m)`` forms.

        ``Gamma(m) = sum_b time(b) Lambda(b) sum_k r_k(b) N(m; mu_k|b, S_k|b)``,
        with ``r_k(b)`` the joint components' responsibilities given position.
        """
        p = self.n_position_dims
        covariances = self._covariances
        gain = np.linalg.solve(covariances[:, :p, :p], covariances[:, :p, p:])
        gain = gain.transpose(0, 2, 1)
        mark_cov = covariances[:, p:, p:] - gain @ covariances[:, :p, p:]
        with np.errstate(divide="ignore"):
            log_weight = (np.log(time_in_bins) + self._log_rate_over_occupancy_at_bins)[
                :, None
            ] + self._log_position_at_bins
        bins, components = np.nonzero(~np.isneginf(log_weight))
        residual = self.bin_centers[bins] - self._means[components, :p]
        conditional_means = self._means[components, p:] + np.einsum(
            "nij,nj->ni", gain[components], residual
        )
        return _log_weighted_mixture_cdf(
            log_weight[bins, components],
            conditional_means,
            mark_cov[components],
            "full",
            marks,
            feature_order,
        )


def _log_weighted_mixture_cdf(
    log_weights, means, covariances, covariance_type, marks, feature_order
):
    """Sequential CDFs of a mixture whose component weights are given as logs.

    The weights stay in log space while the mixture is conditioned on each
    mark. A component of tiny weight can dominate once the first marks are
    observed, so only components of zero weight are dropped.
    """
    keep = ~np.isneginf(log_weights)
    log_weights = log_weights[keep]
    n_components = log_weights.size
    mixture = GaussianMixtureConditionalCDF(
        np.ones(n_components),
        means[keep],
        covariances[keep],
        covariance_type,
        batch_size=max(1, min(128, 2**20 // max(n_components, 1))),
    )
    return mixture.transform(
        marks,
        feature_order=feature_order,
        component_log_weights=lambda first, last: np.broadcast_to(
            log_weights, (last - first, n_components)
        ),
    )


def _log_matmul_exp(log_a, log_b):
    """``log(exp(log_a) @ exp(log_b))`` in float64.

    Rows of ``log_a`` and columns of ``log_b`` are shifted by their maxima
    for a matrix product. An entry whose terms are all small relative to
    both shifts underflows there, so entries whose shifted product is below
    1e-250 are recomputed with log-sum-exp. Above it, terms lost to
    underflow are below 1e-58 of the entry.
    """
    row_shift = log_a.max(axis=1)
    row_shift = np.where(np.isfinite(row_shift), row_shift, 0.0)
    column_shift = log_b.max(axis=0)
    column_shift = np.where(np.isfinite(column_shift), column_shift, 0.0)
    product = np.exp(log_a - row_shift[:, None]) @ np.exp(log_b - column_shift[None])
    with np.errstate(divide="ignore"):
        result = np.log(product) + row_shift[:, None] + column_shift[None]
    rows, columns = np.nonzero(product < 1e-250)
    block = max(1, 2**20 // max(log_a.shape[1], 1))
    for first in range(0, rows.size, block):
        row, column = rows[first : first + block], columns[first : first + block]
        result[row, column] = logsumexp(log_a[row] + log_b[:, column].T, axis=1)
    return result
