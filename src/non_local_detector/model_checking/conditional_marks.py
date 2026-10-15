"""Analytical conditional CDFs for fixed Gaussian-mixture mark models.

The sequential transform conditions component responsibilities and Gaussian
parameters on observed position and preceding marks. Query marks never fit a
distribution or empirical CDF. Temporary query arrays scale with batch size.
"""

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp, ndtr


class GaussianMixtureConditionalCDF:
    """A training-fixed callable ``(times, observed_marks, feature_order)``.

    Parameters
    ----------
    weights : array_like, shape (n_components,)
        Nonnegative fitted component weights with positive total mass.
    means : array_like, shape (n_components, n_position_dims + n_features)
        Fitted means. Position coordinates precede mark coordinates.
    covariances : array_like
        Full ``(K,D,D)``, tied ``(D,D)``, diagonal ``(K,D)``, or spherical
        ``(K,)`` fitted covariances, according to ``covariance_type``.
    covariance_type : {"full", "tied", "diag", "spherical"}, default="full"
        Covariance parameterization. Diagonal/spherical evaluation avoids full
        covariance matrices, which is useful for KDE components.
    n_position_dims : int, default=0
        Number of leading position coordinates to condition on.
    position_at_time : callable or None
        Maps a query time vector to its ``(n_queries,n_position_dims)`` observed
        positions. Required by the callable interface when position is modeled;
        ``transform`` can instead accept positions directly.
    batch_size : int, default=128
        Maximum query rows evaluated together. Workspace is proportional to
        ``batch_size * n_components * n_conditioning_coordinates``.

    Notes
    -----
    Parameters are copied and made read-only. A diagnostic remains fixed when
    the original encoding model is subsequently refitted. Positions and all
    preceding marks condition mixture weights, rather than merely rearranging
    marginal CDF output columns. Correct held-out transforms have independent
    uniform coordinates for a correctly specified continuous mark model.
    """

    def __init__(
        self,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
        covariance_type: str = "full",
        *,
        n_position_dims: int = 0,
        position_at_time: Callable[[np.ndarray], ArrayLike] | None = None,
        batch_size: int = 128,
    ) -> None:
        weights = np.array(weights, dtype=float, copy=True)
        means = np.array(means, dtype=float, copy=True)
        covariance = np.array(covariances, dtype=float, copy=True)
        if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
            raise ValueError(
                "means must be a finite nonempty component-by-coordinate matrix"
            )
        if (
            weights.shape != (means.shape[0],)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "weights must be finite nonnegative component weights with positive total mass"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(n_position_dims, bool)
            or not isinstance(n_position_dims, Integral)
            or not 0 <= n_position_dims < means.shape[1]
        ):
            raise ValueError("n_position_dims must leave at least one mark feature")
        n_components, n_dims = means.shape
        if covariance_type == "full":
            expected = (n_components, n_dims, n_dims)
        elif covariance_type == "tied":
            expected = (n_dims, n_dims)
        elif covariance_type == "diag":
            expected = (n_components, n_dims)
        elif covariance_type == "spherical":
            expected = (n_components,)
        else:
            raise ValueError("covariance_type must be full, tied, diag, or spherical")
        if covariance.shape != expected:
            raise ValueError(f"covariances must be finite with shape {expected}")
        keep = weights > 0
        self._kept_components = np.flatnonzero(keep)
        means, weights = means[keep], weights[keep]
        weights = weights / weights.max()
        weights = weights / weights.sum()
        if covariance_type == "tied":
            covariance = np.broadcast_to(
                covariance, (len(weights), n_dims, n_dims)
            ).copy()
        else:
            covariance = covariance[keep]
        if not np.isfinite(means).all() or not np.isfinite(covariance).all():
            raise ValueError(
                "Positive-weight Gaussian component parameters must be finite"
            )
        self.variances: np.ndarray | None = None
        self.covariances: np.ndarray | None = None
        if covariance_type in ("diag", "spherical"):
            if covariance_type == "spherical":
                covariance = np.broadcast_to(
                    covariance[:, None], (len(weights), n_dims)
                ).copy()
            if np.any(covariance <= 0):
                raise ValueError("Gaussian component variances must be positive")
            self.variances = covariance
        else:
            # Fitted covariances are symmetric only to rounding (scikit-learn's
            # differ from their transposes by about 1e-14); symmetrize those.
            transposed = np.swapaxes(covariance, 1, 2)
            scale = np.abs(covariance).max(axis=(1, 2), keepdims=True)
            if not np.all(np.abs(covariance - transposed) <= 1e-10 * scale):
                raise ValueError("Gaussian component covariances must be symmetric")
            covariance = 0.5 * (covariance + transposed)
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Gaussian component covariances must be positive definite"
                ) from exc
            self.covariances = covariance
        self.weights = weights
        self.means = means
        self.n_position_dims = int(n_position_dims)
        self.n_features = n_dims - self.n_position_dims
        self.position_at_time = position_at_time
        self.batch_size = int(batch_size)
        self._recipes: dict[
            tuple[int, ...], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        self._position_precision: np.ndarray | None = None
        self._position_log_normalizer: np.ndarray | None = None
        if self.n_position_dims and self.covariances is not None:
            chol = np.linalg.cholesky(
                self.covariances[:, : self.n_position_dims, : self.n_position_dims]
            )
            self._position_precision = np.swapaxes(np.linalg.inv(chol), 1, 2)
            self._position_log_normalizer = np.log(
                np.diagonal(chol, axis1=1, axis2=2)
            ).sum(axis=1) + 0.5 * self.n_position_dims * np.log(2 * np.pi)
        for fitted in (self.weights, self.means, self.variances, self.covariances):
            if fitted is not None:
                fitted.setflags(write=False)

    @classmethod
    def from_gmm(
        cls,
        model: Any,
        *,
        n_position_dims: int = 0,
        position_at_time: Callable[[np.ndarray], ArrayLike] | None = None,
        batch_size: int = 128,
    ) -> "GaussianMixtureConditionalCDF":
        """Capture a fitted GMM's parameters, supporting every covariance type."""
        model._check_fitted()
        return cls(
            model.weights_,
            model.means_,
            model.covariances_,
            model.covariance_type,
            n_position_dims=n_position_dims,
            position_at_time=position_at_time,
            batch_size=batch_size,
        )

    @classmethod
    def from_kde(
        cls,
        encoding_marks: ArrayLike,
        waveform_std: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        encoding_positions: ArrayLike | None = None,
        position_std: ArrayLike | None = None,
        position_at_time: Callable[[np.ndarray], ArrayLike] | None = None,
        batch_size: int = 128,
    ) -> "GaussianMixtureConditionalCDF":
        """Build the exact Gaussian-mixture CDF of a fixed mark or joint KDE.

        The component means are encoding samples and component weights are
        encoding spike weights. With positions, the conditional weights include
        their position-kernel densities. No query sample fits a component.
        """
        marks = np.asarray(encoding_marks, dtype=float)
        if marks.ndim != 2:
            raise ValueError("encoding_marks must be a two-dimensional matrix")
        std = np.broadcast_to(np.asarray(waveform_std, dtype=float), (marks.shape[1],))
        n_position_dims = 0
        means = marks
        if encoding_positions is not None:
            positions = np.asarray(encoding_positions, dtype=float)
            if (
                positions.ndim != 2
                or positions.shape[0] != marks.shape[0]
                or position_std is None
            ):
                raise ValueError(
                    "encoding_positions must align with marks and have position_std"
                )
            n_position_dims = positions.shape[1]
            std = np.concatenate(
                (
                    np.broadcast_to(
                        np.asarray(position_std, dtype=float), (n_position_dims,)
                    ),
                    std,
                )
            )
            means = np.concatenate((positions, marks), axis=1)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("KDE standard deviations must be finite and positive")
        mixture_weights = np.ones(marks.shape[0]) if weights is None else weights
        variances = np.broadcast_to(std**2, means.shape)
        return cls(
            mixture_weights,
            means,
            variances,
            "diag",
            n_position_dims=n_position_dims,
            position_at_time=position_at_time,
            batch_size=batch_size,
        )

    def _feature_order(self, feature_order: Sequence[int] | None) -> np.ndarray:
        order = (
            np.arange(self.n_features)
            if feature_order is None
            else np.asarray(feature_order)
        )
        if (
            order.shape != (self.n_features,)
            or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(self.n_features))
        ):
            raise ValueError(
                "feature_order must be a permutation of every mark coordinate"
            )
        return order

    def _conditional_recipes(
        self, order: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        key = tuple(int(i) for i in order)
        if key not in self._recipes:
            # Keep only the latest order; random permutations must not grow a
            # factorial number of component-parameter caches.
            self._recipes.clear()
            prefix = list(range(self.n_position_dims))
            recipes = []
            for coordinate in self.n_position_dims + order:
                if self.variances is not None:
                    gain = np.empty((len(self.weights), 0))
                    variance = self.variances[:, coordinate]
                elif not prefix:
                    gain = np.zeros((len(self.weights), 0))
                    variance = self.covariances[:, coordinate, coordinate]
                else:
                    covariance_prefix = self.covariances[:, prefix][:, :, prefix]
                    cross_covariance = self.covariances[:, prefix, coordinate]
                    gain = np.linalg.solve(
                        covariance_prefix, cross_covariance[..., None]
                    )[..., 0]
                    # Last Cholesky diagonal is the conditional standard deviation.
                    # This avoids cancellation in variance - cross*inverse*cross.
                    indices = [*prefix, int(coordinate)]
                    chol = np.linalg.cholesky(
                        self.covariances[:, indices][:, :, indices]
                    )
                    variance = chol[:, -1, -1] ** 2
                recipes.append((np.asarray(prefix, dtype=int), gain, np.sqrt(variance)))
                prefix.append(int(coordinate))
            self._recipes[key] = recipes
        return self._recipes[key]

    def _position_log_density(self, positions: np.ndarray) -> np.ndarray:
        if self.n_position_dims == 0:
            return np.zeros((len(positions), len(self.weights)))
        residual = positions[:, None, :] - self.means[None, :, : self.n_position_dims]
        if self.variances is not None:
            variance = self.variances[:, : self.n_position_dims]
            return -0.5 * np.sum(
                residual**2 / variance[None] + np.log(2 * np.pi * variance)[None],
                axis=2,
            )
        whitened = np.einsum("bkp,kpq->bkq", residual, self._position_precision)
        return -0.5 * np.sum(whitened**2, axis=2) - self._position_log_normalizer[None]

    def transform(
        self,
        observed_marks: ArrayLike,
        *,
        feature_order: Sequence[int] | None = None,
        given_positions: ArrayLike | None = None,
        component_log_weights: Callable[[int, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Evaluate sequential fitted CDFs in the requested conditioning order.

        Parameters
        ----------
        observed_marks : array_like, shape (n_queries, n_features)
        feature_order : sequence of int, optional
            Conditioning order of the mark features; identity by default.
        given_positions : array_like, shape (n_queries, n_position_dims), optional
            Positions conditioning the mixture when it models position.
        component_log_weights : callable, optional
            ``component_log_weights(first, last)`` returns the log weight of
            every fitted component, shape ``(last - first, n_components)``,
            for query rows ``first:last``; it multiplies the fitted weights.
            This conditions a mark-only mixture on something other than a
            Gaussian position, such as a heat kernel. Columns follow the
            components passed to the constructor, including zero-weight ones.

        Returns
        -------
        cdf : np.ndarray, shape (n_queries, n_features)
            Column ``l`` is the CDF of feature ``feature_order[l]`` given the
            position and the features before it in that order.
        """
        marks = np.asarray(observed_marks, dtype=float)
        if (
            marks.ndim != 2
            or marks.shape[1] != self.n_features
            or not np.isfinite(marks).all()
        ):
            raise ValueError(
                "observed_marks must be finite with the fitted mark dimensionality"
            )
        positions = (
            np.empty((len(marks), 0))
            if given_positions is None
            else np.asarray(given_positions, dtype=float)
        )
        if (
            positions.shape != (len(marks), self.n_position_dims)
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "given_positions must be finite and match query rows and fitted position dimensions"
            )
        order = self._feature_order(feature_order)
        recipes = self._conditional_recipes(order)
        result = np.empty(marks.shape, dtype=float)
        for first in range(0, len(marks), self.batch_size):
            last = min(first + self.batch_size, len(marks))
            values = np.concatenate((positions[first:last], marks[first:last]), axis=1)
            log_responsibility = np.log(self.weights)[
                None
            ] + self._position_log_density(positions[first:last])
            if component_log_weights is not None:
                given = np.asarray(component_log_weights(first, last), dtype=float)
                log_responsibility = (
                    log_responsibility + given[:, self._kept_components]
                )
            for feature, coordinate in enumerate(self.n_position_dims + order):
                prefix, gain, std = recipes[feature]
                if self.variances is not None:
                    mean = self.means[None, :, coordinate]
                else:
                    mean = self.means[None, :, coordinate] + np.sum(
                        (values[:, None, prefix] - self.means[None, :, prefix])
                        * gain[None],
                        axis=2,
                    )
                z = (values[:, coordinate, None] - mean) / std[None]
                log_normalizer = logsumexp(log_responsibility, axis=1)
                if not np.isfinite(log_normalizer).all():
                    raise ValueError(
                        "conditioning density is outside the float64 numerical range"
                    )
                responsibility = np.exp(log_responsibility - log_normalizer[:, None])
                result[first:last, feature] = np.sum(responsibility * ndtr(z), axis=1)
                if feature + 1 < self.n_features:
                    log_responsibility = (
                        log_responsibility
                        - 0.5 * z * z
                        - np.log(std)[None]
                        - 0.5 * np.log(2 * np.pi)
                    )
        # Summing normalized positive component probabilities can stray a few
        # ulps beyond a CDF endpoint. Restore the mathematical [0, 1] bounds.
        return np.clip(result, 0.0, 1.0)

    def __call__(
        self, times: np.ndarray, observed_marks: np.ndarray, feature_order: np.ndarray
    ) -> np.ndarray:
        """Adapt fixed mixture CDFs to ``interval_rescaling_transform``."""
        times = np.asarray(times, dtype=float)
        if times.shape != (len(observed_marks),) or not np.isfinite(times).all():
            raise ValueError("times must be finite and have one value per mark")
        positions = None
        if self.n_position_dims:
            if self.position_at_time is None:
                raise ValueError(
                    "Position-conditioned CDF needs position_at_time or transform(given_positions=...)"
                )
            positions = self.position_at_time(times)
        return self.transform(
            observed_marks, feature_order=feature_order, given_positions=positions
        )


def gaussian_mixture_rosenblatt_transform(
    samples: ArrayLike,
    weights: ArrayLike,
    means: ArrayLike,
    covariances: ArrayLike,
    *,
    covariance_type: str = "full",
    feature_order: Sequence[int] | None = None,
    given_positions: ArrayLike | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Evaluate a fitted mixture's sequential CDFs with optional given positions.

    ``means`` contain leading position columns when ``given_positions`` is
    supplied. Reuse ``GaussianMixtureConditionalCDF`` for repeated evaluations
    rather than copying/revalidating component parameters on every call.
    """
    if given_positions is not None and np.ndim(given_positions) != 2:
        raise ValueError("given_positions must be a two-dimensional matrix")
    n_position_dims = 0 if given_positions is None else np.shape(given_positions)[1]
    fitted = GaussianMixtureConditionalCDF(
        weights,
        means,
        covariances,
        covariance_type,
        n_position_dims=n_position_dims,
        batch_size=batch_size,
    )
    return fitted.transform(
        samples, feature_order=feature_order, given_positions=given_positions
    )
