"""Highest posterior density (HPD) thresholds and spatial coverage of posteriors."""

import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dims

__all__ = [
    "get_HPD_spatial_coverage",
    "get_highest_posterior_threshold",
]


def get_highest_posterior_threshold(
    posterior: xr.DataArray | np.ndarray,
    coverage: float = 0.95,
    bin_width: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the posterior threshold for highest posterior density (HPD) regions.

    Computes the threshold values that define the highest posterior density
    regions containing a specified coverage probability. This approach can
    handle multimodal distributions by selecting regions with highest density
    rather than requiring a single contiguous interval.

    Parameters
    ----------
    posterior : xarray.DataArray or np.ndarray, shape (n_time, n_position_bins) or (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over position at each time point.
        The non-time dimensions are flattened for processing.
    coverage : float, optional
        Desired coverage probability for the HPD region. Must be between 0 and 1.
        Default is 0.95 for 95% coverage.
    bin_width : np.ndarray, shape (n_position_bins,), optional
        Positive width (or measure) of each position bin, in the flattened
        order of the non-time dimensions. Pass it when the widths differ,
        e.g. ``np.diff(environment.edges_[0])`` on a track graph; the
        threshold is then on density, ``posterior / bin_width``. Pass the
        same widths to `get_HPD_spatial_coverage`.

    Returns
    -------
    threshold : np.ndarray, shape (n_time,)
        Threshold values for each time point. Posterior values (densities,
        ``posterior / bin_width``, when `bin_width` is given) at or above
        this threshold define the HPD region with the specified coverage.
        Rows without any posterior mass have a NaN threshold. Missing bins
        are ignored independently in each row.

    Notes
    -----
    The algorithm sorts posterior values in descending order and finds the
    threshold where the cumulative probability first exceeds the desired coverage.
    For reference see: https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    With `bin_width`, bins are sorted by density and their mass is
    accumulated in that order, so the region has the smallest total width
    for its coverage. Sorting by mass instead favours wide bins: a wide bin
    can hold more mass than a narrow one at a higher density. Bins of equal
    density are all in or all out of the region, as bins of equal mass are
    without widths.
    """
    # Reshape non-time dimensions into a single dimension
    n_time = posterior.shape[0]
    posterior_array = np.asarray(posterior).reshape((n_time, -1))
    # Missing bins in one frame must not remove those bins from other frames.
    posterior_array = np.where(np.isnan(posterior_array), 0.0, posterior_array)
    threshold = np.full(n_time, np.nan)
    const = np.sum(posterior_array, axis=1, keepdims=True)
    has_mass = np.isfinite(const[:, 0]) & (const[:, 0] > 0.0)
    if not np.any(has_mass):
        return threshold
    if bin_width is not None:
        return _highest_density_threshold(
            posterior_array, has_mass, const, coverage, bin_width, threshold
        )

    # Sort the posterior values in descending order
    sorted_norm_posterior = (
        np.sort(posterior_array[has_mass], axis=1)[:, ::-1] / const[has_mass]
    )

    # Find the threshold that corresponds to the coverage
    # by finding the first index where the cumulative sum is greater than the coverage
    posterior_less_than_coverage = np.cumsum(sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)

    # Rounding can leave the full cumulative mass just below one. Fall back
    # to the last positive bin, so missing/zero-mass bins stay outside the HPD.
    no_cutoff = ~posterior_less_than_coverage.any(axis=1)
    last_positive = np.count_nonzero(sorted_norm_posterior > 0.0, axis=1) - 1
    crit_ind[no_cutoff] = last_positive[no_cutoff]

    threshold[has_mass] = (
        sorted_norm_posterior[np.arange(np.count_nonzero(has_mass)), crit_ind]
        * const[has_mass, 0]
    )
    return threshold


def _highest_density_threshold(
    posterior_array: np.ndarray,
    has_mass: np.ndarray,
    const: np.ndarray,
    coverage: float,
    bin_width: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    """Fill `threshold` with the density threshold of each row that has mass.

    Parameters
    ----------
    posterior_array : np.ndarray, shape (n_time, n_position_bins)
        Posterior mass, missing bins set to zero.
    has_mass : np.ndarray, shape (n_time,)
        Rows with positive, finite total mass.
    const : np.ndarray, shape (n_time, 1)
        Total mass of each row.
    coverage : float
        Mass the region must reach.
    bin_width : np.ndarray, shape (n_position_bins,)
        Width of each bin.
    threshold : np.ndarray, shape (n_time,)
        NaN-filled output.

    Returns
    -------
    threshold : np.ndarray, shape (n_time,)
        Density ``posterior / bin_width`` at the region's boundary.
    """
    bin_width = np.asarray(bin_width, dtype=float).ravel()
    if bin_width.shape != (posterior_array.shape[1],):
        msg = (
            f"bin_width must have {posterior_array.shape[1]} entries, one per "
            f"position bin; got shape {np.shape(bin_width)}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(bin_width) & (bin_width > 0)):
        msg = "bin_width must be finite and positive."
        raise ValueError(msg)
    # The same division as get_HPD_spatial_coverage, so the boundary bin
    # compares equal to the threshold.
    density = posterior_array[has_mass] / bin_width
    order = np.argsort(-density, axis=1, kind="stable")
    sorted_density = np.take_along_axis(density, order, axis=1)
    sorted_mass = np.take_along_axis(posterior_array[has_mass], order, axis=1)
    reached = np.cumsum(sorted_mass / const[has_mass], axis=1) >= coverage
    crit_ind = np.argmax(reached, axis=1)
    # As for mass: if rounding leaves the total just below one, stop at the
    # last bin with mass.
    no_cutoff = ~reached.any(axis=1)
    last_positive = np.count_nonzero(sorted_density > 0.0, axis=1) - 1
    crit_ind[no_cutoff] = last_positive[no_cutoff]
    threshold[has_mass] = sorted_density[np.arange(len(crit_ind)), crit_ind]
    return threshold


def get_HPD_spatial_coverage(
    posterior: xr.DataArray,
    hpd_threshold: np.ndarray,
    bin_width: np.ndarray | None = None,
) -> np.ndarray:
    """Compute total spatial measure covered by highest posterior density regions.

    Calculates the total spatial measure (length in 1D, area in 2D, volume in
    3D) of the environment covered by posterior values that exceed the HPD
    threshold at each time point. This provides a measure of spatial
    uncertainty in the posterior distribution.

    Parameters
    ----------
    posterior : xarray.DataArray
        Shape ``(n_time, *n_position_bins)`` with time as the leading axis and
        the position dimensions trailing. Position dims follow the canonical
        naming: ``position`` (1D) or names ending in ``_position``
        (``x_position``/``y_position`` for 2D, ``z_position`` and so on for
        higher dimensions).
    hpd_threshold : np.ndarray, shape (n_time,)
        HPD threshold values for each time point, typically obtained from
        `get_highest_posterior_threshold`. With `bin_width`, a threshold on
        density from ``get_highest_posterior_threshold(posterior, coverage,
        bin_width=bin_width)``.
    bin_width : np.ndarray, shape (n_position_bins,), optional
        Exact width of each position bin, used when the bins are *not*
        uniformly spaced — e.g. ``np.diff(environment.edges_[0])`` for a
        linearized track-graph environment whose segments have different
        lengths (different per-arm bin widths) or ``edge_spacing`` gap bins.
        A bin is in the region when its density ``posterior / bin_width`` is
        at or above `hpd_threshold`. Only valid when the posterior has a
        single position dimension. When omitted, uniform spacing is assumed
        (see Notes); this is exact for open-field grids and single-segment
        tracks, which the ``Environment`` builds with equal-width bins.

    Returns
    -------
    spatial_coverage : np.ndarray, shape (n_time,)
        Total spatial measure covered by the highest posterior density regions
        at each time point. Units depend on the spatial coordinate system and
        the dimensionality of the posterior (e.g., cm for 1D, cm² for 2D,
        cm³ for 3D). A NaN threshold produces NaN coverage for that frame.

    Notes
    -----
    With ``bin_width=None`` the function assumes uniform spatial bin spacing
    along each axis and uses the first difference of each position coordinate;
    the bin measure is the product of the per-dimension bin widths (a length in
    1D, an area in 2D, a volume in 3D, and so on). Each position coordinate must
    be strictly increasing (so the first difference is a valid bin width); a
    non-monotonic or descending coordinate, or a degenerate single-bin axis,
    raises a ``ValueError``. Pass ``bin_width`` to integrate exact per-bin widths
    instead (1D only); the widths cannot be recovered from the bin centers alone
    when the spacing is non-uniform.
    """
    position_dims = get_position_dims(posterior)
    if not position_dims:
        msg = (
            "posterior must have a 'position' dim or dims ending in "
            f"'_position'; got {tuple(posterior.dims)}"
        )
        raise ValueError(msg)
    # Broadcast the per-time threshold over the trailing position axes.
    threshold = hpd_threshold[(slice(None), *([np.newaxis] * len(position_dims)))]

    if bin_width is not None:
        if len(position_dims) != 1:
            msg = (
                "bin_width is only supported for a single (linearized) position "
                f"dimension; the posterior has {len(position_dims)} position "
                f"dims {tuple(position_dims)}. Multidimensional grids are "
                "uniformly spaced, so omit bin_width."
            )
            raise ValueError(msg)
        (dim,) = position_dims
        bin_width = np.asarray(bin_width, dtype=float)
        if bin_width.shape != (posterior.sizes[dim],):
            msg = (
                f"bin_width must have shape ({posterior.sizes[dim]},) to match "
                f"the posterior's '{dim}' dimension; got {bin_width.shape}."
            )
            raise ValueError(msg)
        # Weight each in-HPD bin by its exact width and reduce over the position
        # axis identified by name, so the result does not depend on whether the
        # position axis is trailing.
        axis = posterior.dims.index(dim)
        weight_shape = [1] * posterior.ndim
        weight_shape[axis] = bin_width.shape[0]
        width = bin_width.reshape(weight_shape)
        isin_hpd = np.asarray(posterior.values) / width >= threshold
        weighted = isin_hpd * width
        return np.where(np.isnan(hpd_threshold), np.nan, weighted.sum(axis=axis))

    isin_hpd = posterior >= threshold
    # Uniform spacing: bin measure is the product of per-dimension bin widths,
    # taken from the first difference of each (strictly increasing) coordinate.
    bin_measure = 1.0
    for dim in position_dims:
        coord = np.asarray(posterior[dim].values)
        if coord.size < 2:
            msg = (
                f"position dimension '{dim}' has size {coord.size}; at least 2 "
                "bins are required to infer the bin width from coordinates. Pass "
                "an explicit bin_width for a single-bin 1D grid."
            )
            raise ValueError(msg)
        diffs = np.diff(coord)
        if np.any(diffs <= 0):
            msg = (
                f"position coordinate '{dim}' must be strictly increasing to "
                "infer a uniform bin width; got non-monotonic or descending "
                "coordinates. Sort the posterior along this dimension, or pass an "
                "explicit bin_width for a 1D non-uniform grid."
            )
            raise ValueError(msg)
        bin_measure *= float(diffs[0])
    coverage = (isin_hpd * bin_measure).sum(position_dims).values
    return np.where(np.isnan(hpd_threshold), np.nan, coverage)
