import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dims


def get_highest_posterior_threshold(
    posterior: xr.DataArray, coverage: float = 0.95
) -> np.ndarray:
    """Estimate the posterior threshold for highest posterior density (HPD) regions.

    Computes the threshold values that define the highest posterior density
    regions containing a specified coverage probability. This approach can
    handle multimodal distributions by selecting regions with highest density
    rather than requiring a single contiguous interval.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over position at each time point.
        The non-time dimensions are flattened for processing.
    coverage : float, optional
        Desired coverage probability for the HPD region. Must be between 0 and 1.
        Default is 0.95 for 95% coverage.

    Returns
    -------
    threshold : np.ndarray, shape (n_time,)
        Threshold values for each time point. Posterior values at or above
        this threshold define the HPD region with the specified coverage.
        Rows without any posterior mass have a NaN threshold. Missing bins
        are ignored independently in each row.

    Notes
    -----
    The algorithm sorts posterior values in descending order and finds the
    threshold where the cumulative probability first exceeds the desired coverage.
    For reference see: https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

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
        `get_highest_posterior_threshold`.
    bin_width : np.ndarray, shape (n_position_bins,), optional
        Exact width of each position bin, used when the bins are *not*
        uniformly spaced — e.g. ``np.diff(environment.edges_[0])`` for a
        linearized track-graph environment whose segments have different
        lengths (different per-arm bin widths) or ``edge_spacing`` gap bins.
        Only valid when the posterior has a single position dimension. When
        omitted, uniform spacing is assumed (see Notes); this is exact for
        open-field grids and single-segment tracks, which the
        ``Environment`` builds with equal-width bins.

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
        raise ValueError(
            "posterior must have a 'position' dim or dims ending in "
            f"'_position'; got {tuple(posterior.dims)}"
        )
    # Broadcast the per-time threshold over the trailing position axes.
    threshold = hpd_threshold[(slice(None), *([np.newaxis] * len(position_dims)))]
    isin_hpd = posterior >= threshold

    if bin_width is not None:
        if len(position_dims) != 1:
            raise ValueError(
                "bin_width is only supported for a single (linearized) position "
                f"dimension; the posterior has {len(position_dims)} position "
                f"dims {tuple(position_dims)}. Multidimensional grids are "
                "uniformly spaced, so omit bin_width."
            )
        (dim,) = position_dims
        bin_width = np.asarray(bin_width, dtype=float)
        if bin_width.shape != (posterior.sizes[dim],):
            raise ValueError(
                f"bin_width must have shape ({posterior.sizes[dim]},) to match "
                f"the posterior's '{dim}' dimension; got {bin_width.shape}."
            )
        # Weight each in-HPD bin by its exact width and reduce over the position
        # axis identified by name, so the result does not depend on whether the
        # position axis is trailing.
        axis = posterior.dims.index(dim)
        weight_shape = [1] * isin_hpd.ndim
        weight_shape[axis] = bin_width.shape[0]
        weighted = np.asarray(isin_hpd.values) * bin_width.reshape(weight_shape)
        return np.where(np.isnan(hpd_threshold), np.nan, weighted.sum(axis=axis))

    # Uniform spacing: bin measure is the product of per-dimension bin widths,
    # taken from the first difference of each (strictly increasing) coordinate.
    bin_measure = 1.0
    for dim in position_dims:
        coord = np.asarray(posterior[dim].values)
        if coord.size < 2:
            raise ValueError(
                f"position dimension '{dim}' has size {coord.size}; at least 2 "
                "bins are required to infer the bin width from coordinates. Pass "
                "an explicit bin_width for a single-bin 1D grid."
            )
        diffs = np.diff(coord)
        if np.any(diffs <= 0):
            raise ValueError(
                f"position coordinate '{dim}' must be strictly increasing to "
                "infer a uniform bin width; got non-monotonic or descending "
                "coordinates. Sort the posterior along this dimension, or pass an "
                "explicit bin_width for a 1D non-uniform grid."
            )
        bin_measure *= float(diffs[0])
    coverage = (isin_hpd * bin_measure).sum(position_dims).values
    return np.where(np.isnan(hpd_threshold), np.nan, coverage)
