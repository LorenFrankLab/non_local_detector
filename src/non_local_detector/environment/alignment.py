"""
Alignment and Probability Mapping Between Spatial Environments.

This module provides functionalities to align and map data, particularly
probability distributions, between different spatial `Environment` instances
defined in the `non_local_detector.environment` package.

Core capabilities include:

1.  **Geometric Transformations**:
    * Applying similarity transformations (rotation, scaling, and translation)
        to sets of points, typically the bin centers of a source `Environment`,
        to align them with a target `Environment`'s coordinate space.
    * Helper functions to create 2D rotation matrices from angles or for
        common rotations (e.g., 90 degrees).

2.  **Probability Mapping**:
    * The primary method, `map_probabilities_to_nearest_target_bin`,
        transfers probabilities from a source environment to a target environment.
        For each bin in the (optionally transformed) source environment, its
        probability is assigned to the spatially nearest bin in the target
        environment. If multiple source bins map to the same target bin,
        their probabilities are summed. This is useful when comparing or
        aggregating data from slightly different discretizations or
        experimental setups of the same underlying space.

This module is designed to assist in scenarios such as:
    * Comparing probability distributions (e.g., place fields, occupancy maps)
        from experiments where the recording environment might have undergone
        slight shifts, rotations, or scaling.
    * Mapping data from one type of spatial discretization (e.g., a fine grid)
        to another (e.g., a coarser grid or a different layout type), while
        attempting to preserve the spatial correspondence of the data.

The functions generally expect `Environment` objects that have been "fitted"
(i.e., their `bin_centers` attribute is populated) and probability arrays
that correspond to the active bins of these environments.
"""

import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from non_local_detector.environment.environment import Environment


def get_2d_rotation_matrix(angle_degrees: float) -> NDArray[np.float64]:
    """
    Creates a 2D counter-clockwise rotation matrix for a given angle.

    Parameters
    ----------
    angle_degrees : float
        The rotation angle in degrees. Positive for counter-clockwise.

    Returns
    -------
    NDArray[np.float64]
        The 2x2 rotation matrix.

    Example
    -------
    >>> rotation_matrix = get_2d_rotation_matrix(90)
    >>> print(rotation_matrix)
    >> [[ 0. -1.]
        [ 1.  0.]]
    """
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotation_matrix


def apply_similarity_transform(
    points: NDArray[np.float64],
    rotation_matrix: NDArray[np.float64],
    scale_factor: float,
    translation_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Applies a similarity transformation (rotation, scaling, translation)
    to a set of points.
    The transformation is applied as: P_transformed = scale * (R @ P.T).T + t

    Parameters
    ----------
    points : NDArray[np.float64]
        Points to transform, shape (n_points, n_dims).
    rotation_matrix : NDArray[np.float64]
        Rotation matrix, shape (n_dims, n_dims).
    scale_factor : float
        Uniform scaling factor.
    translation_vector : NDArray[np.float64]
        Translation vector, shape (n_dims,).

    Returns
    -------
    NDArray[np.float64]
        Transformed points, shape (n_points, n_dims).
        If `points` is empty, returns an empty array of shape (0, n_dims).

    Raises
    ------
    ValueError
        If dimensionality mismatches occur or rotation matrix is not square.
    """
    n_dims = points.shape[1]
    if points.ndim != 2:
        raise ValueError(f"Points must be a 2D array, got shape {points.shape}")

    if points.shape[0] == 0:
        return np.zeros((0, n_dims), dtype=points.dtype)

    if rotation_matrix.shape != (n_dims, n_dims):
        raise ValueError(
            f"Rotation matrix shape {rotation_matrix.shape} "
            f"is not compatible with points_dims {n_dims}."
        )
    if not np.isscalar(scale_factor):
        raise ValueError("Scale factor must be a scalar.")
    if translation_vector.shape != (n_dims,):
        raise ValueError(
            f"Translation vector shape {translation_vector.shape} "
            f"is not compatible with points_dims {n_dims}."
        )

    # 1. Rotate
    rotated_points = (rotation_matrix @ points.T).T
    # 2. Scale
    scaled_points = scale_factor * rotated_points
    # 3. Translate
    transformed_points = scaled_points + translation_vector
    return transformed_points


def map_probabilities_to_nearest_target_bin(
    source_env: Environment,
    target_env: Environment,
    source_probs: NDArray[np.float64],
    *,
    mode: str = "nearest",
    n_neighbors: int = 1,
    eps: float = 1e-8,
    source_scale_factor: float = 1.0,
    source_rotation_matrix: Optional[NDArray[np.float64]] = None,
    source_translation_vector: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Map probabilities on source_env onto target_env, with optional scaling/translation
    of the source bin-centers beforehand.

    Parameters
    ----------
    source_env : Environment
        A fitted Environment whose bins currently have centers `source_env.bin_centers`.
    target_env : Environment
        A fitted Environment onto whose bins we want to map probabilities.
    source_probs : NDArray[np.float64]
        1D array of length source_env.n_bins (nonnegative).
    mode : {'nearest', 'inverse-distance-weighted'}
        - "nearest": each source bin's mass → its single nearest target bin.
        - "inverse-distance-weighted" : spread each source bin's mass over
            its `n_neighbors` nearest target bins,
            weighted by inverse distance, and summing contributions.
    n_neighbors : int
        Number of neighbors to use when mode="inverse-distance-weighted"
        (ignored if mode="nearest").
    eps : float
        Small constant to avoid division by zero in IDW weights.
    source_scale_factor : float
        Multiply every source bin-center by this scalar before querying.
    source_rotation_matrix : Optional[NDArray[np.float64]]
        If not None, must be a length-n_dims array; we add it to every source bin-center
        (after scaling) before querying.
    source_translation_vector : Optional[NDArray[np.float64]]
        If not None, must be a length-n_dims array; we apply it to every source bin-center
        (after scaling) before querying.

    Returns
    -------
    target_probabilities : array, shape (n_target_bins,)
        Each entry is the sum of `source_probabilities` whose (transformed)
        source bin center is nearest to that target bin center. If `n_target_bins == 0`,
        returns an empty array of shape (0,).

    Raises
    ------
    RuntimeError
        If either `source_env` or `target_env` is not fitted.
    ValueError
        If `source_probabilities` has incorrect shape, or if dims mismatch.
    """
    import warnings

    import numpy as np
    from scipy.spatial import cKDTree

    # 1) Basic sanity checks (same as before) -----------------------------
    if not getattr(source_env, "_is_fitted", False) or not getattr(
        target_env, "_is_fitted", False
    ):
        raise ValueError(
            "Both source_env and target_env must be fitted before mapping probabilities."
        )

    n_src = source_env.n_bins
    n_tgt = target_env.n_bins

    if n_src == 0 or n_tgt == 0:
        warnings.warn(
            "One of the environments has zero bins; returning zeros.", UserWarning
        )
        return np.zeros(n_tgt, dtype=float)

    if source_env.n_dims != target_env.n_dims:
        raise ValueError(
            f"Source and target environments must have the same number of dimensions: "
            f"{source_env.n_dims} != {target_env.n_dims}."
        )

    if source_probs.ndim != 1 or source_probs.shape[0] != n_src:
        raise ValueError(f"source_probs must be a 1D array of length {n_src}.")

    if np.any(source_probs < 0):
        raise ValueError("source_probs must be nonnegative.")

    if mode not in ("nearest", "inverse-distance-weighted"):
        raise ValueError(
            f"Unrecognized mode '{mode}'. Supported: 'nearest' or 'inverse-distance-weighted'."
        )

    if mode == "inverse-distance-weighted" and (
        n_neighbors < 1 or not isinstance(n_neighbors, int)
    ):
        raise ValueError(
            f"In 'inverse-distance-weighted' mode, n_neighbors must be an integer ≥ 1 (got n_neighbors={n_neighbors})."
        )

    # 2) Fetch and “pre-transform” source bin centers -----------------------
    #    (i.e. apply scale then translation if provided)
    src_centers = source_env.bin_centers.copy().astype(float)  # shape = (n_src, n_dims)
    # (a) scale
    if source_scale_factor != 1.0:
        src_centers *= float(source_scale_factor)
    if source_rotation_matrix is not None:
        source_rotation_matrix = np.asarray(source_rotation_matrix, dtype=float)
        if source_rotation_matrix.shape != (2, 2):
            raise ValueError("source_rotation_matrix must be a 2x2 array.")
        src_centers = src_centers @ source_rotation_matrix.T

    # (b) translate if provided
    if source_translation_vector is not None:
        vec = np.asarray(source_translation_vector, dtype=float)
        if vec.ndim != 1 or vec.shape[0] != src_centers.shape[1]:
            raise ValueError(
                f"source_translation_vector must be 1D of length {src_centers.shape[1]}"
            )
        src_centers += vec  # broadcast to every row

    # 3) Build KDTree on target bin centers --------------------------------
    try:
        tgt_centers = target_env.bin_centers  # shape = (n_tgt, n_dims)
        tree = cKDTree(tgt_centers, leafsize=16)
    except Exception as e:
        warnings.warn(
            f"KDTree construction on target_env failed: {e}. Returning zeros.",
            RuntimeWarning,
        )
        return np.zeros(n_tgt, dtype=float)

    # 4) Prepare output array ------------------------------------------------
    target_probs = np.zeros(n_tgt, dtype=float)

    # 5) Perform the requested mapping ---------------------------------------
    if mode == "nearest":
        # Query exactly one neighbor per source
        try:
            _, idxs = tree.query(src_centers, k=1)  # shape = (n_src,)
        except Exception as e:
            warnings.warn(
                f"KDTree.query (nearest) failed: {e}. Returning zeros.", RuntimeWarning
            )
            return np.zeros(n_tgt, dtype=float)

        # Accumulate each source's entire probability into its nearest target
        np.add.at(target_probs, idxs, source_probs)

    elif mode == "inverse-distance-weighted":
        # Clamp n_neighbors if larger than number of target bins
        k_eff = min(n_neighbors, n_tgt)
        try:
            dists, idxs = tree.query(
                src_centers, k=k_eff
            )  # each shape = (n_src, k_eff)
        except Exception as e:
            warnings.warn(
                f"KDTree.query (inverse-distance-weighted) failed: {e}. Returning zeros.",
                RuntimeWarning,
            )
            return np.zeros(n_tgt, dtype=float)

        # If k_eff == 1, we may get 1D arrays; force them to 2D
        if k_eff == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)

        # Compute inverse-distance weights w_ij = 1/(d_ij + eps), row-normalize
        weights = 1.0 / (dists + eps)  # shape = (n_src, k_eff)
        sums = np.sum(weights, axis=1, keepdims=True)  # shape = (n_src, 1)
        normed = weights / sums  # each row now sums to 1

        # Multiply each source's probability by its weight vector
        src_col = source_probs.reshape(-1, 1)  # shape = (n_src, 1)
        contribs = src_col * normed  # shape = (n_src, k_eff)

        # Flatten indices & contributions so we can call add.at once
        flat_idxs = idxs.ravel()  # length = n_src * k_eff
        flat_contribs = contribs.ravel()  # same length

        np.add.at(target_probs, flat_idxs, flat_contribs)
    else:
        raise ValueError(f"Unrecognized mode '{mode}'.")

    return target_probs
