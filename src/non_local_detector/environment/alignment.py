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

    Raises
    ------
    ValueError
        If dimensionality mismatches occur or rotation matrix is not square.
    """
    if points.shape[0] == 0:
        return np.array([])
    if points.ndim != 2:
        raise ValueError(f"Points must be a 2D array, got shape {points.shape}")
    n_dims = points.shape[1]

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
    source_probabilities: NDArray[np.float64],
    source_rotation_matrix: Optional[NDArray[np.float64]] = None,
    source_scale_factor: float = 1.0,
    source_translation_vector: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Maps probabilities from source_env to target_env by finding the nearest
    target bin for each source bin and summing probabilities if multiple
    source bins map to the same target bin.

    An optional similarity transformation (rotation, scaling, translation)
    can be applied to the source environment's bin centers before mapping
    to align them with the target environment's coordinate space.

    Parameters
    ----------
    source_env : Environment
        The source environment with defined probabilities. Must be fitted
        and have `bin_centers`.
    target_env : Environment
        The target environment to map probabilities onto. Must be fitted
        and have `bin_centers`.
    source_probabilities : NDArray[np.float64]
        Probabilities associated with each active bin in `source_env`.
        Shape must be (n_source_active_bins,).
    source_rotation_matrix : Optional[NDArray[np.float64]], optional
        Rotation matrix to apply to `source_env.bin_centers`.
        Shape (n_dims, n_dims). Defaults to identity if None.
    source_scale_factor : float, optional
        Uniform scaling factor for `source_env.bin_centers`. Defaults to 1.0.
    source_translation_vector : Optional[NDArray[np.float64]], optional
        Translation vector for `source_env.bin_centers`.
        Shape (n_dims,). Defaults to zero vector if None.

    Returns
    -------
    target_probabilities : NDArray[np.float64]
        Probabilities mapped to the active bins of `target_env`.
        Shape (n_target_active_bins,). Bins in `target_env` that are not
        the nearest neighbor to any source bin will have a probability of 0.

    Raises
    ------
    RuntimeError
        If environments are not fitted.
    ValueError
        If `bin_centers` are missing, or if `source_probabilities`
        shape mismatches, or if transformation parameters have incorrect dimensions.
    """
    if not source_env._is_fitted or not target_env._is_fitted:
        raise RuntimeError("Both source and target environments must be fitted.")
    if source_env.bin_centers is None:
        raise ValueError("Source environment is missing 'bin_centers'.")
    if target_env.bin_centers is None:
        raise ValueError("Target environment is missing 'bin_centers'.")

    n_source_bins = source_env.bin_centers.shape[0]
    n_target_bins = target_env.bin_centers.shape[0]
    n_dims = source_env.n_dims

    if source_probabilities.shape != (n_source_bins,):
        raise ValueError(
            f"source_probabilities shape {source_probabilities.shape} "
            f"must match source_env.bin_centers shape ({n_source_bins},)."
        )
    if source_env.n_dims != target_env.n_dims:
        raise ValueError(
            f"Source (dims={source_env.n_dims}) and Target (dims={target_env.n_dims}) "
            "environments must have the same number of dimensions."
        )

    # Prepare transformed source bin centers
    active_source_bin_centers = source_env.bin_centers
    if (
        source_rotation_matrix is not None
        or not np.isclose(source_scale_factor, 1.0)
        or source_translation_vector is not None
    ):

        R = (
            source_rotation_matrix
            if source_rotation_matrix is not None
            else np.eye(n_dims)
        )
        s = source_scale_factor
        t = (
            source_translation_vector
            if source_translation_vector is not None
            else np.zeros(n_dims)
        )

        active_source_bin_centers = apply_similarity_transform(
            source_env.bin_centers, R, s, t
        )

    # Handle cases with no bins
    if n_source_bins == 0:
        warnings.warn(
            "Source environment has no active bins. Returning all zeros for target.",
            UserWarning,
        )
        return np.zeros(n_target_bins)
    if n_target_bins == 0:
        warnings.warn("Target environment has no active bins to map to.", UserWarning)
        return np.array([])

    # Build KDTree on target_env bin centers for efficient nearest neighbor lookup
    try:
        target_kdtree = KDTree(target_env.bin_centers)
    except Exception as e:
        # This can happen if target_env.bin_centers is degenerate (e.g., all points identical)
        # or other KDTree construction issues.
        warnings.warn(
            f"KDTree construction on target_env.bin_centers failed: {e}. "
            "Cannot perform nearest neighbor mapping. Returning zeros.",
            UserWarning,
        )
        return np.zeros(n_target_bins)

    # For each (transformed) source bin, find the index of the nearest target bin
    try:
        # query() returns (distances, indices)
        _, nearest_target_indices_for_each_source = target_kdtree.query(
            active_source_bin_centers
        )
    except Exception as e:
        warnings.warn(
            f"KDTree query failed: {e}. "
            "Cannot perform nearest neighbor mapping. Returning zeros.",
            UserWarning,
        )
        return np.zeros(n_target_bins)

    # Ensure indices are integer type for np.add.at
    nearest_target_indices_for_each_source = (
        nearest_target_indices_for_each_source.astype(np.intp)
    )

    # Initialize target probabilities
    target_probabilities = np.zeros(n_target_bins, dtype=float)

    # Sum probabilities from source bins into their assigned nearest target bin
    # np.add.at(array, indices, values_to_add)
    # This efficiently handles cases where multiple source bins map to the same target_bin_index
    np.add.at(
        target_probabilities,
        nearest_target_indices_for_each_source,
        source_probabilities,
    )

    return target_probabilities
