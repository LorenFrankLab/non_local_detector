"""Probe contact geometry, local contact neighborhoods, and noise whitening.

High-density probes (polymer shanks, silicon probes, Neuropixels) have physical
contact coordinates that make each spike's extracellular footprint spatially
local. This module turns contact coordinates and optional measured channel noise
covariances into:

- per-contact local channel neighborhoods ``(indices, mask)`` padded to a static
  width ``C_loc`` for JAX kernels;
- greedy physical probe tiles for scalable whole-shank initialization;
- symmetric ZCA noise whitening ``Sigma_noise^{-1/2}``, which decorrelates
  channel noise without depending on channel ordering and preserves spatial
  locality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import validate_positive_int

__all__ = [
    "ProbeGeometry",
    "build_probe_tiles",
    "compute_local_neighborhoods",
    "compute_zca_whitener",
    "resolve_probe_geometry",
    "validate_contact_positions",
    "validate_noise_covariance",
]

# Default physical neighborhood radius as a multiple of median nearest-neighbor
# contact pitch (e.g. 60 um on 20 um Neuropixels, 75 um on 25 um polymer).
DEFAULT_PITCH_RADIUS_MULTIPLIER = 3.0
# Default cap on contacts per local component neighborhood.
DEFAULT_MAX_LOCAL_CHANNELS = 16
# Largest accepted condition number (max / min eigenvalue) of a measured noise
# covariance. The ZCA whitener scales the weakest noise direction by
# 1 / sqrt(min eigenvalue), so rounding errors in the marks grow with the square
# root of the condition number. Single-reference recordings of 26-32-contact
# shanks measured 127-874; a common-average-referenced covariance is singular
# (about 1e15 in float64). At 1e6, rounding float64 marks to float32 moved a
# whitened Gaussian log density by at most 7e-4 nats per spike in simulation
# (31 channels, 150 uV spikes, 8 uV noise; median 5e-5), against 9e-3
# (median 4e-4) at 1e8.
MAX_NOISE_CONDITION_NUMBER = 1e6


def _read_only_copy(array: np.ndarray) -> np.ndarray:
    """Return a read-only contiguous float64 copy."""
    copied = np.array(array, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def validate_contact_positions(
    contact_positions: ArrayLike,
    *,
    expected_channels: int | None = None,
    electrode: int | None = None,
) -> np.ndarray:
    """Validate contact coordinate array and return a float64 copy.

    Parameters
    ----------
    contact_positions : array_like, shape (n_contacts, n_probe_dims)
        Physical coordinates of each contact (e.g. in microns).
    expected_channels : int, optional
        If provided, require ``n_contacts == expected_channels``.
    electrode : int, optional
        Electrode index for error messages.

    Returns
    -------
    positions : np.ndarray, shape (n_contacts, n_probe_dims)
        Validated float64 contact coordinates.

    Raises
    ------
    ValidationError
        If the array is not 2-D, empty, non-finite, or mismatches
        ``expected_channels``.
    """
    positions = np.asarray(contact_positions, dtype=np.float64)
    label = (
        f"contact_positions of electrode {electrode}"
        if electrode is not None
        else "contact_positions"
    )
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] < 1:
        msg = f"{label} must be a non-empty 2-D array"
        raise ValidationError(
            msg,
            expected="shape (n_contacts, n_probe_dims) with n_contacts >= 1",
            got=f"shape {positions.shape}",
        )
    if not np.all(np.isfinite(positions)):
        msg = f"{label} must contain only finite values"
        raise ValidationError(msg)
    if expected_channels is not None and positions.shape[0] != expected_channels:
        msg = f"{label} channel count does not match waveform features"
        raise ValidationError(
            msg,
            expected=f"{expected_channels} contacts",
            got=f"{positions.shape[0]} contacts",
        )
    return positions


def validate_noise_covariance(
    noise_covariance: ArrayLike,
    n_channels: int,
    *,
    electrode: int | None = None,
) -> np.ndarray:
    """Validate a well-conditioned symmetric positive-definite noise covariance.

    The covariance is used through its ZCA whitener ``Sigma^{-1/2}``, which
    scales the weakest noise direction by ``1 / sqrt(min eigenvalue)``. A
    numerically singular covariance, such as that of a common-average-
    referenced recording (whose channels sum to zero), would make the
    likelihood depend on rounding noise, so the condition number (largest
    over smallest eigenvalue) must be at most ``MAX_NOISE_CONDITION_NUMBER``
    (1e6). Measured single-reference covariances of 26-32-contact shanks have
    condition numbers of 127-874. At 1e6, rounding the marks to float32 moves
    a whitened Gaussian log density by at most about 7e-4 nats per spike in
    simulation.

    Parameters
    ----------
    noise_covariance : array_like, shape (n_channels, n_channels)
        Measured channel noise covariance matrix.
    n_channels : int
        Expected number of channels.
    electrode : int, optional
        Electrode index for error messages.

    Returns
    -------
    covariance : np.ndarray, shape (n_channels, n_channels)
        Symmetrized float64 positive-definite covariance matrix.

    Raises
    ------
    ValidationError
        If shape mismatches, values are non-finite, asymmetric, not
        symmetric positive definite, or ill-conditioned (condition number
        above ``MAX_NOISE_CONDITION_NUMBER``).
    """
    cov = np.asarray(noise_covariance, dtype=np.float64)
    label = (
        f"noise_covariance of electrode {electrode}"
        if electrode is not None
        else "noise_covariance"
    )
    if cov.shape != (n_channels, n_channels):
        msg = f"{label} shape does not match channel count"
        raise ValidationError(
            msg,
            expected=f"shape ({n_channels}, {n_channels})",
            got=f"shape {cov.shape}",
        )
    if not np.all(np.isfinite(cov)):
        msg = f"{label} must contain only finite values"
        raise ValidationError(msg)
    if not np.allclose(cov, cov.T, rtol=1e-5, atol=1e-8):
        msg = f"{label} must be symmetric"
        raise ValidationError(msg)
    sym = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(sym)
    if not (np.all(np.isfinite(eigvals)) and float(np.min(eigvals)) > 0.0):
        msg = f"{label} must be symmetric positive definite"
        raise ValidationError(
            msg,
            expected="strictly positive eigenvalues",
            got=f"min eigenvalue {float(np.min(eigvals)):.3e}",
        )
    condition_number = float(eigvals[-1] / eigvals[0])
    if condition_number > MAX_NOISE_CONDITION_NUMBER:
        ridge = 1e-5 * float(eigvals[-1])
        msg = f"{label} is ill-conditioned (numerically singular)"
        raise ValidationError(
            msg,
            expected=f"condition number <= {MAX_NOISE_CONDITION_NUMBER:.0e}",
            got=f"condition number {condition_number:.3e} (eigenvalues "
            f"{float(eigvals[0]):.3e} to {float(eigvals[-1]):.3e})",
            hint="A common-average-referenced recording has a singular noise "
            "covariance because its channels sum to zero: drop one channel of "
            "each referenced group (it is a combination of the others). "
            "Otherwise regularize it, e.g. add r * np.eye(n_channels) with "
            f"r = {ridge:.3g} (1e-5 of the largest eigenvalue).",
        )
    return sym


def compute_zca_whitener(
    noise_covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute symmetric ZCA whitener ``Sigma^{-1/2}`` and its log-determinant.

    ZCA whitening is symmetric and permutation-equivariant: permuting channels
    permutes the whitener identically, preserving spatial locality across
    contacts regardless of channel ordering.

    Parameters
    ----------
    noise_covariance : np.ndarray, shape (n_channels, n_channels)
        Validated symmetric positive-definite channel noise covariance.

    Returns
    -------
    whitener : np.ndarray, shape (n_channels, n_channels)
        Symmetric positive-definite matrix ``W`` with ``W @ Sigma @ W = I``.
    log_det_whitener : float
        ``log |det W| = -0.5 * log det Sigma``.
    """
    eigvals, eigvecs = np.linalg.eigh(noise_covariance)
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    whitener = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T
    whitener = 0.5 * (whitener + whitener.T)
    log_det = -0.5 * float(np.sum(np.log(eigvals)))
    return whitener, log_det


@dataclass(frozen=True, eq=False)
class ProbeGeometry:
    """Physical contact layout and optional channel noise covariance of a probe.

    Can describe a single shank (``shank_ids=None``) or a multi-shank probe
    (``shank_ids`` assigning each contact to a shank). Stored as read-only
    NumPy arrays so instances pickle and round-trip cleanly in ``model_format``.

    Attributes
    ----------
    contact_positions : np.ndarray, shape (n_contacts, n_probe_dims)
        Physical coordinates of each contact (e.g. microns).
    shank_ids : np.ndarray or None, shape (n_contacts,), optional
        Integer shank identifier per contact, by default None (single shank).
    noise_covariance : np.ndarray or None, shape (n_contacts, n_contacts), optional
        Measured channel noise covariance, by default None.
    """

    contact_positions: np.ndarray
    shank_ids: np.ndarray | None = None
    noise_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = validate_contact_positions(self.contact_positions)
        n_contacts = positions.shape[0]
        object.__setattr__(self, "contact_positions", _read_only_copy(positions))

        if self.shank_ids is not None:
            shanks = np.asarray(self.shank_ids)
            if shanks.ndim != 1 or shanks.shape[0] != n_contacts:
                msg = "shank_ids must be a 1-D array with one entry per contact"
                raise ValidationError(
                    msg,
                    expected=f"shape ({n_contacts},)",
                    got=f"shape {shanks.shape}",
                )
            shanks_copy = np.array(shanks, copy=True)
            shanks_copy.setflags(write=False)
            object.__setattr__(self, "shank_ids", shanks_copy)

        if self.noise_covariance is not None:
            cov = validate_noise_covariance(self.noise_covariance, n_contacts)
            object.__setattr__(self, "noise_covariance", _read_only_copy(cov))

    def split_by_shank(
        self, shank_order: Sequence[object] | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """Split positions and noise covariance into one entry per shank.

        Parameters
        ----------
        shank_order : sequence, optional
            Explicit ordering of shank identifiers. By default sorted unique
            values of ``shank_ids`` (or a single-element list when
            ``shank_ids`` is None).

        Returns
        -------
        positions_per_shank : list of np.ndarray
            Per-shank contact coordinates, each shape ``(n_shank_contacts, n_probe_dims)``.
        noise_per_shank : list of np.ndarray or None
            Per-shank diagonal blocks of ``noise_covariance`` (or None entries).
        """
        if self.shank_ids is None:
            return (
                [np.array(self.contact_positions, copy=True)],
                [
                    None
                    if self.noise_covariance is None
                    else np.array(self.noise_covariance, copy=True)
                ],
            )
        unique_shanks = (
            list(np.unique(self.shank_ids))
            if shank_order is None
            else list(shank_order)
        )
        positions_list: list[np.ndarray] = []
        noise_list: list[np.ndarray | None] = []
        for shank in unique_shanks:
            idx = np.flatnonzero(self.shank_ids == shank)
            if idx.size == 0:
                msg = f"shank_order contains shank {shank!r} not present in shank_ids"
                raise ValidationError(msg)
            positions_list.append(np.array(self.contact_positions[idx], copy=True))
            if self.noise_covariance is None:
                noise_list.append(None)
            else:
                noise_list.append(
                    np.array(self.noise_covariance[np.ix_(idx, idx)], copy=True)
                )
        return positions_list, noise_list


def resolve_probe_geometry(
    contact_positions: Sequence[ArrayLike | None] | ProbeGeometry | None,
    noise_covariances: Sequence[ArrayLike | None] | None,
    mark_dimensions: Sequence[int],
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Resolve user geometry arguments into validated per-electrode lists.

    Parameters
    ----------
    contact_positions : sequence of array_like or None, ProbeGeometry, or None
        Per-electrode contact coordinates, a ``ProbeGeometry`` to split across
        shanks, or None.
    noise_covariances : sequence of array_like or None, or None
        Optional per-electrode channel noise covariances. Overrides any
        covariance attached to a ``ProbeGeometry``.
    mark_dimensions : sequence of int
        Number of waveform features (channels) per electrode.

    Returns
    -------
    positions : list of np.ndarray or None, length n_electrodes
    covariances : list of np.ndarray or None, length n_electrodes
    """
    n_electrodes = len(mark_dimensions)
    geom_covariances: list[np.ndarray | None] = [None] * n_electrodes

    if isinstance(contact_positions, ProbeGeometry):
        pos_list, geom_covariances = contact_positions.split_by_shank()
        if len(pos_list) != n_electrodes:
            msg = "ProbeGeometry shank count does not match electrode count"
            raise ValidationError(
                msg,
                expected=f"{n_electrodes} shanks",
                got=f"{len(pos_list)} shanks",
            )
        raw_positions: list[ArrayLike | None] = list(pos_list)
    elif contact_positions is None:
        raw_positions = [None] * n_electrodes
    else:
        if len(contact_positions) != n_electrodes:
            msg = "contact_positions length does not match electrode count"
            raise ValidationError(
                msg,
                expected=f"{n_electrodes} entries",
                got=f"{len(contact_positions)} entries",
            )
        raw_positions = list(contact_positions)

    if noise_covariances is None:
        raw_covariances: list[ArrayLike | None] = list(geom_covariances)
    else:
        if len(noise_covariances) != n_electrodes:
            msg = "noise_covariances length does not match electrode count"
            raise ValidationError(
                msg,
                expected=f"{n_electrodes} entries",
                got=f"{len(noise_covariances)} entries",
            )
        raw_covariances = list(noise_covariances)

    resolved_positions: list[np.ndarray | None] = []
    resolved_covariances: list[np.ndarray | None] = []
    for electrode, (n_ch, pos, cov) in enumerate(
        zip(mark_dimensions, raw_positions, raw_covariances, strict=True)
    ):
        resolved_positions.append(
            None
            if pos is None
            else validate_contact_positions(
                pos, expected_channels=n_ch, electrode=electrode
            )
        )
        resolved_covariances.append(
            None
            if cov is None
            else validate_noise_covariance(cov, n_ch, electrode=electrode)
        )
    return resolved_positions, resolved_covariances


def compute_local_neighborhoods(
    contact_positions: ArrayLike,
    *,
    neighborhood_radius: float | None = None,
    max_local_channels: int | None = DEFAULT_MAX_LOCAL_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build padded per-contact local neighborhoods from physical coordinates.

    Parameters
    ----------
    contact_positions : array_like, shape (n_contacts, n_probe_dims)
        Physical coordinates of each contact.
    neighborhood_radius : float or None, optional
        Physical radius selecting neighbor contacts. By default None, which
        auto-infers ``3.0 * median_nearest_neighbor_pitch`` (or infinity when
        all contacts coincide or there is a single contact).
    max_local_channels : int or None, optional
        Maximum contacts retained per neighborhood (closest by distance), by
        default ``DEFAULT_MAX_LOCAL_CHANNELS`` (16). None keeps all contacts
        within ``neighborhood_radius``.

    Returns
    -------
    indices : np.ndarray of int32, shape (n_contacts, c_loc)
        Neighbor channel indices per contact, sorted by ascending distance
        with contact ``c`` always at column 0. Padded entries hold 0.
    mask : np.ndarray of bool, shape (n_contacts, c_loc)
        True for real neighbors, False for padding.
    resolved_radius : float
        Physical radius used.
    """
    positions = validate_contact_positions(contact_positions)
    n_contacts = positions.shape[0]
    validate_positive_int("max_local_channels", max_local_channels, allow_none=True)

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    if neighborhood_radius is None:
        if n_contacts == 1:
            resolved_radius = 0.0
        else:
            dist_no_self = np.where(np.eye(n_contacts, dtype=bool), np.inf, dist)
            nearest = np.min(dist_no_self, axis=1)
            pitch = float(np.median(nearest))
            resolved_radius = (
                DEFAULT_PITCH_RADIUS_MULTIPLIER * pitch if pitch > 0.0 else float("inf")
            )
    else:
        radius_val = float(neighborhood_radius)
        if not (np.isfinite(radius_val) and radius_val > 0.0):
            msg = "neighborhood_radius must be None or a finite positive number"
            raise ValidationError(
                msg,
                expected="None or 0 < neighborhood_radius < inf",
                got=repr(neighborhood_radius),
            )
        resolved_radius = radius_val

    cap = (
        n_contacts
        if max_local_channels is None
        else min(int(max_local_channels), n_contacts)
    )
    neighbors: list[np.ndarray] = []
    for c in range(n_contacts):
        in_radius = np.flatnonzero(dist[c] <= resolved_radius + 1e-12)
        # Ensure self contact is first, then sort others by (distance, index).
        others = in_radius[in_radius != c]
        order = np.lexsort((others, dist[c, others]))
        ordered = np.concatenate(([c], others[order]))[:cap]
        neighbors.append(ordered.astype(np.int32))

    c_loc = max(len(nb) for nb in neighbors)
    indices = np.zeros((n_contacts, c_loc), dtype=np.int32)
    mask = np.zeros((n_contacts, c_loc), dtype=bool)
    for c, nb in enumerate(neighbors):
        indices[c, : len(nb)] = nb
        mask[c, : len(nb)] = True
    return indices, mask, resolved_radius


def build_probe_tiles(
    contact_positions: ArrayLike,
    *,
    radius: float,
    max_local_channels: int | None = DEFAULT_MAX_LOCAL_CHANNELS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Partition contacts into physical probe tiles for local initialization.

    Each contact is assigned to its nearest greedy tile center (spaced by at
    least ``radius``), giving disjoint peak-contact tiles. Each tile's local
    fitting support includes its own contacts plus nearby boundary contacts
    within ``radius`` so units near tile edges see their full local footprint.

    Parameters
    ----------
    contact_positions : array_like, shape (n_contacts, n_probe_dims)
        Physical coordinates of each contact.
    radius : float
        Physical neighborhood radius.
    max_local_channels : int or None, optional
        Local neighborhood cap; when ``n_contacts <= cap`` and all pairwise
        distances are at most ``2 * radius``, a single tile covering all
        contacts is returned.

    Returns
    -------
    tiles : list of np.ndarray of int32
        Disjoint contact-index arrays partitioning ``0..n_contacts-1``.
    tile_supports : list of np.ndarray of int32
        Sorted contact indices used to fit each tile's local initial mixture
        (a superset of the corresponding tile).
    """
    positions = validate_contact_positions(contact_positions)
    n_contacts = positions.shape[0]
    all_contacts = np.arange(n_contacts, dtype=np.int32)
    cap = n_contacts if max_local_channels is None else int(max_local_channels)

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    diameter = float(np.max(dist)) if n_contacts > 1 else 0.0

    if n_contacts <= cap and (not np.isfinite(radius) or diameter <= 2.0 * radius):
        return [all_contacts], [all_contacts]

    # Deterministic greedy covering: order contacts along primary probe axis.
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis_proj = centered @ vt[0]
    order = np.lexsort((np.arange(n_contacts), axis_proj))

    # Pick tile centers separated by at least ``radius`` (or median pitch if 0).
    effective_radius = radius if (np.isfinite(radius) and radius > 0.0) else 1.0
    centers: list[int] = []
    for c in order:
        c_int = int(c)
        if not centers or all(
            dist[c_int, center] > effective_radius for center in centers
        ):
            centers.append(c_int)

    # Assign each contact to its nearest tile center (tie-break by center order).
    dist_to_centers = dist[:, centers]
    owner = np.argmin(dist_to_centers, axis=1)

    tiles: list[np.ndarray] = []
    tile_supports: list[np.ndarray] = []
    # Bound support size so local initialization stays O(C_loc^3) even on dense grids.
    support_cap = min(n_contacts, max(cap * 2, cap))
    for tile_idx, center in enumerate(centers):
        members = np.flatnonzero(owner == tile_idx).astype(np.int32)
        if members.size == 0:
            continue
        # Union of tile contacts and their neighbors within ``effective_radius``.
        min_dist_to_tile = np.min(dist[:, members], axis=1)
        in_support = np.flatnonzero(min_dist_to_tile <= effective_radius + 1e-12)
        if in_support.size > support_cap:
            # Always keep tile members, then closest boundary contacts to tile center.
            extras = np.setdiff1d(in_support, members, assume_unique=False)
            extra_order = np.lexsort((extras, dist[center, extras]))
            keep_extras = extras[extra_order[: max(0, support_cap - members.size)]]
            support = np.sort(np.concatenate([members, keep_extras])).astype(np.int32)
        else:
            support = np.sort(in_support).astype(np.int32)
        tiles.append(members)
        tile_supports.append(support)

    return tiles, tile_supports
