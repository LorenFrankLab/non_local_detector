"""Training-fitted reference density for clusterless waveform marks.

Active clusterless states score each spike with a marked intensity
``lambda(x, m)``, a density over waveform features ``m``. Its value carries
the inverse volume of the waveform units. The No-Spike state models a
population firing at ``no_spike_rate`` with marks distributed as a
normalized reference density ``q(m)``, so each of its spikes scores
``log(no_spike_rate * dt) + log q(m)``. Under an affine change of waveform
units, with mark bandwidths transformed alike, every state then gains the same
log Jacobian per spike and their odds are unchanged.

``q`` is a diagonal Gaussian whose location and scale are the weighted moments
of the electrode's permitted training marks. It is a fixed reference measure,
not a fitted model of firing, and stays frozen during posterior-weighted
encoding updates.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    _SpikeTimeOrder,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
    validate_population_lengths,
)


@dataclass(frozen=True, eq=False)
class MarkReference:
    """Normalized diagonal Gaussian over one electrode's waveform features.

    Attributes
    ----------
    location : np.ndarray, shape (n_features,)
        Mean in waveform units.
    scale : np.ndarray, shape (n_features,)
        Positive standard deviation in waveform units.
    """

    location: np.ndarray
    scale: np.ndarray

    def log_density(self, marks: np.ndarray) -> np.ndarray:
        """Log reference density at each mark.

        Parameters
        ----------
        marks : np.ndarray, shape (n_spikes, n_features)

        Returns
        -------
        log_density : np.ndarray, shape (n_spikes,)
        """
        # Standardize on the host in float64 so large waveform offsets do not
        # lose digits to float32 subtraction.
        z = (np.asarray(marks, dtype=np.float64) - self.location) / self.scale
        return (
            -0.5 * np.sum(z * z, axis=1)
            - np.sum(np.log(self.scale))
            - 0.5 * z.shape[1] * np.log(2.0 * np.pi)
        )


def fit_mark_reference(
    marks: np.ndarray,
    weights: np.ndarray,
    fallback_scale: float | np.ndarray,
    electrode: int | None = None,
) -> MarkReference | None:
    """Fit one electrode's reference from its weighted training marks.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_features)
        Training waveform features.
    weights : np.ndarray, shape (n_spikes,)
        Non-negative training weight of each spike. Spikes with zero weight
        (held out, masked or outside encoding support) are ignored.
    fallback_scale : float or np.ndarray, shape (n_features,)
        Scale, in waveform units, for a coordinate with zero empirical
        variance, such as the configured mark-kernel bandwidth.
    electrode : int, optional
        Electrode index, used in error messages.

    Returns
    -------
    reference : MarkReference or None
        None when no spike has positive weight: the electrode then has no
        identifiable reference.
    """
    marks = np.asarray(marks, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = weights > 0
    if not np.any(keep):
        return None
    marks, weights = marks[keep], weights[keep]
    if not np.all(np.isfinite(marks)):
        raise ValidationError(
            f"Training waveform features of electrode {electrode} must be finite"
        )
    weights = weights / weights.sum()
    location = weights @ marks
    variance = weights @ (marks - location) ** 2
    fallback = np.broadcast_to(
        np.asarray(fallback_scale, dtype=np.float64), location.shape
    )
    # Decide constancy exactly: the weighted mean of a constant that is not
    # representable can differ from it by an ulp, leaving a roundoff variance.
    is_constant = np.ptp(marks, axis=0) == 0
    scale = np.where(is_constant, fallback, np.sqrt(variance))
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValidationError(
            "Mark reference scales must be finite and positive",
            hint="A constant training waveform feature uses the mark bandwidth; check waveform_std or gmm_reg_covar.",
        )
    return MarkReference(location, scale)


def mark_reference_log_likelihood(
    references: list[MarkReference | None],
    spike_times: list[np.ndarray],
    spike_waveform_features: list[np.ndarray],
    time_edges: np.ndarray,
    row_slice: slice | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
) -> jnp.ndarray:
    """Sum each decode bin's reference log densities over its spikes.

    Electrodes without a reference contribute nothing.

    Parameters
    ----------
    references : list of MarkReference or None, length n_electrodes
    spike_times : list of np.ndarray, each shape (n_spikes,)
    spike_waveform_features : list of np.ndarray, each shape (n_spikes, n_features)
    time_edges : np.ndarray, shape (n_bins + 1,)
        Full decoding bin edges.
    row_slice : slice, optional
        Requested decode bins; all bins by default.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1)
    """
    validate_population_lengths(
        "electrode",
        mark_reference=references,
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    row_start, row_stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    total = jnp.zeros((row_stop - row_start,))
    for electrode, (reference, times, features) in enumerate(
        zip(references, spike_times, spike_waveform_features, strict=True)
    ):
        if reference is None:
            continue
        if np.shape(features)[1] != reference.location.shape[0]:
            raise ValidationError(
                f"Electrode {electrode} has {np.shape(features)[1]} waveform "
                f"features but its mark reference was fitted with "
                f"{reference.location.shape[0]}"
            )
        selection = select_spikes_in_rows(
            times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        log_density = reference.log_density(select_spike_rows(features, selection))
        total = total + sum_spikes_into_rows(jnp.asarray(log_density), selection)
    return total[:, None]
