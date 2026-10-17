"""Training-fitted reference density for clusterless waveform marks.

Active clusterless states score each spike with a marked intensity
``lambda(x, m)``, a density over waveform features ``m``. Its value carries
the inverse volume of the waveform units. The No-Spike state models a
population firing at ``no_spike_rate`` with marks distributed as a
normalized reference density ``q(m)``, so each of its spikes scores
``log(no_spike_rate * dt) + log q(m)``. Under a per-feature affine change of
waveform units, with mark bandwidths transformed alike, every state then gains
the same log Jacobian per spike and their odds are unchanged.

``q`` is a product of per-feature Student-t densities centred on the weighted
mean of the electrode's permitted training marks, with scale
``sqrt(training variance + kernel_scale**2)``: the spread of the marks after
the model's mark kernel. Its polynomial tails exceed any Gaussian kernel or
mixture far from the training marks, so a spike no state explains is scored by
the background ``RATE_EPS_HZ * q(mark)`` and favours No-Spike. It is a fixed
reference measure, not a fitted model of firing, and stays frozen during
posterior-weighted encoding updates. The encoding fits do not fit it: a detector fits it once
(``mark_reference_``) and passes it to every state, and direct callers fit it
with ``fit_mark_references``.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    RATE_EPS_HZ,
    _SpikeTimeOrder,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
    validate_finite,
    validate_population_lengths,
)

# Degrees of freedom of each feature's Student-t: tails fall as |z|**-4, so
# far from the training marks they outlast any Gaussian kernel, while the
# variance stays finite.
REFERENCE_DEGREES_OF_FREEDOM = 3.0


@dataclass(frozen=True, eq=False)
class MarkReference:
    """Normalized product of Student-t densities over one electrode's features.

    Attributes
    ----------
    location : np.ndarray, shape (n_features,)
        Finite centre in waveform units; read-only.
    scale : np.ndarray, shape (n_features,)
        Finite, positive Student-t scale in waveform units; read-only.
    degrees_of_freedom : float
        Finite, positive Student-t degrees of freedom of each feature, by
        default ``REFERENCE_DEGREES_OF_FREEDOM``.

    Raises
    ------
    ValidationError
        If the parameters break these constraints.
    """

    location: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: float = REFERENCE_DEGREES_OF_FREEDOM

    def __post_init__(self) -> None:
        location = np.array(self.location, dtype=np.float64)
        scale = np.array(self.scale, dtype=np.float64)
        if location.ndim != 1 or scale.shape != location.shape:
            raise ValidationError(
                "A mark reference needs one scale per feature",
                got=f"location shape {location.shape}, scale shape {scale.shape}",
            )
        if not np.all(np.isfinite(location)):
            raise ValidationError("Mark reference locations must be finite")
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValidationError(
                "Mark reference scales must be finite and positive",
                hint="The scale adds the mark bandwidth to the training spread; "
                "check waveform_std or gmm_reg_covar.",
            )
        dof = float(self.degrees_of_freedom)
        if not (np.isfinite(dof) and dof > 0):
            raise ValidationError(
                "Mark reference degrees of freedom must be finite and positive",
                got=repr(self.degrees_of_freedom),
            )
        location.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "degrees_of_freedom", dof)

    def log_density(self, marks: np.ndarray) -> np.ndarray:
        """Log reference density at each mark.

        Parameters
        ----------
        marks : np.ndarray, shape (n_spikes, n_features)

        Returns
        -------
        log_density : np.ndarray, shape (n_spikes,)
        """
        marks = np.asarray(marks, dtype=np.float64)
        if marks.ndim != 2 or marks.shape[1] != self.location.shape[0]:
            raise ValidationError(
                f"Marks with {marks.shape[1:]} waveform features do not match a mark "
                f"reference fitted with {self.location.shape[0]}"
            )
        # Standardize on the host in float64 so large waveform offsets do not
        # lose digits to float32 subtraction.
        z = (marks - self.location) / self.scale
        nu = self.degrees_of_freedom
        log_normalizer = (
            gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
        )
        return (
            z.shape[1] * log_normalizer
            - np.sum(np.log(self.scale))
            - 0.5 * (nu + 1.0) * np.sum(np.log1p(z * z / nu), axis=1)
        )


def fit_mark_reference(
    marks: np.ndarray,
    weights: np.ndarray,
    kernel_scale: float | np.ndarray,
    electrode: int | None = None,
) -> MarkReference | None:
    """Fit one electrode's reference from its weighted training marks.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_features)
        Training waveform features.
    weights : np.ndarray, shape (n_spikes,)
        Finite, non-negative training weight of each spike. Spikes with zero
        weight (held out, masked or outside encoding support) are ignored.
    kernel_scale : float or np.ndarray, shape (n_features,)
        Width of the model's mark kernel in waveform units (its bandwidth),
        added in quadrature to the training spread. It also sets the scale of
        a feature without training variance.
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
    if not np.all(np.isfinite(weights) & (weights >= 0)):
        raise ValidationError(
            f"Training weights of electrode {electrode} must be finite and non-negative"
        )
    keep = weights > 0
    if not np.any(keep):
        return None
    marks, weights = marks[keep], weights[keep]
    validate_finite(marks, f"Training waveform features of electrode {electrode}")
    weights = weights / weights.sum()
    location = weights @ marks
    variance = weights @ (marks - location) ** 2
    kernel = np.broadcast_to(np.asarray(kernel_scale, dtype=np.float64), location.shape)
    # The marks' spread after the model's mark kernel.
    return MarkReference(location, np.sqrt(variance + kernel**2))


def fit_mark_references(
    spike_waveform_features: list[np.ndarray],
    kernel_scale: float | np.ndarray,
    spike_weights: list[np.ndarray] | None = None,
) -> list[MarkReference | None]:
    """Fit every electrode's reference from its training spikes.

    Pass the result as ``mark_reference`` to a clusterless predict function.
    Every state decoded together must use the same references; a detector fits
    them once, as ``mark_reference_``, from the training spikes of every
    encoding group in the modelled environments. Direct callers should likewise
    pass training spikes only.

    Parameters
    ----------
    spike_waveform_features : list of np.ndarray, each shape (n_spikes, n_features)
        Training waveform features of each electrode.
    kernel_scale : float or np.ndarray, shape (n_features,)
        The mark-kernel width, added in quadrature to each feature's training
        spread: ``<backend>.mark_reference_scale(**fit_params)``, which is
        ``waveform_std`` for KDE and diffusion and ``sqrt(gmm_reg_covar)`` for
        GMM.
    spike_weights : list of np.ndarray, each shape (n_spikes,), optional
        Training weight of each spike; spikes with zero weight are ignored.
        Uniform by default.

    Returns
    -------
    mark_reference : list of MarkReference or None, length n_electrodes
        None for an electrode without a positively weighted training spike.
    """
    if spike_weights is None:
        spike_weights = [
            np.ones(np.shape(features)[0]) for features in spike_waveform_features
        ]
    validate_population_lengths(
        "electrode",
        spike_waveform_features=spike_waveform_features,
        spike_weights=spike_weights,
    )
    return [
        fit_mark_reference(features, weights, kernel_scale, electrode)
        for electrode, (features, weights) in enumerate(
            zip(spike_waveform_features, spike_weights, strict=True)
        )
    ]


def background_log_intensity(
    reference: MarkReference | None, marks: np.ndarray, log_rate: float
) -> np.ndarray:
    """Per-spike log background intensity ``log_rate + log q(m)``.

    Without a reference the electrode has no training marks; No-Spike then adds
    no mark term either, so the background is the bare rate.

    Parameters
    ----------
    reference : MarkReference or None
    marks : np.ndarray, shape (n_spikes, n_features)
    log_rate : float
        Log background rate in the caller's intensity units.

    Returns
    -------
    log_background : np.ndarray, shape (n_spikes,)
    """
    if reference is None:
        return np.full(np.shape(marks)[0], log_rate)
    return log_rate + reference.log_density(marks)


def background_rate(n_electrodes: int) -> float:
    """Summed background rate of ``n_electrodes`` electrodes, in Hz.

    Every electrode, including a zero-rate one, has background rate
    ``RATE_EPS_HZ``. Encoding fits store only the encoding model's ground
    intensity; every clusterless predict path adds this to its expected count.

    Parameters
    ----------
    n_electrodes : int

    Returns
    -------
    rate : float
    """
    return n_electrodes * RATE_EPS_HZ


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
    for reference, times, features in zip(
        references, spike_times, spike_waveform_features, strict=True
    ):
        if reference is None:
            continue
        selection = select_spikes_in_rows(
            times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        log_density = reference.log_density(select_spike_rows(features, selection))
        total = total + sum_spikes_into_rows(log_density, selection)
    return total[:, None]
