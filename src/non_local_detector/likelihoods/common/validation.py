"""Input validation and the encoding contract checked between fit and predict."""

import hashlib
from collections.abc import Sequence, Sized
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError


def as_std_array(std: "jax.Array | np.ndarray | float | int", n_dims: int) -> jax.Array:
    """Broadcast a scalar kernel bandwidth to a per-dimension array.

    A single ``jnp.ndim`` check covers Python ``int``/``float``, NumPy scalars
    (including ``np.float32``), 0-d arrays, and JAX scalars uniformly, avoiding
    the ``isinstance(std, int | float)`` pattern that silently misses
    ``np.float32`` and 0-d arrays. Array-valued ``std`` (``ndim >= 1``) is
    returned via ``jnp.asarray`` (a no-op for JAX arrays; lists/NumPy arrays are
    converted to a JAX array).

    Parameters
    ----------
    std : jax.Array or float or int
        Scalar (broadcast to ``n_dims``) or per-dimension standard deviation(s).
    n_dims : int
        Number of dimensions to broadcast a scalar to.

    Returns
    -------
    std_array : jax.Array
        Shape ``(n_dims,)`` for scalar input; the original array otherwise.
    """
    if jnp.ndim(std) == 0:
        return jnp.full((n_dims,), std)
    return jnp.asarray(std)


def validate_bandwidth(std: Any, n_dims: int, name: str) -> jax.Array:
    """Kernel bandwidths as a per-dimension array, rejecting unusable values.

    A zero, negative or non-finite bandwidth would fit a degenerate kernel that
    yields plausible-looking floor rates instead of an error.

    Parameters
    ----------
    std : float or array-like, shape () or (n_dims,)
        Python, NumPy or JAX scalar, or one bandwidth per dimension.
    n_dims : int
        Number of kernel dimensions.
    name : str
        Parameter name for error messages.

    Returns
    -------
    std : jax.Array, shape (n_dims,)
        In JAX's default floating dtype.

    Raises
    ------
    ValidationError
        If a value is not finite and positive, the shape is neither scalar nor
        ``(n_dims,)``, or a value overflows or underflows to zero in JAX's
        default floating dtype.
    """
    values = np.asarray(std, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        msg = f"{name} must contain only finite positive values"
        raise ValidationError(
            msg,
            expected="finite bandwidths greater than zero",
            got=f"{name}={std!r}",
            hint="Kernel bandwidths are standard deviations in the data's units.",
        )
    if values.ndim == 0:
        values = np.full(n_dims, values.item())
    if values.shape != (n_dims,):
        msg = f"{name} has the wrong shape"
        raise ValidationError(
            msg,
            expected=f"a scalar or shape ({n_dims},)",
            got=f"shape {values.shape}",
        )
    dtype = jax.dtypes.canonicalize_dtype(values.dtype)
    with np.errstate(over="ignore", under="ignore"):
        values = values.astype(dtype)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        msg = f"{name} is not representable as finite positive {dtype} values"
        raise ValidationError(
            msg,
            got=f"{name}={std!r}",
            hint="Rescale the data or enable JAX float64 (jax_enable_x64).",
        )
    return jnp.asarray(values)


def validate_positive_int(name: str, value: Any, *, allow_none: bool = False) -> None:
    """Reject a setting that is not a positive integer (or None, if allowed).

    Parameters
    ----------
    name : str
        The parameter name, for the error message.
    value : object
        Setting to check.
    allow_none : bool, optional
        Whether None is accepted, by default False.

    Raises
    ------
    ValidationError
        If ``value`` is a bool, not an integer, or below 1.
    """
    if allow_none and value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        expected = "None or a positive integer" if allow_none else "a positive integer"
        msg = f"{name} must be {expected}"
        raise ValidationError(msg, expected=expected, got=repr(value))


def validate_finite(array: np.ndarray | jax.Array, name: str) -> None:
    """Raise if ``array`` contains any non-finite (NaN/inf) value.

    Shared fit-input finiteness check for the encoding fits: a non-finite
    position or spike waveform feature is invalid encoding data that would
    otherwise propagate into the density estimate as a silent NaN. Fail fast at
    the fit entry, consistent with ``validate_weights`` and the GMM/MRF fits
    (which already reject non-finite inputs).

    Parameters
    ----------
    array : np.ndarray | jax.Array
        The array to check.
    name : str
        Name used in the error message (e.g. ``"position"``).

    Raises
    ------
    ValidationError
        If ``array`` contains any non-finite value.
    """
    # Reduce on the array's own device: for a JAX array this keeps the full
    # (potentially GPU-resident) array on-device and transfers only the scalar
    # result, avoiding a full device->host copy per electrode on every refit.
    if isinstance(array, jax.Array):
        all_finite = bool(jnp.all(jnp.isfinite(array)))
    else:
        all_finite = bool(np.all(np.isfinite(np.asarray(array))))
    if not all_finite:
        msg = f"{name} must contain only finite values"
        raise ValidationError(msg)


def as_array(array: ArrayLike) -> np.ndarray | jax.Array:
    """Return a JAX array as is and any other array-like as a NumPy array.

    A detector passes NumPy arrays, returned as is. A JAX array stays on its
    devices, so a sharded input is never gathered to the host.

    Parameters
    ----------
    array : array_like
        The array.

    Returns
    -------
    array : np.ndarray or jax.Array
        ``array`` if it is a JAX array, else ``np.asarray(array)``.
    """
    return array if isinstance(array, jax.Array) else np.asarray(array)


def as_arrays(arrays: Sequence[ArrayLike]) -> list[np.ndarray | jax.Array]:
    """Apply ``as_array`` to each array of a per-neuron or per-electrode sequence.

    Parameters
    ----------
    arrays : sequence of array_like
        One array per neuron or electrode.

    Returns
    -------
    arrays : list of np.ndarray or jax.Array
        The arrays, in order.
    """
    return [as_array(array) for array in arrays]


def require_position(
    position_time: ArrayLike | None, position: ArrayLike | None, likelihood: str
) -> tuple[np.ndarray | jax.Array, np.ndarray | jax.Array]:
    """Return the decode position samples a likelihood reads, through ``as_array``.

    A detector passes None for both when no state reads position (no Local
    state and no non-local position penalty), so only the non-local
    likelihoods are evaluated without them.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,), or None
        Time of each position sample, in seconds.
    position : array_like, shape (n_time_position, n_position_dims), or None
        Position samples.
    likelihood : str
        Name of the likelihood that reads them, for the error message.

    Returns
    -------
    position_time : np.ndarray or jax.Array, shape (n_time_position,)
        ``as_array(position_time)``.
    position : np.ndarray or jax.Array, shape (n_time_position, n_position_dims)
        ``as_array(position)``.

    Raises
    ------
    ValidationError
        If ``position_time`` or ``position`` is None.
    """
    if position_time is None or position is None:
        missing = [
            name
            for name, value in (
                ("position_time", position_time),
                ("position", position),
            )
            if value is None
        ]
        verb = "is" if len(missing) == 1 else "are"
        msg = (
            f"The {likelihood} reads position, but {' and '.join(missing)} {verb} None"
        )
        raise ValidationError(
            msg,
            hint="Pass the tracked position_time and position the local "
            "likelihood is evaluated at.",
        )
    return as_array(position_time), as_array(position)


def validate_population_lengths(unit_name: str, **populations: Sized | None) -> int:
    """Require parallel population collections to contain the same number of units.

    Likelihood predictors combine observed spike trains with fitted per-unit models.
    Plain ``zip`` silently truncates when one collection is shorter, producing a
    plausible likelihood from only part of the recorded population. Validate the
    collections once at the entry point and return the common length.

    Parameters
    ----------
    unit_name : str
        Human-readable population unit, such as ``"electrode"`` or ``"neuron"``.
    **populations
        Named sized collections that should be parallel. ``None`` marks an
        optional collection that was not supplied and is skipped.

    Returns
    -------
    n_units : int
        Shared collection length, or 0 when no collections are supplied.

    Raises
    ------
    ValidationError
        If the collection lengths differ.
    """
    lengths = {
        name: len(values) for name, values in populations.items() if values is not None
    }
    if not lengths:
        return 0
    expected = next(iter(lengths.values()))
    if any(length != expected for length in lengths.values()):
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"{unit_name} population lengths do not match"
        raise ValidationError(
            msg,
            expected=f"all per-{unit_name} collections to have length {expected}",
            got=details,
            hint=(
                f"Provide one entry in every collection for each {unit_name}; do not "
                "drop empty units."
            ),
        )
    return expected


def environment_grid_fingerprint(environment: Environment) -> dict[str, Any]:
    """Identify the fitted place grid that an encoding model's arrays describe.

    Encoding models store occupancy, rates and interior masks per bin of the
    grid the environment had at fit time, and keep a reference to the
    environment, which ``Environment.fit_place_grid`` refits in place. The
    fingerprint lets prediction detect that the grid has changed since.

    Parameters
    ----------
    environment : Environment
        A fitted environment.

    Returns
    -------
    environment_grid : dict
        ``centers_shape`` (tuple of int), ``n_interior_bins`` (int) and
        ``sha256`` (str), a SHA-256 digest of the bin centers, the bin edges of
        each dimension and the interior mask. It identifies bin positions and
        sizes as well as which bins are interior, in 64 characters rather than
        a copy of every bin, and holds only Python values, so it pickles with
        the encoding model.
    """
    centers_shape = tuple(int(n) for n in environment.centers_shape_)
    is_track_interior = np.asarray(environment.is_track_interior_, dtype=bool)
    digest = hashlib.sha256()
    for array in (
        np.asarray(environment.place_bin_centers_, dtype="<f8"),
        is_track_interior,
        *(np.asarray(edge, dtype="<f8") for edge in environment.edges_),
    ):
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "centers_shape": centers_shape,
        "n_interior_bins": int(np.sum(is_track_interior)),
        "sha256": digest.hexdigest(),
    }


def validate_encoding_contract(
    rate_units: str | None,
    environment: Environment,
    environment_grid: dict[str, Any] | None,
) -> None:
    """Require Hz rates fit on the environment's current grid.

    Every predict calls this first with its encoding model's contract keys.
    Direct predict calls accept a dict missing either key: each predict
    defaults ``rate_units`` to ``"Hz"`` and ``environment_grid`` to None, which
    skips the grid comparison. Callers may assemble the keyword arguments
    themselves, and a model fitted before the grid was recorded predicts as it
    did when fitted. Detectors are stricter about units: they reject a fitted
    entry without ``rate_units`` before decoding (``_validate_time_contract``).

    Parameters
    ----------
    rate_units : str | None
        The model's ``rate_units``; must be ``"Hz"``.
    environment : Environment
        The environment the encoding model references now.
    environment_grid : dict | None
        The model's ``environment_grid``, from
        :func:`environment_grid_fingerprint` at fit time, or None to skip the
        comparison.

    Raises
    ------
    ValidationError
        If the rates are not in Hz, or the environment's grid has changed
        since the fit.
    """
    if rate_units != "Hz":
        msg = "Encoding rates must be in Hz; refit legacy encoding models before decoding."
        raise ValidationError(
            msg,
            expected='rate_units="Hz"',
            got=f"rate_units={rate_units!r}",
        )
    if environment_grid is None:
        return
    current = environment_grid_fingerprint(environment)
    if current == environment_grid:
        return
    msg = (
        "Encoding model was fit on a different environment grid; its per-bin "
        "occupancy and rates no longer match the environment's bins."
    )
    raise ValidationError(
        msg,
        expected=(
            f"the fit-time grid: shape {environment_grid['centers_shape']}, "
            f"{environment_grid['n_interior_bins']} interior bins"
        ),
        got=(
            f"the current grid: shape {current['centers_shape']}, "
            f"{current['n_interior_bins']} interior bins, with different bin "
            "positions, sizes or interior mask"
        ),
        hint=(
            "The environment was refit (Environment.fit_place_grid) after this "
            "encoding model was fit, e.g. on other positions or by another "
            "detector sharing the same Environment, or was rebuilt differently, "
            "e.g. a saved model loaded on another platform or version. Refit the "
            "encoding model against the current environment, or give each "
            "fitted model its own Environment."
        ),
    )


def validate_spike_feature_pair(
    spike_times: np.ndarray | jax.Array,
    spike_features: np.ndarray | jax.Array,
    electrode: int,
) -> None:
    """Validate one electrode's parallel spike/mark arrays without copying them.

    Parameters
    ----------
    spike_times : np.ndarray or jax.Array, shape (n_spikes,)
        Spike times of the electrode.
    spike_features : np.ndarray or jax.Array, shape (n_spikes, n_features)
        Waveform features (marks) of the electrode, one row per spike.
    electrode : int
        Electrode index, used in the error message.

    Raises
    ------
    ValidationError
        If either array has the wrong rank or their row counts differ.
    """
    times_shape = np.shape(spike_times)
    features_shape = np.shape(spike_features)
    if len(times_shape) != 1:
        msg = f"spike_times for electrode {electrode} must be 1-D"
        raise ValidationError(
            msg,
            expected="shape (n_spikes,)",
            got=f"shape {times_shape}",
        )
    if len(features_shape) != 2:
        msg = f"spike_waveform_features for electrode {electrode} must be 2-D"
        raise ValidationError(
            msg,
            expected="shape (n_spikes, n_features)",
            got=f"shape {features_shape}",
        )
    if features_shape[0] != times_shape[0]:
        msg = f"spike times and waveform features disagree for electrode {electrode}"
        raise ValidationError(
            msg,
            expected=f"{times_shape[0]} waveform-feature rows",
            got=f"{features_shape[0]} rows",
            hint="Provide exactly one waveform-feature row for every spike time.",
        )


def validate_spike_feature_population(
    spike_times: Sequence[np.ndarray | jax.Array],
    spike_waveform_features: Sequence[np.ndarray | jax.Array],
) -> int:
    """Require clusterless spike times and features to describe the same electrodes.

    Checks the electrode count and each electrode's spike/feature row alignment
    before anything pairs the collections, so a mismatch cannot silently drop an
    electrode or misalign features with spike times.

    Parameters
    ----------
    spike_times : sequence of arrays, each shape (n_spikes,)
        Spike times for each electrode.
    spike_waveform_features : sequence of arrays, each shape (n_spikes, n_features)
        Waveform features (marks) for each electrode.

    Returns
    -------
    n_electrodes : int
        Number of electrodes.

    Raises
    ------
    ValidationError
        If the electrode counts differ or any electrode's arrays disagree.
    """
    n_electrodes = validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    for electrode, (times, features) in enumerate(
        zip(spike_times, spike_waveform_features, strict=True)
    ):
        validate_spike_feature_pair(times, features, electrode)
    return n_electrodes
