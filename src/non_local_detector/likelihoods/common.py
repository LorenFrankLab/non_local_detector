from collections.abc import Sized
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from jax.nn import logsumexp
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError

EPS = 1e-15
LOG_EPS = np.log(EPS)


def as_std_array(std: "jnp.ndarray | float | int", n_dims: int) -> jnp.ndarray:
    """Broadcast a scalar kernel bandwidth to a per-dimension array.

    A single ``jnp.ndim`` check covers Python ``int``/``float``, NumPy scalars
    (including ``np.float32``), 0-d arrays, and JAX scalars uniformly, avoiding
    the ``isinstance(std, int | float)`` pattern that silently misses
    ``np.float32`` and 0-d arrays. Array-valued ``std`` (``ndim >= 1``) is
    returned via ``jnp.asarray`` (a no-op for JAX arrays; lists/NumPy arrays are
    converted to a JAX array).

    Parameters
    ----------
    std : jnp.ndarray or float or int
        Scalar (broadcast to ``n_dims``) or per-dimension standard deviation(s).
    n_dims : int
        Number of dimensions to broadcast a scalar to.

    Returns
    -------
    std_array : jnp.ndarray
        Shape ``(n_dims,)`` for scalar input; the original array otherwise.
    """
    if jnp.ndim(std) == 0:
        return jnp.full((n_dims,), std)
    return jnp.asarray(std)


def validate_weights(weights: np.ndarray, n_time: int) -> np.ndarray:
    """Validate per-sample weights shared by the sorted-spikes and clusterless fits.

    Weights feed both the occupancy field and the spike counts, so an invalid
    array corrupts the whole encoding silently; validate once at the fit entry.

    Parameters
    ----------
    weights : np.ndarray, shape (n_time,)
        Per-sample weights (e.g. posterior state probabilities during EM).
    n_time : int
        Expected number of samples (``position.shape[0]``).

    Returns
    -------
    weights : np.ndarray, shape (n_time,)
        The weights as a float array.

    Raises
    ------
    ValidationError
        If ``weights`` is not 1-D of length ``n_time``, or contains non-finite or
        negative values.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.shape != (n_time,):
        raise ValidationError(
            f"weights must have shape ({n_time},), got {weights.shape}"
        )

    if not np.all(np.isfinite(weights)):
        raise ValidationError("weights must contain only finite values")

    if np.any(weights < 0):
        raise ValidationError("weights must be non-negative")

    return weights


def validate_finite(array: np.ndarray | jnp.ndarray, name: str) -> None:
    """Raise if ``array`` contains any non-finite (NaN/inf) value.

    Shared fit-input finiteness check for the encoding fits: a non-finite
    position or spike waveform feature is invalid encoding data that would
    otherwise propagate into the density estimate as a silent NaN. Fail fast at
    the fit entry, consistent with ``validate_weights`` and the GMM/MRF fits
    (which already reject non-finite inputs).

    Parameters
    ----------
    array : np.ndarray | jnp.ndarray
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
    if isinstance(array, jnp.ndarray):
        all_finite = bool(jnp.all(jnp.isfinite(array)))
    else:
        all_finite = bool(np.all(np.isfinite(np.asarray(array))))
    if not all_finite:
        raise ValidationError(f"{name} must contain only finite values")


def validate_population_lengths(unit_name: str, **populations: Sized) -> int:
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
        Named sized collections that should be parallel.

    Returns
    -------
    n_units : int
        Shared collection length, or 0 when no collections are supplied.

    Raises
    ------
    ValidationError
        If the collection lengths differ.
    """
    if not populations:
        return 0

    lengths = {name: len(values) for name, values in populations.items()}
    expected = next(iter(lengths.values()))
    if any(length != expected for length in lengths.values()):
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValidationError(
            f"{unit_name} population lengths do not match",
            expected=f"all per-{unit_name} collections to have length {expected}",
            got=details,
            hint=(
                f"Provide one entry in every collection for each {unit_name}; do not "
                "drop empty units."
            ),
        )
    return expected


def interpolate_weights_at_spike_times(
    spike_times: np.ndarray, position_time: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Per-spike weights: the per-sample ``weights`` linearly interpolated onto times.

    Spike times are assumed already clipped to ``[position_time[0], position_time[-1]]``,
    so 1-D ``np.interp`` matches ``interpn`` without building an interpolator (and does
    not extrapolate). Shared by the clusterless and sorted-spikes encoding fits.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    position_time : np.ndarray, shape (n_time_position,)
    weights : np.ndarray, shape (n_time_position,)

    Returns
    -------
    spike_weights : np.ndarray, shape (n_spikes,)
    """
    return np.interp(
        np.asarray(spike_times), np.asarray(position_time), np.asarray(weights)
    )


def weighted_mean_rate(spike_weights: np.ndarray, weight_sum: float) -> float:
    """Weighted mean firing rate: weighted spike count / weighted occupancy time.

    Returns 0.0 when ``weight_sum`` is not positive (no effective training data), so a
    zero-weight electrode/group contributes nothing rather than dividing by zero.

    Parameters
    ----------
    spike_weights : np.ndarray, shape (n_spikes,)
        Per-spike weights (see :func:`interpolate_weights_at_spike_times`).
    weight_sum : float
        Sum of the per-sample weights (the weighted occupancy time).

    Returns
    -------
    mean_rate : float
    """
    return float(np.sum(spike_weights) / weight_sum) if weight_sum > 0 else 0.0


def get_position_at_time(
    time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: jnp.ndarray,
    env: Environment | None = None,
) -> np.ndarray:
    """Get the position at the time of each spike.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
    position : jnp.ndarray, shape (n_time_position, n_dims_position)
    spike_times : jnp.ndarray, shape (n_spikes,)
    env : Environment | None, optional
        The spatial environment, by default None

    Returns
    -------
    position_at_spike_times : np.ndarray, shape (n_spikes, n_dims_position)
    """
    position_at_spike_times = scipy.interpolate.interpn(
        (time,), position, spike_times, bounds_error=False, fill_value=None
    )
    if env is not None and env.track_graph is not None:
        if position_at_spike_times.shape[0] > 0:
            position_at_spike_times = get_linearized_position(
                position_at_spike_times,
                env.track_graph,
                edge_order=env.edge_order,
                edge_spacing=env.edge_spacing,
            ).linear_position.to_numpy()[:, None]
        else:
            position_at_spike_times = np.array([])[:, None]

    return position_at_spike_times


def get_spike_time_bin_ind(spike_times: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Get the index of the time bin for each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time : np.ndarray, shape (n_time_bins,)
        Bin edges.

    Returns
    -------
    ind : np.ndarray, shape (n_spikes,)
    """
    return np.digitize(spike_times, time[1:-1])


def resolve_row_slice(row_slice: slice | None, n_time: int) -> tuple[int, int]:
    """Normalize a likelihood row request against the full decoding timeline.

    A backend's ``row_slice`` selects which rows of the FULL-time likelihood to
    return; ``time`` always stays the full decoding timeline so that spike-to-row
    ownership is independent of how the rows were chunked.

    Parameters
    ----------
    row_slice : slice | None
        Contiguous (unit-step) range of output rows, or None for all rows.
    n_time : int
        Length of the full decoding ``time`` array.

    Returns
    -------
    row_start : int
    row_stop : int
        Half-open row range with ``0 <= row_start <= row_stop <= n_time``.
    """
    if row_slice is None:
        return 0, n_time
    if row_slice.step not in (None, 1):
        raise ValidationError(
            "row_slice must select a contiguous range of rows",
            expected="a slice with step None or 1",
            got=f"step={row_slice.step}",
            hint="Chunked prediction only requests contiguous row ranges.",
            example="    predict_..._log_likelihood(time, ..., row_slice=slice(0, 100))",
        )
    row_start, row_stop, _ = row_slice.indices(n_time)
    return row_start, max(row_start, row_stop)


def select_spikes_in_rows(
    spike_times: np.ndarray,
    time: np.ndarray,
    row_start: int,
    row_stop: int,
) -> tuple[slice | np.ndarray, np.ndarray]:
    """Select the spikes owned by the global rows ``[row_start, row_stop)``.

    Row ownership is the unchunked convention evaluated on the full timeline: a
    spike is in range iff ``time[0] <= t <= time[-1]`` and it belongs to row
    ``np.digitize(t, time[1:-1])``. Only rows ``0 .. max(n_time - 2, 0)`` can own
    a spike, so the final row of a multi-row timeline always stays empty.

    Because ``np.digitize(t, time[1:-1])`` is ``searchsorted(time[1:-1], t,
    "right")``, owning row ``>= row_start`` is exactly ``t >= time[row_start]``
    and owning row ``< row_stop`` is exactly ``t < time[row_stop]`` (inclusive of
    ``time[-1]`` when the range reaches the last owning row). The selection is
    therefore a contiguous range in time, found with ``np.searchsorted`` when the
    spike times are ascending; only the selected subset is digitized, so no spike
    is re-binned for every chunk.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Decoding spike times for one neuron/electrode.
    time : np.ndarray, shape (n_time,)
        FULL decoding timeline (not the chunk's rows).
    row_start : int
    row_stop : int
        Half-open range of requested rows, as returned by ``resolve_row_slice``.

    Returns
    -------
    indexer : slice | np.ndarray
        Index into ``spike_times`` -- and, by the identical index, into any
        per-spike array such as waveform features -- selecting the owned spikes
        in their original order. A ``slice`` when the spike times are verified
        ascending, otherwise a boolean mask.
    bin_ind : np.ndarray, shape (n_selected,)
        Row of each selected spike, LOCAL to the requested range
        (``global_row - row_start``), for ``num_segments = row_stop - row_start``.
    """
    time = np.asarray(time)
    spike_times = np.asarray(spike_times)
    n_time = time.shape[0]
    # Rows that can own a spike. A length-1 timeline owns only t == time[0].
    n_owning_rows = max(n_time - 1, 1)

    if spike_times.size == 0 or row_start >= min(row_stop, n_owning_rows):
        return slice(0, 0), np.zeros((0,), dtype=int)

    lower = time[row_start]
    # Reaching the last owning row extends the range to time[-1] inclusive,
    # matching the unchunked ``spike_times <= time[-1]`` clip.
    upper_is_inclusive = row_stop >= n_owning_rows
    upper = time[n_time - 1] if upper_is_inclusive else time[row_stop]

    # Establish, never assume, the ordering the range lookup needs.
    is_ascending = spike_times.size < 2 or bool(
        np.all(spike_times[1:] >= spike_times[:-1])
    )
    if is_ascending:
        start = int(np.searchsorted(spike_times, lower, side="left"))
        stop = int(
            np.searchsorted(
                spike_times, upper, side="right" if upper_is_inclusive else "left"
            )
        )
        indexer: slice | np.ndarray = slice(start, max(start, stop))
        selected = spike_times[indexer]
    else:
        in_rows = spike_times >= lower
        in_rows &= spike_times <= upper if upper_is_inclusive else spike_times < upper
        indexer = in_rows
        selected = spike_times[in_rows]

    return indexer, np.digitize(selected, time[1:-1]) - row_start


@jax.jit
def log_gaussian_pdf(
    x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log of the Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_samples,)
    """
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def gaussian_pdf(x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    pdf : jnp.ndarray, shape (n_samples,)
    """
    return jnp.exp(log_gaussian_pdf(x, mean, sigma))


def _log_kernel_matrix(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Log Gaussian kernel matrix summed over dimensions.

    Shared by :func:`kde` and :func:`log_kde` so both build the kernel the same
    way: accumulate per-dimension log densities, defer the single ``exp`` (or
    ``logsumexp``) to the caller.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)

    Returns
    -------
    log_kernel : jnp.ndarray, shape (n_samples, n_eval_points)
        ``log_kernel[i, j] = sum_d log N(eval_points[j, d] | samples[i, d], std[d])``.
    """
    log_kernel = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=True):
        log_kernel += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),
            jnp.expand_dims(dim_samp, axis=1),
            dim_std,
        )
    return log_kernel


@jax.jit
def kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Kernel density estimation.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    weights : jnp.ndarray, shape (n_samples,)
        Weights for each sample.

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)

    Notes
    -----
    The kernel is accumulated in log-space and exponentiated once, so per-sample
    densities only underflow to 0 in float32 at the final value (where 0 is
    typically the correct result). :func:`log_kde` (or
    :meth:`KDEModel.predict_log`) remains the fully underflow-safe path when the
    log-density itself is needed.
    """
    distance = jnp.exp(_log_kernel_matrix(eval_points, samples, std))
    # Double-where pattern: substitute safe denominator, then select result.
    # This avoids NaN in both forward pass and gradients.
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    return jnp.where(weight_sum > 0, (weights @ distance) / safe_weight_sum, 0.0)


def block_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Kernel density estimation split into blocks.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Size of blocks to do computation over, by default 100
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)

    Notes
    -----
    Wraps :func:`kde`, whose underflow is deferred to the final density value;
    prefer :func:`block_log_kde` when the log-density itself is needed.
    """
    n_eval_points = eval_points.shape[0]

    if n_eval_points == 0:
        return jnp.zeros((n_eval_points,))

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block densities and concatenate once. The blocks tile
    # [0, n_eval_points) exactly, so this is identical to filling a preallocated
    # array but avoids the O(n_eval * n_blocks) copies of a per-block
    # dynamic_update_slice.
    blocks = [
        kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval_points, block_size)
    ]
    return jnp.concatenate(blocks)


@jax.jit
def log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log kernel density estimate:
        log p(x) = logsumexp_i [ log w_i + sum_d log N(x_d | s_{i,d}, std_d) ] - logsumexp_i [log w_i]
    Shapes:
      eval_points: (n_eval, n_dims)
      samples:     (n_samp, n_dims)
      std:         (n_dims,)
      weights:     (n_samp,)
    Returns: (n_eval,)
    """
    if eval_points.ndim == 1:
        eval_points = jnp.expand_dims(eval_points, axis=1)

    # log-kernel matrix K_log with shape (n_samp, n_eval)
    K_log = _log_kernel_matrix(eval_points, samples, std)

    # True log-weight: a zero weight (log = -inf) drops the sample from both the
    # numerator and the denominator, exactly like the linear ``kde``. ``safe_log``
    # would floor a zero weight to LOG_EPS and leak that sample's kernel back in
    # (visible at an eval point near a zero-weight sample but far from the rest).
    safe_weights = jnp.where(weights > 0, weights, 1.0)
    log_w = jnp.where(weights > 0, jnp.log(safe_weights), -jnp.inf)  # (n_samp,)
    log_num = logsumexp(log_w[:, None] + K_log, axis=0)  # (n_eval,)
    log_den = logsumexp(log_w)  # scalar; -inf only when every weight is 0
    # All-zero weights -> no support anywhere. Return LOG_EPS (matching the linear
    # kde's 0.0, which callers floor) instead of NaN from -inf - (-inf).
    return jnp.where(jnp.isneginf(log_den), LOG_EPS, log_num - log_den)


def block_log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Log kernel density estimation split into blocks over eval points.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Size of blocks to do computation over, by default 100
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    log_density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    n_eval = eval_points.shape[0]

    if n_eval == 0:
        return jnp.full((n_eval,), LOG_EPS)

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block log-densities and concatenate once (see block_kde).
    blocks = [
        log_kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval, block_size)
    ]
    return jnp.concatenate(blocks)


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: int | None = None
    samples_: jnp.ndarray | None = field(init=False, default=None)
    weights_: jnp.ndarray | None = field(init=False, default=None)

    def fit(
        self, samples: jnp.ndarray, weights: jnp.ndarray | None = None
    ) -> "KDEModel":
        """Fit the model.

        Parameters
        ----------
        samples : jnp.ndarray, shape (n_samples, n_dims)
            Training samples.

        Returns
        -------
        self : KDEModel
        """
        samples = jnp.asarray(samples)
        if samples.ndim == 1:
            samples = jnp.expand_dims(samples, axis=1)
        self.samples_ = samples
        if weights is None:
            self.weights_ = jnp.ones((samples.shape[0],))
        else:
            self.weights_ = jnp.asarray(weights)

        return self

    def predict(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Predict the density at the evaluation points.

        Parameters
        ----------
        eval_points : jnp.ndarray, shape (n_eval_points, n_dims)

        Returns
        -------
        density : jnp.ndarray, shape (n_eval_points,)
        """
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size, self.weights_)

    def predict_log(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density version of predict(). Same inputs, returns log p(eval_points).
        """
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )
        return block_log_kde(eval_points, self.samples_, std, block_size, self.weights_)


def select_spike_rows(
    array: "np.ndarray | jnp.ndarray", indexer: "slice | np.ndarray"
) -> np.ndarray:
    """Take the rows an indexer names without materializing the whole array.

    ``select_spikes_in_rows`` returns one indexer per unit/electrode, applied to
    the spike times and to the per-spike waveform features. Those arrays are
    recording-length; the selection is one chunk's worth. Converting the array
    before selecting therefore pays for the whole recording on every chunk call.
    For a ``jax.Array`` it is worse than a copy: ``np.asarray`` gathers the array
    to the host **and caches that host copy on the array object**
    (``jax.Array._npy_value``), so the memory is retained for as long as the
    caller holds its inputs.

    Indexing on device avoids both. It is not always available: under a mesh with
    ``AxisType.Explicit`` axes JAX refuses a gather whose output sharding it
    cannot infer, so that case falls back to gather-then-slice — the behaviour
    every backend had before, correct but paying the host copy.

    Parameters
    ----------
    array : np.ndarray or jnp.ndarray, shape (n_spikes, ...)
        Per-spike array for one unit/electrode.
    indexer : slice or np.ndarray
        The indexer from ``select_spikes_in_rows`` (a ``slice`` for ascending
        spike times, otherwise a boolean mask).

    Returns
    -------
    selected : np.ndarray, shape (n_selected, ...)
        The selected rows, always on the host. A selection that kept a sharded
        input's sharding would carry it into the jitted kernels downstream, which
        also take single-device encoding-model arrays -- JAX rejects that mix.
        The selection is request-sized, so this costs the chunk, not the
        recording; consumers that want a device array convert it themselves.

    Raises
    ------
    ValidationError
        If the indexer does not address this array: a boolean mask of a different
        length, or a slice reaching past the end. ``jnp.take`` would otherwise
        silently NaN-fill out-of-range rows (its default ``mode='fill'``), where
        NumPy raises -- a silent-NaN path into the likelihood.
    """
    n_rows = array.shape[0]
    if isinstance(indexer, slice):
        stop = indexer.stop
        if stop is not None and stop > n_rows:
            raise ValidationError(
                "spike selection reaches past the end of a per-spike array",
                expected=f"at least {stop} rows",
                got=f"{n_rows} rows",
                hint="Spike times and waveform features must be paired row for row.",
            )
    elif indexer.shape[0] != n_rows:
        raise ValidationError(
            "spike selection does not match the length of a per-spike array",
            expected=f"{indexer.shape[0]} rows (one per spike time)",
            got=f"{n_rows} rows",
            hint="Spike times and waveform features must be paired row for row.",
        )

    if not isinstance(array, jax.Array):
        return array[indexer]

    try:
        if isinstance(indexer, slice):
            selected = array[indexer]
        else:
            # A boolean mask becomes a gather of integer positions: ``take`` is
            # shardable, while boolean indexing needs the mask's popcount and
            # would force a host sync. The length check above is what keeps
            # ``take``'s default ``mode='fill'`` from turning a mismatch into
            # NaN rows.
            selected = jnp.take(array, jnp.asarray(np.flatnonzero(indexer)), axis=0)
    except Exception:
        # jax raises a private ShardingTypeError (jax._src.core, no public base
        # class in jax 0.9) when the operand is explicitly sharded and the output
        # sharding is ambiguous. Gathering first always works and is what this
        # code did before; it just costs the host copy this function avoids.
        return np.asarray(array)[indexer]

    # See Returns: the selection comes back on the host so the jitted kernels
    # downstream all see one device ("Received incompatible devices for jitted
    # computation" otherwise).
    return np.asarray(selected)


def get_spikecount_per_time_bin(
    spike_times: np.ndarray,
    time: np.ndarray,
    row_slice: slice | None = None,
) -> np.ndarray:
    """Get the number of spikes in each requested time bin.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time : np.ndarray, shape (n_time,)
        FULL decoding timeline, which defines the bins a spike belongs to.
    row_slice : slice | None, optional
        Contiguous range of rows to count, by default None (all rows). Counting
        rows ``[a, b)`` of the full timeline gives the same values as counting
        all rows and slicing ``[a:b]``, so concatenating the chunks of a row
        partition reproduces the full-time counts exactly.

    Returns
    -------
    count : np.ndarray, shape (n_rows,)
        ``n_rows`` is ``n_time`` by default, else the length of ``row_slice``.
    """
    row_start, row_stop = resolve_row_slice(row_slice, time.shape[0])
    _, bin_ind = select_spikes_in_rows(spike_times, time, row_start, row_stop)
    return np.bincount(bin_ind, minlength=row_stop - row_start)


def safe_divide(numerator, denominator, eps=EPS, condition=None):
    """Safely divide two arrays, avoiding division by zero.

    Parameters
    ----------
    numerator : jnp.ndarray
        Numerator array of any shape.
    denominator : jnp.ndarray
        Denominator array, must be broadcastable with numerator.
    eps : float, optional
        Small value to avoid division by zero, by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the division, by default None.
        If None, condition is computed as abs(denominator) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Result of safe division with same shape as broadcast of inputs.
        Where condition is True, returns eps instead of dividing.
    """
    if condition is None:
        condition = jnp.abs(denominator) < eps

    # Double-where: substitute safe denominator first, then select result.
    # This avoids NaN in both forward pass and gradients.
    safe_denominator = jnp.where(condition, 1.0, denominator)
    return jnp.where(condition, eps, numerator / safe_denominator)


def safe_log(x, eps=EPS, condition=None):
    """Safely compute the logarithm of an array, avoiding log(0).

    Parameters
    ----------
    x : jnp.ndarray
        Input array of any shape.
    eps : float, optional
        Small value to avoid log(0), by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the logarithm, by default None.
        If None, condition is computed as abs(x) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Logarithm of input array with same shape as x.
        Where condition is True, returns log(eps) instead of log(0).
    """
    if condition is None:
        condition = jnp.abs(x) < eps

    return jnp.log(jnp.where(condition, eps, x))
