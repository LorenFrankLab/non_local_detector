"""Log-likelihood computation for a very low firing rate Poisson firing model.

This module provides a function to calculate the log-likelihood of observed spike
counts under a simple baseline model. This model is intended to quiescent
times when the population of neurons is not firing, or is firing at a very low rate.
Examples of this are times when the animal is immobile and the hippocampus has
burst like activity.

The primary function `predict_no_spike_log_likelihood` computes this value
for specified time bins based on the provided spike times and baseline rate.
It utilizes JAX for efficient computation.
"""

from typing import cast

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    SpikeTimeOrder,
    concatenate_row_blocks,
    poisson_row_log_likelihood,
    prepare_decode_rows,
    spike_counts_matrix,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    requires_time_edges,
)


def no_spike_time_bin_sizes(time_edges: np.ndarray) -> np.ndarray:
    """Compute the duration of every bin of the full decode grid.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges, in seconds.

    Returns
    -------
    durations : np.ndarray, shape (n_bins,)
        Duration of each decoding bin, in seconds.
    """
    return np.diff(time_edges)


@requires_time_edges
def predict_no_spike_log_likelihood(
    spike_times: list[list[float]],
    no_spike_rate: float | jax.Array = 1e-10,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    disable_progress_bar: bool = False,
    _time_bin_sizes: np.ndarray | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    r"""Return the log likelihood of low spike rate for each time bin.

    This function computes the log-likelihood under a Poisson model with
    very low firing rates, typically used during quiescent periods when
    neural activity is minimal or during immobility periods.

    Parameters
    ----------
    spike_times : list[list[float]]
        Nested list where each inner list contains spike times for one neuron.
        Length equals number of neurons in the population.
    no_spike_rate : float, default=1e-10
        Expected firing rate during no-spike periods in Hz. Should be very
        small to represent baseline/quiescent activity levels.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: the bin duration
        and the spike-to-row assignment are both taken from them, so the result
        equals the full-time result sliced by ``row_slice``.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Full decoding bin edges. The output has one row per bin before
        applying ``row_slice``.
    disable_progress_bar : bool, default=False
        Turn off the per-block progress bar. Detectors pass their
        ``disable_progress_bar`` algorithm parameter.
    _time_bin_sizes : np.ndarray | None, optional
        Internal precomputed ``no_spike_time_bin_sizes(time_edges)`` for these edges.
        Detector predictions prepare it once and reuse it across chunks. Direct
        callers can omit it; only the requested durations are computed here.
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, 1)
        Log-likelihood values for each requested time bin under the no-spike
        model. ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.

    Notes
    -----
    The model assumes Poisson firing with rate `no_spike_rate` scaled by
    the time bin duration. This is appropriate for modeling background
    activity during periods of behavioral quiescence such as slow-wave sleep
    or immobile periods when place cells show minimal spatial selectivity.

    The log-likelihood is computed as:

    .. math::
        \\log P(n|\\lambda) = n \\log(\\lambda \\Delta t) - \\lambda \\Delta t

    where n is the spike count, λ is the firing rate, and Δt is the bin duration.

    Bins are left-closed and right-open, and the final bin also holds a spike
    at ``time_edges[-1]``. ``Δt`` is each bin's actual duration.

    Examples
    --------
    >>> import numpy as np
    >>> time_edges = np.linspace(0, 10, 101)  # 100 bins
    >>> spike_times = [[] for _ in range(5)]  # 5 neurons, no spikes
    >>> log_lik = predict_no_spike_log_likelihood(spike_times, time_edges=time_edges)
    >>> log_lik.shape
    (100, 1)

    >>> # With some sparse spikes
    >>> spike_times = [[1.0, 5.0], [], [8.5], [], []]
    >>> log_lik = predict_no_spike_log_likelihood(
    ...     spike_times, time_edges=time_edges, no_spike_rate=1e-8
    ... )
    >>> log_lik.shape
    (100, 1)
    """
    if np.ndim(no_spike_rate) != 0 or (
        not isinstance(no_spike_rate, jax.core.Tracer)
        and (not np.isfinite(no_spike_rate) or no_spike_rate < 0)
    ):
        msg = "no_spike_rate must be a finite nonnegative scalar"
        raise ValidationError(msg)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    if _time_bin_sizes is None:
        durations = rows.durations()
    else:
        durations = _time_bin_sizes[rows.start : rows.stop]
    no_spike_rates = no_spike_rate * jnp.asarray(durations)

    def evaluate(start: int, stop: int) -> jax.Array:
        """Return the log likelihood of rows ``start`` to ``stop`` (shape (stop - start, 1))."""
        counts = spike_counts_matrix(
            cast("list[np.ndarray]", spike_times),
            rows.time_edges,
            "No Spike Likelihood",
            True,
            slice(start, stop),
            _spike_time_order=rows.spike_time_order,
        )
        return cast(
            jax.Array,
            poisson_row_log_likelihood(
                jnp.asarray(counts),
                no_spike_rates[start - rows.start : stop - rows.start, None],
            ),
        )

    return concatenate_row_blocks(
        rows.start,
        rows.stop,
        len(spike_times),
        evaluate,
        "No Spike Likelihood",
        disable_progress_bar,
    )[:, None]
