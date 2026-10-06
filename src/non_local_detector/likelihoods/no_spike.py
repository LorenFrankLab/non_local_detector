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

import jax.numpy as jnp
import jax.scipy
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.likelihoods.common import (
    _SpikeTimeOrder,
    get_spikecount_per_time_bin,
    resolve_row_slice,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)


def no_spike_time_bin_sizes(time_edges: np.ndarray) -> np.ndarray:
    """Durations in seconds for every bin of the full decode grid."""
    return np.diff(time_edges)


@requires_time_edges
def predict_no_spike_log_likelihood(
    spike_times: list[list[float]],
    no_spike_rate: float = 1e-10,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    _time_bin_sizes: np.ndarray | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Return the log likelihood of low spike rate for each time bin.

    This function computes the log-likelihood under a Poisson model with
    very low firing rates, typically used during quiescent periods when
    neural activity is minimal or during immobility periods.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Full decoding bin edges. The output has one row per bin before
        applying ``row_slice``.
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
    _time_bin_sizes : np.ndarray | None, optional
        Internal precomputed ``no_spike_time_bin_sizes(time_edges)`` for these edges.
        Detector predictions prepare it once and reuse it across chunks. Direct
        callers can omit it; only the requested durations are computed here.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1)
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
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    if _time_bin_sizes is None:
        durations = _time_grid.durations(row_start, row_stop)
    else:
        durations = _time_bin_sizes[row_start:row_stop]
    no_spike_rates = no_spike_rate * jnp.asarray(durations)
    no_spike_log_likelihood = jnp.zeros((row_stop - row_start,))

    for neuron_spike_times in tqdm(
        spike_times, unit="cell", desc="No Spike Likelihood"
    ):
        no_spike_log_likelihood += (
            jax.scipy.special.xlogy(
                get_spikecount_per_time_bin(
                    neuron_spike_times,
                    time_edges=time_edges,
                    row_slice=row_slice,
                    _spike_time_order=_spike_time_order,
                ),
                no_spike_rates,
            )
            - no_spike_rates
        )

    return no_spike_log_likelihood[:, None]
