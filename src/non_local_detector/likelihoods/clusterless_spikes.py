"""Per-electrode accumulation of clusterless spike terms into the log likelihood.

Each clusterless prediction selects every electrode's decoding spikes in the
requested rows, checks their marks, forms their background intensity and adds
the algorithm's per-spike log intensities to the rows. This module holds that
loop; the algorithms supply the per-electrode term.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import jax
import numpy as np
from tqdm.autonotebook import tqdm

from non_local_detector.likelihoods.common import (
    DecodeRows,
    SpikeSelection,
    select_decode_marks,
    select_spikes_in_rows,
    sum_spikes_into_rows,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
)

# Internal module: no public API.
__all__: list[str] = []


_Model = TypeVar("_Model")


@dataclass(frozen=True)
class ElectrodeSpikes:
    """One electrode's decoding spikes in the requested rows.

    Attributes
    ----------
    electrode : int
        Electrode index.
    times : np.ndarray or jax.Array, shape (n_electrode_spikes,)
        The electrode's whole decoding spike train, in seconds;
        ``select_spike_rows(times, selection)`` gives the selected times.
    selection : SpikeSelection
        The spikes' rows, from ``select_spikes_in_rows``.
    marks : np.ndarray, shape (n_spikes, n_features)
        Their waveform features, checked finite, on the host.
    log_background : np.ndarray, shape (n_spikes,)
        Their log background intensity, ``log_background_floor + log q(mark)``.
    """

    electrode: int
    times: np.ndarray | jax.Array
    selection: SpikeSelection
    marks: np.ndarray
    log_background: np.ndarray

    @property
    def n_spikes(self) -> int:
        """Number of selected spikes, at least one."""
        return int(self.selection.bin_ind.shape[0])


def add_background_only(
    log_likelihood: jax.Array, spikes: ElectrodeSpikes
) -> jax.Array:
    """Add the spikes of an electrode explained by the background alone.

    A zero-rate electrode has no encoding spikes, so each observed spike adds
    its background intensity, the same at every position bin.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows,) or (n_rows, n_bins)
    spikes : ElectrodeSpikes

    Returns
    -------
    log_likelihood : jax.Array, same shape
    """
    contribution = sum_spikes_into_rows(spikes.log_background, spikes.selection)
    return log_likelihood + (
        contribution if log_likelihood.ndim == 1 else contribution[:, None]
    )


def add_electrode_spike_terms(
    log_likelihood: jax.Array,
    spike_times: Sequence[np.ndarray | jax.Array],
    spike_waveform_features: Sequence[np.ndarray | jax.Array],
    mark_reference: Sequence[MarkReference | None],
    electrode_models: Sequence[_Model],
    add_electrode: Callable[[jax.Array, ElectrodeSpikes, _Model], jax.Array],
    *,
    rows: DecodeRows,
    log_background_floor: float,
    desc: str,
    disable_progress_bar: bool,
) -> jax.Array:
    """Add every electrode's spike terms to the log likelihood of the rows.

    For each electrode in order: select its spikes in ``rows``, select and
    check their marks, skip the electrode if it has no spikes there, form the
    background intensity and call ``add_electrode``.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows,) or (n_rows, n_bins)
        The starting log likelihood, usually the expected-count term.
    spike_times : sequence of array, each shape (n_spikes,)
        Decoding spike times per electrode, in seconds.
    spike_waveform_features : sequence of array, each shape (n_spikes, n_features)
        Decoding waveform features per electrode, row-paired with the times.
    mark_reference : sequence of MarkReference or None
        Reference mark density per electrode (None: the bare background rate).
    electrode_models : sequence
        The algorithm's fitted terms per electrode, passed to ``add_electrode``.
    add_electrode : callable
        ``add_electrode(log_likelihood, spikes, model)`` returns
        ``log_likelihood`` plus the electrode's spike terms.
    rows : DecodeRows
        The prediction's decode grid, requested rows and spike preparation.
    log_background_floor : float
        Log background rate in the algorithm's intensity units.
    desc : str
        Progress bar label.
    disable_progress_bar : bool

    Returns
    -------
    log_likelihood : jax.Array, same shape as the input
    """
    for electrode, (times, features, reference, model) in enumerate(
        tqdm(
            zip(
                spike_times,
                spike_waveform_features,
                mark_reference,
                electrode_models,
                strict=True,
            ),
            desc=desc,
            unit="electrode",
            disable=disable_progress_bar,
            total=len(spike_times),
        )
    ):
        selection = select_spikes_in_rows(
            times,
            rows.start,
            rows.stop,
            time_edges=rows.time_edges,
            _spike_time_order=rows.spike_time_order,
        )
        # Checked even without spikes in the rows: the features must pair
        # with the spike times.
        marks = select_decode_marks(features, selection, electrode)
        if marks.shape[0] == 0:
            continue
        log_background = background_log_intensity(
            reference, marks, log_background_floor
        )
        log_likelihood = add_electrode(
            log_likelihood,
            ElectrodeSpikes(electrode, times, selection, marks, log_background),
            model,
        )
    return log_likelihood
