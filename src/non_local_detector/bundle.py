"""
non_local_detector.bundle
=========================

`DataBundle` is a **typed, immutable container** that holds every raw data
stream your detector might need:

* spike counts or spike times (clustered or unclustered),
* spike waveforms,
* continuous traces (LFP, Ca²⁺, etc.),
* behavioural covariates (speed, head-direction, theta phase, …).

It guarantees that **all time-series arrays share the same first-dimension
length**, so downstream algorithms can trust `bundle.n_time`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

Array = np.ndarray
SpikeTimes = List[Array]  # one array per neuron
SpikeWaveforms = List[Array]  # shape[i] = (n_spikes_i, n_features)


def _common_length(arrays: Sequence[Array]) -> Optional[int]:
    """Return common length along axis-0, or ``None`` if no arrays given."""
    lengths = {arr.shape[0] for arr in arrays if arr is not None}
    if not lengths:
        return None
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent time lengths found: {lengths}")
    return lengths.pop()


@dataclass(slots=True)
class DataBundle:
    """
    Typed container for *one contiguous time chunk*.

    All **time-based** arrays must have equal length along axis-0.
    """

    # ---- spike data ----------------------------------------------------
    counts: Optional[Array] = None  # (n_time,) or (n_time, n_cells)
    spikes: Optional[SpikeTimes] = None  # list of spike-time arrays
    waveforms: Optional[SpikeWaveforms] = None  # list aligned with `spikes`

    # ---- continuous recordings ----------------------------------------
    lfp: Optional[Array] = None  # (n_time,) or (n_time, n_ch)
    calcium: Optional[Array] = None  # same

    # ---- behavioural / covariate data ---------------------------------
    position: Optional[Array] = None  # (n_time, dims)
    covariates: Optional[Dict[str, Array]] = None

    # internal cache (filled in __post_init__)
    _n_time: int = field(init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Validation                                                        #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        time_axes: list[Array] = []

        # Collect all arrays that imply a time axis
        for arr in (
            self.counts,
            self.lfp,
            self.calcium,
            self.position,
        ):
            if arr is not None:
                time_axes.append(arr)

        if self.covariates:
            time_axes.extend(self.covariates.values())

        # Derive common length from arrays, else from spikes / waveforms
        n_time_arrays = _common_length(time_axes)
        if n_time_arrays is None:
            # Use spike-count bins length if present
            if self.spikes is not None or self.waveforms is not None:
                raise ValueError(
                    "Provide at least one time-binned array (e.g. `counts`) "
                    "so `n_time` can be inferred from spike data."
                )
            raise ValueError("Cannot infer n_time — no time-indexed arrays provided.")

        self._n_time = n_time_arrays

    # ------------------------------------------------------------------ #
    #  Convenience properties                                            #
    # ------------------------------------------------------------------ #
    @property
    def n_time(self) -> int:  # number of time steps
        return self._n_time

    # Aliased for coherence with earlier docs
    T = n_time

    # ------------------------------------------------------------------ #
    #  Utility methods                                                   #
    # ------------------------------------------------------------------ #
    def slice(self, start: int, stop: int) -> "DataBundle":
        """
        Return a **shallow** copy containing [`start:stop`] along the time axis.

        All non-time-indexed attributes (e.g., spike times) are shared
        unchanged because their slicing semantics are experiment-specific.
        """

        def _slice_or_same(arr: Optional[Array]) -> Optional[Array]:
            return None if arr is None else arr[start:stop]

        return DataBundle(
            counts=_slice_or_same(self.counts),
            spikes=self.spikes,  # left untouched
            waveforms=self.waveforms,
            lfp=_slice_or_same(self.lfp),
            calcium=_slice_or_same(self.calcium),
            position=_slice_or_same(self.position),
            covariates=(
                {k: v[start:stop] for k, v in self.covariates.items()}
                if self.covariates
                else None
            ),
        )

    # Simple pretty-print
    def __repr__(self) -> str:  # pragma: no cover
        attrs = ", ".join(
            f"{name}={getattr(self, name).__class__.__name__}"
            for name in self.__dataclass_fields__  # type: ignore
            if getattr(self, name) is not None and not name.startswith("_")
        )
        return f"DataBundle(n_time={self.n_time}, {attrs})"
