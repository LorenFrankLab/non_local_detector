"""
non_local_detector.bundle
=========================

* RecordingBundle —— immutable, native-rate archive.
* DecoderBatch     —— time-aligned slice at the decoder's Δt.

All *modalities* live in the open ``signals`` dictionary.
Common ones (`counts`, `lfp`, `calcium`) remain discoverable through
read-only properties for neuroscientist convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

Array = np.ndarray


@dataclass(frozen=True, slots=True)
class SpikeTrain:
    times_s: np.ndarray  # strictly increasing
    unit_id: int | str  # cluster or channel label
    channel_position: tuple[float, ...] | None = None  # (x, y, z) if known
    quality_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WaveformSeries:
    data: np.ndarray  # (n_spikes, n_channels, n_samples)
    channel_positions: np.ndarray | None = None  # (n_channels, ndim)
    channel_ids: tuple[int, ...] | None = None
    feature_names: tuple[str, ...] | None = None  # e.g. ("amp", "width")


# --------------------------------------------------------------------------- #
#  Helper: generic time-series container                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TimeSeries:
    data: Array  # shape (n_samples, …)
    sampling_rate_hz: float  # samples per second
    start_s: float = 0.0  # session time of first sample

    # OPTIONAL metadata  ──────────────────────────────────────────────
    channel_ids: tuple[int, ...] | None = None  # length = n_channels
    channel_positions: np.ndarray | None = None
    units: str | None = None  # "uV", "raw_adc", "deg"

    def __post_init__(self):
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive.")
        if self.data.ndim < 1:
            raise ValueError("TimeSeries.data must be at least 1-D.")


# --------------------------------------------------------------------------- #
#  Native-rate archive                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecordingBundle:
    """
    Immutable store for raw session data at native sampling rates.
    """

    # List of spike times for each neuron, in seconds.
    spike_times_s: Optional[List[SpikeTrain]] = None
    # List of spike waveforms, one for each neuron.
    spike_waveforms: Optional[List[WaveformSeries]] = None
    signals: Dict[str, TimeSeries] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ---- 1. waveforms without timestamps → fatal -----------------
        if self.spike_waveforms is not None and self.spike_times_s is None:
            raise ValueError(
                "spike_waveforms were provided but spike_times_s is None. "
                "Waveforms cannot be aligned without spike times."
            )

        # ---- 2. no timestamps → nothing else to check ----------------
        if self.spike_times_s is None:
            return  # (case 1 above)

        # ---- 3. timestamps present, waveforms missing → fine ---------
        if self.spike_waveforms is None:
            return  # (case 2 above)

        # ---- 4. both present → validate lengths & per-unit matches ---
        if len(self.spike_times_s) != len(self.spike_waveforms):
            raise ValueError(
                "Length mismatch between spike_times_s "
                f"({len(self.spike_times_s)}) and spike_waveforms "
                f"({len(self.spike_waveforms)})."
            )

        for idx, (t, w) in enumerate(zip(self.spike_times_s, self.spike_waveforms)):
            ntimes = t.times_s.size
            nwaves = w.data.shape[0]
            if ntimes != nwaves:
                raise ValueError(
                    f"Neuron {idx}: {ntimes} spike times but "
                    f"{nwaves} waveform rows."
                )


# --------------------------------------------------------------------------- #
#  Decoder-aligned batch                                                      #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DecoderBatch:
    """
    Time-aligned view at the decoder's bin width Δt (e.g. 50 Hz).
    Every array in ``signals`` **must** share the first-axis length ``n_time``.
    signals can be any time-series modality, such as:
    - `counts` (spike counts, shape (n_time,))
    - `lfp` (local field potential, shape (n_time, n_channels))
    - `calcium` (fluorescence, shape (n_time, n_channels))
    Optionally, it can also contain raw spike times and waveforms.

    Attributes
    ----------
    signals : Dict[str, Array]
        Dictionary of time-series arrays, where each key is a modality name
        and the value is the corresponding data array.
    bin_edges_s : Optional[Array]
        Optional array of bin edges in seconds, with length equal to `n_time + 1`.
        If provided, it defines the time intervals for each bin.
    spike_times_s : Optional[List[Array]]
        Optional list of spike times for each neuron, where each array
        corresponds to a different neuron and contains spike times in seconds.
    spike_waveforms : Optional[List[Array]]
        Optional list of spike waveforms, where each array corresponds to
        a different neuron and contains the waveforms of spikes.
    """

    signals: Dict[str, Array] = field(default_factory=dict)

    # Per-bin metadata
    bin_edges_s: Optional[Array] = None  # len = n_time + 1

    # Raw spikes
    spike_times_s: Optional[List[Array]] = None
    spike_waveforms: Optional[List[Array]] = None

    _n_time: int = field(init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Validation at construction                                        #
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        # 1) Neither signals nor spikes → error
        if not self.signals and self.spike_times_s is None:
            raise ValueError(
                "DecoderBatch must contain at least one signal or spike_times_s."
            )

        if self.signals:
            # a) Disallow any non-ndarray
            bad_type = [
                k for k, v in self.signals.items() if not isinstance(v, np.ndarray)
            ]
            if bad_type:
                raise TypeError(f"signals must be np.ndarray; offenders: {bad_type}")

            # b) Now each v is an ndarray → check dtype
            bad_dtype = [
                k for k, v in self.signals.items() if v.dtype.kind not in "fi?uO"
            ]
            if bad_dtype:
                raise TypeError(
                    f"signals must be numeric or bool; offenders: {bad_dtype}"
                )

            # c) Ensure all signals share the same n_time
            lengths = {k: v.shape[0] for k, v in self.signals.items()}
            if len(set(lengths.values())) != 1:
                detail = ", ".join(f"{k}={n}" for k, n in lengths.items())
                raise ValueError(f"Time-axis mismatch among signals: {detail}")
            self._n_time = next(iter(lengths.values()))

        else:
            # No signals, but spike_times_s is not None (otherwise we would have raised above).
            # We allow “spikes + bin_edges only” → set n_time from bin_edges_s.
            if self.spike_times_s is not None and self.bin_edges_s is not None:
                self._n_time = len(self.bin_edges_s) - 1
            else:
                # If spike_times_s exists but bin_edges_s is missing, we cannot infer n_time.
                raise ValueError(
                    "DecoderBatch must contain either signals or both spike_times_s and bin_edges_s."
                )

        # Final check on bin_edges_s length (if provided)
        if self.bin_edges_s is not None:
            if len(self.bin_edges_s) != self._n_time + 1:
                raise ValueError("len(bin_edges_s) must equal n_time + 1.")

    # ------------------------------------------------------------------ #
    #  Convenience read-only attributes                                  #
    # ------------------------------------------------------------------ #
    @property
    def counts(self) -> Optional[Array]:
        return self.signals.get("counts")

    @property
    def lfp(self) -> Optional[Array]:
        return self.signals.get("lfp")

    @property
    def calcium(self) -> Optional[Array]:
        return self.signals.get("calcium")

    @property
    def n_time(self) -> int:
        return self._n_time

    # ------------------------------------------------------------------ #
    #  Shallow slice helper                                              #
    # ------------------------------------------------------------------ #
    def slice(
        self, start: int, stop: int, *, slice_spikes: bool = True
    ) -> "DecoderBatch":
        new_signals = {k: v[start:stop] for k, v in self.signals.items()}

        st, wf = self.spike_times_s, self.spike_waveforms
        if slice_spikes and st is not None and self.bin_edges_s is not None:
            t0, t1 = self.bin_edges_s[start], self.bin_edges_s[stop]
            st_new = []
            wf_new = [] if wf is not None else None

            for i, times in enumerate(st):
                mask = (times >= t0) & (times < t1)
                st_new.append(times[mask])
                if wf is not None:
                    wf_new.append(wf[i][mask])

            st, wf = st_new, (wf_new if wf is not None else None)

        new_edges = (
            None if self.bin_edges_s is None else self.bin_edges_s[start : stop + 1]
        )
        return DecoderBatch(
            signals=new_signals,
            bin_edges_s=new_edges,
            spike_times_s=st,
            spike_waveforms=wf,
        )

    def select_signals(self, keys: Iterable[str]) -> "DecoderBatch":
        return DecoderBatch(
            signals={k: self.signals[k] for k in keys},
            bin_edges_s=self.bin_edges_s,
            spike_times_s=self.spike_times_s,
            spike_waveforms=self.spike_waveforms,
        )

    def select_spikes(self, keys: Iterable[int]) -> "DecoderBatch":
        if self.spike_times_s is None:
            raise ValueError("No spike_times_s available in this DecoderBatch.")

        orig_st = self.spike_times_s
        orig_wf = self.spike_waveforms

        new_times = []
        for i in keys:
            t = orig_st[i]
            if hasattr(t, "times_s"):
                new_times.append(t.times_s)
            else:
                new_times.append(t)

        if orig_wf is not None:
            new_wf = []
            for i in keys:
                w = orig_wf[i]
                if hasattr(w, "data"):
                    new_wf.append(w.data)
                else:
                    new_wf.append(w)
        else:
            new_wf = None

        return DecoderBatch(
            signals=self.signals,
            bin_edges_s=self.bin_edges_s,
            spike_times_s=new_times,
            spike_waveforms=new_wf,
        )

    # ------------------------------------------------------------------ #
    #  Pretty-print                                                      #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        sigs = ", ".join(self.signals.keys())
        return f"<DecoderBatch n_time={self.n_time} signals=[{sigs}]>"


# --------------------------------------------------------------------------- #
#  Public helper: bundle-level validation                                     #
# --------------------------------------------------------------------------- #
def validate_sources(batch: "DecoderBatch", observation_models: list) -> None:
    """
    Check that every ObservationModel's ``required_sources`` are
    present in *this* batch.

    Usage
    -----
    >>> validate_sources(batch, detector.observation_models)
    """
    missing: set[str] = set()
    for model in observation_models:
        for src in getattr(model, "required_sources", ()):
            # OK if it’s in signals dict
            if src in batch.signals:
                continue
            # OK if there is an attribute with that name AND it’s not None
            if hasattr(batch, src) and getattr(batch, src) is not None:
                continue
            # Otherwise, it’s missing
            missing.add(src)
    if missing:
        raise ValueError(
            "DecoderBatch missing required fields:\n  • "
            + "\n  • ".join(sorted(missing))
        )
