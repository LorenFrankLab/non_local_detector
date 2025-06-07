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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

Array = np.ndarray


def _is_array_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


@dataclass(frozen=True, slots=True)
class SpikeTrain:
    times_s: Sequence[float]  # strictly increasing
    unit_id: int | str  # cluster or channel label
    channel_position: tuple[float, ...] | None = None  # (x, y, z) if known
    quality_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        times = np.asarray(self.times_s, dtype=float)
        if times.ndim != 1:
            raise ValueError(f"times_s must be 1D; got shape {times.shape}")
        if not np.all(np.diff(times) >= 0):
            raise ValueError("times_s must be strictly increasing.")
        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "times_s", times)

    def __len__(self) -> int:
        return self.times_s.size

    def __getitem__(self, idx):
        # Allow st[i] or st[2:10] to return times_s sliced
        return self.times_s[idx]

    def __repr__(self) -> str:
        uid = f", unit_id={self.unit_id}" if self.unit_id is not None else ""
        return f"SpikeTrain(times_s=array(shape={self.times_s.shape}){uid})"


@dataclass(frozen=True, slots=True)
class WaveformSeries:
    data: np.ndarray  # (n_spikes, n_channels, n_features)
    channel_positions: np.ndarray | None = None  # (n_channels, ndim)
    channel_ids: tuple[int, ...] | None = None
    feature_names: tuple[str, ...] | None = None  # e.g. ("amp", "width")

    def __post_init__(self):
        data = np.asarray(self.data, dtype=float)
        if data.ndim == 2:
            # If 2D, assume (n_spikes, n_features) and add a dummy channel axis
            data = data[:, np.newaxis, :]
        elif data.ndim != 3:
            raise ValueError(f"data must be 2D or 3D; got shape {data.shape}")
        if data.shape[0] == 0:
            raise ValueError("data must have at least one spike (n_spikes > 0).")
        if self.channel_ids is not None:
            if len(self.channel_ids) != data.shape[1]:
                raise ValueError(
                    f"channel_ids length {len(self.channel_ids)} does not match "
                    f"data shape {data.shape[1]} (n_channels)."
                )
            # Ensure channel_ids is a tuple of integers or None
            if not all(isinstance(cid, int) for cid in self.channel_ids):
                raise TypeError("channel_ids must be a tuple of integers or None.")
            self.channel_ids = tuple(self.channel_ids)
        if self.channel_positions is not None:
            pos = np.asarray(self.channel_positions, dtype=float)
            if pos.ndim != 2:
                raise ValueError(f"channel_positions must be 2D; got shape {pos.shape}")
            if pos.shape[0] != data.shape[1]:
                raise ValueError(
                    f"channel_positions has {pos.shape[0]} rows, "
                    f"but data has {data.shape[1]} channels."
                )
            object.__setattr__(self, "channel_positions", pos)
        if self.feature_names is not None:
            if not isinstance(self.feature_names, tuple):
                raise TypeError("feature_names must be a tuple of strings.")
            if len(self.feature_names) != data.shape[2]:
                raise ValueError(
                    f"feature_names length {len(self.feature_names)} does not match "
                    f"data shape {data.shape[2]} (n_features)."
                )
        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "data", data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Allow wf[i] or wf[2:10] to return data sliced
        return self.data[idx]

    def __repr__(self) -> str:
        ch_ids = f", channel_ids={self.channel_ids}" if self.channel_ids else ""
        return f"WaveformSeries(data=array(shape={self.data.shape}){ch_ids})"


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
        if self.data.ndim == 1:
            # If 1D, assume (n_samples,) and add a dummy channel axis
            data = self.data[:, np.newaxis]
        else:
            data = self.data
        if data.shape[0] == 0:
            raise ValueError(
                "TimeSeries.data must have at least one sample (n_samples > 0)."
            )
        if self.channel_ids is not None:

            if len(self.channel_ids) != data.shape[1]:
                raise ValueError(
                    f"channel_ids length {len(self.channel_ids)} does not match "
                    f"data shape {data.shape[1]} (n_channels)."
                )
            if not all(isinstance(cid, int) for cid in self.channel_ids):
                raise TypeError("channel_ids must be a tuple of integers or None.")
            # Use object.__setattr__ to set the tuple
            object.__setattr__(self, "channel_ids", tuple(self.channel_ids))
        if self.channel_positions is not None:
            if self.channel_positions.ndim != 2:
                raise ValueError(
                    f"channel_positions must be 2D; got shape {self.channel_positions.shape}"
                )
            if self.channel_positions.shape[0] != data.shape[1]:
                raise ValueError(
                    f"channel_positions has {self.channel_positions.shape[0]} rows, "
                    f"but data has {data.shape[1]} channels."
                )
        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "start_s", float(self.start_s))
        object.__setattr__(self, "sampling_rate_hz", float(self.sampling_rate_hz))
        if self.units is not None and not isinstance(self.units, str):
            raise TypeError("units must be a string or None.")
        if self.channel_ids is not None and not isinstance(self.channel_ids, tuple):
            raise TypeError("channel_ids must be a tuple of integers or None.")


# --------------------------------------------------------------------------- #
#  Native-rate archive                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecordingBundle:
    """
    Immutable store for raw session data at native sampling rates.
    """

    # List of spike times for each neuron, in seconds.
    spike_times_s: Optional[Sequence[SpikeTrain | np.ndarray]] = None
    # List of spike waveforms, one for each neuron.
    spike_waveforms: Optional[Sequence[WaveformSeries]] = None
    signals: Dict[str, TimeSeries] = field(default_factory=dict)

    def __post_init__(self) -> None:

        # 1) If the user supplied anything for spike_times_s, it must be a sequence
        if self.spike_times_s is not None:
            if not isinstance(self.spike_times_s, Sequence):
                raise TypeError(
                    "spike_times_s must be a sequence of SpikeTrain or np.ndarray."
                )
            if len(self.spike_times_s) == 0:
                raise ValueError("spike_times_s cannot be an empty list.")
            new_spike_times = []
            for idx, st in enumerate(self.spike_times_s):
                if isinstance(st, SpikeTrain):
                    # Already a SpikeTrain → just keep it
                    new_spike_times.append(st)
                elif _is_array_like(st):
                    # Wrap raw array into a SpikeTrain
                    new_spike_times.append(SpikeTrain(times_s=st, unit_id=idx))
                else:
                    raise TypeError(
                        f"spike_times_s[{idx}] must be a SpikeTrain or numpy.ndarray, "
                        f"got {type(st).__name__}"
                    )

            # Replace with our wrapped list of SpikeTrain objects
            object.__setattr__(self, "spike_times_s", new_spike_times)

        if self.spike_waveforms is not None:
            if not isinstance(self.spike_waveforms, list):
                raise TypeError("spike_waveforms must be a list of WaveformSeries.")
            if len(self.spike_waveforms) == 0:
                raise ValueError("spike_waveforms cannot be an empty list.")
            new_waveforms = []
            for idx, wf in enumerate(self.spike_waveforms):
                if isinstance(wf, WaveformSeries):
                    new_waveforms.append(wf)
                elif _is_array_like(wf):
                    new_waveforms.append(WaveformSeries(data=wf))
                else:
                    raise TypeError(
                        f"spike_waveforms[{idx}] must be a WaveformSeries or numpy.ndarray, "
                        f"got {type(wf).__name__}"
                    )

            # Replace with our wrapped list of WaveformSeries objects
            object.__setattr__(self, "spike_waveforms", new_waveforms)

        # 2) waveforms without timestamps → fatal
        if self.spike_waveforms is not None and self.spike_times_s is None:
            raise ValueError(
                "spike_waveforms provided but spike_times_s is None. "
                "Waveforms require spike times."
            )

        # 3) no spike_times → nothing more to check
        if self.spike_times_s is None:
            return

        # 4) spike_times present, no waveforms → fine
        if self.spike_waveforms is None:
            return

        # 5) both present → check lengths
        if len(self.spike_times_s) != len(self.spike_waveforms):
            raise ValueError(
                f"Length mismatch: spike_times_s has {len(self.spike_times_s)} entries "
                f"but spike_waveforms has {len(self.spike_waveforms)}."
            )

        # 6) per‐unit row counts must match
        for idx, (t, w) in enumerate(zip(self.spike_times_s, self.spike_waveforms)):
            ntimes = t.times_s.size
            nwaves = w.data.shape[0]
            if ntimes != nwaves:
                raise ValueError(
                    f"Neuron {idx}: {ntimes} spike times but {nwaves} waveform rows."
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

    _categorical_maps: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

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
            bad_type = [k for k, v in self.signals.items() if not _is_array_like(v)]
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

    def get_categorical_encoder(self, signal_name: str) -> Any | None:
        """
        Retrieves the scikit-learn encoder or label map for a given
        categorical signal.

        Parameters
        ----------
        signal_name : str
            The name of the categorical signal.

        Returns
        -------
        Any or None
            The fitted encoder instance (e.g., OneHotEncoder, OrdinalEncoder)
            or a list of labels, if it exists. Otherwise, None.
        """
        return self._categorical_maps.get(signal_name)

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
        """
        Creates a new DecoderBatch containing only a subset of spike data.

        This method preserves the SpikeTrain and WaveformSeries object types
        in the new batch.

        Parameters
        ----------
        keys : Iterable[int]
            An iterable of integer indices for the spikes to select.

        Returns
        -------
        DecoderBatch
            A new batch with the selected spike data objects.

        Raises
        ------
        ValueError
            If the batch does not contain any spike times to select from.
        """
        if self.spike_times_s is None:
            raise ValueError("No spike_times_s available in this DecoderBatch.")

        orig_st = self.spike_times_s
        orig_wf = self.spike_waveforms

        # Create a new list containing the selected SpikeTrain objects
        new_times = [orig_st[i] for i in keys]

        new_wf = None
        if orig_wf is not None:
            # Create a new list containing the selected WaveformSeries objects
            new_wf = [orig_wf[i] for i in keys]

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
