import numpy as np
import pytest

from non_local_detector.bundle import (
    DecoderBatch,
    RecordingBundle,
    SpikeTrain,
    TimeSeries,
    WaveformSeries,
    validate_sources,
)


def make_spiketrain(n, unit_id=0):
    return SpikeTrain(
        times_s=np.linspace(0, 1, n),
        unit_id=unit_id,
        channel_position=(0.0, 0.0),
        quality_metrics={"isi_viol": 0.1},
    )


def make_waveformseries(n, n_ch=2, n_samp=10):
    return WaveformSeries(
        data=np.random.randn(n, n_ch, n_samp),
        channel_positions=np.random.randn(n_ch, 2),
        channel_ids=tuple(range(n_ch)),
        feature_names=("amp", "width"),
    )


def make_timeseries(n, n_ch=1, rate=1000.0):
    return TimeSeries(
        data=np.random.randn(n, n_ch),
        sampling_rate_hz=rate,
        channel_ids=tuple(range(n_ch)),
        channel_positions=np.random.randn(n_ch, 2),
        units="uV",
    )


def test_timeseries_validation():
    # Valid
    ts = make_timeseries(100)
    assert ts.data.shape[0] == 100
    # Invalid sampling rate
    with pytest.raises(ValueError):
        TimeSeries(data=np.zeros((10,)), sampling_rate_hz=0)
    # Invalid data shape
    with pytest.raises(ValueError):
        TimeSeries(data=np.array(5), sampling_rate_hz=1000)


def test_recordingbundle_waveforms_without_spikes():
    wf = [make_waveformseries(5)]
    with pytest.raises(ValueError):
        RecordingBundle(spike_times_s=None, spike_waveforms=wf)


def test_recordingbundle_length_mismatch():
    st = [make_spiketrain(5), make_spiketrain(6)]
    wf = [make_waveformseries(5)]
    with pytest.raises(ValueError):
        RecordingBundle(spike_times_s=st, spike_waveforms=wf)


def test_recordingbundle_waveform_row_mismatch():
    st = [make_spiketrain(5)]
    wf = [make_waveformseries(6)]
    with pytest.raises(ValueError):
        RecordingBundle(spike_times_s=st, spike_waveforms=wf)


def test_recordingbundle_valid():
    st = [make_spiketrain(5), make_spiketrain(6)]
    wf = [make_waveformseries(5), make_waveformseries(6)]
    ts = {"lfp": make_timeseries(10, 2)}
    rb = RecordingBundle(spike_times_s=st, spike_waveforms=wf, signals=ts)
    assert isinstance(rb, RecordingBundle)


def test_decoderbatch_signals_validation():
    arr1 = np.ones((10,))
    arr2 = np.ones((10, 2))
    batch = DecoderBatch(signals={"counts": arr1, "lfp": arr2})
    assert batch.n_time == 10
    assert batch.counts is arr1
    assert batch.lfp is arr2
    assert batch.calcium is None


def test_decoderbatch_signals_type_and_dtype():
    arr = np.ones((5,))
    arr_bad = [[1, 2, 3, 4, 5]]
    # Not ndarray
    with pytest.raises(TypeError):
        DecoderBatch(signals={"counts": arr, "bad": arr_bad})
    # Bad dtype
    arr_str = np.array(["a", "b", "c", "d", "e"])
    with pytest.raises(TypeError):
        DecoderBatch(signals={"counts": arr, "bad": arr_str})


def test_decoderbatch_time_axis_mismatch():
    arr1 = np.ones((10,))
    arr2 = np.ones((11,))
    with pytest.raises(ValueError):
        DecoderBatch(signals={"a": arr1, "b": arr2})


def test_decoderbatch_bin_edges_validation():
    arr = np.ones((5,))
    edges = np.linspace(0, 1, 7)  # Should be 6 for 5 bins
    with pytest.raises(ValueError):
        DecoderBatch(signals={"counts": arr}, bin_edges_s=edges)
    # Valid with correct length
    edges = np.linspace(0, 1, 6)
    batch = DecoderBatch(signals={"counts": arr}, bin_edges_s=edges)
    assert batch.n_time == 5


def test_decoderbatch_no_signals_bin_edges_required():
    edges = np.linspace(0, 1, 6)
    with pytest.raises(ValueError):
        # No signals or spike_times_s, but bin_edges_s provided
        # should raise ValueError
        batch = DecoderBatch(signals={}, bin_edges_s=edges)
    # No signals and no bin_edges_s
    with pytest.raises(ValueError):
        DecoderBatch(signals={})


def test_decoderbatch_slice_and_select():
    arr = np.arange(10)
    lfp = np.arange(20).reshape(10, 2)
    bin_edges = np.linspace(0, 1, 11)
    spike_times = [np.array([0.05, 0.15, 0.25, 0.95])]
    spike_waveforms = [np.random.randn(4, 2, 10)]
    batch = DecoderBatch(
        signals={"counts": arr, "lfp": lfp},
        bin_edges_s=bin_edges,
        spike_times_s=spike_times,
        spike_waveforms=spike_waveforms,
    )
    sliced = batch.slice(2, 5)
    assert sliced.n_time == 3
    assert sliced.counts.shape[0] == 3
    assert sliced.lfp.shape[0] == 3
    assert sliced.bin_edges_s.shape[0] == 4
    # Test select_signals
    sel = batch.select_signals(["counts"])
    assert "counts" in sel.signals and "lfp" not in sel.signals
    # Test select_spikes
    sel2 = batch.select_spikes([0])
    assert len(sel2.spike_times_s) == 1


def test_decoderbatch_slice_spikes_false():
    arr = np.arange(10)
    bin_edges = np.linspace(0, 1, 11)
    spike_times = [np.array([0.05, 0.15, 0.25, 0.95])]
    batch = DecoderBatch(
        signals={"counts": arr},
        bin_edges_s=bin_edges,
        spike_times_s=spike_times,
    )
    sliced = batch.slice(2, 5, slice_spikes=False)
    # Should not filter spikes
    assert np.all(sliced.spike_times_s[0] == spike_times[0])


def test_decoderbatch_select_spikes_no_spikes():
    arr = np.arange(10)
    batch = DecoderBatch(signals={"counts": arr})
    with pytest.raises(ValueError):
        batch.select_spikes([0])


def test_validate_sources_success_and_failure():
    arr = np.ones((5,))
    batch = DecoderBatch(signals={"counts": arr, "lfp": arr})

    class DummyModel:
        required_sources = ["counts", "lfp"]

    # Should not raise
    validate_sources(batch, [DummyModel()])

    # Missing source
    class DummyModel2:
        required_sources = ["counts", "lfp", "calcium"]

    with pytest.raises(ValueError) as e:
        validate_sources(batch, [DummyModel2()])
    assert "calcium" in str(e.value)


def test_recordingbundle_spikes_only():
    st = [make_spiketrain(4), make_spiketrain(3)]  # two neurons, lengths 4 and 3
    rb = RecordingBundle(spike_times_s=st, spike_waveforms=None, signals={})
    # Should succeed, and .spike_waveforms stays None
    assert rb.spike_waveforms is None
    assert [s.times_s.size for s in rb.spike_times_s] == [4, 3]


def test_recordingbundle_waveforms_without_spikes():
    wf = [make_waveformseries(5)]
    with pytest.raises(ValueError):
        RecordingBundle(spike_times_s=None, spike_waveforms=wf, signals={})


def test_recordingbundle_signals_only():
    ts1 = make_timeseries(n=50, rate=100.0)
    ts2 = make_timeseries(n=25, n_ch=2, rate=25.0)
    rb = RecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"a": ts1, "b": ts2}
    )
    assert "a" in rb.signals and "b" in rb.signals
    assert rb.spike_times_s is None


def test_decoderbatch_two_signals_same_length():
    sig1 = np.arange(10.0)  # float
    sig2 = np.ones((10, 3))  # 3‐channel LFP
    batch = DecoderBatch(signals={"counts": sig1, "lfp": sig2})
    assert batch.n_time == 10
    assert batch.counts is sig1
    assert batch.lfp is sig2


def test_decoderbatch_spikes_only_without_errors():
    # Build two fake spike‐trains at arbitrary times:
    class FakeSpikeTrain:
        def __init__(self, times):
            self.times_s = np.array(times)

    st_list = [FakeSpikeTrain([0.1, 0.5, 1.2]), FakeSpikeTrain([0.0, 0.3])]
    edges = np.array([0.0, 1.0, 2.0])
    batch2 = DecoderBatch(
        signals={}, spike_times_s=st_list, spike_waveforms=None, bin_edges_s=edges
    )
    assert batch2.n_time == 2


def test_decoderbatch_slice_spikes_flag():
    # Build a small batch with:
    #  • signals = {"a": np.arange(10)}  (n_time=10)
    #  • bin_edges_s = np.linspace(0, 1, 11)  (10 bins of width 0.1)
    #  • spike_times_s = two arrays: [0.05, 0.25, 0.95] and [0.15, 0.7]
    st0 = np.array([0.05, 0.25, 0.95])
    st1 = np.array([0.15, 0.7])
    edges = np.linspace(0, 1, 11)
    batch = DecoderBatch(
        signals={"a": np.arange(10)},
        bin_edges_s=edges,
        spike_times_s=[st0, st1],
        spike_waveforms=None,
    )
    # Now slice [2:5) → times [0.2, 0.5)
    sliced1 = batch.slice(2, 5, slice_spikes=True)
    #   For st0 = [0.05, 0.25, 0.95], only 0.25 remains
    assert np.allclose(sliced1.spike_times_s[0], [0.25])
    #   For st1 = [0.15, 0.7], none remain
    assert sliced1.spike_times_s[1].size == 0
    #   Also check that signals["a"] has length (5−2)=3
    assert sliced1.signals["a"].shape == (3,)

    # Now do slice_spikes=False → spikes are untouched
    sliced2 = batch.slice(2, 5, slice_spikes=False)
    assert np.allclose(sliced2.spike_times_s[0], st0)
    assert np.allclose(sliced2.spike_times_s[1], st1)


def test_decoderbatch_select_signals():
    sig1 = np.arange(5.0)
    sig2 = np.arange(5.0) * 2
    edges = np.linspace(0, 1, 6)
    batch = DecoderBatch(
        signals={"counts": sig1, "lfp": sig2},
        bin_edges_s=edges,
        spike_times_s=None,
        spike_waveforms=None,
    )
    sub = batch.select_signals(["lfp"])
    assert "lfp" in sub.signals and "counts" not in sub.signals
    assert np.allclose(sub.lfp, sig2)
    assert np.allclose(sub.bin_edges_s, edges)


def test_decoderbatch_select_spikes_and_errors():

    class FakeSpikeTrain:
        def __init__(self, times):
            self.times_s = np.array(times)

    st_list = [FakeSpikeTrain([0.1, 0.2]), FakeSpikeTrain([0.3])]
    wf_list = [np.zeros((2, 4)), np.zeros((1, 4))]
    edges = np.linspace(0, 1, 4)
    batch = DecoderBatch(
        signals={"a": np.arange(3)},
        bin_edges_s=edges,
        spike_times_s=st_list,
        spike_waveforms=wf_list,
    )
    # valid selection
    sub = batch.select_spikes([1])
    assert len(sub.spike_times_s) == 1
    assert np.allclose(sub.spike_times_s[0].times_s, [0.3])
    # no spike_times → should raise
    empty_batch = DecoderBatch(signals={"a": np.arange(3)}, bin_edges_s=edges)
    with pytest.raises(ValueError):
        empty_batch.select_spikes([0])


def test_validate_sources_requires_non_none_attribute():
    sig = np.ones(5)
    batch = DecoderBatch(signals={"counts": sig})

    # forcing batch.calcium to be None (no “calcium” key and no `calcium` prop), so requiring it should fail
    class ModelReq:
        required_sources = ["counts", "calcium"]

    with pytest.raises(ValueError) as exc:
        validate_sources(batch, [ModelReq()])
    assert "calcium" in str(exc.value)
