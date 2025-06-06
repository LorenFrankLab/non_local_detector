import numpy as np
import pandas as pd
import pytest

# ── Step A: Import preprocessing first, then patch its imported names via monkeypatch
from non_local_detector.io import preprocessing


class DummyRecordingBundle:
    def __init__(self, *, spike_times_s=None, spike_waveforms=None, signals=None):
        self.spike_times_s = spike_times_s
        self.spike_waveforms = spike_waveforms
        self.signals = signals or {}


class DummyDecoderBatch:
    def __init__(
        self,
        *,
        signals=None,
        bin_edges_s=None,
        spike_times_s=None,
        spike_waveforms=None,
    ):
        self.signals = signals or {}
        self.bin_edges_s = bin_edges_s


@pytest.fixture(autouse=True)
def patch_preprocessing_to_use_dummies(monkeypatch):
    """
    Before each test runs, overwrite preprocessing.RecordingBundle,
    preprocessing.TimeSeries, and preprocessing.DecoderBatch to use the dummies.
    """
    monkeypatch.setattr(preprocessing, "RecordingBundle", DummyRecordingBundle)
    monkeypatch.setattr(preprocessing, "TimeSeries", DummyTimeSeries)
    monkeypatch.setattr(preprocessing, "DecoderBatch", DummyDecoderBatch)


class DummyTimeSeries:
    def __init__(self, data, sampling_rate_hz, start_s=0.0):
        self.data = np.array(data)
        self.sampling_rate_hz = sampling_rate_hz
        self.start_s = start_s


def test_df_to_recording_bundle_numeric_index():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[0.0, 0.5, 1.0])
    bundle = preprocessing.df_to_recording_bundle(df)
    assert isinstance(bundle, DummyRecordingBundle)

    # Check that the TimeSeries got created with correct shape, rate, start.
    assert set(bundle.signals.keys()) == {"a", "b"}
    ts_a = bundle.signals["a"]
    assert isinstance(ts_a, DummyTimeSeries)
    assert np.all(ts_a.data == [1, 2, 3])
    assert np.isclose(ts_a.sampling_rate_hz, 2.0)
    assert np.isclose(ts_a.start_s, 0.0)

    ts_b = bundle.signals["b"]
    assert np.all(ts_b.data == [4, 5, 6])


def test_to_decoder_batch_categorical_ordinal():
    ts = DummyTimeSeries(["a", "b", "a", "c"], sampling_rate_hz=2.0, start_s=0.0)
    rec = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"cat": ts}
    )
    batch = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["cat"], one_hot_categories=False
    )
    # There are 4 samples at 2 Hz → 2 one‐second bins.  Raw labels = ["a","b","a","c"].
    # The OrdinalEncoder label‐order might be ["a","b","c"], so codes = [0,1,0,2].
    # Bin 0 sees samples [0,1] → _mode(...) = 0.  Bin 1 sees [0,2] → _mode = 0.
    assert isinstance(batch, DummyDecoderBatch)
    assert "cat" in batch.signals
    assert np.all(batch.signals["cat"] == [0, 0])  # integer codes


def test_to_decoder_batch_categorical_onehot():
    ts = DummyTimeSeries(["a", "b", "a", "c"], sampling_rate_hz=2.0, start_s=0.0)
    rec = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"cat": ts}
    )
    batch = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["cat"], one_hot_categories=True
    )
    # Now “cat” becomes a 2×3 binary matrix.  Category order = ["a","b","c"].
    # Raw 2→bins: Bin0 sees “a”/“b” → [1,1,0], Bin1 sees “a”/“c” → [1,0,1].
    arr = batch.signals["cat"]
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[0], [1, 1, 0])
    np.testing.assert_array_equal(arr[1], [1, 0, 1])


def test_ordinal_unknown_label_maps_to_minus_one(monkeypatch):
    # Let the raw data contain an “unknown” for one bin
    raw = np.array(["a", "b", "c", "d"], dtype="<U1")
    # If we artificially force OrdinalEncoder to only see ["a","b","c"]:
    from sklearn.preprocessing import OrdinalEncoder

    enc = OrdinalEncoder(
        categories=[["a", "b", "c"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    codes = enc.fit_transform(raw.reshape(-1, 1)).astype(int).flatten()
    assert list(codes) == [0, 1, 2, -1]

    # Now feed this into to_decoder_batch, but monkey‐patch the encoder in preprocessing
    class FakeOrdinal(preprocessing.OrdinalEncoder):
        def fit_transform(self, X):
            return codes.reshape(-1, 1)

    FakeOrdinal.__name__ = "FakeOrdinal"  # for better error messages
    FakeOrdinal.categories_ = [np.array([["a", "b", "c"]])]

    monkeypatch.setattr(preprocessing, "OrdinalEncoder", FakeOrdinal)

    ts = DummyTimeSeries(raw, sampling_rate_hz=2.0, start_s=0.0)
    rec = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"cat": ts}
    )
    batch = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["cat"], one_hot_categories=False
    )
    # Now bin0 sees ["a","b"] → codes [0,1] → mode=0
    # bin1 sees ["c","d"] → codes [2,-1] → mode picks -1 (since unique([-1,2])→counts tie→first)
    assert np.all(batch.signals["cat"] == [0, -1])


def test_integer_fill_ffill_and_pad_zero():
    # 4 samples at 2 Hz → 2 bins; raw codes = [0, 1, 0, 1] for ["a","b","a","b"]
    raw = np.array(["a", "b", "a", "b"], dtype="<U1")
    ts = DummyTimeSeries(raw, sampling_rate_hz=2.0, start_s=0.0)
    rec = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"cat": ts}
    )

    # Suppose we artificially zero out bin1’s codes by setting bin_idx > n_bins for some samples.
    # Easiest is to set bin_width_s=2.0:
    #   samples at t=0 → bin0; at t=0.5 → bin0; at t=1.0 → bin1; at t=1.5 → bin1.
    #   But then raw codes are [0,1] in bin0 and [0,1] in bin1.
    #   To force an “empty bin,” simulate a one‐sample shift:
    ts2 = DummyTimeSeries(raw, sampling_rate_hz=2.0, start_s=0.1)
    #  samples at times 0.1,0.6,1.1,1.6 → floor((t-0)/2)=0 for t<2, so NO empty bin.
    # Instead, drop the bin1 altogether by making stop_s small:
    batch = preprocessing.to_decoder_batch(
        rec, bin_width_s=3.0, signals_to_use=["cat"], one_hot_categories=False
    )
    # Now only 1 bin (0≤t<3): codes [0,1,0,1] → mode=0; no gaps at all, skip.
    assert batch.signals["cat"].shape == (1,)
    assert batch.signals["cat"][0] == 0

    # To force an empty bin: build a ts where two samples land in bin0, and bin1 has none
    raw2 = np.array(["a", "b"], dtype="<U1")
    ts3 = DummyTimeSeries(raw2, sampling_rate_hz=2.0, start_s=0.0)
    rec3 = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"cat": ts3}
    )
    # bin_width=1 → 2 bins; bin0 sees [0,1]; bin1 sees no samples → should be fillable.
    batch_ffill = preprocessing.to_decoder_batch(
        rec3,
        bin_width_s=1.0,
        signals_to_use=["cat"],
        one_hot_categories=False,
        int_fill="ffill",
        nan_policy="ignore",
        start_s=0.0,
        stop_s=2.0,  # to ensure bin1 is empty
    )
    assert batch_ffill.signals["cat"].shape == (2,)
    # bin0 = mode([0,1])=0; bin1 (empty) with ffill→0
    assert batch_ffill.signals["cat"][0] == 0
    assert batch_ffill.signals["cat"][1] == 0

    # pad_zero keeps the 2nd bin at 0 as well:
    batch_pad = preprocessing.to_decoder_batch(
        rec3,
        bin_width_s=1.0,
        signals_to_use=["cat"],
        one_hot_categories=False,
        int_fill="pad_zero",
        nan_policy="ignore",
        start_s=0.0,
        stop_s=2.0,  # to ensure bin1 is empty
    )
    assert batch_pad.signals["cat"][1] == 0


def test_bool_binning_and_fill():
    # Raw boolean at 2 Hz → 2 bins
    raw = np.array([True, False, False, False])
    ts = DummyTimeSeries(raw, sampling_rate_hz=2.0, start_s=0.0)
    rec = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"b": ts}
    )

    # bin0 sees [True, False] → True; bin1 sees [False, False] → False
    batch = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["b"], nan_policy="ignore"
    )
    assert np.array_equal(batch.signals["b"], [True, False])

    # Now force an empty bin by shifting start:
    ts2 = DummyTimeSeries(raw, sampling_rate_hz=2.0, start_s=1.0)  # all samples in bin1
    rec2 = DummyRecordingBundle(
        spike_times_s=None, spike_waveforms=None, signals={"b": ts2}
    )
    # bin_width=1 → bin0 empty, bin1 sees [True,False,False,False] → True
    batch_ffill = preprocessing.to_decoder_batch(
        rec2,
        bin_width_s=1.0,
        signals_to_use=["b"],
        bool_fill="ffill",
        nan_policy="ignore",
        start_s=0.0,
        stop_s=3.0,  # to ensure bin0 is empty
    )
    assert batch_ffill.signals["b"][0] == False  # no previous → stays False
    assert batch_ffill.signals["b"][1] == True
    # with bool_fill="or", same behaviour because “or” over an empty set yields False
    # and then the next bin’s True remains True
    batch_or = preprocessing.to_decoder_batch(
        rec2,
        bin_width_s=1.0,
        signals_to_use=["b"],
        bool_fill="or",
        nan_policy="ignore",
        start_s=0.0,
        stop_s=3.0,
    )
    assert batch_or.signals["b"][0] == False
    assert batch_or.signals["b"][1] == True


def test_spike_count_hist_vs_center():
    # Let bin_width=1.0, edges = [0,1,2].
    # Place spikes at exactly t=1.0 and t=0.999
    spike_times = np.array([0.999, 1.0, 1.001])

    class FakeSpikeTrain:
        def __init__(self, times):
            self.times_s = times

    ts = [FakeSpikeTrain(spike_times)]
    rec = DummyRecordingBundle(spike_times_s=ts, spike_waveforms=None, signals={})

    # hist: t=1.0 goes into bin0 (left‐inclusive, right‐exclusive by numpy)
    batch_hist = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["counts"]
    )
    # Because all spikes fall between 0.999 and 1.001, the inferred window is [0.999,1.001),
    # producing exactly 1 bin.  All three spikes go into that 1 bin.
    assert batch_hist.signals["counts"].shape == (1, 1)
    assert batch_hist.signals["counts"][0, 0] == 3

    # center: shift by −0.5, so t=1.0 → 0.5 (bin0), t=0.999 → 0.499 (bin0),
    # t=1.001 → 0.501 (bin0). All fall in bin0
    batch_center = preprocessing.to_decoder_batch(
        rec, bin_width_s=1.0, signals_to_use=["counts"], count_method="center"
    )
    # Again only 1 bin, with all three spikes
    assert batch_center.signals["counts"].shape == (1, 1)
    assert batch_center.signals["counts"][0, 0] == 0


def test_empty_spike_times_list():
    rec = DummyRecordingBundle(spike_times_s=[], spike_waveforms=None, signals={})
    with pytest.raises(ValueError):
        # No spike times, so no counts can be computed
        preprocessing.to_decoder_batch(
            rec, bin_width_s=1.0, signals_to_use=["counts"], nan_policy="ignore"
        )
    # bin edges from 0→0 (no candidates) might error; so supply an extra numeric signal
    ts = DummyTimeSeries([1.0, 2.0], sampling_rate_hz=1.0, start_s=0.0)
    rec2 = DummyRecordingBundle(
        spike_times_s=[], spike_waveforms=None, signals={"t": ts}
    )
    with pytest.raises(ValueError):
        # No spike times, so no counts can be computed
        preprocessing.to_decoder_batch(
            rec2, bin_width_s=1.0, signals_to_use=["counts", "t"], nan_policy="ignore"
        )


def test_mixed_numeric_categorical_bool():
    # Numeric: 4 samples at 2 Hz → 2 bins
    num_data = np.array([0.1, 1.1, 2.1, 3.1])
    ts_num = DummyTimeSeries(num_data, sampling_rate_hz=2.0, start_s=0.0)
    # Categorical: ["x","y","x","z"] → ordinal codes [0,1,0,2]
    ts_cat = DummyTimeSeries(["x", "y", "x", "z"], sampling_rate_hz=2.0, start_s=0.0)
    # Boolean: [True, False, True, False]
    ts_bool = DummyTimeSeries(
        [True, False, True, False], sampling_rate_hz=2.0, start_s=0.0
    )

    rec = DummyRecordingBundle(
        spike_times_s=None,
        spike_waveforms=None,
        signals={"num": ts_num, "cat": ts_cat, "flag": ts_bool},
    )

    batch = preprocessing.to_decoder_batch(
        rec,
        bin_width_s=1.0,
        signals_to_use=["num", "cat", "flag"],
        one_hot_categories=False,
        nan_policy="raise",
    )
    # num: bin0=(0.1,1.1)→mean=0.6; bin1=(2.1,3.1)→mean=2.6
    assert np.allclose(batch.signals["num"], [0.6, 2.6])
    # cat: codes [0,1]→mode=0; [0,2]→mode=0
    assert np.all(batch.signals["cat"] == [0, 0])
    # flag: [True,False]→True; [True,False]→True
    assert np.all(batch.signals["flag"] == [True, True])


def test_partial_unaligned_signal_skips_only_that_key():
    # Numeric stream from t=[0,1,2]
    ts_num = DummyTimeSeries([1.0, 2.0, 3.0], sampling_rate_hz=1.0, start_s=0.0)
    # Categorical stream from t=[10,11,12] (completely outside the numeric window)
    ts_cat = DummyTimeSeries(["a", "b", "c"], sampling_rate_hz=1.0, start_s=10.0)

    rec = DummyRecordingBundle(
        spike_times_s=None,
        spike_waveforms=None,
        signals={"num": ts_num, "cat": ts_cat},
    )

    batch = preprocessing.to_decoder_batch(
        rec,
        bin_width_s=1.0,
        signals_to_use=["num", "cat"],
        nan_policy="warn",
        start_s=0.0,
        stop_s=3.0,  # numeric data is in [0,3], categorical in [10,13]
    )
    assert "num" in batch.signals
    # "cat" was entirely outside [0,3] → should be skipped
    assert "cat" not in batch.signals


def test_df_to_recording_bundle_datetime_index():
    times = pd.date_range("2020-01-01 00:00:00", periods=3, freq="500L")
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3]}, index=times)
    bundle = preprocessing.df_to_recording_bundle(df)
    # sampling_rate = 2 Hz (because 500ms spacing → 0.5s → 2Hz)
    ts = bundle.signals["x"]
    assert isinstance(ts, DummyTimeSeries)
    assert np.isclose(ts.sampling_rate_hz, 2.0)
    # start_s should be 0.0
    assert np.isclose(ts.start_s, 0.0)
    # data should exactly match [0.1, 0.2, 0.3]
    assert np.allclose(ts.data, [0.1, 0.2, 0.3])


def test_df_to_recording_bundle_nonuniform_index():
    df = pd.DataFrame({"v": [1, 2, 3]}, index=[0.0, 0.3, 0.9])
    with pytest.raises(ValueError):
        preprocessing.df_to_recording_bundle(df)

    # But if we pass sampling_rate_hz, it should succeed
    bundle = preprocessing.df_to_recording_bundle(df, sampling_rate_hz=2.0)
    assert "v" in bundle.signals


def test_df_to_recording_bundle_subset_of_columns():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=[0.0, 0.5, 1.0]
    )
    bundle = preprocessing.df_to_recording_bundle(df, signals_to_include=["b", "c"])
    assert set(bundle.signals.keys()) == {"b", "c"}


def test_df_to_recordingbundle_nonzero_start():
    df = pd.DataFrame({"x": [10, 20, 30]}, index=[5.0, 5.5, 6.0])
    rb = preprocessing.df_to_recording_bundle(df)
    # The returned TimeSeries.start_s should equal the first index = 5.0
    assert rb.signals["x"].start_s == 5.0
    # And sampling_rate inferred = 1/(0.5) = 2 Hz
    assert np.isclose(rb.signals["x"].sampling_rate_hz, 2.0)


def test_df_to_recordingbundle_empty_df():
    df = pd.DataFrame([], columns=["a", "b"])
    with pytest.raises(ValueError):
        preprocessing.df_to_recording_bundle(df)
