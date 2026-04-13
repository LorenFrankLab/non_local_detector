"""Tests for dense probe simulation and probe geometry.

Verifies probe geometry generation, amplitude physics, and that the
dense-probe simulator produces outputs conforming to the
ClusterlessSimOutput contract.
"""

import numpy as np
import pytest

from non_local_detector.simulate.dense_probe_simulation import (
    make_dense_probe_run_data,
    make_dimensionality_sweep,
)
from non_local_detector.simulate.probe_geometry import (
    compute_amplitude_falloff,
    make_linear_probe,
    make_neuron_locations,
    select_channels,
)

# ---------------------------------------------------------------------------
# Probe geometry tests
# ---------------------------------------------------------------------------


class TestMakeLinearProbe:
    def test_shape(self):
        pos = make_linear_probe(n_channels=100, n_columns=2)
        assert pos.shape == (100, 2)

    def test_neuropixels_defaults(self):
        pos = make_linear_probe()
        assert pos.shape == (384, 2)
        # Two distinct x values for 2-column layout
        unique_x = np.unique(pos[:, 0])
        assert len(unique_x) == 2
        assert np.isclose(unique_x[1] - unique_x[0], 16.0)

    def test_vertical_spacing(self):
        pos = make_linear_probe(n_channels=10, n_columns=1, vertical_spacing=25.0)
        dz = np.diff(pos[:, 1])
        np.testing.assert_allclose(dz, 25.0)

    def test_single_column(self):
        pos = make_linear_probe(n_channels=10, n_columns=1)
        assert np.all(pos[:, 0] == 0)


class TestMakeNeuronLocations:
    def test_shape(self):
        ch = make_linear_probe(n_channels=50)
        locs = make_neuron_locations(20, ch, rng=np.random.default_rng(0))
        assert locs.shape == (20, 3)

    def test_depth_bounds(self):
        ch = make_linear_probe(n_channels=50)
        depth_range = (15.0, 60.0)
        locs = make_neuron_locations(
            200, ch, depth_range=depth_range, rng=np.random.default_rng(0)
        )
        assert np.all(locs[:, 1] >= depth_range[0])
        assert np.all(locs[:, 1] <= depth_range[1])

    def test_z_within_probe_extent(self):
        ch = make_linear_probe(n_channels=50)
        locs = make_neuron_locations(200, ch, rng=np.random.default_rng(0))
        z_min, z_max = ch[:, 1].min(), ch[:, 1].max()
        assert np.all(locs[:, 2] >= z_min)
        assert np.all(locs[:, 2] <= z_max)


class TestComputeAmplitudeFalloff:
    @pytest.fixture()
    def simple_setup(self):
        ch = make_linear_probe(n_channels=20, n_columns=1, vertical_spacing=20.0)
        # Place one neuron right next to channel 5
        neuron = np.array([[0.0, 10.0, 5 * 20.0]])
        return neuron, ch

    def test_shape(self, simple_setup):
        neuron, ch = simple_setup
        amp = compute_amplitude_falloff(neuron, ch)
        assert amp.shape == (1, 20)

    def test_peak_normalized_to_one(self, simple_setup):
        neuron, ch = simple_setup
        amp = compute_amplitude_falloff(neuron, ch)
        np.testing.assert_allclose(amp.max(axis=1), 1.0)

    def test_amplitude_decreases_with_distance(self, simple_setup):
        neuron, ch = simple_setup
        amp = compute_amplitude_falloff(neuron, ch)[0]
        # Channel 5 should be closest → highest amplitude
        peak_ch = np.argmax(amp)
        # Amplitudes should decrease moving away from peak
        max_offset = min(peak_ch, len(amp) - peak_ch - 1, 5)
        assert max_offset >= 2, (
            f"peak_ch={peak_ch} too close to edge for meaningful test"
        )
        for offset in range(1, max_offset):
            assert amp[peak_ch - offset] <= amp[peak_ch - offset + 1] or np.isclose(
                amp[peak_ch - offset], amp[peak_ch - offset + 1]
            )
            assert amp[peak_ch + offset] <= amp[peak_ch + offset - 1] or np.isclose(
                amp[peak_ch + offset], amp[peak_ch + offset - 1]
            )

    def test_exponential_model(self):
        ch = make_linear_probe(n_channels=10, n_columns=1)
        neuron = np.array([[0.0, 10.0, 0.0]])
        amp = compute_amplitude_falloff(neuron, ch, decay_model="exponential")
        assert amp.shape == (1, 10)
        np.testing.assert_allclose(amp.max(axis=1), 1.0)

    def test_invalid_model_raises(self):
        ch = make_linear_probe(n_channels=10, n_columns=1)
        neuron = np.array([[0.0, 10.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown decay_model"):
            compute_amplitude_falloff(neuron, ch, decay_model="linear")


class TestSelectChannels:
    def test_all_returns_all(self):
        templates = np.random.default_rng(0).random((5, 100))
        idx = select_channels(templates, method="all")
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_none_n_active_returns_all(self):
        templates = np.random.default_rng(0).random((5, 100))
        idx = select_channels(templates, n_active_channels=None)
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_top_k_count(self):
        templates = np.random.default_rng(0).random((5, 100))
        idx = select_channels(templates, n_active_channels=20, method="top_k")
        assert len(idx) == 20
        assert np.all(np.diff(idx) > 0)  # sorted

    def test_uniform_count(self):
        templates = np.random.default_rng(0).random((5, 100))
        idx = select_channels(templates, n_active_channels=10, method="uniform")
        assert len(idx) == 10

    def test_top_k_selects_highest(self):
        # Create templates where channels 90-99 clearly dominate
        templates = np.zeros((3, 100))
        templates[:, 90:] = 10.0
        idx = select_channels(templates, n_active_channels=10, method="top_k")
        np.testing.assert_array_equal(idx, np.arange(90, 100))

    def test_invalid_method_raises(self):
        templates = np.random.default_rng(0).random((5, 100))
        with pytest.raises(ValueError, match="Unknown method"):
            select_channels(templates, n_active_channels=10, method="bad")


# ---------------------------------------------------------------------------
# Dense probe simulation contract tests
# ---------------------------------------------------------------------------


class TestDenseProbeContract:
    """Verify ClusterlessSimOutput contract compliance for dense probe sim."""

    @pytest.fixture(scope="class")
    def sim(self):
        return make_dense_probe_run_data(
            n_channels=50,
            n_active_channels=10,
            sampling_frequency=500,
            n_runs=2,
            place_field_means=np.arange(0, 100, 10),  # 10 neurons
            seed=0,
        )

    def test_position_shape(self, sim):
        assert sim.position.ndim == 2
        assert sim.position.shape[1] == 1
        assert sim.position.shape[0] == sim.position_time.shape[0]

    def test_single_electrode(self, sim):
        assert len(sim.spike_times) == 1
        assert len(sim.spike_waveform_features) == 1

    def test_feature_dimensionality(self, sim):
        assert sim.spike_waveform_features[0].shape[1] == 10

    def test_spike_count_matches(self, sim):
        assert sim.spike_times[0].shape[0] == sim.spike_waveform_features[0].shape[0]

    def test_has_spikes(self, sim):
        assert sim.spike_times[0].size > 0, "Simulation should produce spikes"

    def test_times_strictly_increasing(self, sim):
        times = sim.spike_times[0]
        if times.size > 1:
            assert np.all(np.diff(times) > 0)

    def test_times_in_bounds(self, sim):
        times = sim.spike_times[0]
        assert np.all(times >= sim.edges[0])
        assert np.all(times <= sim.edges[-1])

    def test_features_finite(self, sim):
        features = sim.spike_waveform_features[0]
        assert np.all(np.isfinite(features))

    def test_position_finite(self, sim):
        assert np.all(np.isfinite(sim.position))
        assert np.all(np.isfinite(sim.position_time))

    def test_edges_sorted(self, sim):
        assert np.all(np.diff(sim.edges) > 0)

    def test_bin_widths_consistency(self, sim):
        if sim.bin_widths is not None:
            np.testing.assert_allclose(sim.bin_widths, np.diff(sim.edges), rtol=1e-10)

    def test_environment_fitted(self, sim):
        assert hasattr(sim.environment, "place_bin_centers_")
        assert sim.environment.place_bin_centers_ is not None

    def test_dtypes_are_float(self, sim):
        assert np.issubdtype(sim.position_time.dtype, np.floating)
        assert np.issubdtype(sim.position.dtype, np.floating)
        assert np.issubdtype(sim.edges.dtype, np.floating)
        assert np.issubdtype(sim.spike_times[0].dtype, np.floating)
        assert np.issubdtype(sim.spike_waveform_features[0].dtype, np.floating)


class TestDenseProbeSeeding:
    def test_deterministic(self):
        kwargs = {
            "n_channels": 30,
            "n_active_channels": 8,
            "sampling_frequency": 500,
            "n_runs": 1,
            "place_field_means": np.arange(0, 50, 10),
            "seed": 42,
        }
        s1 = make_dense_probe_run_data(**kwargs)
        s2 = make_dense_probe_run_data(**kwargs)
        np.testing.assert_array_equal(s1.spike_times[0], s2.spike_times[0])
        np.testing.assert_array_equal(
            s1.spike_waveform_features[0], s2.spike_waveform_features[0]
        )

    def test_different_seeds_differ(self):
        kwargs = {
            "n_channels": 30,
            "n_active_channels": 8,
            "sampling_frequency": 500,
            "n_runs": 2,
            "place_field_means": np.arange(0, 50, 10),
        }
        s1 = make_dense_probe_run_data(seed=10, **kwargs)
        s2 = make_dense_probe_run_data(seed=20, **kwargs)
        assert not np.array_equal(s1.spike_times[0], s2.spike_times[0])


class TestDenseProbeFeatures:
    """Test physical properties of the generated marks."""

    def test_marks_form_clusters(self):
        """Spikes from the same neuron should cluster in mark space."""
        # Use a small probe so we can reason about the structure
        sim = make_dense_probe_run_data(
            n_channels=20,
            n_active_channels=20,
            sampling_frequency=500,
            n_runs=3,
            place_field_means=np.array([25.0, 150.0]),  # 2 well-separated neurons
            amplitude_noise_std=0.01,  # low noise
            seed=0,
        )
        marks = sim.spike_waveform_features[0]
        assert marks.shape[0] > 10, "Need enough spikes to check clustering"

        # With 2 neurons, marks should have 2 clusters.
        # Check that variance within rows is much less than variance across rows,
        # which indicates structure (not uniform noise).
        row_stds = marks.std(axis=1)  # within-spike variation (across channels)
        overall_std = marks.std()
        assert overall_std > row_stds.mean(), "Marks should have cross-spike structure"

    def test_dimensionality_controls_features(self):
        """n_active_channels should control feature width."""
        for n_ch in [4, 20, 50]:
            sim = make_dense_probe_run_data(
                n_channels=50,
                n_active_channels=n_ch,
                sampling_frequency=500,
                n_runs=1,
                place_field_means=np.arange(0, 50, 10),
                seed=0,
            )
            assert sim.spike_waveform_features[0].shape[1] == n_ch

    def test_all_channels_when_none(self):
        """n_active_channels=None should use all channels."""
        sim = make_dense_probe_run_data(
            n_channels=30,
            n_active_channels=None,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 50, 10),
            seed=0,
        )
        assert sim.spike_waveform_features[0].shape[1] == 30


class TestBackgroundSpikes:
    def test_background_adds_spikes(self):
        """Background rate > 0 should add more spikes than rate = 0."""
        kwargs = {
            "n_channels": 20,
            "n_active_channels": 10,
            "sampling_frequency": 500,
            "n_runs": 2,
            "place_field_means": np.arange(0, 50, 10),
            "seed": 0,
        }
        s_no_bg = make_dense_probe_run_data(background_rate=0.0, **kwargs)
        s_bg = make_dense_probe_run_data(background_rate=50.0, **kwargs)
        assert s_bg.spike_times[0].size > s_no_bg.spike_times[0].size

    def test_background_only(self):
        """With no place field neurons but background, should still get spikes."""
        sim = make_dense_probe_run_data(
            n_channels=20,
            n_active_channels=10,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.array([]),  # no neurons
            background_rate=100.0,
            seed=0,
        )
        assert sim.spike_times[0].size > 0


class TestFeatureTransform:
    def test_transform_applied(self):
        """Feature transform should change output dimensionality."""

        def reduce_to_3d(marks, channel_positions):
            # Fake spike localisation: weighted centroid + peak amplitude
            weights = np.abs(marks)
            total = weights.sum(axis=1, keepdims=True)
            total = np.where(total > 0, total, 1.0)
            x_est = (weights @ channel_positions[:, 0:1]) / total
            z_est = (weights @ channel_positions[:, 1:2]) / total
            amp = marks.max(axis=1, keepdims=True)
            return np.hstack([x_est, z_est, amp])

        sim = make_dense_probe_run_data(
            n_channels=30,
            n_active_channels=30,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 50, 10),
            feature_transform=reduce_to_3d,
            seed=0,
        )
        assert sim.spike_waveform_features[0].shape[1] == 3


class TestDimensionalitySweep:
    def test_returns_correct_count(self):
        dims = [4, 10, 20]
        outputs = make_dimensionality_sweep(
            dims,
            seed=0,
            n_channels=30,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 50, 10),
        )
        assert len(outputs) == 3

    def test_feature_widths_match(self):
        dims = [4, 10, 20]
        outputs = make_dimensionality_sweep(
            dims,
            seed=0,
            n_channels=30,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 50, 10),
        )
        for out, d in zip(outputs, dims, strict=True):
            assert out.spike_waveform_features[0].shape[1] == d

    def test_spike_times_identical_across_dimensions(self):
        """Spike times must be identical across runs — only marks change."""
        dims = [4, 10, 20]
        outputs = make_dimensionality_sweep(
            dims,
            seed=0,
            n_channels=30,
            sampling_frequency=500,
            n_runs=2,
            place_field_means=np.arange(0, 50, 10),
        )
        ref_times = outputs[0].spike_times[0]
        for i, out in enumerate(outputs[1:], 1):
            np.testing.assert_array_equal(
                out.spike_times[0],
                ref_times,
                err_msg=f"dim={dims[i]}: spike times diverged from dim={dims[0]}",
            )
