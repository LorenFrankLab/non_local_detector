"""Tests for probe presets, multi-shank simulation, and spike localization.

Covers:
- ProbeConfig presets (Neuropixels, polymer, tetrode)
- Multi-shank simulation via make_probe_run_data
- Spike localization and local amplitude extraction
- Integration: localization as feature_transform in simulation
"""

from functools import partial

import numpy as np
import pytest

from non_local_detector.simulate.dense_probe_simulation import make_probe_run_data
from non_local_detector.simulate.probe_geometry import (
    neuropixels_config,
    polymer_probe_config,
    tetrode_config,
)
from non_local_detector.simulate.spike_localization import (
    estimate_spike_position,
    extract_local_amplitudes,
    localize_spikes,
)

# ---------------------------------------------------------------------------
# Probe preset tests
# ---------------------------------------------------------------------------


class TestProbePresets:
    def test_neuropixels_config(self):
        cfg = neuropixels_config()
        assert cfg.n_channels_per_shank == 384
        assert cfg.n_shanks == 1
        assert cfg.n_columns == 2

    def test_polymer_config(self):
        cfg = polymer_probe_config()
        assert cfg.n_channels_per_shank == 32
        assert cfg.n_shanks == 4
        assert cfg.n_columns == 1

    def test_tetrode_config(self):
        cfg = tetrode_config(n_tetrodes=8)
        assert cfg.n_channels_per_shank == 4
        assert cfg.n_shanks == 8

    def test_make_channel_positions_single_shank(self):
        cfg = neuropixels_config()
        positions = cfg.make_channel_positions()
        assert len(positions) == 1
        assert positions[0].shape == (384, 2)

    def test_make_channel_positions_multi_shank(self):
        cfg = polymer_probe_config()
        positions = cfg.make_channel_positions()
        assert len(positions) == 4
        for pos in positions:
            assert pos.shape == (32, 2)

    def test_shanks_are_spatially_separated(self):
        cfg = polymer_probe_config()
        positions = cfg.make_channel_positions()
        # Each shank should be offset by shank_spacing in x
        for i in range(1, len(positions)):
            x_offset = positions[i][:, 0].mean() - positions[0][:, 0].mean()
            np.testing.assert_allclose(x_offset, i * cfg.shank_spacing, atol=1.0)


# ---------------------------------------------------------------------------
# Multi-shank simulation tests
# ---------------------------------------------------------------------------


class TestMultiShankSimulation:
    @pytest.fixture(scope="class")
    def polymer_sim(self):
        cfg = polymer_probe_config()
        return make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=2,
            place_field_means=np.arange(0, 100, 5),  # 20 neurons
            seed=0,
        )

    def test_electrode_count_matches_shanks(self, polymer_sim):
        assert len(polymer_sim.spike_times) == 4
        assert len(polymer_sim.spike_waveform_features) == 4

    def test_features_per_shank(self, polymer_sim):
        for features in polymer_sim.spike_waveform_features:
            if features.shape[0] > 0:
                assert features.shape[1] == 32  # n_channels_per_shank

    def test_has_spikes_on_some_shanks(self, polymer_sim):
        total_spikes = sum(t.size for t in polymer_sim.spike_times)
        assert total_spikes > 0

    def test_times_strictly_increasing_per_shank(self, polymer_sim):
        for i, times in enumerate(polymer_sim.spike_times):
            if times.size > 1:
                assert np.all(np.diff(times) > 0), f"Shank {i}: not increasing"

    def test_features_finite(self, polymer_sim):
        for i, features in enumerate(polymer_sim.spike_waveform_features):
            assert np.all(np.isfinite(features)), f"Shank {i}: non-finite"

    def test_position_shape(self, polymer_sim):
        assert polymer_sim.position.ndim == 2
        assert polymer_sim.position.shape[1] == 1

    def test_environment_fitted(self, polymer_sim):
        assert hasattr(polymer_sim.environment, "place_bin_centers_")

    def test_deterministic_seeding(self):
        cfg = polymer_probe_config()
        kwargs = {
            "sampling_frequency": 500,
            "n_runs": 1,
            "place_field_means": np.arange(0, 50, 10),
            "seed": 42,
        }
        s1 = make_probe_run_data(cfg, **kwargs)
        s2 = make_probe_run_data(cfg, **kwargs)
        for i in range(cfg.n_shanks):
            np.testing.assert_array_equal(s1.spike_times[i], s2.spike_times[i])
            np.testing.assert_array_equal(
                s1.spike_waveform_features[i], s2.spike_waveform_features[i]
            )


class TestTetrodeSimulation:
    def test_tetrode_produces_correct_electrode_count(self):
        cfg = tetrode_config(n_tetrodes=5)
        sim = make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 100, 10),
            seed=0,
        )
        assert len(sim.spike_times) == 5
        for features in sim.spike_waveform_features:
            if features.shape[0] > 0:
                assert features.shape[1] == 4


class TestNeuropixelsSimulation:
    def test_single_electrode_output(self):
        cfg = neuropixels_config()
        sim = make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 100, 20),  # 5 neurons
            seed=0,
        )
        assert len(sim.spike_times) == 1
        if sim.spike_waveform_features[0].shape[0] > 0:
            assert sim.spike_waveform_features[0].shape[1] == 384


# ---------------------------------------------------------------------------
# Spike localization tests
# ---------------------------------------------------------------------------


class TestExtractLocalAmplitudes:
    @pytest.fixture()
    def simple_marks(self):
        """Marks where each spike has a clear peak on one channel."""
        rng = np.random.default_rng(0)
        n_channels = 10
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 35.0,
            ]
        )
        # 3 spikes with peaks at channels 2, 5, 8
        marks = rng.normal(0, 0.1, (3, n_channels))
        marks[0, 2] = 10.0
        marks[1, 5] = 10.0
        marks[2, 8] = 10.0
        return marks, channel_positions

    def test_shape_n_neighbors_1(self, simple_marks):
        marks, pos = simple_marks
        result = extract_local_amplitudes(marks, pos, n_neighbors=1)
        assert result.shape == (3, 3)

    def test_shape_n_neighbors_2(self, simple_marks):
        marks, pos = simple_marks
        result = extract_local_amplitudes(marks, pos, n_neighbors=2)
        assert result.shape == (3, 5)

    def test_peak_is_first_feature(self, simple_marks):
        marks, pos = simple_marks
        result = extract_local_amplitudes(marks, pos, n_neighbors=1)
        # First feature should be the peak amplitude
        np.testing.assert_allclose(result[:, 0], 10.0, atol=0.5)

    def test_handles_edge_channels(self):
        """Peak at first or last channel should not crash."""
        n_channels = 5
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 35.0,
            ]
        )
        marks = np.zeros((2, n_channels))
        marks[0, 0] = 10.0  # peak at first channel
        marks[1, 4] = 10.0  # peak at last channel
        result = extract_local_amplitudes(marks, channel_positions, n_neighbors=1)
        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))


class TestEstimateSpikePosition:
    def test_position_near_peak_channel(self):
        """Estimated position should be near the peak channel position."""
        n_channels = 10
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 35.0,
            ]
        )
        # Spike with peak at channel 5 (z=175)
        marks = np.zeros((1, n_channels))
        marks[0, 5] = 10.0
        marks[0, 4] = 2.0
        marks[0, 6] = 2.0

        pos = estimate_spike_position(marks, channel_positions, n_neighbors=1)
        assert pos.shape == (1, 2)
        # z should be near 175 (channel 5), pulled slightly toward 4 and 6
        assert abs(pos[0, 1] - 175.0) < 10.0

    def test_position_between_channels(self):
        """Spike between two channels should estimate intermediate position."""
        n_channels = 10
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 35.0,
            ]
        )
        # Equal amplitude on channels 4 and 5
        marks = np.zeros((1, n_channels))
        marks[0, 4] = 10.0
        marks[0, 5] = 10.0

        pos = estimate_spike_position(marks, channel_positions, n_neighbors=1)
        # z should be halfway between ch4 (140) and ch5 (175)
        expected_z = (140.0 + 175.0) / 2.0
        assert abs(pos[0, 1] - expected_z) < 5.0

    def test_shape(self):
        n_channels = 20
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 20.0,
            ]
        )
        marks = np.random.default_rng(0).random((50, n_channels))
        pos = estimate_spike_position(marks, channel_positions, n_neighbors=2)
        assert pos.shape == (50, 2)


class TestLocalizeSpikes:
    @pytest.fixture()
    def setup(self):
        n_channels = 20
        channel_positions = np.column_stack(
            [
                np.zeros(n_channels),
                np.arange(n_channels) * 35.0,
            ]
        )
        rng = np.random.default_rng(0)
        marks = rng.normal(0, 0.1, (10, n_channels))
        for i in range(10):
            marks[i, i * 2] = 10.0  # spread peaks across channels
        return marks, channel_positions

    def test_mode_a_position_only(self, setup):
        marks, pos = setup
        result = localize_spikes(
            marks,
            pos,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=False,
        )
        # (x, z, peak_amp) = 3 features
        assert result.shape == (10, 3)

    def test_mode_b_local_amps_only(self, setup):
        marks, pos = setup
        result = localize_spikes(
            marks,
            pos,
            n_neighbors=1,
            include_position=False,
            include_local_amplitudes=True,
        )
        # 2*1+1 = 3 features
        assert result.shape == (10, 3)

    def test_mode_c_combined(self, setup):
        marks, pos = setup
        result = localize_spikes(
            marks,
            pos,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=True,
        )
        # 2 (position) + 3 (local amps) = 5 features
        assert result.shape == (10, 5)

    def test_mode_c_n_neighbors_2(self, setup):
        marks, pos = setup
        result = localize_spikes(
            marks,
            pos,
            n_neighbors=2,
            include_position=True,
            include_local_amplitudes=True,
        )
        # 2 (position) + 5 (local amps) = 7 features
        assert result.shape == (10, 7)

    def test_raises_when_nothing_selected(self, setup):
        marks, pos = setup
        with pytest.raises(ValueError, match="At least one"):
            localize_spikes(
                marks,
                pos,
                include_position=False,
                include_local_amplitudes=False,
            )

    def test_all_finite(self, setup):
        marks, pos = setup
        for kwargs in [
            {"include_position": True, "include_local_amplitudes": False},
            {"include_position": False, "include_local_amplitudes": True},
            {"include_position": True, "include_local_amplitudes": True},
        ]:
            result = localize_spikes(marks, pos, n_neighbors=1, **kwargs)
            assert np.all(np.isfinite(result)), f"Non-finite with {kwargs}"


# ---------------------------------------------------------------------------
# Integration: localization as feature_transform in simulation
# ---------------------------------------------------------------------------


class TestLocalizationAsTransform:
    def test_polymer_probe_with_localization(self):
        """Localization transform applied to multi-shank polymer probe sim."""
        cfg = polymer_probe_config()
        transform = partial(
            localize_spikes,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=True,
        )
        sim = make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=2,
            place_field_means=np.arange(0, 100, 5),
            feature_transform=transform,
            seed=0,
        )
        for i, features in enumerate(sim.spike_waveform_features):
            if features.shape[0] > 0:
                # 2 (position) + 3 (local amps with n_neighbors=1) = 5
                assert features.shape[1] == 5, (
                    f"Shank {i}: expected 5 features, got {features.shape[1]}"
                )

    def test_tetrode_with_localization(self):
        """Localization on tetrode (4 channels, n_neighbors=1 gives 3 local)."""
        cfg = tetrode_config(n_tetrodes=3)
        transform = partial(
            localize_spikes,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=True,
        )
        sim = make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=2,
            place_field_means=np.arange(0, 60, 10),
            feature_transform=transform,
            seed=0,
        )
        for features in sim.spike_waveform_features:
            if features.shape[0] > 0:
                # 2 (position) + 3 (local amps) = 5
                assert features.shape[1] == 5

    def test_position_only_transform(self):
        """Position-only mode gives 3 features (x, z, peak_amp)."""
        cfg = polymer_probe_config()
        transform = partial(
            localize_spikes,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=False,
        )
        sim = make_probe_run_data(
            cfg,
            sampling_frequency=500,
            n_runs=1,
            place_field_means=np.arange(0, 50, 10),
            feature_transform=transform,
            seed=0,
        )
        for features in sim.spike_waveform_features:
            if features.shape[0] > 0:
                assert features.shape[1] == 3
