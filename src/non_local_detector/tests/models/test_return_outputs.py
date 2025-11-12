"""Tests for the return_outputs parameter in model predict() methods.

Tests the new string-based interface for controlling which outputs are returned
from predict() methods, as well as backward compatibility with the old boolean flags.
"""

import warnings

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def simple_fitted_detector():
    """Create a simple fitted detector for testing using simulated data."""
    # Generate simulated data
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=10)  # Small for speed

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    return detector, spike_times, time, position


@pytest.mark.unit
class TestReturnOutputsParameter:
    """Test the return_outputs parameter with various inputs."""

    def test_default_returns_smoother_only(self, simple_fitted_detector):
        """Test that default (None) returns only smoother and marginal likelihood."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
        )

        # Should have smoother outputs
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results
        assert "marginal_log_likelihoods" in results.attrs

        # Should NOT have filter, predictive, or log_likelihood
        assert "causal_posterior" not in results
        assert "causal_state_probabilities" not in results
        assert "predictive_state_probabilities" not in results
        assert "log_likelihood" not in results

    def test_return_filter_string(self, simple_fitted_detector):
        """Test return_outputs='filter' returns filter distribution."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs="filter",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results

        # Should have filter
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results

        # Should NOT have predictive or log_likelihood
        assert "predictive_state_probabilities" not in results
        assert "log_likelihood" not in results

        # Verify shapes
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        n_states = results.acausal_state_probabilities.shape[1]

        assert results.causal_posterior.shape == (n_time, n_state_bins)
        assert results.causal_state_probabilities.shape == (n_time, n_states)

    def test_return_predictive_string(self, simple_fitted_detector):
        """Test return_outputs='predictive' returns both predictive distributions."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs="predictive",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have both predictive outputs
        assert "predictive_state_probabilities" in results
        assert "predictive_posterior" in results

        # Should NOT have filter or log_likelihood
        assert "causal_posterior" not in results
        assert "log_likelihood" not in results

        # Verify shapes
        n_time = len(time)
        n_states = results.acausal_state_probabilities.shape[1]
        n_state_bins = results.acausal_posterior.shape[1]

        assert results.predictive_state_probabilities.shape == (n_time, n_states)
        assert results.predictive_posterior.shape == (n_time, n_state_bins)

        # Verify probabilities sum to 1 (both versions)
        predictive_sums = results.predictive_state_probabilities.sum(dim="states")
        assert np.allclose(predictive_sums.values, 1.0, atol=1e-10)

        predictive_posterior_sums = results.predictive_posterior.sum(dim="state_bins")
        assert np.allclose(predictive_posterior_sums.values, 1.0, atol=1e-10)

    def test_return_predictive_posterior_string(self, simple_fitted_detector):
        """Test return_outputs='predictive_posterior' returns full predictive posterior."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs="predictive_posterior",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have predictive_posterior but NOT predictive_state_probabilities
        assert "predictive_posterior" in results
        assert "predictive_state_probabilities" not in results

        # Should NOT have filter or log_likelihood
        assert "causal_posterior" not in results
        assert "log_likelihood" not in results

        # Verify shape of full version (state bins)
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        assert results.predictive_posterior.shape == (n_time, n_state_bins)

        # Verify probabilities sum to 1 (full version)
        predictive_posterior_sums = results.predictive_posterior.sum(dim="state_bins")
        assert np.allclose(predictive_posterior_sums.values, 1.0, atol=1e-10)

    def test_return_log_likelihood_string(self, simple_fitted_detector):
        """Test return_outputs='log_likelihood' returns log likelihoods."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs="log_likelihood",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have log_likelihood
        assert "log_likelihood" in results

        # Should NOT have filter or predictive
        assert "causal_posterior" not in results
        assert "predictive_state_probabilities" not in results

        # Verify shape
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        assert results.log_likelihood.shape == (n_time, n_state_bins)

    def test_return_all_string(self, simple_fitted_detector):
        """Test return_outputs='all' returns all outputs."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs="all",
        )

        # Should have everything
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results
        assert "predictive_state_probabilities" in results
        assert "predictive_posterior" in results
        assert "log_likelihood" in results
        assert "marginal_log_likelihoods" in results.attrs

    def test_return_multiple_outputs_list(self, simple_fitted_detector):
        """Test return_outputs with list of strings."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs=["filter", "log_likelihood"],
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have filter and log_likelihood
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results
        assert "log_likelihood" in results

        # Should NOT have predictive outputs
        assert "predictive_state_probabilities" not in results
        assert "predictive_posterior" not in results

    def test_return_multiple_outputs_set(self, simple_fitted_detector):
        """Test return_outputs with set of strings."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=time,
            return_outputs={"filter", "log_likelihood"},
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have filter and log_likelihood
        assert "causal_posterior" in results
        assert "log_likelihood" in results

        # Should NOT have predictive
        assert "predictive_state_probabilities" not in results

    def test_invalid_string_raises_error(self, simple_fitted_detector):
        """Test that invalid return_outputs string raises ValueError."""
        detector, spike_times, time, position = simple_fitted_detector

        with pytest.raises(ValueError, match="Invalid return_outputs"):
            detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                return_outputs="invalid_option",
            )

    def test_invalid_set_element_raises_error(self, simple_fitted_detector):
        """Test that invalid element in set raises ValueError."""
        detector, spike_times, time, position = simple_fitted_detector

        with pytest.raises(ValueError, match="Invalid outputs"):
            detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                return_outputs={"filter", "invalid_option"},
            )

    def test_invalid_type_raises_error(self, simple_fitted_detector):
        """Test that invalid type for return_outputs raises TypeError."""
        detector, spike_times, time, position = simple_fitted_detector

        with pytest.raises(TypeError, match="return_outputs must be"):
            detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                return_outputs=123,  # Invalid type
            )


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with old boolean flags."""

    def test_save_log_likelihood_to_results_still_works(self, simple_fitted_detector):
        """Test that old save_log_likelihood_to_results flag still works with warning."""
        detector, spike_times, time, position = simple_fitted_detector

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                save_log_likelihood_to_results=True,
            )

            # Should have issued deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Should have log_likelihood in results
        assert "log_likelihood" in results

    def test_save_causal_posterior_to_results_still_works(self, simple_fitted_detector):
        """Test that old save_causal_posterior_to_results flag still works with warning."""
        detector, spike_times, time, position = simple_fitted_detector

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                save_causal_posterior_to_results=True,
            )

            # Should have issued deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        # Should have causal posterior in results
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results

    def test_both_old_flags_work_together(self, simple_fitted_detector):
        """Test that both old flags work together."""
        detector, spike_times, time, position = simple_fitted_detector

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                save_log_likelihood_to_results=True,
                save_causal_posterior_to_results=True,
            )

            # Should have issued deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        # Should have both outputs
        assert "log_likelihood" in results
        assert "causal_posterior" in results

    def test_cannot_mix_old_and_new_interface(self, simple_fitted_detector):
        """Test that mixing old flags and new parameter raises error."""
        detector, spike_times, time, position = simple_fitted_detector

        with pytest.raises(ValueError, match="Cannot specify both"):
            detector.predict(
                spike_times=spike_times,
                time=time,
                position=position,
                position_time=time,
                return_outputs="filter",
                save_log_likelihood_to_results=True,
            )
