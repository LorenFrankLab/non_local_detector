"""Test that estimate_parameters returns posterior consistent with the final parameters.

After EM finishes (convergence or max_iter), the returned posterior must
reflect the final fitted parameters — i.e., it must match what predict()
would compute on the same data with the saved model. See issue #26.
"""

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


@pytest.mark.slow
@pytest.mark.integration
class TestEstimateParametersPredictConsistency:
    """estimate_parameters must return the posterior of the final params."""

    def test_clusterless_decoder_posterior_matches_predict(self):
        """Posterior returned by estimate_parameters must equal predict() with
        the same data after fitting.

        Before the fix, estimate_parameters returns the posterior from the
        E-step at the *start* of the final iteration — computed with
        parameters from the *previous* M-step — while predict() uses the
        final M-step parameters. The two diverge whenever EM does not
        converge exactly within tolerance.
        """
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        decoder = ClusterlessDecoder()
        est_results = decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=3,
            estimate_encoding_model=False,
        )

        pred_results = decoder.predict(
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            position_time=sim.position_time,
            position=sim.position,
        )

        np.testing.assert_allclose(
            est_results.acausal_posterior.values,
            pred_results.acausal_posterior.values,
            atol=1e-10,
            err_msg=(
                "Posterior from estimate_parameters does not match predict() "
                "with the same fitted parameters. estimate_parameters may be "
                "returning a posterior computed before the final M-step."
            ),
        )
        np.testing.assert_allclose(
            est_results.acausal_state_probabilities.values,
            pred_results.acausal_state_probabilities.values,
            atol=1e-10,
        )

        # The final marginal log-likelihood stored in estimate_parameters
        # must match the marginal log-likelihood reported by predict()
        est_final_ll = est_results.attrs["marginal_log_likelihoods"][-1]
        pred_ll = pred_results.attrs["marginal_log_likelihoods"]
        pred_ll_scalar = float(pred_ll[-1]) if np.ndim(pred_ll) > 0 else float(pred_ll)
        assert np.isclose(est_final_ll, pred_ll_scalar, atol=1e-10), (
            f"Final marginal LL from estimate_parameters ({est_final_ll}) does "
            f"not match predict() LL ({pred_ll_scalar})."
        )


@pytest.mark.unit
class TestEncodingModelDataCleanup:
    """estimate_parameters must delete the cached _encoding_model_data attribute.

    The subclass ``estimate_parameters`` unconditionally sets
    ``self._encoding_model_data`` (spike times, waveform features, position)
    before delegating to the base EM loop, which deletes it on completion.
    Earlier code used the wrong attribute name in the ``hasattr``/``del``
    cleanup, so these (potentially large) arrays were silently retained for the
    lifetime of every fitted model. This is a fast, isolated regression test so
    the cleanup contract is checked even when slow/integration tests are
    deselected.
    """

    def test_estimate_parameters_deletes_encoding_model_data(self):
        sim = make_simulated_run_data(
            n_tetrodes=1,
            place_field_means=np.arange(0, 40, 20),  # 2 neurons on 1 tetrode
            n_runs=1,
            seed=0,
        )

        decoder = ClusterlessDecoder()

        # A freshly-constructed decoder has not cached anything yet.
        assert not hasattr(decoder, "_encoding_model_data")

        decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=1,
            estimate_encoding_model=False,
        )

        # The attribute is always created during estimate_parameters, so its
        # absence afterward is a genuine, non-vacuous check: pre-fix the
        # mismatched cleanup name left it attached to the model.
        assert not hasattr(decoder, "_encoding_model_data"), (
            "estimate_parameters must clean up the cached _encoding_model_data "
            "attribute when EM completes; it is still present on the model."
        )
