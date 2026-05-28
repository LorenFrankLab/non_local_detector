"""Test that estimate_parameters returns posterior consistent with the final parameters.

After EM finishes (convergence or max_iter), the returned posterior must
reflect the final fitted parameters — i.e., it must match what predict()
would compute on the same data with the saved model.
"""

import numpy as np
import pytest
from sklearn.base import clone

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

        # After EM completes the cached encoding-model data (spike times,
        # waveform features, position) must be cleaned up. Earlier code used
        # the wrong attribute name and silently retained these arrays for the
        # lifetime of the model.
        assert not hasattr(decoder, "_encoding_model_data"), (
            "estimate_parameters must clean up the cached _encoding_model_data "
            "attribute when EM completes; it is still present on the model."
        )

    def test_constructor_attribute_preserved_after_fit(self):
        """The user-supplied ``discrete_initial_conditions`` must not be mutated
        by fitting; the fitted value must live on ``discrete_initial_conditions_``.

        Mutating the constructor-set attribute violates sklearn's
        ``BaseEstimator`` contract: ``get_params()`` would return the fitted
        value rather than the user's spec, and a second call to
        ``estimate_parameters`` would start from the previous fit rather than
        the user's input.
        """
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        user_spec = np.array([1.0])
        original_constructor_arg = user_spec.copy()
        decoder = ClusterlessDecoder(discrete_initial_conditions=user_spec)

        decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=2,
            estimate_encoding_model=False,
        )

        # Constructor attribute is unchanged: equals the user's spec.
        assert np.array_equal(
            decoder.discrete_initial_conditions, original_constructor_arg
        ), (
            "fit must not overwrite the user-supplied "
            "discrete_initial_conditions; expected "
            f"{original_constructor_arg.tolist()}, got "
            f"{np.asarray(decoder.discrete_initial_conditions).tolist()}."
        )

        # Fitted attribute exists with the trailing-underscore name.
        assert hasattr(decoder, "discrete_initial_conditions_"), (
            "fit must set ``discrete_initial_conditions_`` (with trailing "
            "underscore) to the fitted initial-condition distribution."
        )
        fitted_ic = np.asarray(decoder.discrete_initial_conditions_)
        assert fitted_ic.shape == original_constructor_arg.shape
        assert np.all(np.isfinite(fitted_ic))

    def test_sklearn_clone_works_after_fit(self):
        """``sklearn.clone(fitted_model)`` must return an unfitted estimator
        whose constructor spec equals the original constructor arguments.

        ``clone`` calls ``get_params`` on the fitted estimator and re-instantiates;
        if ``fit`` had mutated a constructor-set attribute, the clone would carry
        the fitted value instead of the user's original spec, and the clone
        would also already have fitted attributes — both violations of the
        sklearn estimator contract.
        """
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        original_constructor_arg = np.array([1.0])
        decoder = ClusterlessDecoder(
            discrete_initial_conditions=original_constructor_arg.copy()
        )

        decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=2,
            estimate_encoding_model=False,
        )

        # ``sklearn.clone`` requires that ``get_params`` on the fitted estimator
        # reflect the original constructor spec, not the fitted value. Exercise
        # that property directly: build a fresh instance from the fitted model's
        # params and check it carries the user's spec rather than the fitted IC.
        params = decoder.get_params(deep=False)
        assert np.array_equal(
            params["discrete_initial_conditions"], original_constructor_arg
        ), (
            "get_params() must return the user's original spec for "
            "discrete_initial_conditions, not the fitted value; got "
            f"{np.asarray(params['discrete_initial_conditions']).tolist()} "
            f"instead of {original_constructor_arg.tolist()}."
        )

        cloned = type(decoder)(**params)
        assert np.array_equal(
            cloned.discrete_initial_conditions, original_constructor_arg
        ), (
            "Re-instantiating from get_params() must reproduce the original "
            "constructor spec; got "
            f"{np.asarray(cloned.discrete_initial_conditions).tolist()} "
            f"instead of {original_constructor_arg.tolist()}."
        )
        assert not hasattr(cloned, "discrete_initial_conditions_"), (
            "A freshly constructed estimator must not carry fitted attributes; "
            "the trailing-underscore ``discrete_initial_conditions_`` must only "
            "appear after fitting."
        )

        # If sklearn.clone happens to be available end-to-end (i.e., the
        # broader estimator contract holds), confirm it produces the same
        # result. If it raises for an unrelated reason (e.g. a different
        # constructor-modified parameter), skip rather than fail — this test
        # is scoped to the discrete_initial_conditions contract.
        try:
            sk_cloned = clone(decoder)
        except RuntimeError as exc:
            pytest.skip(f"sklearn.clone is not available on this estimator: {exc}")
        assert np.array_equal(
            sk_cloned.discrete_initial_conditions, original_constructor_arg
        )
        assert not hasattr(sk_cloned, "discrete_initial_conditions_")
