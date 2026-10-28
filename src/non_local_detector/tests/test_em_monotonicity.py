"""Test that EM log-likelihood is monotonically non-decreasing.

This is a fundamental guarantee of the EM algorithm: the marginal
log-likelihood must never decrease across iterations. Violation
indicates a bug in the E-step or M-step.

Tests cover:
- A four-state sorted-spikes detector with discrete-transition updates
- The local-mass/ESS guard that skips encoding-model refits
- Sorted spikes (GLM) encoding refit end-to-end smoke test
- Surfacing of monotonicity violations and max-iter exits via
  warnings and model attributes (``converged_``, ``n_iter_``,
  ``em_monotonicity_violations_``).
"""

import inspect
import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    DiscreteNonStationaryDiagonal,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.core import EM_MONOTONICITY_SLACK, check_converged
from non_local_detector.discrete_state_transitions import _transition_log_prior
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def sim():
    """Small 2-tetrode clusterless run dataset shared across the EM tests."""
    return make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 80, 20),
        n_runs=3,
        seed=42,
    )


@pytest.fixture(scope="module")
def sorted_sim():
    """Five-neuron sorted-spikes recording: its speed and the EM fit arguments.

    The first 10,000 samples of the seed-42 simulation, with spike trains cut
    to the same window. Long enough that the discrete transitions move, and that the default
    stickiness lowers the marginal log-likelihood within four EM iterations.
    """
    speed, position, spike_times, time, _, _, is_event, _ = make_simulated_data(
        seed=42, n_neurons=5
    )
    n = 10_000
    time = time[:n]
    return SimpleNamespace(
        speed=speed[:n],
        fit_args={
            "position_time": time,
            "position": position[:n],
            "spike_times": [st[st <= time[-1]] for st in spike_times],
            "time_edges": time_edges_from_centers(time),
            "is_training": ~is_event[:n],
        },
    )


def _shift_marginal(monkeypatch, model, shift):
    """Add ``shift(call_number)`` to the marginal log-likelihood of each E-step.

    E-steps are numbered from 1, and the final E-step after the loop counts.
    """
    original = model._predict
    calls = itertools.count(1)

    def shifted(*args, **kwargs):
        acausal, state_probabilities, marginal, *rest = original(*args, **kwargs)
        return (acausal, state_probabilities, marginal + shift(next(calls)), *rest)

    monkeypatch.setattr(model, "_predict", shifted)


def _assert_penalized_objective_rises(detector, results, caught):
    """Every EM check, including the final E-step's, rose by more than float
    rounding, and EM recorded and warned about no decrease."""
    objectives = np.asarray(results.attrs["em_objectives"])
    previous = np.asarray(results.attrs["em_previous_objectives"])
    assert objectives.shape == np.shape(results.attrs["marginal_log_likelihoods"])
    np.testing.assert_array_less(
        -EM_MONOTONICITY_SLACK * np.abs(objectives[1:]), objectives[1:] - previous
    )
    assert detector.em_monotonicity_violations_ == []
    assert not [w for w in caught if "decreased" in str(w.message)], [
        str(w.message) for w in caught
    ]


def _assert_monotonic(lls, label=""):
    """Assert that log-likelihoods are non-decreasing (with float tolerance)."""
    assert len(lls) >= 2, f"{label}: Expected at least 2 EM iterations, got {len(lls)}"
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - 1e-6, (
            f"{label}: EM log-likelihood decreased at iteration {i}: "
            f"{lls[i]:.6f} < {lls[i - 1]:.6f} "
            f"(diff={lls[i] - lls[i - 1]:.2e})"
        )


@pytest.mark.unit
def test_encoding_refit_is_opt_in():
    """The encoding refit is not an exact M-step, so no ``estimate_parameters``
    runs it by default."""
    from non_local_detector.models.base import (
        ClusterlessDetector,
        SortedSpikesDetector,
        _DetectorBase,
    )

    for cls in (_DetectorBase, ClusterlessDetector, SortedSpikesDetector):
        parameters = inspect.signature(cls.estimate_parameters).parameters
        assert parameters["estimate_encoding_model"].default is False, cls


@pytest.mark.slow
@pytest.mark.integration
class TestEMMonotonicity:
    """EM marginal log-likelihood must be non-decreasing across iterations."""

    @pytest.mark.parametrize("representation", ["dense", "structured"])
    def test_convergence_is_judged_per_decoded_time_bin(
        self, monkeypatch, sorted_sim, representation
    ):
        """Both EM paths pass ``check_converged`` the number of decoded time
        bins, and the max_iter warning reports the change per bin, the units
        of ``tolerance``."""
        from non_local_detector.models import base as base_module

        n_time_bins = len(sorted_sim.fit_args["time_edges"]) - 1
        calls = []

        def spy(current, previous, tolerance, **kwargs):
            calls.append(kwargs.get("n_time_bins"))
            return check_converged(current, previous, tolerance, **kwargs)

        monkeypatch.setattr(base_module, "check_converged", spy)
        detector = NonLocalSortedSpikesDetector()
        with pytest.warns(UserWarning, match="did not converge") as caught:
            results = detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=3,
                tolerance=0.0,
                estimate_encoding_model=False,
                transition_representation=representation,
            )

        # Two in-loop checks and the final E-step's.
        assert calls == [n_time_bins] * 3
        objectives = np.asarray(results.attrs["em_objectives"])
        previous = np.asarray(results.attrs["em_previous_objectives"])
        per_bin = (objectives[2] - previous[1]) / n_time_bins
        (message,) = [
            str(w.message) for w in caught if "did not converge" in str(w.message)
        ]
        assert f"change per time bin={per_bin:.3e}" in message

    def test_detector_em_monotonicity_with_transition_updates(self, sorted_sim):
        """Fit a four-state NonLocalSortedSpikesDetector without a transition
        prior (so the M-step maximizes the likelihood alone) and verify that
        marginal_log_likelihoods is non-decreasing while the discrete
        transitions are actually re-estimated."""
        detector = NonLocalSortedSpikesDetector(
            discrete_transition_concentration=1.0,
            discrete_transition_stickiness=np.zeros(4),
        )
        initial_transitions, _, _ = (
            detector.discrete_transition_type.make_state_transition()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EM stops at max_iter
            results = detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=5,
                estimate_encoding_model=False,
            )

        lls = results.attrs["marginal_log_likelihoods"]
        _assert_monotonic(lls, "NonLocalSortedSpikesDetector (discrete only)")
        assert detector.em_monotonicity_violations_ == []
        # The transition M-step ran: the fitted matrix moved off its start.
        assert (
            np.max(np.abs(detector.discrete_state_transitions_ - initial_transitions))
            > 1e-3
        )


@pytest.mark.slow
@pytest.mark.integration
class TestEncodingUpdateGuards:
    """Encoding-model refits during EM: the local-mass/ESS guard that skips a
    refit when the Local state has too little posterior support, an
    end-to-end GLM refit, and convergence surfacing for a well-behaved fit."""

    @pytest.mark.parametrize(
        ("min_encoding_local_mass", "refits_every_iteration"),
        [(1.0, True), (np.inf, False)],
        ids=["default-threshold", "unreachable-threshold"],
    )
    def test_local_mass_guard_skips_encoding_refit(
        self, monkeypatch, sorted_sim, min_encoding_local_mass, refits_every_iteration
    ):
        """With ``estimate_encoding_model=True`` the encoding model is refit
        once per EM iteration, unless the Local state's posterior mass is
        below ``min_encoding_local_mass``; an unreachable threshold skips
        every refit. The refit is replaced by a recorder, so only the guard's
        decision is under test."""
        detector = NonLocalSortedSpikesDetector()
        local_masses = []
        monkeypatch.setattr(
            detector,
            "_refit_encoding_model",
            lambda weights: local_masses.append(float(np.sum(weights))),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EM stops at max_iter
            detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=2,
                estimate_encoding_model=True,
                min_encoding_local_mass=min_encoding_local_mass,
            )

        if refits_every_iteration:
            assert len(local_masses) == detector.n_iter_ == 2
            assert all(mass >= min_encoding_local_mass for mass in local_masses)
        else:
            assert local_masses == []

    def test_well_behaved_em_no_warnings(self, sim):
        """A standard converging fit sets ``converged_`` to True, records
        no monotonicity violations, and emits no ``UserWarning``."""

        decoder = ClusterlessDecoder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=20,
                estimate_encoding_model=False,
            )

        assert decoder.converged_ is True
        assert decoder.em_monotonicity_violations_ == []
        assert decoder.n_iter_ >= 1
        # The forward-pass degenerate-step collector is wired through _predict
        # into the fitted attribute; a well-behaved fit yields an empty array.
        assert isinstance(decoder.degenerate_timesteps_, np.ndarray)
        assert decoder.degenerate_timesteps_.size == 0
        unexpected = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and (
                "did not converge" in str(w.message)
                or "decreased" in str(w.message)
                or "inconsistency between the E-step" in str(w.message)
            )
        ]
        assert unexpected == [], (
            "Unexpected EM convergence warnings emitted: "
            f"{[str(w.message) for w in unexpected]}"
        )

    def test_sorted_spikes_glm_encoding_runs_end_to_end(self, sorted_sim):
        """NonLocalSortedSpikesDetector with GLM encoding updates should
        run to completion without errors."""
        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_glm",
            sorted_spikes_algorithm_params={},
        )
        results = detector.estimate_parameters(
            **sorted_sim.fit_args,
            max_iter=3,
            estimate_encoding_model=True,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results


@pytest.mark.unit
class TestEMAttributeInitialization:
    """The three EM-result attributes are set all-or-nothing."""

    def test_validation_failure_leaves_no_partial_em_attributes(self):
        """A fit that raises during input validation must not leave a partial
        set of EM-result attributes on the model.

        Calls the base ``_DetectorBase.estimate_parameters`` directly with
        ``time_edges=None`` so validation fails immediately (before any model state
        or data is needed), isolating the attribute-initialization order.
        """
        from non_local_detector.exceptions import ValidationError
        from non_local_detector.models.base import _DetectorBase

        decoder = ClusterlessDecoder()
        with pytest.raises(ValidationError):
            _DetectorBase.estimate_parameters(
                decoder, time_edges=None, estimate_encoding_model=False
            )

        for attr in ("converged_", "n_iter_", "em_monotonicity_violations_"):
            assert not hasattr(decoder, attr), (
                f"{attr} was set despite a validation failure (partial state)"
            )


@pytest.mark.slow
@pytest.mark.integration
class TestEMConvergenceSurfacing:
    """Surfacing of EM monotonicity violations and max-iter exits."""

    def test_em_violation_warned_for_buggy_mstep(self, monkeypatch, sim):
        """Force ``check_converged`` to report a monotonicity violation
        on iteration 2 and verify a ``UserWarning`` is emitted and the
        iteration index is recorded in ``em_monotonicity_violations_``.

        Monotonicity violations are surfaced via ``warnings.warn`` (the same
        channel as the other EM fit-quality diagnostics), not ``logger``.
        """
        from non_local_detector.core import check_converged as _real_check_converged
        from non_local_detector.models import base as base_module

        call_count = {"n": 0}

        def fake_check_converged(curr, prev, tolerance, **kwargs):
            call_count["n"] += 1
            is_converged, _ = _real_check_converged(curr, prev, tolerance, **kwargs)
            # Report a non-monotonic step on the first comparison
            # (i.e., when n_iter==2 in the EM loop). Subsequent comparisons
            # report monotonic so the loop is allowed to converge.
            is_increasing = call_count["n"] != 1
            return is_converged, is_increasing

        monkeypatch.setattr(base_module, "check_converged", fake_check_converged)

        decoder = ClusterlessDecoder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=3,
                estimate_encoding_model=False,
            )

        assert decoder.em_monotonicity_violations_ == [2]
        violation_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "log-likelihood decreased" in str(w.message)
        ]
        assert violation_warnings, (
            "Expected a UserWarning describing the monotonicity violation; "
            f"got: {[str(w.message) for w in caught]}"
        )

    def test_final_e_step_decrease_is_not_convergence(self, monkeypatch, sim):
        """The final E-step evaluates the returned parameters, so a decrease
        there is a monotonicity violation and leaves the fit unconverged, even
        when the loop met its tolerance."""

        decoder = ClusterlessDecoder()
        # A tolerance any increase meets ends the loop at iteration 2, so the
        # final E-step is the third.
        _shift_marginal(monkeypatch, decoder, lambda n: -1000.0 if n == 3 else 0.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=20,
                tolerance=1.0,
                estimate_encoding_model=False,
            )

        messages = [str(w.message) for w in caught]
        # The loop met its tolerance: no max-iter warning.
        assert decoder.n_iter_ == 2
        assert not [m for m in messages if "did not converge" in m]
        assert decoder.converged_ is False
        assert decoder.em_monotonicity_violations_ == [decoder.n_iter_ + 1]
        final = [m for m in messages if "inconsistency between the E-step" in m]
        assert len(final) == 1
        assert "converged_ is False" in final[0]

    def test_max_iter_warning_emitted(self, sim):
        """A fit that exits because ``max_iter`` was reached (with an
        unreachable tolerance) sets ``converged_=False``, ``n_iter_==max_iter``,
        and emits a ``UserWarning`` mentioning "did not converge".
        """

        decoder = ClusterlessDecoder()
        with pytest.warns(UserWarning, match="did not converge"):
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=1,
                tolerance=1e-20,
                estimate_encoding_model=False,
            )

        assert decoder.converged_ is False
        assert decoder.n_iter_ == 1

    def test_max_iter_after_a_decrease_does_not_suggest_a_looser_tolerance(
        self, monkeypatch, sim
    ):
        """A decrease never counts as convergence at any tolerance, so the
        max-iter warning after one must not suggest relaxing tolerance."""

        decoder = ClusterlessDecoder()
        # Far larger than any real EM gain, so every E-step decreases.
        _shift_marginal(monkeypatch, decoder, lambda n: -1e6 * n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=3,
                estimate_encoding_model=False,
            )

        # Iterations 2 and 3, and the final E-step as iteration 4.
        assert decoder.em_monotonicity_violations_ == [2, 3, 4]
        max_iter_messages = [
            str(w.message) for w in caught if "did not converge" in str(w.message)
        ]
        assert len(max_iter_messages) == 1
        assert "decreased in the last iteration" in max_iter_messages[0]
        assert "relaxing tolerance" not in max_iter_messages[0]

    def test_negative_infinity_objective_is_not_a_violation(self, monkeypatch, sim):
        """A degenerate timestep makes every E-step's marginal log-likelihood
        ``-inf`` while the posterior stays finite. EM then records no
        monotonicity violations, and its max-iter warning says the objective
        is ``-inf`` instead of suggesting a looser tolerance."""

        decoder = ClusterlessDecoder()
        _shift_marginal(monkeypatch, decoder, lambda n: -np.inf)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=3,
                estimate_encoding_model=False,
            )

        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert decoder.em_monotonicity_violations_ == []
        assert not [message for message in messages if "decreased" in message]
        assert decoder.converged_ is False
        max_iter_messages = [m for m in messages if "did not converge" in m]
        assert len(max_iter_messages) == 1
        assert "is -inf" in max_iter_messages[0]
        assert "relaxing tolerance" not in max_iter_messages[0]

    @pytest.mark.parametrize(
        ("prior_weight", "expected_n_iter"),
        # A fixed prior stops at the first check that passes; a data-adaptive
        # prior keeps going while the transitions still move.
        [(0.0, 2), (0.5, 4)],
    )
    def test_adaptive_prior_also_requires_settled_transitions(
        self, monkeypatch, sorted_sim, prior_weight, expected_n_iter
    ):
        """With ``discrete_transition_prior_weight > 0`` the prior changes
        with each E-step, so a flat objective is not enough: every
        transition probability must also have moved by less than
        ``tolerance`` in the last M-step."""
        from non_local_detector.models import base as base_module

        # Every objective check passes, so only the transition change can
        # keep EM going.
        monkeypatch.setattr(
            base_module,
            "check_converged",
            lambda curr, prev, tolerance, **kwargs: (True, True),
        )
        detector = NonLocalSortedSpikesDetector(
            discrete_transition_prior_weight=prior_weight
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=4,
                tolerance=1e-12,
                estimate_encoding_model=False,
            )

        assert detector.n_iter_ == expected_n_iter
        assert detector.converged_ is (prior_weight == 0.0)
        max_iter_messages = [
            str(w.message) for w in caught if "did not converge" in str(w.message)
        ]
        if prior_weight > 0:
            (message,) = max_iter_messages
            assert "must also move every transition probability" in message
        else:
            assert max_iter_messages == []

    @pytest.mark.parametrize("transition_representation", ["dense", "structured"])
    def test_zero_iterations_returns_posterior(self, sim, transition_representation):
        """``max_iter=0`` skips EM but still returns the final E-step posterior."""

        decoder = ClusterlessDecoder()
        with pytest.warns(UserWarning, match="did not converge"):
            results = decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time_edges=time_edges_from_centers(sim.position_time),
                max_iter=0,
                estimate_encoding_model=False,
                transition_representation=transition_representation,
            )

        assert decoder.n_iter_ == 0
        assert "acausal_posterior" in results
        assert results.attrs["em_objectives"].shape == (1,)
        np.testing.assert_array_equal(
            results.attrs["em_objectives"],
            results.attrs["marginal_log_likelihoods"],
        )


@pytest.mark.slow
@pytest.mark.integration
class TestEMMonitorsMapObjective:
    """With a Dirichlet transition prior the M-step is a MAP update, so EM is
    guaranteed to increase marginal log-likelihood + log prior, not the
    marginal log-likelihood alone."""

    @pytest.mark.parametrize("transition_representation", ["dense", "structured"])
    def test_prior_driven_likelihood_decrease_is_not_a_violation(
        self, sorted_sim, transition_representation
    ):
        """The default stickiness (1e5 pseudo-counts on two diagonals) pulls the
        marginal log-likelihood down, while every check of the penalized
        log-likelihood rises."""
        detector = NonLocalSortedSpikesDetector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=4,
                tolerance=1e-14,
                estimate_encoding_model=False,
                transition_representation=transition_representation,
            )

        # The scenario under test: the prior pulls the likelihood down.
        assert np.any(np.diff(results.attrs["marginal_log_likelihoods"]) < 0)
        _assert_penalized_objective_rises(detector, results, caught)

    @pytest.mark.parametrize("transition_representation", ["dense", "structured"])
    def test_covariate_transitions_raise_the_penalized_objective(
        self, sorted_sim, transition_representation
    ):
        """Speed-dependent transitions with the default stickiness: every
        check of the penalized log-likelihood rises, and the final objective
        includes the covariate transitions' float64 log prior, without the
        default frozen No-Spike row."""
        detector = NonLocalSortedSpikesDetector(
            discrete_transition_type=DiscreteNonStationaryDiagonal(
                np.full(4, 0.98), formula="1 + speed"
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = detector.estimate_parameters(
                **sorted_sim.fit_args,
                discrete_transition_covariate_data={"speed": sorted_sim.speed},
                max_iter=4,
                tolerance=1e-14,
                estimate_encoding_model=False,
                transition_representation=transition_representation,
            )

        _assert_penalized_objective_rises(detector, results, caught)
        n_transitions = len(sorted_sim.fit_args["time_edges"]) - 2
        log_prior = _transition_log_prior(
            np.empty((n_transitions, 4, 4)),
            None,
            detector.discrete_transition_coefficients_,
            detector.discrete_transition_design_matrix_,
            detector.discrete_transition_concentration,
            detector.discrete_transition_stickiness,
            detector.discrete_transition_regularization,
            free_rows=~detector._frozen_discrete_transition_rows_mask_,
        )
        assert log_prior < 0
        np.testing.assert_allclose(
            results.attrs["em_objectives"][-1]
            - results.attrs["marginal_log_likelihoods"][-1],
            log_prior,
            rtol=1e-9,
        )

    @pytest.mark.parametrize("prior_weight", [0.0, 0.5])
    def test_saved_objectives_reproduce_every_check(
        self, monkeypatch, sorted_sim, prior_weight
    ):
        """``em_objectives[k + 1]`` and ``em_previous_objectives[k]`` are the
        two values check ``k`` compared, under one M-step's prior, so they
        reproduce the recorded violations. A data-adaptive prior changes with
        each E-step, so the previous objective then differs from the saved
        objective of the same E-step."""
        tolerance = 1e-14
        detector = NonLocalSortedSpikesDetector(
            discrete_transition_prior_weight=prior_weight
        )
        # This fit has no violations of its own, so lower the fourth E-step's
        # marginal log-likelihood to give the checks one to reproduce.
        _shift_marginal(monkeypatch, detector, lambda n: -1000.0 if n == 4 else 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = detector.estimate_parameters(
                **sorted_sim.fit_args,
                max_iter=6,
                tolerance=tolerance,
                estimate_encoding_model=False,
            )

        objectives = np.asarray(results.attrs["em_objectives"])
        previous = np.asarray(results.attrs["em_previous_objectives"])
        # One check per in-loop iteration after the first, plus the final E-step.
        assert previous.shape == (detector.n_iter_,)
        assert objectives.shape == (detector.n_iter_ + 1,)
        increasing = [
            check_converged(
                current,
                earlier,
                tolerance,
                n_time_bins=len(sorted_sim.fit_args["time_edges"]) - 1,
            )[1]
            for current, earlier in zip(objectives[1:], previous, strict=True)
        ]
        assert detector.em_monotonicity_violations_ == [4]
        assert detector.em_monotonicity_violations_ == [
            k + 2 for k, is_increasing in enumerate(increasing) if not is_increasing
        ]
        if prior_weight == 0.0:
            np.testing.assert_array_equal(previous, objectives[:-1])
        else:
            assert np.all(previous[1:] != objectives[1:-1])
