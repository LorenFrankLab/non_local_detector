"""A detector decodes with the likelihood algorithm that fitted its encoding model.

A detector unpacks the fit's encoding model into the algorithm's predict, so
the predict of another algorithm either rejects the keys or reads keys that
happen to share a name. Each encoding model records the algorithm that fitted
it, and a detector whose algorithm parameter no longer names that algorithm
refuses to decode. Unknown algorithm names are rejected at construction.
"""

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)

pytestmark = pytest.mark.unit

_N_DECODE = 200


@pytest.mark.parametrize(
    ("detector_cls", "parameter", "registry"),
    [
        (SortedSpikesDecoder, "sorted_spikes_algorithm", _SORTED_SPIKES_ALGORITHMS),
        (
            NonLocalSortedSpikesDetector,
            "sorted_spikes_algorithm",
            _SORTED_SPIKES_ALGORITHMS,
        ),
        (ClusterlessDecoder, "clusterless_algorithm", _CLUSTERLESS_ALGORITHMS),
        (
            NonLocalClusterlessDetector,
            "clusterless_algorithm",
            _CLUSTERLESS_ALGORITHMS,
        ),
    ],
)
def test_unknown_algorithm_is_rejected_at_construction(
    detector_cls, parameter, registry
):
    """The error names the bad value and every registered algorithm."""
    with pytest.raises(ValidationError, match="'nope'") as error:
        detector_cls(**{parameter: "nope"})
    for name in registry:
        assert name in str(error.value)


def test_removed_log_kde_name_is_rejected_at_construction_with_a_hint():
    """The removed ``clusterless_kde_log`` name points to ``clusterless_kde``."""
    with pytest.raises(ValidationError, match='Use "clusterless_kde"'):
        ClusterlessDecoder(clusterless_algorithm="clusterless_kde_log")


def _fit_sorted(algorithm, sorted_sim):
    time, position, spike_times = sorted_sim
    params = {"disable_progress_bar": True}
    if algorithm == "sorted_spikes_mrf":
        params["penalty"] = 1.0
    detector = SortedSpikesDecoder(
        sorted_spikes_algorithm=algorithm, sorted_spikes_algorithm_params=params
    )
    return detector.fit(time, position, spike_times)


def _predict_sorted(detector, sorted_sim):
    time, position, spike_times = sorted_sim
    return detector.predict(
        spike_times,
        time_edges=time_edges_from_centers(time[:_N_DECODE]),
        position_time=time,
        position=position,
        return_outputs="log_likelihood",
    )


@pytest.mark.parametrize(
    ("fitted", "switched"),
    [
        # The GLM predict rejects the KDE's keys with a TypeError.
        ("sorted_spikes_kde", "sorted_spikes_glm"),
        # The diffusion predict used to accept the KDE's keys and decode.
        ("sorted_spikes_kde", "sorted_spikes_diffusion"),
        ("sorted_spikes_diffusion", "sorted_spikes_mrf"),
    ],
)
def test_switching_the_algorithm_after_fit_raises(sorted_sim, fitted, switched):
    """Predict names both algorithms and says to refit; switching back decodes."""
    detector = _fit_sorted(fitted, sorted_sim)
    expected = _predict_sorted(detector, sorted_sim)

    detector.set_params(sorted_spikes_algorithm=switched)
    with pytest.raises(ValidationError, match="refit") as error:
        _predict_sorted(detector, sorted_sim)
    assert fitted in str(error.value)
    assert switched in str(error.value)

    detector.set_params(sorted_spikes_algorithm=fitted)
    restored = _predict_sorted(detector, sorted_sim)
    np.testing.assert_array_equal(
        np.asarray(restored.log_likelihood), np.asarray(expected.log_likelihood)
    )


def test_switching_the_clusterless_algorithm_after_fit_raises(clusterless_sim):
    """The clusterless detectors check the recorded algorithm too."""
    detector = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        clusterless_sim.position_time,
        clusterless_sim.position,
        clusterless_sim.spike_times,
        clusterless_sim.spike_waveform_features,
    )
    detector.set_params(clusterless_algorithm="clusterless_diffusion")
    with pytest.raises(ValidationError, match="clusterless_kde"):
        detector.predict(
            clusterless_sim.spike_times,
            clusterless_sim.spike_waveform_features,
            time_edges=time_edges_from_centers(
                clusterless_sim.position_time[:_N_DECODE]
            ),
        )


def test_encoding_model_without_a_recorded_algorithm_raises(sorted_sim):
    """A model whose encoding entry lacks the algorithm must be refit."""
    detector = _fit_sorted("sorted_spikes_kde", sorted_sim)
    (encoding_model,) = detector.encoding_model_.values()
    del encoding_model["algorithm"]
    with pytest.raises(ValidationError, match="refit") as error:
        _predict_sorted(detector, sorted_sim)
    assert "=None" not in str(error.value)


@pytest.mark.filterwarnings("ignore:EM did not converge:UserWarning")
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_em_refit_does_not_go_through_the_public_fit_encoding_model(
    family, sorted_sim, clusterless_sim, monkeypatch
):
    """Both families' EM refits call the private fit body, so an override of
    the public ``fit_encoding_model`` changes EM refits in neither."""
    if family == "sorted":
        time, position, spike_times = sorted_sim
        detector = SortedSpikesDecoder(
            sorted_spikes_algorithm_params={"disable_progress_bar": True}
        )
        data = {"spike_times": spike_times}
    else:
        time = clusterless_sim.position_time
        position = clusterless_sim.position
        detector = ClusterlessDecoder(
            clusterless_algorithm_params={"disable_progress_bar": True}
        )
        data = {
            "spike_times": clusterless_sim.spike_times,
            "spike_waveform_features": clusterless_sim.spike_waveform_features,
        }
    public_calls = []
    original = detector.fit_encoding_model

    def spy(*args, **kwargs):
        public_calls.append(kwargs.get("weights"))
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "fit_encoding_model", spy)
    detector.estimate_parameters(
        position_time=time,
        position=position,
        **data,
        time_edges=time_edges_from_centers(time),
        max_iter=1,
        estimate_encoding_model=True,
        min_encoding_local_mass=0.0,
        min_encoding_local_ess=0.0,
    )
    # The initial fit is public; the posterior-weighted refit is not.
    assert public_calls == [None]
