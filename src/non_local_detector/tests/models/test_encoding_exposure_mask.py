"""Sorted-spike encoding fits must receive the training/group/environment mask.

The detector computes ``is_group = is_training & is_encoding & is_environment``
and selects that group's spikes with it, but historically passed ``weights=None``
to the encoding fit whenever the caller supplied no explicit weights. The fits
then substituted uniform weights, so occupancy accumulated over held-out samples,
other environments, and other encoding groups while only in-group spikes were
counted — biasing every place field low.

``weights`` is the model's *exposure*: ``weights[i]`` is how much position sample
``i`` counts toward occupancy, spike density, and the mean rate. ``None`` means
"uniform over every supplied sample", which is only true when the mask selects
everything.
"""

import warnings

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.common import RATE_EPS_HZ
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.observation_models import ObservationModel

SAMPLING_FREQUENCY = 100.0
DURATION = 40.0


@pytest.fixture
def run_data():
    """Position sweeping the track, one place cell, and a half-coverage mask."""
    time = np.arange(0.0, DURATION, 1.0 / SAMPLING_FREQUENCY)
    position = (50.0 + 40.0 * np.sin(2.0 * np.pi * time / 12.0))[:, np.newaxis]
    rng = np.random.default_rng(0)
    rate = 8.0 * np.exp(-0.5 * ((position[:, 0] - 70.0) / 8.0) ** 2) + 0.2
    fired = rng.random(time.shape[0]) < rate / SAMPLING_FREQUENCY
    # Spikes fall at continuous times inside a sample interval, as real ones do.
    spike_times = np.sort(
        time[fired] + rng.uniform(0.0, 1.0 / SAMPLING_FREQUENCY, int(fired.sum()))
    )
    spike_times = spike_times[spike_times <= time[-1]]
    is_training = time < DURATION / 2.0
    return time, position, [spike_times], is_training


@pytest.mark.unit
def test_detector_passes_exposure_mask_as_weights(monkeypatch, run_data):
    """The mask reaches the encoding fit as weights, not as ``None``.

    This is the regression guard for the defect itself: it inspects what the
    detector hands the registered fit function, so it fails whenever the mask is
    dropped, regardless of how the downstream fit happens to behave.
    """
    time, position, spike_times, is_training = run_data
    captured = {}

    def spy(*args, **kwargs):
        captured["weights"] = kwargs.get("weights")
        return fit_sorted_spikes_kde_encoding_model(*args, **kwargs)

    monkeypatch.setitem(
        _SORTED_SPIKES_ALGORITHMS,
        "sorted_spikes_kde",
        (spy, _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"][1]),
    )

    SortedSpikesDecoder(sampling_frequency=SAMPLING_FREQUENCY).fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=is_training,
    )

    weights = captured["weights"]
    assert weights is not None, "the detector dropped the exposure mask"
    np.testing.assert_allclose(weights, is_training.astype(float))


@pytest.mark.integration
def test_glm_group_without_training_coverage_returns_eps_model():
    """A group observed only outside training fits defined rates and can decode."""
    time = np.arange(400, dtype=float) / SAMPLING_FREQUENCY
    position = (50.0 + 40.0 * np.sin(2.0 * np.pi * time / 2.0))[:, None]
    is_training = time < 2.0
    encoding_groups = (~is_training).astype(int)
    # Group 1 has real spikes and position samples, but none belong to training.
    spike_times = [time[210:390:10] + 0.002]
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=10.0),
        observation_models=[ObservationModel(encoding_group=1)],
        sorted_spikes_algorithm="sorted_spikes_glm",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
        sampling_frequency=SAMPLING_FREQUENCY,
        infer_track_interior=False,
    )

    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=is_training,
        encoding_group_labels=encoding_groups,
    )

    encoding_model = detector.encoding_model_[("", 1)]
    assert np.all(np.isfinite(encoding_model["coefficients"]))
    interior = np.asarray(encoding_model["is_track_interior"])
    place_fields = np.asarray(encoding_model["place_fields"])
    np.testing.assert_allclose(
        place_fields[:, interior], RATE_EPS_HZ, rtol=1e-5, atol=0.0
    )

    results = detector.predict(
        spike_times=spike_times, time_edges=time_edges_from_centers(time[200:])
    )
    posterior = results.acausal_posterior.values
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, rtol=1e-5, atol=0.0)


@pytest.mark.unit
def test_masked_fit_matches_subset_fit(run_data):
    """Weighting the full arrays by the mask equals fitting the subset arrays.

    Exact only away from mask transitions: per-spike weights are interpolated, so
    a spike within one sample of a transition gets a fractional weight the subset
    fit would round to 0 or 1. The fixture's single transition is at the midpoint,
    and the assertion tolerance covers the one spike that can land beside it.
    """
    time, position, spike_times, is_training = run_data
    environment = Environment(place_bin_size=4.0).fit_place_grid(position)

    masked = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=environment,
        weights=is_training.astype(float),
        disable_progress_bar=True,
    )
    subset = fit_sorted_spikes_kde_encoding_model(
        position_time=time[is_training],
        position=position[is_training],
        spike_times=spike_times,
        environment=environment,
        weights=None,
        disable_progress_bar=True,
    )

    np.testing.assert_allclose(
        np.asarray(masked["place_fields"]),
        np.asarray(subset["place_fields"]),
        rtol=1e-3,
    )
    assert masked["mean_rates"][0] == pytest.approx(subset["mean_rates"][0], rel=1e-3)


def _group_exposure_warnings(kind, run_data, *, position, is_training, **fit_kwargs):
    """Fit a one-group decoder, returning its detector-level exposure warnings."""
    time, _, spike_times, _ = run_data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "sorted":
            SortedSpikesDecoder(
                sorted_spikes_algorithm_params={"disable_progress_bar": True},
                sampling_frequency=SAMPLING_FREQUENCY,
            ).fit(
                position_time=time,
                position=position,
                spike_times=spike_times,
                is_training=is_training,
                **fit_kwargs,
            )
        else:
            marks = [np.zeros((len(spike_times[0]), 2))]
            ClusterlessDecoder(
                clusterless_algorithm_params={"disable_progress_bar": True},
                sampling_frequency=SAMPLING_FREQUENCY,
            ).fit(
                time,
                position,
                spike_times,
                marks,
                is_training=is_training,
                **fit_kwargs,
            )
    return [w for w in caught if "Encoding group" in str(w.message)]


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["sorted", "clusterless"])
def test_group_trained_only_on_nan_positions_warns(run_data, kind):
    """Positive group weights on NaN-position samples are zero exposure."""
    time, position, _, _ = run_data
    position = position.copy()
    is_missing = (time > 10.0) & (time < 12.0)
    position[is_missing] = np.nan

    caught = _group_exposure_warnings(
        kind, run_data, position=position, is_training=is_missing
    )

    assert len(caught) == 1
    assert "no training samples with positive exposure" in str(caught[0].message)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["sorted", "clusterless"])
def test_group_trained_only_outside_encoding_time_range_warns(run_data, kind):
    """Positive group weights outside the encoding support are zero exposure."""
    time, position, _, _ = run_data

    caught = _group_exposure_warnings(
        kind,
        run_data,
        position=position,
        is_training=time > 30.0,
        encoding_time_range=(0.0, 20.0),
    )

    assert len(caught) == 1


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["sorted", "clusterless"])
def test_group_with_training_exposure_does_not_warn(run_data, kind):
    _, position, _, is_training = run_data

    caught = _group_exposure_warnings(
        kind, run_data, position=position, is_training=is_training
    )

    assert not caught


@pytest.mark.unit
def test_all_ones_weights_match_unweighted_fit(run_data):
    """All-ones weights give exactly the unweighted (``weights=None``) fit.

    A single group with full training coverage passes
    ``is_group.astype(float)``, which is all ones, so its encoding model is
    identical to the unweighted one.
    """
    time, position, spike_times, _ = run_data
    environment = Environment(place_bin_size=4.0).fit_place_grid(position)

    explicit_ones = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=environment,
        weights=np.ones(time.shape[0]),
        disable_progress_bar=True,
    )
    implicit_none = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=environment,
        weights=None,
        disable_progress_bar=True,
    )

    np.testing.assert_array_equal(
        np.asarray(explicit_ones["place_fields"]),
        np.asarray(implicit_none["place_fields"]),
    )
