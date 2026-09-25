"""Encoding spikes are owned by interpolated group weights, not hard windows.

Each encoding spike's weight for a group is the group's per-sample weight
(training & encoding-group & environment mask, times any EM weight) linearly
interpolated to the spike time on the full position timeline. That weight is
the whole ownership rule: a spike with zero weight contributes nothing, a spike
near a mask transition contributes a fraction, and complementary groups
partition every spike exactly.

The detector used to select spikes with run windows widened by the first
position-time difference. The windows duplicated spikes that fell inside two
runs' windows, dropped spikes with positive weight when sampling was
nonuniform, and, for clusterless fits, passed a subset timeline across which
interpolation bridged mask gaps. These tests inspect the spikes, timeline, and
weights each registered fit receives and compare the implied per-event weights
with the canonical reference.
"""

from collections import defaultdict

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, SortedSpikesDecoder
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

FAMILIES = ["sorted", "clusterless"]


def _event_weights(call: dict) -> dict[float, float]:
    """Total weight the backend gives each spike time, as the backends compute it.

    Backends drop spikes outside ``[position_time[0], position_time[-1]]`` and
    weight the rest by ``np.interp(t, position_time, weights)`` (uniform when
    ``weights`` is None). Repeated spike times add, so a duplicated spike shows
    up as a doubled weight.
    """
    position_time = np.asarray(call["position_time"])
    weights = call["weights"]
    if weights is None:
        weights = np.ones_like(position_time)
    totals: dict[float, float] = defaultdict(float)
    for times in call["spike_times"]:
        times = np.asarray(times)
        times = times[(times >= position_time[0]) & (times <= position_time[-1])]
        for t, w in zip(
            times, np.interp(times, position_time, np.asarray(weights)), strict=True
        ):
            if w > 0:
                totals[float(t)] += float(w)
    return dict(totals)


def _canonical(spike_times, position_time, sample_weights) -> dict[float, float]:
    """Reference: interpolate the per-sample weights on the full timeline."""
    weights = np.interp(spike_times, position_time, sample_weights)
    return {
        float(t): float(w) for t, w in zip(spike_times, weights, strict=True) if w > 0
    }


def _assert_same_weights(actual, expected):
    assert sorted(actual) == pytest.approx(sorted(expected))
    for t, w in expected.items():
        assert actual[t] == pytest.approx(w, abs=1e-12), f"spike at t={t}"


def _spy(monkeypatch, family):
    """Replace the default registered fit with a recorder; return the call log."""
    calls: list[dict] = []

    def record(**kwargs):
        calls.append(kwargs)
        return {}

    registry, name = (
        (_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde")
        if family == "sorted"
        else (_CLUSTERLESS_ALGORITHMS, "clusterless_kde")
    )
    monkeypatch.setitem(registry, name, (record, registry[name][1]))
    return calls


def _fit(detector, family, position_time, spike_times, **fit_kwargs):
    """Fit with a linear position ramp; clusterless features encode spike time."""
    position = np.linspace(0.0, 100.0, position_time.shape[0])[:, np.newaxis]
    if family == "sorted":
        return detector.fit(position_time, position, spike_times, **fit_kwargs)
    features = [np.column_stack([t, -t]) for t in spike_times]
    return detector.fit(position_time, position, spike_times, features, **fit_kwargs)


def _single_group_detector(family):
    cls = SortedSpikesDecoder if family == "sorted" else ClusterlessDecoder
    return cls(
        environments=Environment(place_bin_size=10.0), infer_track_interior=False
    )


def _assert_features_aligned(call):
    """Clusterless features must still belong to their spikes after selection."""
    for times, features in zip(
        call["spike_times"], call["spike_waveform_features"], strict=True
    ):
        np.testing.assert_array_equal(np.asarray(features)[:, 0], np.asarray(times))


@pytest.mark.unit
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("position_time", "spike_times"),
    [
        # A gap sample: the boundary spikes are half-owned, the gap spike is not
        # owned at all. Clusterless used to count t=2 twice at full weight.
        (np.arange(5.0), np.array([1.5, 2.0, 2.5])),
        # Nonuniform sampling: t=1.2 has weight 0.53 but lay outside the
        # first-difference (0.5) window.
        (np.array([0.0, 0.5, 2.0, 3.0, 4.0]), np.array([1.2, 2.5])),
        # A short gap after a long step: both run windows contain t=1.05 (weight
        # 0.5), so it was counted twice.
        (np.array([0.0, 1.0, 1.1, 1.2, 2.2]), np.array([0.5, 1.05, 1.15, 2.0])),
    ],
    ids=["gap-sample", "nonuniform-sampling", "overlapping-windows"],
)
def test_group_events_carry_interpolated_mask_weight(
    monkeypatch, family, position_time, spike_times
):
    calls = _spy(monkeypatch, family)
    is_training = np.array([True, True, False, True, True])

    _fit(
        _single_group_detector(family),
        family,
        position_time,
        [spike_times],
        is_training=is_training,
    )

    (call,) = calls
    # The full timeline and the mask reach the backend, so its occupancy and
    # interpolation see the gap.
    np.testing.assert_array_equal(call["position_time"], position_time)
    np.testing.assert_array_equal(call["weights"], is_training.astype(float))
    _assert_same_weights(
        _event_weights(call),
        _canonical(spike_times, position_time, is_training.astype(float)),
    )
    if family == "clusterless":
        _assert_features_aligned(call)


@pytest.mark.unit
@pytest.mark.parametrize("family", FAMILIES)
def test_complementary_groups_partition_every_event(monkeypatch, family):
    """Two complementary groups split each spike's EM weight exactly.

    Jittered sample times, fractional EM weights, and several group transitions
    exercise fractional boundary events; there is no disjoint-window requirement.
    """
    calls = _spy(monkeypatch, family)
    rng = np.random.default_rng(0)
    n = 60
    position_time = np.cumsum(rng.uniform(0.5, 1.5, n))
    groups = (np.arange(n) // 7) % 2
    spike_times = np.sort(rng.uniform(position_time[0], position_time[-1], 200))
    em_weights = rng.uniform(0.0, 1.0, n)
    cls = (
        ContFragSortedSpikesClassifier
        if family == "sorted"
        else (ContFragClusterlessClassifier)
    )
    detector = cls(
        observation_models=[
            ObservationModel(encoding_group=0),
            ObservationModel(encoding_group=1),
        ],
        environments=Environment(place_bin_size=10.0),
        infer_track_interior=False,
    )
    _fit(detector, family, position_time, [spike_times], encoding_group_labels=groups)
    calls.clear()

    data = {
        "position_time": position_time,
        "position": np.linspace(0.0, 100.0, n)[:, np.newaxis],
        "spike_times": [spike_times],
        "encoding_group_labels": groups,
        "weights": em_weights,
    }
    if family == "clusterless":
        data["spike_waveform_features"] = [np.column_stack([spike_times, -spike_times])]
    detector.fit_encoding_model(**data)

    assert len(calls) == 2
    totals: dict[float, float] = defaultdict(float)
    for call, group in zip(calls, (0, 1), strict=True):
        expected = _canonical(
            spike_times, position_time, em_weights * (groups == group)
        )
        _assert_same_weights(_event_weights(call), expected)
        for t, w in _event_weights(call).items():
            totals[t] += w
        if family == "clusterless":
            _assert_features_aligned(call)
    _assert_same_weights(totals, _canonical(spike_times, position_time, em_weights))


CLUSTERLESS_ALGORITHMS = [
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
]


@pytest.fixture(scope="module")
def gapped_clusterless_run():
    """Simulated clusterless run with a training mask that has many short gaps."""
    sim = make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.arange(0, 80, 20), n_runs=2, seed=0
    )
    n = sim.position_time.shape[0]
    is_training = (np.arange(n) // 150) % 4 != 3
    return sim, is_training


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", CLUSTERLESS_ALGORITHMS)
def test_clusterless_exposure_uses_full_timeline_mask(
    gapped_clusterless_run, algorithm
):
    """Every clusterless backend takes its occupancy and mean rate from the
    mask-weighted full timeline.

    A zero-weight sample contributes no occupancy, so occupancy matches a fit on
    the masked samples alone. The mean rate is the canonical weighted event
    count per weighted sample, so gap spikes are excluded and boundary spikes
    count fractionally, instead of being bridged by a subset timeline.
    """
    sim, is_training = gapped_clusterless_run
    detector = ClusterlessDecoder(clusterless_algorithm=algorithm).fit(
        sim.position_time,
        sim.position,
        sim.spike_times,
        sim.spike_waveform_features,
        is_training=is_training,
    )
    (model,) = detector.encoding_model_.values()

    mask = is_training.astype(float)
    expected_rates = [
        sum(_canonical(times, sim.position_time, mask).values()) / mask.sum()
        for times in sim.spike_times
    ]
    np.testing.assert_allclose(model["mean_rates"], expected_rates, rtol=1e-5)

    fit, _ = _CLUSTERLESS_ALGORITHMS[algorithm]
    subset = fit(
        position_time=sim.position_time[is_training],
        position=sim.position[is_training],
        spike_times=[t[:0] for t in sim.spike_times],
        spike_waveform_features=[f[:0] for f in sim.spike_waveform_features],
        environment=detector.environments[0],
        **detector._resolve_clusterless_algorithm_params(),
    )
    key = "log_occupancy" if algorithm == "clusterless_gmm" else "occupancy"
    np.testing.assert_allclose(model[key], subset[key], rtol=1e-4, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", CLUSTERLESS_ALGORITHMS)
def test_clusterless_group_without_training_coverage(gapped_clusterless_run, algorithm):
    """A group observed only outside training owns no spikes and has zero
    exposure. It warns, fits zero-rate electrodes, and still decodes. The subset
    timeline was empty here and the backends raised instead."""
    sim, is_training = gapped_clusterless_run
    decoder = ClusterlessDecoder(
        clusterless_algorithm=algorithm,
        observation_models=[ObservationModel(encoding_group=1)],
    )
    with pytest.warns(UserWarning, match="no training samples"):
        decoder.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
            is_training=is_training,
            encoding_group_labels=(~is_training).astype(int),
        )

    (model,) = decoder.encoding_model_.values()
    np.testing.assert_array_equal(np.asarray(model["mean_rates"]), 0.0)
    posterior = decoder.predict(
        sim.spike_times, sim.spike_waveform_features, time=sim.position_time[:500]
    ).acausal_posterior.values
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, rtol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm",
    [
        "sorted_spikes_kde",
        "sorted_spikes_glm",
        "sorted_spikes_diffusion",
        "sorted_spikes_mrf",
    ],
)
def test_sorted_group_without_training_coverage(algorithm):
    """The sorted family warns for a group with no training samples and still
    decodes a normalized posterior."""
    sim = make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.arange(0, 80, 20), n_runs=1, seed=0
    )
    n = sim.position_time.shape[0]
    is_training = np.arange(n) < n // 2
    spike_times = [t[t < sim.position_time[-1]] for t in sim.spike_times]
    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm=algorithm,
        observation_models=[ObservationModel(encoding_group=1)],
    )
    with pytest.warns(UserWarning, match="no training samples"):
        decoder.fit(
            sim.position_time,
            sim.position,
            spike_times,
            is_training=is_training,
            encoding_group_labels=(~is_training).astype(int),
        )

    posterior = decoder.predict(
        spike_times, time=sim.position_time[:500]
    ).acausal_posterior.values
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, rtol=1e-5)
