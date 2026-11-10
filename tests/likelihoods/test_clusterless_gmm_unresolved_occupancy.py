"""Clusterless GMM at positions the encoding occupancy does not resolve.

The animal covers only part of the environment: 0-150 cm of a 0-300 cm track
(one or three electrodes), or the left 55 cm of a 100 x 100 cm arena. Beyond
the encoding data the occupancy and spike-position mixtures are tails of
different Gaussians, so their ratio ``p_gpi / p_occ`` grows without bound
(``inf`` in float32 on the track). Where the occupancy log density is below
the fit's ``log_occupancy_threshold`` (``log(1e-6)`` below its maximum over
the interior bins), the place intensity is zero and spikes are scored by the
background alone, as the diffusion likelihoods do.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    _ground_process_intensity,
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.common import RATE_EPS_HZ
from non_local_detector.likelihoods.mark_reference import (
    background_rate,
    fit_mark_references,
)
from non_local_detector.model_checking._clusterless_models import GMMElectrode
from non_local_detector.model_checking.clusterless import (
    ircm_transform,
    mdci_transform,
)
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = pytest.mark.unit

DT = 0.002
# 20,000 position samples per recording.
DURATION = 40.0
DECODE_DURATION = 0.5
LAYOUTS = ["track_one_electrode", "track_three_electrodes", "arena_two_electrodes"]


def _track(rng, fields):
    """0-150 cm of a 0-300 cm track; ``fields`` is ((center, mark amplitude), ...)."""
    environment = Environment(
        place_bin_size=3.0, position_range=[(0.0, 300.0)]
    ).fit_place_grid(np.linspace(0, 300, 101)[:, None])
    time = np.arange(0.0, DURATION, DT)
    position = (75 + 75 * np.sin(2 * np.pi * time / 20))[:, None]
    electrodes = []
    for electrode_fields in fields:
        rates = [
            (15 * np.exp(-0.5 * ((position[:, 0] - center) / 10) ** 2), amplitude)
            for center, amplitude in electrode_fields
        ]
        electrodes.append(rates)
    return environment, time, position, electrodes


def _arena(rng):
    """The left 55 cm of a 100 x 100 cm arena, two electrodes."""
    environment = Environment(
        place_bin_size=5.0, position_range=[(0.0, 100.0), (0.0, 100.0)]
    )
    grid = np.stack(
        np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41)), axis=-1
    ).reshape(-1, 2)
    environment = environment.fit_place_grid(grid, infer_track_interior=False)
    time = np.arange(0.0, DURATION, DT)
    position = np.column_stack(
        [
            27.5 + 27.5 * np.sin(2 * np.pi * time / 13.0),
            50 + 50 * np.sin(2 * np.pi * time / 7.3),
        ]
    )
    electrodes = []
    for (cx, cy), amplitude in (((15.0, 30.0), 60.0), ((40.0, 70.0), 90.0)):
        distance2 = (position[:, 0] - cx) ** 2 + (position[:, 1] - cy) ** 2
        electrodes.append([(20 * np.exp(-0.5 * distance2 / 12.0**2), amplitude)])
    return environment, time, position, electrodes


@pytest.fixture(scope="module", params=LAYOUTS)
def coverage(request):
    """An encoding fit on partial coverage, decoding spikes and far positions."""
    rng = np.random.default_rng(0)
    if request.param == "track_one_electrode":
        environment, time, position, electrodes = _track(
            rng, [((40.0, 60.0), (120.0, 90.0))]
        )
        far, edge_path = [250.0], ([140.0], [170.0])
    elif request.param == "track_three_electrodes":
        environment, time, position, electrodes = _track(
            rng, [((40.0, 60.0),), ((120.0, 90.0),), ((80.0, 75.0),)]
        )
        far, edge_path = [250.0], ([140.0], [170.0])
    else:
        environment, time, position, electrodes = _arena(rng)
        far, edge_path = [90.0, 50.0], ([40.0, 50.0], [95.0, 50.0])
    spike_times, marks, decode_spikes, decode_marks = [], [], [], []
    for fields in electrodes:
        times, features = [], []
        for rate, amplitude in fields:
            spiking = rng.random(time.size) < (rate + 0.5 / len(fields)) * DT
            times.append(time[spiking])
            features.append(amplitude + rng.normal(0, 5, (spiking.sum(), 4)))
        order = np.argsort(np.concatenate(times))
        spike_times.append(np.concatenate(times)[order])
        marks.append(np.concatenate(features)[order])
        decode_spikes.append(np.sort(rng.uniform(0.0, DECODE_DURATION, 10)))
        decode_marks.append(fields[0][1] + rng.normal(0, 5, (10, 4)))
    encoding = fit_clusterless_gmm_encoding_model(
        time, position, spike_times, marks, environment, disable_progress_bar=True
    )
    log_occupancy = np.asarray(encoding["log_occupancy"])
    unresolved = log_occupancy < encoding["log_occupancy_threshold"]
    assert unresolved.any()
    far_logp = encoding["occupancy_model"].score_samples(jnp.asarray([far]))
    assert float(far_logp[0]) < encoding["log_occupancy_threshold"]
    return {
        "layout": request.param,
        "environment": environment,
        "encoding": encoding,
        "mark_reference": fit_mark_references(marks, mark_reference_scale()),
        "unresolved": unresolved,
        "decode_spikes": decode_spikes,
        "decode_marks": decode_marks,
        "far": np.asarray(far),
        "edge_path": edge_path,
    }


def _predict(data, encoding, *, spikes=None, marks=None, **kwargs):
    """Decode ``data``'s decoding spikes (or the given ones) in 2 ms bins."""
    spikes = data["decode_spikes"] if spikes is None else spikes
    marks = data["decode_marks"] if marks is None else marks
    centers = np.arange(0.0, DECODE_DURATION, DT)
    return np.asarray(
        predict_clusterless_gmm_log_likelihood(
            kwargs.pop("position_time", None),
            kwargs.pop("position", None),
            spikes,
            marks,
            **{**encoding, **kwargs},
            mark_reference=data["mark_reference"],
            time_edges=time_edges_from_centers(centers),
        )
    )


def _local(data, encoding, position, *, spikes=None, marks=None):
    """Local log likelihood with the animal held at ``position``."""
    position_time = np.arange(0.0, DECODE_DURATION, DT)
    return _predict(
        data,
        encoding,
        spikes=spikes,
        marks=marks,
        position_time=position_time,
        position=np.tile(position, (position_time.size, 1)),
        is_local=True,
    ).ravel()


def _no_spikes(data):
    return [np.zeros(0) for _ in data["decode_spikes"]], [
        np.zeros((0, 4)) for _ in data["decode_marks"]
    ]


def _background_only(encoding):
    """The same encoding with every electrode zero-rate: background alone."""
    n_electrodes = len(encoding["joint_models"])
    return {
        **encoding,
        "joint_models": [None] * n_electrodes,
        "gpi_models": [None] * n_electrodes,
        "mean_rates": np.zeros(n_electrodes),
        "summed_ground_process_intensity": np.zeros_like(
            encoding["summed_ground_process_intensity"]
        ),
    }


def _independent_background(data, spikes, marks):
    """Background-only log likelihood per 2 ms row, computed here in float64.

    Each spike contributes ``log(beta q(m) dt)``, with ``q`` the electrode's
    Student-t mark reference, and each row ``-n_electrodes beta dt``.
    """
    edges = time_edges_from_centers(np.arange(0.0, DECODE_DURATION, DT))
    expected = np.full(edges.size - 1, -len(spikes) * RATE_EPS_HZ * DT)
    for times, features, reference in zip(
        spikes, marks, data["mark_reference"], strict=True
    ):
        log_q = scipy.stats.t.logpdf(
            features,
            reference.degrees_of_freedom,
            loc=reference.location,
            scale=reference.scale,
        ).sum(axis=1)
        rows = np.searchsorted(edges, times, side="right") - 1
        np.add.at(expected, rows, np.log(RATE_EPS_HZ * DT) + log_q)
    return expected


def _unthresholded_intensity(encoding):
    """Summed ``rate * p_gpi / p_occ`` at the interior bins with no threshold.

    The fit's own helper on the raw log occupancy, accumulated over electrodes
    in order, which is how fits formed the stored intensity before they kept a
    threshold (the mixtures they fit are unchanged).
    """
    bins = jnp.asarray(encoding["interior_place_bin_centers"])
    log_occupancy = jnp.asarray(encoding["log_occupancy"])
    summed = jnp.zeros_like(log_occupancy)
    for gpi, rate in zip(encoding["gpi_models"], encoding["mean_rates"], strict=True):
        if gpi is not None:
            summed += _ground_process_intensity(
                jnp.asarray(rate), gpi.score_samples(bins), log_occupancy
            )
    return np.asarray(summed)


def _encoding_without_threshold(encoding):
    """The encoding model as fits stored it before they kept a threshold."""
    old = {k: v for k, v in encoding.items() if k != "log_occupancy_threshold"}
    old["summed_ground_process_intensity"] = _unthresholded_intensity(encoding)
    return old


@pytest.mark.parametrize("bin_tile_size", [None, 7], ids=["untiled", "tiled"])
def test_non_local_log_likelihood_is_finite_at_unvisited_bins(coverage, bin_tile_size):
    log_likelihood = _predict(
        coverage, coverage["encoding"], bin_tile_size=bin_tile_size
    )

    assert np.all(np.isfinite(log_likelihood))


@pytest.mark.parametrize("bin_tile_size", [None, 7], ids=["untiled", "tiled"])
def test_unresolved_bins_score_only_the_background(coverage, bin_tile_size):
    """At unresolved bins the log likelihood is that of zero-rate electrodes,
    which is ``sum log(beta q(m) dt) - n_electrodes beta dt`` per row."""
    data = coverage
    unresolved = data["unresolved"]

    log_likelihood = _predict(data, data["encoding"], bin_tile_size=bin_tile_size)
    background = _predict(
        data, _background_only(data["encoding"]), bin_tile_size=bin_tile_size
    )
    independent = _independent_background(
        data, data["decode_spikes"], data["decode_marks"]
    )

    np.testing.assert_allclose(
        log_likelihood[:, unresolved], background[:, unresolved], rtol=1e-5
    )
    np.testing.assert_allclose(
        log_likelihood[:, unresolved],
        np.broadcast_to(independent[:, None], log_likelihood[:, unresolved].shape),
        rtol=1e-5,
    )


def test_fitted_intensity_is_zero_only_where_occupancy_is_unresolved(coverage):
    """Resolved bins keep the fitted ``rate * p_gpi / p_occ`` to float32 rounding."""
    encoding = coverage["encoding"]
    unresolved = coverage["unresolved"]
    intensity = np.asarray(encoding["summed_ground_process_intensity"])

    assert np.all(intensity[unresolved] == 0.0)
    np.testing.assert_allclose(
        intensity[~unresolved],
        _unthresholded_intensity(encoding)[~unresolved],
        rtol=1e-6,
    )


@pytest.mark.parametrize("with_spikes", [False, True], ids=["no_spikes", "spikes"])
def test_local_likelihood_outside_coverage_is_the_background(coverage, with_spikes):
    """The animal held at a position never visited during encoding."""
    data = coverage
    spikes, marks = (
        (data["decode_spikes"], data["decode_marks"])
        if with_spikes
        else _no_spikes(data)
    )

    log_likelihood = _local(
        data, data["encoding"], data["far"], spikes=spikes, marks=marks
    )
    background = _local(
        data,
        _background_only(data["encoding"]),
        data["far"],
        spikes=spikes,
        marks=marks,
    )

    np.testing.assert_allclose(log_likelihood, background, rtol=1e-5)
    np.testing.assert_allclose(
        log_likelihood, _independent_background(data, spikes, marks), rtol=1e-5
    )


def test_resolved_bins_decode_as_without_the_threshold(coverage):
    """The threshold changes nothing at bins whose occupancy is resolved."""
    data = coverage
    resolved = ~data["unresolved"]

    thresholded = _predict(data, data["encoding"])
    unthresholded = _predict(data, _encoding_without_threshold(data["encoding"]))

    np.testing.assert_array_equal(thresholded[:, resolved], unthresholded[:, resolved])


def _unthresholded_non_local(data, encoding):
    """Non-local log likelihood with no occupancy threshold, computed here.

    Per row ``-dt (stored intensity + n_electrodes beta)``, and per spike
    ``log(rate p_joint(x, m) / p_occ(x) + beta q(m)) + log dt``, in float64
    from the fitted mixtures' log densities.
    """
    edges = time_edges_from_centers(np.arange(0.0, DECODE_DURATION, DT))
    bins = np.asarray(encoding["interior_place_bin_centers"], float)
    log_occupancy = np.asarray(encoding["log_occupancy"], float)
    stored = np.asarray(encoding["summed_ground_process_intensity"], float)
    n_electrodes = len(encoding["joint_models"])
    expected = np.tile(-DT * (stored + n_electrodes * RATE_EPS_HZ), (edges.size - 1, 1))
    for joint, rate, times, features, reference in zip(
        encoding["joint_models"],
        encoding["mean_rates"],
        data["decode_spikes"],
        data["decode_marks"],
        data["mark_reference"],
        strict=True,
    ):
        log_background = np.log(RATE_EPS_HZ) + scipy.stats.t.logpdf(
            features,
            reference.degrees_of_freedom,
            loc=reference.location,
            scale=reference.scale,
        ).sum(axis=1)
        points = np.concatenate(
            [np.tile(bins, (len(times), 1)), np.repeat(features, len(bins), axis=0)],
            axis=1,
        )
        joint_logp = np.asarray(
            joint.score_samples(jnp.asarray(points)), float
        ).reshape(len(times), len(bins))
        place = np.log(float(rate)) + joint_logp - log_occupancy
        terms = np.logaddexp(place, log_background[:, None]) + np.log(DT)
        np.add.at(expected, np.searchsorted(edges, times, side="right") - 1, terms)
    return expected


def test_encoding_model_without_threshold_decodes_as_fitted(coverage):
    """A model stored before fits kept a threshold keeps its fitted values.

    Its non-local log likelihood is the unthresholded one at every bin, and its
    local expected count far from the data is the unthresholded ratio. With
    one electrode on the track that ratio overflows to ``inf``, as it did
    before thresholds, so the log likelihood is ``-inf`` there.
    """
    data = coverage
    old = _encoding_without_threshold(data["encoding"])
    stored = old["summed_ground_process_intensity"]

    non_local = _predict(data, old)
    np.testing.assert_allclose(
        non_local, _unthresholded_non_local(data, old), rtol=1e-5, atol=1e-5
    )

    no_spikes, no_marks = _no_spikes(data)
    local = _local(data, old, data["far"], spikes=no_spikes, marks=no_marks)
    at = jnp.asarray([data["far"]])
    log_occupancy = old["occupancy_model"].score_samples(at)
    unthresholded_rate = sum(
        float(
            _ground_process_intensity(
                jnp.asarray(rate), gpi.score_samples(at), log_occupancy
            )[0]
        )
        for gpi, rate in zip(old["gpi_models"], old["mean_rates"], strict=True)
    )
    expected = -DT * (unthresholded_rate + background_rate(len(old["gpi_models"])))
    np.testing.assert_allclose(local, expected, rtol=1e-5)
    if data["layout"] == "track_one_electrode":
        assert np.isposinf(stored).any()
        assert np.all(np.isneginf(non_local[:, np.isposinf(stored)]))
        assert np.isposinf(unthresholded_rate)


def test_model_checking_intensity_applies_the_decoder_threshold(coverage):
    """Model checking's GMM intensity is zero where decoding's is, and equal
    elsewhere; without a stored threshold it keeps the unthresholded ratio."""
    data = coverage
    unresolved = data["unresolved"]
    encoding = data["encoding"]
    n_electrodes = len(encoding["joint_models"])
    electrodes = [GMMElectrode(encoding, e) for e in range(n_electrodes)]
    bins = electrodes[0].bin_centers

    ground = sum(electrode.ground_intensity(bins) for electrode in electrodes)

    # Float64 here, float32 in the fit: intensities below float32's smallest
    # subnormal (1.4e-45 Hz) are zero there.
    np.testing.assert_allclose(
        ground, encoding["summed_ground_process_intensity"], rtol=1e-5, atol=1e-40
    )
    assert np.all(ground[unresolved] == 0.0)
    old = _encoding_without_threshold(encoding)
    log_occupancy = np.asarray(encoding["log_occupancy"], float)
    for e, electrode in enumerate(electrodes):
        log_intensity = electrode.log_intensity_at_bins(data["decode_marks"][e][:3])
        assert np.all(np.isneginf(log_intensity[unresolved]))
        assert np.all(np.isfinite(log_intensity[~unresolved]))
        with np.errstate(over="ignore"):  # the unthresholded ratio overflows far out
            old_ground = GMMElectrode(old, e).ground_intensity(bins)
            unthresholded = np.exp(
                np.log(float(encoding["mean_rates"][e]))
                + np.asarray(encoding["gpi_models"][e].score_samples(bins), float)
                - log_occupancy
            )
        new_ground = electrode.ground_intensity(bins)
        np.testing.assert_allclose(old_ground, unthresholded, rtol=1e-12)
        np.testing.assert_array_equal(old_ground[~unresolved], new_ground[~unresolved])


def _edge_crossing(data, n_spikes=30):
    """A trajectory from covered to uncovered positions with spikes along it.

    Returns the transform arguments, the spike positions and the place bin
    centers holding them.
    """
    start, end = (np.asarray(p, float) for p in data["edge_path"])
    position_time = np.arange(0.0, 1.0, DT)
    fraction = position_time / position_time[-1]
    position = start + fraction[:, None] * (end - start)
    spike_times = np.linspace(0.01, position_time[-1] - 0.01, n_spikes)
    spike_position = start + (spike_times / position_time[-1])[:, None] * (end - start)
    environment = data["environment"]
    interior = np.flatnonzero(environment.is_track_interior_flat_)
    spike_bins = environment.get_bin_ind(spike_position)
    assert np.isin(spike_bins, interior).all()
    bin_centers = np.asarray(environment.place_bin_centers_)[spike_bins]
    marks = data["decode_marks"][0][:1] + np.zeros((n_spikes, 4))
    return (position_time, position, spike_times, marks), spike_position, bin_centers


def _n_unresolved(data, points):
    log_occupancy = np.asarray(
        data["encoding"]["occupancy_model"].score_samples(jnp.asarray(points))
    )
    return int(np.sum(log_occupancy < data["encoding"]["log_occupancy_threshold"]))


@pytest.mark.parametrize("transform", [ircm_transform, mdci_transform])
def test_transforms_refuse_spikes_at_unresolved_positions(coverage, transform):
    """Both goodness-of-fit transforms refuse spikes the place intensity gives
    no probability, naming how many and the occupancy threshold. IRCM reads
    the intensity at each spike's position, MDCI at its place bin."""
    arguments, spike_position, bin_centers = _edge_crossing(coverage)
    n_unresolved = _n_unresolved(
        coverage, spike_position if transform is ircm_transform else bin_centers
    )
    assert 0 < n_unresolved < 30

    with pytest.raises(
        ValueError, match=rf"{n_unresolved} of 30 spikes .*log_occupancy_threshold"
    ):
        transform(coverage["encoding"], 0, *arguments)


@pytest.mark.parametrize("transform", [ircm_transform, mdci_transform])
def test_transforms_accept_an_encoding_model_without_threshold(coverage, transform):
    """Without a stored threshold the same spikes keep their unthresholded
    intensity and are transformed, as before."""
    arguments, _, _ = _edge_crossing(coverage)

    with np.errstate(over="ignore", under="ignore"):
        u, v = transform(
            _encoding_without_threshold(coverage["encoding"]), 0, *arguments
        )

    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
