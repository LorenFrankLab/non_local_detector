"""Every likelihood against simulations with known place fields.

1-D: sixteen cells with Gaussian place fields (sigma 4 cm, 25 Hz peak, 0.1 Hz
baseline) sit on the first two thirds of a 120 cm track. The animal runs laps
at 20 cm/s.

2-D: sixteen cells with anisotropic Gaussian fields (sigma 8 cm in x, 6 cm in
y) on a 4 x 4 grid in a 90 x 60 cm open field, which the animal covers on a
Lissajous path (more time near the walls than in the middle). The box,
the field grid and the field shapes are all asymmetric in x and y, so swapped
axes or a bandwidth applied to one dimension only cannot pass.

In both, cells are grouped four per tetrode, with Gaussian marks 80 units
apart (sd 8). Because the true rates are known, two things can be checked
directly:

- recovery: fitted place fields (sorted) or the summed ground-process
  intensity (clusterless) match the true ones;
- the likelihood itself: on held-out data, each model's non-local log
  likelihood per 20 ms bin tracks the exact Poisson log likelihood computed
  from the true rates (and, for clusterless, the true mark densities), both in
  shape across position bins and in absolute level.

A negative control scores each model against the truth with the cells
reassigned; the same metrics must then fail, so the thresholds discriminate.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.likelihoods.common import DEFAULT_WAVEFORM_STD

pytestmark = [
    pytest.mark.integration,
    # The simulated track is deliberately a 1-D open field.
    pytest.mark.filterwarnings("ignore:Position data appears to be 1D"),
    # The 2-D MRF keeps its default reduced rank (250 of 600 bins).
    pytest.mark.filterwarnings("ignore:MRF basis auto-capped"),
]

FS = 500.0
TRACK, SPEED = 120.0, 20.0
MEANS = np.linspace(10.0, 80.0, 16)
SIGMA, PEAK, BASELINE = 4.0, 25.0, 0.1
MARK_SPACING, MARK_SD, N_MARK_DIMS = 80.0, 8.0, 4
CELLS_PER_TETRODE = 4
DECODE_BIN = 0.02

FS_2D = 125.0
WIDTH, HEIGHT = 90.0, 60.0
LISSAJOUS_PERIODS = (17.3, 11.1)  # seconds, x and y
_GRID_X, _GRID_Y = np.meshgrid([15.0, 35.0, 55.0, 75.0], [12.0, 24.0, 36.0, 48.0])
MEANS_2D = np.c_[_GRID_X.ravel(), _GRID_Y.ravel()]  # (n_cells, 2)
SIGMA_2D = np.array([8.0, 6.0])
PLACE_BIN_SIZE_2D = 3.0

SORTED = (
    "sorted_spikes_kde",
    "sorted_spikes_glm",
    "sorted_spikes_diffusion",
    "sorted_spikes_mrf",
)
CLUSTERLESS = (
    "clusterless_kde",
    "clusterless_gmm",
    "clusterless_diffusion",
)
# Mark-kernel bandwidth of each clusterless backend at the detector defaults.
# A kernel estimate converges to the true mark density convolved with its
# kernel, so the KDE and diffusion backends are compared in absolute level
# against an exact likelihood whose mark sd is sqrt(MARK_SD**2 + h**2); the GMM
# fits the mark clusters themselves.
MARK_KERNEL_SD = {
    "clusterless_kde": DEFAULT_WAVEFORM_STD,
    "clusterless_gmm": 0.0,
    "clusterless_diffusion": DEFAULT_WAVEFORM_STD,
}

# Nats per spike; see test_nonlocal_likelihood_tracks_the_exact_likelihood.
LEVEL_TOLERANCE = {"clusterless_gmm": 1.5}


def _true_rates(x):
    """True 1-D rates, shape (n_positions, n_cells), Hz."""
    return BASELINE + PEAK * np.exp(-((x[:, None] - MEANS) ** 2) / (2 * SIGMA**2))


def _true_rates_2d(xy):
    """True 2-D rates at ``xy`` (n_positions, 2), shape (n_positions, n_cells), Hz."""
    z = (((xy[:, None, :] - MEANS_2D[None]) / SIGMA_2D) ** 2).sum(-1)
    return BASELINE + PEAK * np.exp(-0.5 * z)


def _spikes_and_marks(rng, time, rates, sampling_frequency, jitter_start):
    """Poisson spikes per cell and per tetrode, with Gaussian marks.

    A sample's spikes are spread uniformly over
    ``[t + jitter_start, t + jitter_start + 1 / sampling_frequency)``.
    """
    counts = rng.poisson(rates / sampling_frequency)
    spike_times = [
        np.sort(
            np.repeat(time, counts[:, cell])
            + rng.uniform(
                jitter_start,
                jitter_start + 1 / sampling_frequency,
                counts[:, cell].sum(),
            )
        )
        for cell in range(rates.shape[1])
    ]
    tetrode_times, tetrode_marks = [], []
    for tetrode in range(rates.shape[1] // CELLS_PER_TETRODE):
        cells = range(tetrode * CELLS_PER_TETRODE, (tetrode + 1) * CELLS_PER_TETRODE)
        times = np.concatenate([spike_times[c] for c in cells])
        marks = np.concatenate(
            [
                rng.normal(
                    MARK_SPACING * (c % CELLS_PER_TETRODE),
                    MARK_SD,
                    (spike_times[c].size, N_MARK_DIMS),
                )
                for c in cells
            ]
        )
        order = np.argsort(times)
        tetrode_times.append(times[order])
        tetrode_marks.append(marks[order])
    return {
        "spike_times": spike_times,
        "tetrode_times": tetrode_times,
        "tetrode_marks": tetrode_marks,
    }


def _simulate(seed, seconds):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, seconds, 1 / FS)
    phase = (time * SPEED) % (2 * TRACK)
    position = np.where(phase < TRACK, phase, 2 * TRACK - phase)
    spikes = _spikes_and_marks(rng, time, _true_rates(position), FS, 0.0)
    return {"time": time, "position": position[:, None], **spikes}


def _simulate_2d(seed, seconds, phase):
    """Lissajous coverage of the box; spikes jittered +/- half a sample."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, seconds, 1 / FS_2D)
    angle = 2 * np.pi * time[:, None] / np.array(LISSAJOUS_PERIODS) + [phase, phase + 1]
    half = np.array([WIDTH, HEIGHT]) / 2
    position = half + (half - 1.0) * np.sin(angle)
    spikes = _spikes_and_marks(rng, time, _true_rates_2d(position), FS_2D, -0.5 / FS_2D)
    return {"time": time, "position": position, **spikes}


@pytest.fixture(scope="module")
def recording():
    train = _simulate(0, 8 * 2 * TRACK / SPEED)
    test = _simulate(1, 2 * 2 * TRACK / SPEED)
    edges = np.arange(0.0, test["time"][-1], DECODE_BIN)
    return train, test, edges


@pytest.fixture(scope="module")
def recording_2d():
    train = _simulate_2d(0, 600.0, phase=0.0)
    test = _simulate_2d(1, 180.0, phase=2.0)
    edges = np.arange(0.0, test["time"][-1], DECODE_BIN)
    return train, test, edges


@pytest.fixture(scope="module")
def fits():
    """Fitted detectors shared by the recovery and likelihood tests."""
    return {}


def _exact_sorted(rates, spike_times, edges):
    """Exact non-local log likelihood, (n_rows, n_bins), without log(n!) terms.

    ``rates`` are the true rates at the bins, shape (n_bins, n_cells), Hz.
    """
    counts = np.stack([np.histogram(s, edges)[0] for s in spike_times], axis=1)
    return counts @ np.log(rates * DECODE_BIN).T - rates.sum(1) * DECODE_BIN


def _exact_clusterless(
    rates, tetrode_times, tetrode_marks, edges, cells, mark_sd=MARK_SD
):
    """Exact clusterless log likelihood with true joint mark intensities.

    ``rates`` are the true rates at the bins, shape (n_bins, n_cells), Hz.
    ``cells`` maps each tetrode's mark identities to the cells whose rates
    they carry (the negative control reassigns them). ``mark_sd`` replaces
    the true mark sd, to score against the kernel-smoothed mark density.
    """
    log_likelihood = np.tile(-rates.sum(1) * DECODE_BIN, (edges.size - 1, 1))
    centers = MARK_SPACING * np.arange(CELLS_PER_TETRODE)
    normalizer = (2 * np.pi * mark_sd**2) ** (N_MARK_DIMS / 2)
    for tetrode, (times, marks) in enumerate(
        zip(tetrode_times, tetrode_marks, strict=True)
    ):
        squared = ((marks[:, None, :] - centers[None, :, None]) ** 2).sum(-1)
        mark_density = np.exp(-squared / (2 * mark_sd**2)) / normalizer
        joint = mark_density @ rates[:, cells[tetrode]].T  # (n_spikes, n_bins)
        rows = np.searchsorted(edges, times, side="right") - 1
        keep = (rows >= 0) & (rows < edges.size - 1)
        np.add.at(log_likelihood, rows[keep], np.log(joint[keep] * DECODE_BIN))
    return log_likelihood


def _agreement(model, exact, centers, radii):
    """Median per-row correlation of centered log likelihoods across bins, and
    the fraction of rows whose most likely bins are within each radius."""
    a = model - model.mean(1, keepdims=True)
    b = exact - exact.mean(1, keepdims=True)
    correlation = (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1))
    distance = np.linalg.norm(
        centers[np.argmax(a, 1)] - centers[np.argmax(b, 1)], axis=-1
    )
    return (float(np.median(correlation)),) + tuple(
        float(np.mean(distance <= radius)) for radius in radii
    )


def _level_offset_per_spike(model, exact, n_spikes):
    """Median over rows of the per-spike gap in log marginal likelihood.

    The log marginal ``logsumexp`` over position bins is set by the bins
    where the spikes are likely, where rates are well estimated. (A row mean
    is instead dominated by the log of the near-zero tail rates: the GMM's
    Gaussian tails sit ~18 nats per spike below the 0.1 Hz baseline.) A constant error in every spike's log
    intensity shifts a row by exactly that constant times its spike count.
    """
    gap = logsumexp(model, axis=1) - logsumexp(exact, axis=1)
    return float(np.median(gap / n_spikes))


def _fit(algorithm, data, place_bin_size):
    """Fit a detector; return it, its encoding model and interior bin centers."""
    environment = Environment(place_bin_size=place_bin_size)
    if algorithm in CLUSTERLESS:
        model = NonLocalClusterlessDetector(
            environments=environment, clusterless_algorithm=algorithm
        )
        model.fit(
            data["time"],
            data["position"],
            data["tetrode_times"],
            data["tetrode_marks"],
        )
    else:
        model = NonLocalSortedSpikesDetector(
            environments=environment, sorted_spikes_algorithm=algorithm
        )
        model.fit(data["time"], data["position"], data["spike_times"])
    encoding = next(iter(model.encoding_model_.values()))
    interior = encoding["environment"].is_track_interior_.ravel()
    centers = encoding["environment"].place_bin_centers_[interior]
    return model, encoding, interior, centers


def _occupancy(centers, position):
    """Samples per interior bin, by nearest bin center, shape (n_bins,)."""
    nearest = cKDTree(centers).query(position)[1]
    return np.bincount(nearest, minlength=centers.shape[0])


def _center_of_mass(fields, centers):
    """Center of each field above its median, shape (n_cells, n_position_dims)."""
    excess = np.clip(fields - np.median(fields, axis=-1, keepdims=True), 0.0, None)
    return excess @ centers / excess.sum(axis=-1, keepdims=True)


def _cached_fit(fits, algorithm, data, place_bin_size):
    key = (algorithm, data["position"].shape[1])
    if key not in fits:
        fits[key] = _fit(algorithm, data, place_bin_size)
    return fits[key]


def _score_nonlocal(model, algorithm, test, edges, centers, true_rates, radii):
    """Agreement with the exact likelihood, with reassigned cells, and level.

    Rows with at least two spikes. ``centers`` (n_bins, n_position_dims) and
    ``true_rates`` (n_bins, n_cells) are at the model's interior bins.

    Returns
    -------
    agreement, wrong_agreement : tuple
        ``_agreement`` against the true and the reassigned cells.
    offset : float
        ``_level_offset_per_spike`` against the exact likelihood (for KDE and
        diffusion clusterless backends, with the kernel-smoothed mark density).
    """
    clusterless = algorithm in CLUSTERLESS
    spikes = (
        (test["tetrode_times"], test["tetrode_marks"])
        if clusterless
        else (test["spike_times"],)
    )
    log_likelihood = np.asarray(
        model.compute_log_likelihood(
            test["time"], test["position"], *spikes, time_edges=edges
        )
    )
    # The returned columns are the interior bins of every state.
    state = list(model.state_names).index("Non-Local Continuous")
    columns = model.state_ind_[model.is_track_interior_state_bins_] == state
    nonlocal_ll = log_likelihood[:, columns]
    assert nonlocal_ll.shape[1] == true_rates.shape[0]
    n_spikes = sum(np.histogram(times, edges)[0] for times in spikes[0])
    rows = n_spikes >= 2

    n_tetrodes = true_rates.shape[1] // CELLS_PER_TETRODE
    identity = [
        np.arange(t * CELLS_PER_TETRODE, (t + 1) * CELLS_PER_TETRODE)
        for t in range(n_tetrodes)
    ]
    reassigned = [cells[::-1] for cells in identity[::-1]]
    if clusterless:
        times, marks = test["tetrode_times"], test["tetrode_marks"]
        exact = _exact_clusterless(true_rates, times, marks, edges, identity)
        wrong = _exact_clusterless(true_rates, times, marks, edges, reassigned)
        smoothed_sd = np.hypot(MARK_SD, MARK_KERNEL_SD[algorithm])
        level_reference = _exact_clusterless(
            true_rates, times, marks, edges, identity, mark_sd=smoothed_sd
        )
    else:
        exact = _exact_sorted(true_rates, test["spike_times"], edges)
        wrong = _exact_sorted(true_rates, test["spike_times"][::-1], edges)
        level_reference = exact

    agreement = _agreement(nonlocal_ll[rows], exact[rows], centers, radii)
    wrong_agreement = _agreement(nonlocal_ll[rows], wrong[rows], centers, radii)
    offset = _level_offset_per_spike(
        nonlocal_ll[rows], level_reference[rows], n_spikes[rows]
    )
    return agreement, wrong_agreement, offset


@pytest.mark.parametrize("algorithm", SORTED)
def test_sorted_place_fields_recover_the_true_fields(algorithm, recording, fits):
    """Measured: correlations 0.88-0.99, peaks within 3 cm, mean-rate ratios
    0.91-1.17 (finite-data spike counts) for every sorted likelihood."""
    train, _, _ = recording
    _, encoding, interior, centers = _cached_fit(fits, algorithm, train, 2.0)
    x = centers[:, 0]
    fitted = np.asarray(encoding["place_fields"])[:, interior]
    truth = _true_rates(x)
    correlation = [np.corrcoef(fitted[c], truth[:, c])[0, 1] for c in range(MEANS.size)]
    assert min(correlation) > 0.8
    # Peaks within two 2 cm bins of the true centers.
    assert np.max(np.abs(x[np.argmax(fitted, axis=1)] - MEANS)) <= 4.0
    ratio = fitted.mean(axis=1) / truth.mean(axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.25))


def test_glm_place_field_peaks_match_the_truth(recording, fits):
    """The GLM's default log-rate prior does not flatten the fields.

    Measured: peak ratios 0.80-1.29, median 1.05, as with a near-zero
    penalty (finite-data noise). The former per-second penalty, which grew
    with the recording, gave 0.19-0.31.
    """
    train, _, _ = recording
    _, encoding, interior, centers = _cached_fit(fits, "sorted_spikes_glm", train, 2.0)
    fitted = np.asarray(encoding["place_fields"])[:, interior]
    peak = fitted.max(axis=1) / _true_rates(centers[:, 0]).max(axis=0)
    assert np.all((peak > 0.7) & (peak < 1.4)), peak
    assert 0.9 < np.median(peak) < 1.15


@pytest.mark.parametrize("algorithm", CLUSTERLESS)
def test_clusterless_ground_process_recovers_the_true_intensity(
    algorithm, recording, fits
):
    """Measured: correlations 0.98-0.99 and mean ratios 1.02-1.03."""
    train, _, _ = recording
    _, encoding, interior, centers = _cached_fit(fits, algorithm, train, 2.0)
    fitted = np.asarray(encoding["summed_ground_process_intensity"])
    if fitted.size != centers.shape[0]:
        fitted = fitted[interior]
    truth = _true_rates(centers[:, 0]).sum(axis=1)
    assert np.corrcoef(fitted, truth)[0, 1] > 0.9
    assert 0.9 < fitted.mean() / truth.mean() < 1.1


@pytest.mark.parametrize("algorithm", SORTED + CLUSTERLESS)
def test_nonlocal_likelihood_tracks_the_exact_likelihood(algorithm, recording, fits):
    """Held-out non-local log likelihoods follow the exact one from the true rates.

    Rows with at least two spikes. Measured median correlations: 0.95 for the
    GLM and MRF, 0.78-0.79 for the KDE and diffusion variants (their 6 cm
    position bandwidth smooths the 4 cm fields), 0.69 for the GMM; the most
    likely bins agree within 2 bins in 97.8-99.6% of rows (GMM: 89% within
    2, 96% within 3). Against reassigned cells the correlations are below
    0.15 and agreement within 3 bins below 0.15.

    The correlation ignores a constant per row, so the absolute level is
    checked too: the median per-spike gap in log marginal likelihood. Measured
    smoothing/fit bias: -0.81 (GMM), within 0.15 for the rest (GLM 0.01; the
    KDE and diffusion clusterless backends against the kernel-smoothed mark
    density; against the true one their 24-unit mark kernel costs 2.9 nats
    per spike in 4-D). The tolerances, 1.5 (GMM) and 0.5 nats, sit above
    those biases and below the constant errors guarded
    against: a missing sqrt(2 pi) in one mark dimension (0.92, caught for
    the kernel backends), a mis-scaled rate reference (log 10 = 2.3), a
    missing (2 pi)^2 in a 4-D mark density (3.7), a dropped log(dt) (3.9).
    """
    train, test, edges = recording
    model, _, _, centers = _cached_fit(fits, algorithm, train, 2.0)
    # Radii 5 and 7 cm: within 2 and 3 of the 2 cm bins.
    (correlation, within_2, within_3), wrong, offset = _score_nonlocal(
        model, algorithm, test, edges, centers, _true_rates(centers[:, 0]), (5.0, 7.0)
    )
    if algorithm == "clusterless_gmm":
        assert correlation > 0.4 and within_3 > 0.9
    else:
        assert correlation > 0.7 and within_2 > 0.95
    wrong_correlation, _, wrong_within_3 = wrong
    assert wrong_correlation < 0.3 and wrong_within_3 < 0.3
    assert abs(offset) < LEVEL_TOLERANCE.get(algorithm, 0.5), (
        f"level off by {offset:.2f} nats per spike"
    )


@pytest.mark.parametrize("algorithm", SORTED)
def test_2d_sorted_place_fields_recover_the_true_fields(algorithm, recording_2d, fits):
    """Fitted 2-D fields have the true shape, location, amplitude and rate.

    Measured: correlations 0.94-0.995; field centers of mass within 1.8 cm of
    the true ones in each axis (fields 20 cm apart in x, 12 in y);
    occupancy-weighted rate ratios (expected training spikes, fitted over
    true) 0.87-1.11; peak ratios 0.48-0.61 for the KDE and diffusion
    backends, as a 6 cm kernel on an (8, 6) cm field predicts (0.57),
    0.88-1.19 for the GLM and 0.94-1.55 for the MRF.
    """
    train, _, _ = recording_2d
    _, encoding, interior, centers = _cached_fit(
        fits, algorithm, train, PLACE_BIN_SIZE_2D
    )
    fitted = np.asarray(encoding["place_fields"])[:, interior]
    truth = _true_rates_2d(centers).T  # (n_cells, n_bins)
    correlation = [np.corrcoef(f, t)[0, 1] for f, t in zip(fitted, truth, strict=True)]
    assert min(correlation) > 0.85
    error = np.abs(_center_of_mass(fitted, centers) - _center_of_mass(truth, centers))
    assert np.all(error <= PLACE_BIN_SIZE_2D), error.max(axis=0)
    occupancy = _occupancy(centers, train["position"])
    ratio = (fitted @ occupancy) / (truth @ occupancy)
    assert np.all((ratio > 0.8) & (ratio < 1.25)), ratio
    peak_ratio = fitted.max(axis=1) / truth.max(axis=1)
    assert np.all(peak_ratio > 0.4), peak_ratio


@pytest.mark.parametrize("algorithm", CLUSTERLESS)
def test_2d_clusterless_ground_process_recovers_the_true_intensity(
    algorithm, recording_2d, fits
):
    """Measured: correlations 0.89-0.92, occupancy-weighted ratios 0.92-0.98."""
    train, _, _ = recording_2d
    _, encoding, interior, centers = _cached_fit(
        fits, algorithm, train, PLACE_BIN_SIZE_2D
    )
    fitted = np.asarray(encoding["summed_ground_process_intensity"])
    if fitted.size != centers.shape[0]:
        fitted = fitted[interior]
    truth = _true_rates_2d(centers).sum(axis=1)
    assert np.corrcoef(fitted, truth)[0, 1] > 0.85
    occupancy = _occupancy(centers, train["position"])
    assert 0.85 < (fitted @ occupancy) / (truth @ occupancy) < 1.15


@pytest.mark.parametrize("algorithm", SORTED + CLUSTERLESS)
def test_2d_nonlocal_likelihood_tracks_the_exact_likelihood(
    algorithm, recording_2d, fits
):
    """Held-out 2-D non-local log likelihoods follow the exact one.

    Measured median correlations: 0.93 (KDE and diffusion), 0.97 (MRF),
    0.92 (GLM), 0.69 (GMM); most likely bins within 6.5 cm (about two 3 cm
    bins) in 95% (KDE, diffusion), 93% (GLM), 91% (MRF) and 65% (GMM) of
    rows. Against reassigned cells: correlations below -0.1 and at most 1.2%
    of rows within 6.5 cm. Level offsets within 0.21 (GLM -0.01);
    tolerances as in 1-D.
    """
    train, test, edges = recording_2d
    model, _, _, centers = _cached_fit(fits, algorithm, train, PLACE_BIN_SIZE_2D)
    (correlation, within), wrong, offset = _score_nonlocal(
        model, algorithm, test, edges, centers, _true_rates_2d(centers), (6.5,)
    )
    if algorithm == "clusterless_gmm":
        assert correlation > 0.5 and within > 0.5
    else:
        assert correlation > 0.8 and within > 0.85
    wrong_correlation, wrong_within = wrong
    assert wrong_correlation < 0.3 and wrong_within < 0.3
    assert abs(offset) < LEVEL_TOLERANCE.get(algorithm, 0.5), (
        f"level off by {offset:.2f} nats per spike"
    )
