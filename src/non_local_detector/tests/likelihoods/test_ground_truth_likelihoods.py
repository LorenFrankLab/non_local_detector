"""Every likelihood against a simulation with known place fields.

Sixteen cells with Gaussian place fields (sigma 4 cm, 25 Hz peak, 0.1 Hz
baseline) sit on the first two thirds of a 120 cm track, four per tetrode,
with Gaussian marks 80 units apart (sd 8). The animal runs laps at 20 cm/s.
Because the true rates are known, two things can be checked directly:

- recovery: fitted place fields (sorted) or the summed ground-process
  intensity (clusterless) match the true ones;
- the likelihood itself: on held-out laps, each model's non-local log
  likelihood per 20 ms bin tracks the exact Poisson log likelihood computed
  from the true rates (and, for clusterless, the true mark densities).

A negative control scores each model against the truth with the cells
reassigned; the same metrics must then fail, so the thresholds discriminate.
"""

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)

pytestmark = [
    pytest.mark.integration,
    # The simulated track is deliberately a 1-D open field.
    pytest.mark.filterwarnings("ignore:Position data appears to be 1D"),
]

FS = 500.0
TRACK, SPEED = 120.0, 20.0
MEANS = np.linspace(10.0, 80.0, 16)
SIGMA, PEAK, BASELINE = 4.0, 25.0, 0.1
MARK_SPACING, MARK_SD, N_MARK_DIMS = 80.0, 8.0, 4
CELLS_PER_TETRODE = 4
DECODE_BIN = 0.02

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


def _true_rates(x):
    """True rates, shape (n_positions, n_cells), Hz."""
    return BASELINE + PEAK * np.exp(-((x[:, None] - MEANS) ** 2) / (2 * SIGMA**2))


def _simulate(seed, seconds):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, seconds, 1 / FS)
    phase = (time * SPEED) % (2 * TRACK)
    position = np.where(phase < TRACK, phase, 2 * TRACK - phase)
    counts = rng.poisson(_true_rates(position) / FS)
    spike_times = [
        np.sort(
            np.repeat(time, counts[:, cell])
            + rng.uniform(0, 1 / FS, counts[:, cell].sum())
        )
        for cell in range(MEANS.size)
    ]
    tetrode_times, tetrode_marks = [], []
    for tetrode in range(MEANS.size // CELLS_PER_TETRODE):
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
        "time": time,
        "position": position[:, None],
        "spike_times": spike_times,
        "tetrode_times": tetrode_times,
        "tetrode_marks": tetrode_marks,
    }


@pytest.fixture(scope="module")
def recording():
    train = _simulate(0, 8 * 2 * TRACK / SPEED)
    test = _simulate(1, 2 * 2 * TRACK / SPEED)
    edges = np.arange(0.0, test["time"][-1], DECODE_BIN)
    return train, test, edges


def _exact_sorted(x, spike_times, edges):
    """Exact non-local log likelihood, (n_rows, n_bins), up to a row constant."""
    rates = _true_rates(x)
    counts = np.stack([np.histogram(s, edges)[0] for s in spike_times], axis=1)
    return counts @ np.log(rates * DECODE_BIN).T - rates.sum(1) * DECODE_BIN


def _exact_clusterless(x, tetrode_times, tetrode_marks, edges, cells):
    """Exact clusterless log likelihood with true joint mark intensities.

    ``cells`` maps each tetrode's mark identities to the cells whose rates
    they carry (the negative control reassigns them).
    """
    rates = _true_rates(x)
    log_likelihood = np.tile(-rates.sum(1) * DECODE_BIN, (edges.size - 1, 1))
    centers = MARK_SPACING * np.arange(CELLS_PER_TETRODE)
    normalizer = (2 * np.pi * MARK_SD**2) ** (N_MARK_DIMS / 2)
    for tetrode, (times, marks) in enumerate(
        zip(tetrode_times, tetrode_marks, strict=True)
    ):
        squared = ((marks[:, None, :] - centers[None, :, None]) ** 2).sum(-1)
        mark_density = np.exp(-squared / (2 * MARK_SD**2)) / normalizer
        joint = mark_density @ rates[:, cells[tetrode]].T  # (n_spikes, n_bins)
        rows = np.searchsorted(edges, times, side="right") - 1
        keep = (rows >= 0) & (rows < edges.size - 1)
        np.add.at(log_likelihood, rows[keep], np.log(joint[keep] * DECODE_BIN))
    return log_likelihood


def _agreement(model, exact):
    """Median per-row correlation of centered log likelihoods across bins, and
    the fraction of rows whose most likely bins are within 2 and 3 bins."""
    a = model - model.mean(1, keepdims=True)
    b = exact - exact.mean(1, keepdims=True)
    correlation = (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1))
    distance = np.abs(np.argmax(a, 1) - np.argmax(b, 1))
    return (
        float(np.median(correlation)),
        float(np.mean(distance <= 2)),
        float(np.mean(distance <= 3)),
    )


def _fit(algorithm, data):
    environment = Environment(place_bin_size=2.0)
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
    x = encoding["environment"].place_bin_centers_[interior, 0]
    return model, encoding, interior, x


@pytest.mark.parametrize("algorithm", SORTED)
def test_sorted_place_fields_recover_the_true_fields(algorithm, recording):
    """Measured: correlations 0.88-0.99, peaks within 3 cm, mean-rate ratios
    0.91-1.17 (finite-data spike counts) for every sorted likelihood."""
    train, _, _ = recording
    _, encoding, interior, x = _fit(algorithm, train)
    fitted = np.asarray(encoding["place_fields"])[:, interior]
    truth = _true_rates(x)
    correlation = [np.corrcoef(fitted[c], truth[:, c])[0, 1] for c in range(MEANS.size)]
    assert min(correlation) > 0.8
    # Peaks within two 2 cm bins of the true centers.
    assert np.max(np.abs(x[np.argmax(fitted, axis=1)] - MEANS)) <= 4.0
    ratio = fitted.mean(axis=1) / truth.mean(axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.25))


def test_glm_place_field_peaks_match_the_truth(recording):
    """The GLM's default log-rate prior does not flatten the fields.

    Measured: peak ratios 0.80-1.29, median 1.05, as with a near-zero
    penalty (finite-data noise). The former per-second penalty, which grew
    with the recording, gave 0.19-0.31.
    """
    train, _, _ = recording
    _, encoding, interior, x = _fit("sorted_spikes_glm", train)
    fitted = np.asarray(encoding["place_fields"])[:, interior]
    peak = fitted.max(axis=1) / _true_rates(x).max(axis=0)
    assert np.all((peak > 0.7) & (peak < 1.4)), peak
    assert 0.9 < np.median(peak) < 1.15


@pytest.mark.parametrize("algorithm", CLUSTERLESS)
def test_clusterless_ground_process_recovers_the_true_intensity(algorithm, recording):
    """Measured: correlations 0.98-0.99 and mean ratios 1.02-1.03."""
    train, _, _ = recording
    _, encoding, interior, x = _fit(algorithm, train)
    fitted = np.asarray(encoding["summed_ground_process_intensity"])
    if fitted.size != x.size:
        fitted = fitted[interior]
    truth = _true_rates(x).sum(axis=1)
    assert np.corrcoef(fitted, truth)[0, 1] > 0.9
    assert 0.9 < fitted.mean() / truth.mean() < 1.1


@pytest.mark.parametrize("algorithm", SORTED + CLUSTERLESS)
def test_nonlocal_likelihood_tracks_the_exact_likelihood(algorithm, recording):
    """Held-out non-local log likelihoods follow the exact one from the true rates.

    Rows with at least two spikes. Measured median correlations: 0.95 for the
    GLM and MRF, 0.78-0.79 for the KDE and diffusion variants (their 6 cm
    position bandwidth smooths the 4 cm fields), 0.51 for the GMM; the most
    likely bins agree within 2 bins in 97.8-99.6% of rows (GMM: 89% within
    2, 96% within 3). Against reassigned cells the correlations are below
    0.15 and agreement within 3 bins below 0.15.
    """
    train, test, edges = recording
    model, _, _, x = _fit(algorithm, train)
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
    state = list(model.state_names).index("Non-Local Continuous")
    columns = (model.state_ind_ == state) & model.is_track_interior_state_bins_
    nonlocal_ll = log_likelihood[:, columns]
    assert nonlocal_ll.shape[1] == x.size
    n_spikes = sum(np.histogram(times, edges)[0] for times in spikes[0])
    rows = n_spikes >= 2

    identity = [
        np.arange(t * CELLS_PER_TETRODE, (t + 1) * CELLS_PER_TETRODE)
        for t in range(MEANS.size // CELLS_PER_TETRODE)
    ]
    reassigned = [cells[::-1] for cells in identity[::-1]]
    if clusterless:
        times, marks = test["tetrode_times"], test["tetrode_marks"]
        exact = _exact_clusterless(x, times, marks, edges, identity)
        wrong = _exact_clusterless(x, times, marks, edges, reassigned)
    else:
        exact = _exact_sorted(x, test["spike_times"], edges)
        wrong = _exact_sorted(x, test["spike_times"][::-1], edges)

    correlation, within_2, within_3 = _agreement(nonlocal_ll[rows], exact[rows])
    if algorithm == "clusterless_gmm":
        assert correlation > 0.4 and within_3 > 0.9
    else:
        assert correlation > 0.7 and within_2 > 0.95
    wrong_correlation, _, wrong_within_3 = _agreement(nonlocal_ll[rows], wrong[rows])
    assert wrong_correlation < 0.3 and wrong_within_3 < 0.3
