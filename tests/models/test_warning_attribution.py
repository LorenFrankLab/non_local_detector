"""Warnings raised deep inside the library point at the caller's code."""

import warnings

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def recording():
    _, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=4)
    n = 2000
    time, position = time[:n], position[:n]
    return {
        "position_time": time,
        "position": position,
        "spike_times": [st[st <= time[-1]] for st in spike_times],
    }


def _sources(match, call):
    """Files that warnings whose message contains ``match`` were attributed to."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    sources = {w.filename for w in caught if match in str(w.message)}
    assert sources, f"no warning matching {match!r}"
    return sources


def _past_tracking(time):
    """Decode edges that run five bins past the tracked period."""
    edges = time_edges_from_centers(time)
    return np.concatenate([edges, edges[-1] + np.diff(edges[-2:]) * np.arange(1, 6)])


@pytest.mark.parametrize("entry", ["fit", "estimate_parameters"])
def test_spikes_outside_the_recording_warning(recording, entry):
    """Raised in fit_encoding_model, under fit or estimate_parameters."""
    in_ms = {**recording, "spike_times": [t * 1_000 for t in recording["spike_times"]]}
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    if entry == "fit":
        call = lambda: detector.fit(**in_ms)  # noqa: E731
    else:
        edges = time_edges_from_centers(recording["position_time"])
        call = lambda: detector.estimate_parameters(  # noqa: E731
            **in_ms, time_edges=edges, max_iter=1
        )
    assert _sources("None of the", call) == {__file__}


def test_em_not_converged_warning(recording):
    edges = time_edges_from_centers(recording["position_time"])
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    call = lambda: detector.estimate_parameters(  # noqa: E731
        **recording, time_edges=edges, max_iter=1
    )
    assert _sources("EM did not converge", call) == {__file__}


def test_deprecated_output_flag_warning(recording):
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE).fit(
        **recording
    )
    edges = time_edges_from_centers(recording["position_time"])
    call = lambda: detector.predict(  # noqa: E731
        spike_times=recording["spike_times"],
        time_edges=edges,
        position=recording["position"],
        position_time=recording["position_time"],
        save_log_likelihood_to_results=True,
    )
    assert _sources("deprecated", call) == {__file__}


@pytest.mark.parametrize(
    "entry", ["predict", "compute_log_likelihood", "most_likely_sequence"]
)
def test_outside_tracking_warning(recording, entry):
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE).fit(
        **recording
    )
    edges = _past_tracking(recording["position_time"])
    if entry == "compute_log_likelihood":
        call = lambda: detector.compute_log_likelihood(  # noqa: E731
            recording["position_time"],
            recording["position"],
            recording["spike_times"],
            time_edges=edges,
        )
    else:
        call = lambda: getattr(detector, entry)(  # noqa: E731
            spike_times=recording["spike_times"],
            time_edges=edges,
            position=recording["position"],
            position_time=recording["position_time"],
        )
    assert _sources("outside the tracked period", call) == {__file__}


def test_encoding_fit_warning(recording):
    """Warnings from the likelihood fits point at the caller too."""
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    n = len(recording["position_time"])
    call = lambda: detector.fit(  # noqa: E731
        **recording, is_training=np.zeros(n, dtype=bool)
    )
    assert _sources("zero training exposure", call) == {__file__}


def test_attribution_through_a_symlinked_package(tmp_path):
    """Importing the package through a symlink still attributes to the caller."""
    import subprocess
    import sys
    from pathlib import Path

    import non_local_detector

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "non_local_detector").symlink_to(Path(non_local_detector.__file__).parent)
    script = tmp_path / "caller.py"
    script.write_text(
        "import warnings\n"
        "import numpy as np\n"
        "import non_local_detector\n"
        "from non_local_detector.models.base import _missing_bins\n"
        f"assert non_local_detector.__file__.startswith({str(linked)!r})\n"
        "edges = np.linspace(0.0, 1.2, 13)\n"
        "time = np.linspace(0.0, 1.0, 11)\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    _missing_bins(edges, None, time, time[:, None], warn_outside_tracking=True)\n"
        "print(caught[0].filename)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(linked), "PATH": ""},
        check=True,
    )
    assert result.stdout.strip() == str(script)


def test_no_fixed_stacklevels_in_library_code():
    """Library warnings use user_stacklevel(), not a depth tuned to one path."""
    import re
    from pathlib import Path

    import non_local_detector

    root = Path(non_local_detector.__file__).parent
    fixed = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"stacklevel\s*=\s*\d", line)
    ]
    assert not fixed, fixed
