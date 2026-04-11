"""Golden regression tests for decoder outputs (input-pinned design).

These tests freeze both the **inputs** (simulated spike/position data)
and the **expected outputs** (decoder posteriors and state probabilities)
to decouple decoder regression detection from simulator drift. A change
to the simulator will not break these tests; only a change to the
decoder, HMM core, likelihoods, or transitions will.

Each test has two pickle files in ``golden_data/``:

- ``<name>_inputs.pkl`` — the simulated arrays that were fed into the
  decoder the first time the test ran. Regenerating these is a
  deliberate act that requires re-establishing the golden outputs.
- ``<name>_golden.pkl`` — the decoder outputs for those exact inputs.
  Regenerating these is a routine response to an intentional algorithm
  change (and must be accompanied by the analysis protocol in
  CLAUDE.md).

On the first run (or after deletion) both files are auto-created using
the existing ``pytest.skip`` pattern. Subsequent runs load the inputs
from disk, run the decoder on them, and compare the outputs against
the golden pickle with tight tolerances.

To regenerate after an intentional decoder change:
    rm src/non_local_detector/tests/golden_data/*_golden.pkl
    uv run pytest src/non_local_detector/tests/test_golden_regression.py

To regenerate the pinned inputs as well (e.g., after an intentional
simulator change you want the tests to track):
    rm src/non_local_detector/tests/golden_data/*_inputs.pkl \\
       src/non_local_detector/tests/golden_data/*_golden.pkl
    uv run pytest src/non_local_detector/tests/test_golden_regression.py
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from non_local_detector.continuous_state_transitions import RandomWalk
from non_local_detector.models.decoder import ClusterlessDecoder, SortedSpikesDecoder
from non_local_detector.models.non_local_model import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

GOLDEN_DIR = Path(__file__).parent / "golden_data"

# Seeds for input regeneration only. Once *_inputs.pkl exists these
# seeds are not consulted; the tests work from the pinned pickle.
CLUSTERLESS_SEED = 12345
RANDOM_WALK_SEED = 33333
SORTED_SPIKES_SEED = 54321
NONLOCAL_SEED = 99999

# Tolerances for comparison. Looser than CLAUDE.md's 1e-10 standard
# because these tests must survive cross-Python-version CI
# (3.10/3.11/3.12/3.13) where JAX's XLA compilation and double-where
# patterns produce ~1e-8 differences for identical algorithms. 1e-6
# catches meaningful regressions while allowing cross-version noise.
RTOL = 1e-6
ATOL = 1e-6


@pytest.fixture
def golden_path() -> Path:
    """Path to golden data directory."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    return GOLDEN_DIR


def save_pickle(path: Path, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def compare_golden_arrays(
    actual: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    name: str,
) -> None:
    """Compare two arrays with the golden-regression tolerance."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} does not match golden data",
    )


def load_or_create_clusterless_inputs(inputs_file: Path, seed: int) -> dict[str, Any]:
    """Load pinned clusterless simulation inputs, or create them.

    When the pickle is missing this calls ``make_simulated_run_data``
    with the recorded seed and stores the arrays it produces. All
    subsequent runs load from disk regardless of what the simulator
    would produce.
    """
    if inputs_file.exists():
        return load_pickle(inputs_file)
    sim = make_simulated_run_data(
        n_tetrodes=4,
        place_field_means=np.arange(0, 160, 10),  # 16 neurons
        sampling_frequency=500,
        n_runs=3,
        seed=seed,
    )
    inputs = {
        "position_time": np.asarray(sim.position_time),
        "position": np.asarray(sim.position),
        "spike_times": [np.asarray(st) for st in sim.spike_times],
        "spike_waveform_features": [
            np.asarray(wf) for wf in sim.spike_waveform_features
        ],
    }
    save_pickle(inputs_file, inputs)
    return inputs


def load_or_create_sorted_spikes_inputs(inputs_file: Path, seed: int) -> dict[str, Any]:
    """Load pinned sorted-spikes simulation inputs, or create them."""
    if inputs_file.exists():
        return load_pickle(inputs_file)
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(
        track_height=180,
        sampling_frequency=500,
        n_neurons=30,
        seed=seed,
    )
    inputs = {
        "position": np.asarray(position),
        "spike_times": [np.asarray(st) for st in spike_times],
        "time": np.asarray(time),
        "is_event": np.asarray(is_event),
    }
    save_pickle(inputs_file, inputs)
    return inputs


def _predict_clusterless_decoder(
    decoder: ClusterlessDecoder,
    inputs: dict[str, Any],
    n_encode: int,
    test_end_idx: int,
) -> Any:
    """Run predict on the held-out window of a pinned clusterless input."""
    position_time = inputs["position_time"]
    spike_times = inputs["spike_times"]
    spike_waveform_features = inputs["spike_waveform_features"]
    test_start_t = position_time[n_encode]
    test_end_t = position_time[test_end_idx]

    pred_spike_times = [
        st[(st >= test_start_t) & (st < test_end_t)] for st in spike_times
    ]
    pred_spike_waveform_features = [
        wf[(st >= test_start_t) & (st < test_end_t)]
        for st, wf in zip(spike_times, spike_waveform_features, strict=False)
    ]
    return decoder.predict(
        spike_times=pred_spike_times,
        spike_waveform_features=pred_spike_waveform_features,
        time=position_time[n_encode:test_end_idx],
        position=inputs["position"][n_encode:test_end_idx],
        position_time=position_time[n_encode:test_end_idx],
    )


@pytest.mark.slow
def test_clusterless_decoder_golden_regression(golden_path: Path) -> None:
    """ClusterlessDecoder on pinned inputs must produce pinned outputs."""
    inputs = load_or_create_clusterless_inputs(
        golden_path / "clusterless_decoder_inputs.pkl",
        seed=CLUSTERLESS_SEED,
    )

    n_encode = int(0.7 * len(inputs["position_time"]))
    is_training = np.ones(len(inputs["position_time"]), dtype=bool)
    is_training[n_encode:] = False

    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    )
    decoder.fit(
        inputs["position_time"],
        inputs["position"],
        inputs["spike_times"],
        inputs["spike_waveform_features"],
        is_training=is_training,
    )

    test_end_idx = min(n_encode + 50, len(inputs["position_time"]))
    results = _predict_clusterless_decoder(decoder, inputs, n_encode, test_end_idx)

    golden_file = golden_path / "clusterless_decoder_golden.pkl"
    if not golden_file.exists():
        save_pickle(golden_file, {"posterior": results.acausal_posterior.values})
        pytest.skip("Golden data created, skipping comparison")

    golden = load_pickle(golden_file)
    compare_golden_arrays(
        results.acausal_posterior.values, golden["posterior"], "posterior"
    )


@pytest.mark.slow
def test_random_walk_transition_golden_regression(golden_path: Path) -> None:
    """Decoder with RandomWalk continuous transition, input-pinned."""
    inputs = load_or_create_clusterless_inputs(
        golden_path / "random_walk_transition_inputs.pkl",
        seed=RANDOM_WALK_SEED,
    )

    n_encode = int(0.7 * len(inputs["position_time"]))
    is_training = np.ones(len(inputs["position_time"]), dtype=bool)
    is_training[n_encode:] = False

    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
        continuous_transition_types=[[RandomWalk(movement_var=25.0)]],
    )
    decoder.fit(
        inputs["position_time"],
        inputs["position"],
        inputs["spike_times"],
        inputs["spike_waveform_features"],
        is_training=is_training,
    )

    test_end_idx = min(n_encode + 50, len(inputs["position_time"]))
    results = _predict_clusterless_decoder(decoder, inputs, n_encode, test_end_idx)

    golden_file = golden_path / "random_walk_transition_golden.pkl"
    if not golden_file.exists():
        save_pickle(golden_file, {"posterior": results.acausal_posterior.values})
        pytest.skip("Golden data created, skipping comparison")

    golden = load_pickle(golden_file)
    compare_golden_arrays(
        results.acausal_posterior.values, golden["posterior"], "posterior"
    )


@pytest.mark.slow
def test_sorted_spikes_decoder_golden_regression(golden_path: Path) -> None:
    """SortedSpikesDecoder on pinned inputs must produce pinned outputs."""
    inputs = load_or_create_sorted_spikes_inputs(
        golden_path / "sorted_spikes_decoder_inputs.pkl",
        seed=SORTED_SPIKES_SEED,
    )

    position = inputs["position"]
    spike_times = inputs["spike_times"]
    time_arr = inputs["time"]
    is_training = ~inputs["is_event"]

    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    )
    decoder.fit(
        time_arr,
        position,
        spike_times,
        is_training=is_training,
    )

    n_test = min(50, len(time_arr))
    results = decoder.predict(
        spike_times=spike_times,
        time=time_arr[:n_test],
        position=position[:n_test],
        position_time=time_arr[:n_test],
    )

    golden_file = golden_path / "sorted_spikes_decoder_golden.pkl"
    if not golden_file.exists():
        save_pickle(golden_file, {"posterior": results.acausal_posterior.values})
        pytest.skip("Golden data created, skipping comparison")

    golden = load_pickle(golden_file)
    compare_golden_arrays(
        results.acausal_posterior.values, golden["posterior"], "posterior"
    )


@pytest.mark.slow
def test_nonlocal_detector_golden_regression(golden_path: Path) -> None:
    """NonLocalClusterlessDetector on pinned inputs must produce pinned outputs."""
    inputs = load_or_create_clusterless_inputs(
        golden_path / "nonlocal_detector_inputs.pkl",
        seed=NONLOCAL_SEED,
    )

    # Use first 5000 samples for faster testing, 70/30 split
    n_samples = min(5000, len(inputs["position_time"]))
    n_encode = int(0.7 * n_samples)
    is_training = np.ones(n_samples, dtype=bool)
    is_training[n_encode:] = False

    position_time = inputs["position_time"]
    cutoff_t = position_time[n_samples]
    fit_spike_times = [st[st < cutoff_t] for st in inputs["spike_times"]]
    fit_spike_waveform_features = [
        wf[st < cutoff_t]
        for st, wf in zip(
            inputs["spike_times"], inputs["spike_waveform_features"], strict=False
        )
    ]

    detector = NonLocalClusterlessDetector(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 1000,
        },
    )
    detector.fit(
        position_time[:n_samples],
        inputs["position"][:n_samples],
        fit_spike_times,
        fit_spike_waveform_features,
        is_training=is_training,
    )

    test_start_idx = n_encode
    test_end_idx = min(n_encode + 50, n_samples)
    test_start_t = position_time[test_start_idx]
    test_end_t = position_time[test_end_idx]

    pred_spike_times = [
        st[(st >= test_start_t) & (st < test_end_t)] for st in inputs["spike_times"]
    ]
    pred_spike_waveform_features = [
        wf[(st >= test_start_t) & (st < test_end_t)]
        for st, wf in zip(
            inputs["spike_times"], inputs["spike_waveform_features"], strict=False
        )
    ]
    results = detector.predict(
        spike_times=pred_spike_times,
        spike_waveform_features=pred_spike_waveform_features,
        time=position_time[test_start_idx:test_end_idx],
        position=inputs["position"][test_start_idx:test_end_idx],
        position_time=position_time[test_start_idx:test_end_idx],
    )

    golden_file = golden_path / "nonlocal_detector_golden.pkl"
    if not golden_file.exists():
        save_pickle(
            golden_file,
            {
                "posterior": results.acausal_posterior.values,
                "state_probs": results.acausal_state_probabilities.values,
            },
        )
        pytest.skip("Golden data created, skipping comparison")

    golden = load_pickle(golden_file)
    compare_golden_arrays(
        results.acausal_posterior.values, golden["posterior"], "posterior"
    )
    compare_golden_arrays(
        results.acausal_state_probabilities.values,
        golden["state_probs"],
        "state_probs",
    )
