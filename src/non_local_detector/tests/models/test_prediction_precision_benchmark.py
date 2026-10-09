"""Benchmark references must run float64 arithmetic, not merely enable x64."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from non_local_detector.core import filter, smoother
from non_local_detector.tests.conftest import precision_mode

pytestmark = pytest.mark.integration

SCRIPT = (
    Path(__file__).resolve().parents[4] / "benchmarks" / "compare_prediction_modes.py"
)


@pytest.fixture
def benchmark() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.skip("benchmarks/ is not part of this installation")
    spec = importlib.util.spec_from_file_location("prediction_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_float64_reference_matches_dense_and_reports_actual_precision(
    benchmark: ModuleType,
    checkpoint_recording: Callable,
    tmp_path: Path,
    family: str,
) -> None:
    with precision_mode(False):
        model, fit, predict = checkpoint_recording(family)
        model.fit(**fit, transition_representation="structured")
        likelihood = model.compute_log_likelihood(**predict)
    with precision_mode(True):
        interior = model.is_track_interior_state_bins_
        initial = np.asarray(model.initial_conditions_[interior], dtype=np.float64)
        transition = (
            model._continuous_transition_operator_.restricted(interior)
            .bind_discrete(model.discrete_state_transitions_)
            .to_dense(dtype=np.float64)
        )
        _, (causal, _) = filter(
            initial, transition, np.asarray(likelihood, dtype=np.float64)
        )
        dense = np.asarray(smoother(transition, causal))
        labels = model.state_ind_[interior]
        expected = np.column_stack(
            [
                dense[:, labels == state].sum(axis=1)
                for state in range(len(model.state_names))
            ]
        )
        reference, dtypes = benchmark.reference64_prediction(
            model, predict, chunk_size=13, checkpoint_dir=tmp_path / "checkpoints"
        )
        import jax

        assert jax.config.x64_enabled
        assert dtypes == {
            "likelihood_compute_dtypes": ["float32"],
            "likelihood_inference_dtypes": ["float64"],
            "transition_inference_dtypes": ["float64"],
        }
        assert reference.acausal_state_probabilities.dtype == np.float64
        assert dense.dtype == np.float64
        np.testing.assert_allclose(
            reference.acausal_state_probabilities,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            reference.acausal_state_probabilities.sum(axis=1),
            1.0,
            rtol=1e-10,
            atol=1e-10,
        )
        native = model.predict(
            **predict, inference_mode="checkpointed", output_mode="compact"
        )
        assert benchmark.output_dtypes(native) == {
            "acausal_state_probabilities": "float32"
        }


def test_reference_requires_x64(
    benchmark: ModuleType, checkpoint_recording: Callable, tmp_path: Path
) -> None:
    with precision_mode(False):
        model, fit, predict = checkpoint_recording("sorted")
        model.fit(**fit, transition_representation="structured")
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            benchmark.reference64_prediction(
                model, predict, chunk_size=13, checkpoint_dir=tmp_path
            )


def test_reference_promotes_float32_likelihood_before_inference(
    benchmark: ModuleType,
    checkpoint_recording: Callable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with precision_mode(True):
        model, fit, predict = checkpoint_recording("sorted")
        model.fit(**fit, transition_representation="structured")

        def downgraded(*args: Any, row_slice: slice, **kwargs: Any) -> np.ndarray:
            rows = row_slice.stop - row_slice.start
            bins = np.count_nonzero(model.is_track_interior_state_bins_)
            return np.zeros((rows, bins), dtype=np.float32)

        monkeypatch.setattr(
            benchmark, "_prepare_reference_likelihood", lambda *a: downgraded
        )
        result, dtypes = benchmark.reference64_prediction(
            model, predict, chunk_size=13, checkpoint_dir=tmp_path
        )
        assert result.acausal_state_probabilities.dtype == np.float64
        assert dtypes["likelihood_compute_dtypes"] == ["float32"]
        assert dtypes["likelihood_inference_dtypes"] == ["float64"]


def test_reference_rejects_float32_posterior(
    benchmark: ModuleType,
    checkpoint_recording: Callable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xarray as xr

    from non_local_detector import checkpointed_inference

    fake = xr.Dataset(
        {
            "acausal_state_probabilities": (
                ("time", "states"),
                np.zeros((1, 4), dtype=np.float32),
            )
        }
    )
    monkeypatch.setattr(
        checkpointed_inference,
        "checkpointed_forward_backward",
        lambda *args, **kwargs: SimpleNamespace(dataset=fake),
    )
    with precision_mode(True):
        model, fit, predict = checkpoint_recording("sorted")
        model.fit(**fit, transition_representation="structured")
        with pytest.raises(ValueError, match="posterior.*float64"):
            benchmark.reference64_prediction(
                model, predict, chunk_size=13, checkpoint_dir=tmp_path
            )
