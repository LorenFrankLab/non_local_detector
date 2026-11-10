"""The likelihoods.common package keeps its pickle path and layering."""

import ast
import importlib
import pickle
from pathlib import Path

import pytest

import non_local_detector.likelihoods.common as common
from non_local_detector.models.base import SortedSpikesDetector

pytestmark = pytest.mark.unit

COMPAT_DATA = Path(__file__).parents[1] / "compat_data"

# Each submodule may import only from submodules before it.
LAYERS = (
    "numerics",
    "validation",
    "weights",
    "padding",
    "position",
    "reductions",
    "spike_rows",
    "marks",
    "poisson",
    "kernel_density",
)


def test_kde_model_pickles_under_the_former_module_path():
    """KDEModel pickles name ``non_local_detector.likelihoods.common.KDEModel``,
    as detectors saved before the split do, so each version loads the other's."""
    model = common.KDEModel(std=2.0)
    assert common.KDEModel.__module__ == "non_local_detector.likelihoods.common"
    data = pickle.dumps(model, protocol=0)  # protocol 0 names classes as text
    assert b"non_local_detector.likelihoods.common\nKDEModel" in data
    assert b"kernel_density" not in data
    loaded = pickle.loads(data)
    assert type(loaded) is common.KDEModel
    assert float(loaded.std) == 2.0


def test_detector_pickled_before_the_split_loads_its_kde_models():
    """A detector saved by main before the split names the former module path."""
    path = COMPAT_DATA / "main-d5f85823" / "nonlocal_sorted_1d_model.pkl"
    data = path.read_bytes()
    assert b"non_local_detector.likelihoods.common" in data
    assert b"kernel_density" not in data
    detector = SortedSpikesDetector.load_model(path)
    models = [
        model
        for encoding_model in detector.encoding_model_.values()
        for model in encoding_model["marginal_models"]
    ]
    assert models
    assert all(type(model) is common.KDEModel for model in models)


@pytest.mark.parametrize("layer", LAYERS)
def test_submodules_import_only_lower_layers(layer):
    module = importlib.import_module(f"non_local_detector.likelihoods.common.{layer}")
    tree = ast.parse(Path(module.__file__).read_text())
    prefix = "non_local_detector.likelihoods"
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(prefix)
    }
    lower = {f"{prefix}.common.{name}" for name in LAYERS[: LAYERS.index(layer)]}
    assert imported <= lower, imported - lower
