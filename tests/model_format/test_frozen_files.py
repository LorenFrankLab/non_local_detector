"""Saved-model files of each format version load and decode as when written.

``tests/compat_data/format_v1`` holds one small ``.nld`` file per detector
family and each one's decode of the whole recording
(``generate_saved_model_files.py``); see ``VERSION.json``. Every later release must
load them, with the family's ``load_model`` as Spyglass calls it, and decode
them to the same posterior; a change to what the format stores or how it is
read fails here first.
"""

import json
import pickle
import zipfile

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.model_format import read_model_summary
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector
from tests.compat_data.generate_saved_model_files import decode, recording
from tests.models.test_stored_artifact_compatibility import COMPAT_DATA

pytestmark = pytest.mark.integration

FILES = {
    ("format_v1", "nonlocal_sorted_1d"): nld.NonLocalSortedSpikesDetector,
    ("format_v1", "nonlocal_clusterless_2d"): nld.NonLocalClusterlessDetector,
}


@pytest.fixture(scope="module")
def inputs():
    return pickle.loads((COMPAT_DATA / "inputs.pkl").read_bytes())


@pytest.mark.parametrize(
    ("version", "name"), list(FILES), ids=[f"{v}-{n}" for v, n in FILES]
)
def test_a_frozen_file_loads_and_decodes_as_when_saved(version, name, inputs):
    path = COMPAT_DATA / version / f"{name}.nld"
    clusterless = "clusterless" in name
    family = ClusterlessDetector if clusterless else SortedSpikesDetector
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    model = family.load_model(path)
    summary = read_model_summary(path)

    assert manifest["format_version"] == int(version.removeprefix("format_v"))
    assert type(model) is FILES[(version, name)]
    assert summary["fitted"]
    position, spikes = recording(inputs, clusterless)
    with np.load(COMPAT_DATA / version / "predict_reference.npz") as reference:
        expected = reference[name]
    # The golden-regression tolerances (test_golden_regression): float32
    # posteriors decoded on another platform differ in the last bits.
    np.testing.assert_allclose(
        decode(model, inputs, position, spikes), expected, rtol=1e-6, atol=2e-6
    )
