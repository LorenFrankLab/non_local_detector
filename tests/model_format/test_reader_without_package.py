"""The documented reader reads a saved model with only the stdlib and NumPy.

The snippet of ``docs/saving_models.md`` runs in a separate isolated
interpreter (``-I -S``) whose only third-party package is NumPy, so
non_local_detector cannot be imported there, and must return the detector's
grid and place fields.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import non_local_detector as nld
from tests.model_format.conftest import algorithm_params

pytestmark = pytest.mark.integration

PAGE = Path(__file__).resolve().parents[2] / "docs" / "saving_models.md"
SNIPPET = re.compile(
    r"<!-- no-nld-reader -->\s*\n```python\n(?P<code>.*?)\n```", re.DOTALL
)


@pytest.mark.skipif(not PAGE.is_file(), reason="docs/ is not part of this installation")
def test_documented_snippet_reads_the_grid_and_place_fields(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    detector = nld.SortedSpikesDecoder(
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde")
    ).fit(time, position, spike_times)
    detector.save_model(tmp_path / "decoder.nld")
    # Only NumPy, through a directory holding nothing else.
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "numpy").symlink_to(Path(np.__file__).parent)
    snippet = SNIPPET.search(PAGE.read_text())["code"]
    script = (
        textwrap.dedent(
            f"""
        import sys
        sys.path.insert(0, {str(packages)!r})
        try:
            import non_local_detector
        except ModuleNotFoundError:
            pass
        else:
            raise SystemExit("non_local_detector is importable")
        """
        )
        + snippet
        + textwrap.dedent(
            """
        np.savez("read.npz", centers=centers, interior=interior,
                 place_fields=place_fields)
        """
        )
    )

    subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=120,
    )

    read = np.load(tmp_path / "read.npz")
    environment = detector.environments_[0]
    np.testing.assert_array_equal(read["centers"], environment.place_bin_centers_)
    np.testing.assert_array_equal(read["interior"], environment.is_track_interior_)
    np.testing.assert_array_equal(
        read["place_fields"], detector.encoding_model_[("", 0)]["place_fields"]
    )
