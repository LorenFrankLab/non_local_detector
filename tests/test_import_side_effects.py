"""Importing the package must not change global library configuration."""

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_import_leaves_sklearn_config_unchanged():
    code = (
        "import sklearn\n"
        "before = sklearn.get_config()\n"
        "import non_local_detector\n"
        "import non_local_detector.models\n"
        "assert sklearn.get_config() == before, sklearn.get_config()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
