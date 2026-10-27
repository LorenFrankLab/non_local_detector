"""Every benchmark script still imports and parses arguments against the package."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Each case spawns a fresh interpreter that imports JAX (2-4 s); the 22 cases
# take about 35 s together.
pytestmark = [pytest.mark.integration, pytest.mark.slow]

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
SCRIPTS = sorted(BENCHMARKS.glob("*.py")) if BENCHMARKS.is_dir() else []


@pytest.mark.skipif(not SCRIPTS, reason="benchmarks/ is not part of this checkout")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_benchmark_script_help_runs(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
