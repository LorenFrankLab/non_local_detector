"""Every benchmark script still imports and parses arguments against the package."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
SCRIPTS = sorted(BENCHMARKS.glob("*.py")) if BENCHMARKS.is_dir() else []
# Reads ``--devices`` from ``sys.argv`` and sets ``XLA_FLAGS`` before it imports
# JAX, so only a fresh interpreter runs it the way a user does.
FRESH_INTERPRETER_SCRIPT = BENCHMARKS / "benchmark_chunk_likelihood_memory.py"


@pytest.fixture
def isolated_script_state(monkeypatch):
    """Undo what running a script as ``__main__`` leaves behind.

    The scripts put their own directory on ``sys.path``, set environment
    defaults such as ``TQDM_DISABLE`` and import sibling scripts as modules.
    """
    saved_environ = dict(os.environ)
    saved_modules = set(sys.modules)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    yield
    os.environ.clear()
    os.environ.update(saved_environ)
    for name in set(sys.modules) - saved_modules:
        module_file = getattr(sys.modules[name], "__file__", None) or ""
        if Path(module_file).parent == BENCHMARKS:
            del sys.modules[name]


@pytest.mark.skipif(not SCRIPTS, reason="benchmarks/ is not part of this checkout")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_benchmark_script_help_runs(script, isolated_script_state, capsys):
    """``python <script> --help`` prints the usage and exits 0.

    Runs the script as ``__main__`` in this process the way the interpreter
    would, with the script's directory first on ``sys.path`` and ``--help`` in
    ``sys.argv``: that imports the script and the package modules it uses, runs
    its module-level code and its ``__main__`` block and builds its argument
    parser, without a new interpreter (and JAX import) per script.
    """
    sys.path.insert(0, str(script.parent))
    sys.argv[:] = [str(script), "--help"]
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code in (0, None)
    assert "usage:" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.skipif(
    not FRESH_INTERPRETER_SCRIPT.is_file(),
    reason="benchmarks/ is not part of this checkout",
)
def test_benchmark_script_help_runs_in_a_fresh_interpreter():
    """One script as a user runs it, in its own interpreter (2-4 s for JAX)."""
    result = subprocess.run(
        [sys.executable, str(FRESH_INTERPRETER_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
