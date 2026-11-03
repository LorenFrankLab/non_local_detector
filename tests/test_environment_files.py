"""The conda environment files must declare the same dependencies as pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ENV_FILES = ["environment.yml", "environment_gpu.yml"]
# Conda-only entries with no pyproject counterpart.
CONDA_ONLY = {"python", "jaxlib", "cuda-nvcc"}

_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*([<>=!~].*)?$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement(text: str) -> tuple[str, str]:
    """Split ``"name >=1.2"`` into a normalized name and a space-free spec."""
    match = _REQUIREMENT.match(text.strip())
    assert match, f"cannot parse requirement {text!r}"
    return _normalize(match.group(1)), (match.group(2) or "").replace(" ", "")


def _parse_environment(path: Path) -> dict[str, str]:
    """Read the ``- name spec`` lines of a conda file (no PyYAML dependency)."""
    requirements = {}
    in_dependencies = False
    for line in path.read_text().splitlines():
        if line.startswith("dependencies:"):
            in_dependencies = True
        elif in_dependencies and line.startswith("- "):
            name, spec = _parse_requirement(line[2:])
            requirements[name] = spec
    return requirements


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if not PYPROJECT.is_file():
        pytest.skip("pyproject.toml is absent from this installation")
    return tomllib.loads(PYPROJECT.read_text())


def _runtime(pyproject: dict) -> dict[str, str]:
    return dict(_parse_requirement(r) for r in pyproject["project"]["dependencies"])


def _tools(pyproject: dict) -> set[str]:
    """Names of the packages in the test and dev dependency groups."""
    groups = pyproject["dependency-groups"]
    return {
        _parse_requirement(requirement)[0]
        for requirement in groups["test"] + groups["dev"]
        if isinstance(requirement, str)
    }


@pytest.mark.parametrize("filename", ENV_FILES)
def test_environment_file_matches_pyproject(filename, pyproject):
    if not (ROOT / filename).is_file():
        pytest.skip(f"{filename} is absent from this installation")
    environment = _parse_environment(ROOT / filename)
    runtime = _runtime(pyproject)

    for name, spec in runtime.items():
        assert environment.get(name) == spec, (
            f"{filename}: {name} is {environment.get(name)!r}, "
            f"pyproject.toml says {spec!r}"
        )

    expected_python = pyproject["project"]["requires-python"].replace(" ", "")
    assert environment["python"] == expected_python

    extra = set(environment) - set(runtime) - CONDA_ONLY
    assert extra == _tools(pyproject), (
        f"{filename}: tools differ from the pyproject.toml test and dev dependency groups"
    )
