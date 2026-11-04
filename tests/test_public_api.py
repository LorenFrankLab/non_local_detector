"""Every public module declares ``__all__`` and every listed name exists."""

import importlib
import pkgutil
from pathlib import Path

import pytest

import non_local_detector
from non_local_detector.visualization._optional import sortingview_available

# Names that exist only when the optional ``sortingview`` is installed.
_NEEDS_SORTINGVIEW = {
    "non_local_detector.visualization": {
        "create_interactive_1D_decoding_figurl",
        "create_interactive_2D_decoding_figurl",
    },
    "non_local_detector.visualization.figurl_1D": {
        "create_1D_decode_view",
        "create_interactive_1D_decoding_figurl",
    },
    "non_local_detector.visualization.figurl_2D": {
        "create_2D_decode_view",
        "create_interactive_2D_decoding_figurl",
    },
}


_INTERNAL = [
    line.strip()
    for line in (Path(__file__).parents[1] / "docs" / "internal_modules.txt")
    .read_text()
    .splitlines()
    if line.strip() and not line.startswith("#")
]


def _is_internal(name: str) -> bool:
    relative = name.removeprefix("non_local_detector.")
    return any(relative == m or relative.startswith(f"{m}.") for m in _INTERNAL)


def _public_module_names() -> list[str]:
    infos = pkgutil.walk_packages(
        non_local_detector.__path__, prefix=f"{non_local_detector.__name__}."
    )
    names = [
        info.name
        for info in infos
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    return sorted([non_local_detector.__name__, *names])


PUBLIC_MODULES = _public_module_names()


@pytest.mark.unit
def test_public_modules_are_discovered():
    assert "non_local_detector.models.base" in PUBLIC_MODULES
    assert _is_internal("non_local_detector.likelihoods.common.kernel_density")
    assert not any("._" in name for name in PUBLIC_MODULES)


@pytest.mark.unit
@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_defines_all(module_name):
    module = importlib.import_module(module_name)
    if _is_internal(module_name) and not hasattr(module, "__all__"):
        return
    assert hasattr(module, "__all__"), f"{module_name} must define __all__"
    names = module.__all__
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names)), f"{module_name}.__all__ has duplicates"


@pytest.mark.unit
@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_every_name_in_all_exists(module_name):
    module = importlib.import_module(module_name)
    if not hasattr(module, "__all__"):
        pytest.skip("internal module without __all__")
    optional = (
        set() if sortingview_available() else _NEEDS_SORTINGVIEW.get(module_name, set())
    )
    missing = [
        name
        for name in module.__all__
        if name not in optional and not hasattr(module, name)
    ]
    assert not missing, f"{module_name}.__all__ names missing: {missing}"


# What Spyglass imports from this package; each must stay listed.
SPYGLASS_IMPORTS = {
    "non_local_detector": [
        "ContFragClusterlessClassifier",
        "ContFragSortedSpikesClassifier",
        "NonLocalClusterlessDetector",
        "NonLocalSortedSpikesDetector",
    ],
    "non_local_detector.analysis": ["align_tracking_to_results"],
    "non_local_detector.environment": ["Environment"],
    "non_local_detector.exceptions": ["ValidationError"],
    "non_local_detector.models.base": ["ClusterlessDetector", "SortedSpikesDetector"],
    "non_local_detector.observation_models": ["ObservationModel"],
    "non_local_detector.visualization.figurl_1D": [
        "create_1D_decode_view",
        "create_interactive_1D_decoding_figurl",
    ],
    "non_local_detector.visualization.figurl_2D": [
        "create_2D_decode_view",
        "create_interactive_2D_decoding_figurl",
    ],
}


@pytest.mark.unit
@pytest.mark.parametrize("module_name", sorted(SPYGLASS_IMPORTS))
def test_spyglass_imports_are_listed(module_name):
    module = importlib.import_module(module_name)
    assert set(SPYGLASS_IMPORTS[module_name]) <= set(module.__all__)
