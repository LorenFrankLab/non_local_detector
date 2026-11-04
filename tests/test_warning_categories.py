"""The package's warnings carry package-specific categories."""

import ast
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import non_local_detector
from non_local_detector import (
    CompatibilityWarning,
    DataQualityWarning,
    NonLocalDetectorWarning,
    NonLocalSortedSpikesDetector,
    NumericalWarning,
    ParameterWarning,
    time_edges_from_centers,
)
from non_local_detector.likelihoods.diffusion import check_smoothing_bandwidth
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.unit

SPECIFIC = [
    DataQualityWarning,
    NumericalWarning,
    ParameterWarning,
    CompatibilityWarning,
]


def test_hierarchy_keeps_replaced_builtin_categories():
    assert issubclass(NonLocalDetectorWarning, Warning)
    assert not issubclass(NonLocalDetectorWarning, UserWarning)
    for category in SPECIFIC:
        assert issubclass(category, NonLocalDetectorWarning)
    for category in (DataQualityWarning, NumericalWarning, ParameterWarning):
        assert issubclass(category, UserWarning)
    assert issubclass(CompatibilityWarning, DeprecationWarning)
    assert not issubclass(CompatibilityWarning, UserWarning)
    for category in (DataQualityWarning, NumericalWarning, ParameterWarning):
        assert not issubclass(category, DeprecationWarning)


def test_exported_from_the_package():
    for category in [NonLocalDetectorWarning, *SPECIFIC]:
        assert category.__name__ in non_local_detector.__all__


PACKAGE_CATEGORIES = {
    "NonLocalDetectorWarning",
    "DataQualityWarning",
    "NumericalWarning",
    "ParameterWarning",
    "CompatibilityWarning",
}


def _warning_calls_without_package_category(source):
    """Line numbers of warn calls whose category is not a package class.

    Follows ``import warnings [as w]`` and ``from warnings import warn [as x]``,
    for both ``warn`` and ``warn_explicit``. A missing category, or one that is
    not a plain name or attribute of a package class, is flagged.
    """
    tree = ast.parse(source)
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name == "warnings"}
        elif isinstance(node, ast.ImportFrom) and node.module == "warnings":
            functions |= {
                a.asname or a.name
                for a in node.names
                if a.name in {"warn", "warn_explicit"}
            }
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_warn = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in modules
            and func.attr in {"warn", "warn_explicit"}
        ) or (isinstance(func, ast.Name) and func.id in functions)
        if not is_warn:
            continue
        category = node.args[1] if len(node.args) > 1 else None
        for keyword in node.keywords:
            if keyword.arg == "category":
                category = keyword.value
        name = (
            category.id
            if isinstance(category, ast.Name)
            else category.attr
            if isinstance(category, ast.Attribute)
            else None
        )
        if name not in PACKAGE_CATEGORIES:
            lines.append(node.lineno)
    return lines


def test_every_library_warning_has_a_package_category():
    """No warning call in ``src`` uses a builtin or missing category."""
    source_dir = Path(non_local_detector.__file__).parent
    offenders = [
        f"{path.relative_to(source_dir)}:{line}"
        for path in source_dir.rglob("*.py")
        for line in _warning_calls_without_package_category(path.read_text())
    ]
    assert not offenders, offenders


@pytest.mark.parametrize(
    "snippet",
    [
        "import warnings\nwarnings.warn('m')",
        "import warnings\nwarnings.warn('m', UserWarning)",
        "import warnings\nwarnings.warn('m', category=FutureWarning)",
        "import warnings\nwarnings.warn('m', PendingDeprecationWarning)",
        "import warnings\nwarnings.warn('m', ResourceWarning)",
        "import warnings\nwarnings.warn('m', get_category())",
        "from warnings import warn\nwarn('m', UserWarning)",
        "from warnings import warn as w\nw('m')",
        "import warnings as w\nw.warn('m', RuntimeWarning)",
        "import warnings\nwarnings.warn_explicit('m', UserWarning, 'f', 1)",
    ],
)
def test_category_scan_flags_non_package_categories(snippet):
    assert _warning_calls_without_package_category(snippet) == [2]


def test_category_scan_accepts_package_categories():
    snippet = (
        "import warnings as w\nfrom warnings import warn\n"
        "w.warn('m', NumericalWarning)\n"
        "warn('m', category=exceptions.DataQualityWarning)\n"
        "w.warn_explicit('m', ParameterWarning, 'f', 1)\n"
    )
    assert _warning_calls_without_package_category(snippet) == []


@pytest.fixture(scope="module")
def recording():
    _, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=4)
    n = 2000
    time, position = time[:n], position[:n]
    return {
        "position_time": time,
        "position": position,
        "spike_times": [st[st <= time[-1]] for st in spike_times],
    }


def _caught(match, call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    matching = [w for w in caught if match in str(w.message)]
    assert matching, f"no warning matching {match!r}"
    return {w.category for w in matching}


def test_zero_exposure_is_data_quality(recording):
    n = len(recording["position_time"])
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    call = lambda: detector.fit(  # noqa: E731
        **recording, is_training=np.zeros(n, dtype=bool)
    )
    assert _caught("zero training exposure", call) == {DataQualityWarning}


def test_spikes_outside_the_recording_is_data_quality(recording):
    in_ms = {**recording, "spike_times": [t * 1_000 for t in recording["spike_times"]]}
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    call = lambda: detector.fit(**in_ms)  # noqa: E731
    assert _caught("None of the", call) == {DataQualityWarning}


def test_em_not_converged_is_numerical(recording):
    edges = time_edges_from_centers(recording["position_time"])
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    call = lambda: detector.estimate_parameters(  # noqa: E731
        **recording, time_edges=edges, max_iter=1
    )
    assert _caught("EM did not converge", call) == {NumericalWarning}


def test_small_bandwidth_is_parameter():
    graph = nx.path_graph(5)
    nx.set_edge_attributes(graph, 5.0, "distance")
    call = lambda: check_smoothing_bandwidth(1.0, graph)  # noqa: E731
    assert _caught("characteristic bin spacing", call) == {ParameterWarning}


def test_deprecated_output_flag_is_compatibility(recording):
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE).fit(
        **recording
    )
    edges = time_edges_from_centers(recording["position_time"])
    call = lambda: detector.predict(  # noqa: E731
        spike_times=recording["spike_times"],
        time_edges=edges,
        position=recording["position"],
        position_time=recording["position_time"],
        save_log_likelihood_to_results=True,
    )
    assert _caught("deprecated", call) == {CompatibilityWarning}


def test_package_base_class_filters_select_package_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", NonLocalDetectorWarning)
        warnings.warn("ours", NumericalWarning, stacklevel=1)
        warnings.warn("theirs", UserWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == ["theirs"]


def test_user_warning_filter_does_not_catch_compatibility_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", UserWarning)
        warnings.warn("old", CompatibilityWarning, stacklevel=1)
        warnings.warn("data", DataQualityWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == ["old"]


def test_deprecation_warning_filter_catches_compatibility_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.warn("old", CompatibilityWarning, stacklevel=1)
    assert caught == []


def test_base_filter_catches_every_package_category():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", NonLocalDetectorWarning)
        for category in SPECIFIC:
            warnings.warn("ours", category, stacklevel=1)
    assert caught == []
