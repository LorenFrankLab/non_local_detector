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

pytestmark = pytest.mark.unit

SPECIFIC = [
    DataQualityWarning,
    NumericalWarning,
    ParameterWarning,
    CompatibilityWarning,
]


def test_hierarchy_keeps_replaced_builtin_categories():
    assert issubclass(NonLocalDetectorWarning, UserWarning)
    for category in SPECIFIC:
        assert issubclass(category, NonLocalDetectorWarning)
        assert issubclass(category, UserWarning)
    assert issubclass(CompatibilityWarning, DeprecationWarning)
    for category in (DataQualityWarning, NumericalWarning, ParameterWarning):
        assert not issubclass(category, DeprecationWarning)


def test_exported_from_the_package():
    for category in [NonLocalDetectorWarning, *SPECIFIC]:
        assert category.__name__ in non_local_detector.__all__


def test_every_library_warning_has_a_package_category():
    """No ``warnings.warn`` in ``src`` falls back to a builtin category."""
    source_dir = Path(non_local_detector.__file__).parent
    builtin = {"UserWarning", "DeprecationWarning", "RuntimeWarning", None}
    offenders = []
    for path in source_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "warn"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "warnings"
            ):
                continue
            category = node.args[1] if len(node.args) > 1 else None
            for keyword in node.keywords:
                if keyword.arg == "category":
                    category = keyword.value
            name = category.id if isinstance(category, ast.Name) else None
            if name in builtin:
                offenders.append(f"{path.relative_to(source_dir)}:{node.lineno}")
    assert not offenders, offenders


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
    detector = NonLocalSortedSpikesDetector()
    call = lambda: detector.fit(  # noqa: E731
        **recording, is_training=np.zeros(n, dtype=bool)
    )
    assert _caught("zero training exposure", call) == {DataQualityWarning}


def test_one_dimensional_position_is_data_quality(recording):
    detector = NonLocalSortedSpikesDetector()
    call = lambda: detector.fit(**recording)  # noqa: E731
    assert _caught("appears to be 1D", call) == {DataQualityWarning}


def test_em_not_converged_is_numerical(recording):
    edges = time_edges_from_centers(recording["position_time"])
    detector = NonLocalSortedSpikesDetector()
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
    detector = NonLocalSortedSpikesDetector().fit(**recording)
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
