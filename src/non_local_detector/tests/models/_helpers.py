"""Helpers shared by detector-level tests.

``_snapshot`` / ``_assert_unchanged`` check that a failed call leaves a fitted
detector and its environments untouched. ``_decoder`` builds a small one-state
decoder for any registered likelihood algorithm, listed in ``ALGORITHMS``.
``make_two_arm_track`` builds a fitted environment with two disconnected arms.
"""

import networkx as nx
import numpy as np

from non_local_detector import (
    ClusterlessDecoder,
    Environment,
    SortedSpikesDecoder,
    Uniform,
)
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)

# Every registered likelihood algorithm, so a new backend is covered without
# editing this list.
ALGORITHMS = sorted(_SORTED_SPIKES_ALGORITHMS) + sorted(_CLUSTERLESS_ALGORITHMS)


def _snapshot(detector):
    """Attribute identities of the detector and of each of its environments."""
    return (
        dict(vars(detector)),
        [(env, dict(vars(env))) for env in detector.environments],
    )


def _assert_unchanged(detector, snapshot):
    """Assert every attribute is the same object as in ``snapshot``."""
    detector_state, environment_states = snapshot
    assert vars(detector).keys() == detector_state.keys()
    for name, value in detector_state.items():
        assert vars(detector)[name] is value, name
    for env, env_state in environment_states:
        assert vars(env).keys() == env_state.keys()
        for name, value in env_state.items():
            assert vars(env)[name] is value, (env.environment_name, name)


def _decoder(algorithm, *, calibration=False):
    """An unfitted one-state decoder with a Uniform transition.

    Parameters
    ----------
    algorithm : str
        A key of ``_SORTED_SPIKES_ALGORITHMS`` or ``_CLUSTERLESS_ALGORITHMS``.
    calibration : bool, optional
        If True, use 1-unit bins on ``(-1.5, 1.5)``; otherwise 5-unit bins on
        ``(40, 60)``.
    """
    environment = Environment(
        place_bin_size=1 if calibration else 5,
        position_range=((-1.5, 1.5),) if calibration else ((40, 60),),
    )
    options = {
        "environments": environment,
        "continuous_transition_types": [[Uniform()]],
        "infer_track_interior": False,
    }
    params = {"disable_progress_bar": True}
    if algorithm == "sorted_spikes_glm" and calibration:
        # Natural cubic knots must fit this small spatial domain.
        params["emission_knot_spacing"] = 1.0
    if algorithm == "sorted_spikes_mrf":
        params["penalty"] = 1.0
    if algorithm == "clusterless_gmm":
        params.update(
            gmm_components_occupancy=1,
            gmm_components_joint=1,
        )
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        return SortedSpikesDecoder(
            sorted_spikes_algorithm=algorithm,
            sorted_spikes_algorithm_params=params,
            **options,
        )
    return ClusterlessDecoder(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params=params,
        **options,
    )


def make_two_arm_track():
    """A fitted 1-D environment of two disconnected 50 cm arms.

    Arm 0 spans linear positions 0-50 and arm 1 spans 60-110, separated by a
    10 cm gap bin. With 5 cm bins the interior bin centers are 2.5, ..., 47.5
    on arm 0 and 62.5, ..., 107.5 on arm 1; no graph path joins the arms.

    Returns
    -------
    environment : Environment
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(50.0, 0.0))
    graph.add_node(2, pos=(60.0, 0.0))
    graph.add_node(3, pos=(110.0, 0.0))
    graph.add_edge(0, 1, distance=50.0, edge_id=0)
    graph.add_edge(2, 3, distance=50.0, edge_id=1)

    environment = Environment(
        environment_name="",
        place_bin_size=5.0,
        track_graph=graph,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
    )
    return environment.fit_place_grid(position_1d, infer_track_interior=True)
