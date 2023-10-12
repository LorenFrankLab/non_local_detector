"""Classes for constructing different types of movement models."""
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment


def _normalize_row_probability(x: np.ndarray) -> np.ndarray:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_rows, n_cols)

    Returns
    -------
    normalized_x : np.ndarray, shape (n_rows, n_cols)

    """
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def estimate_movement_var(
    position: np.ndarray, sampling_frequency: int = 1
) -> np.ndarray:
    """Estimates the movement variance based on position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dim)
        Position of the animal
    sampling_frequency : int, optional
        Number of samples per second.

    Returns
    -------
    movement_var : np.ndarray, shape (n_position_dim,)
        Variance of the movement.

    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_var = np.cov(np.diff(position, axis=0), rowvar=False)
    return movement_var * sampling_frequency


def _random_walk_on_track_graph(
    place_bin_centers: np.ndarray,
    movement_mean: float,
    movement_var: float,
    place_bin_center_ind_to_node: np.ndarray,
    distance_between_nodes: dict[dict],
) -> np.ndarray:
    """Estimates the random walk probabilities based on the spatial
    topology given by the track graph.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins,)
    movement_mean : float
    movement_var : float
    place_bin_center_ind_to_node : np.ndarray
        Mapping of place bin center to track graph node
    distance_between_nodes : dict[dict]
        Distance between each pair of track graph nodes with an edge.

    Returns
    -------
    random_walk : np.ndarray, shape (n_position_bins, n_position_bins)

    """
    state_transition = np.zeros((place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)

    for bin_ind1, node1 in enumerate(place_bin_center_ind_to_node):
        for bin_ind2, node2 in enumerate(place_bin_center_ind_to_node):
            try:
                state_transition[bin_ind1, bin_ind2] = gaussian.pdf(
                    distance_between_nodes[node1][node2]
                )
            except KeyError:
                # bins not on track interior will be -1 and not in distance
                # between nodes
                continue

    return state_transition


def _euclidean_random_walk(environment, movement_mean, movement_var):
    return np.stack(
        [
            multivariate_normal(mean=center + movement_mean, cov=movement_var).pdf(
                environment.place_bin_centers_
            )
            for center in environment.place_bin_centers_
        ],
        axis=1,
    )


@dataclass
class RandomWalk:
    """A transition where the movement stays locally close in space

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    movement_mean : float, optional
    use_manifold_distance : bool, optional
    direction : ("inward", "outward"), optional

    """

    environment_name: str = ""
    movement_var: float = 6.0
    movement_mean: float = 0.0
    use_manifold_distance: bool = False
    direction: Optional[str] = None

    def make_state_transition(self, environments: Tuple[Environment]) -> np.ndarray:
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        if self.environment.track_graph is None:
            transition_matrix = self._handle_no_track_graph()
        else:
            # Linearized position
            transition_matrix = self._handle_with_track_graph()
        is_track_interior = self.environment.is_track_interior_.ravel()
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0
        return _normalize_row_probability(transition_matrix)

    def _handle_no_track_graph(self) -> np.ndarray:
        if not self.use_manifold_distance:
            transition_matrix = _euclidean_random_walk(
                self.environment, self.movement_mean, self.movement_var
            )
        else:
            transition_matrix = (
                multivariate_normal(mean=self.movement_mean, cov=self.movement_var)
                .pdf(self.environment.distance_between_nodes_.flat)
                .reshape(self.environment.distance_between_nodes_.shape)
            )

            if self.direction is not None:
                direction_func = {
                    "inward": np.greater_equal,
                    "outward": np.less_equal,
                }.get(self.direction.lower(), None)

                centrality = nx.closeness_centrality(
                    self.environment.track_graphDD, distance="distance"
                )
                center_node_id = list(centrality.keys())[
                    np.argmax(list(centrality.values()))
                ]
                transition_matrix *= direction_func(
                    self.environment.distance_between_nodes_[:, [center_node_id]],
                    self.environment.distance_between_nodes_[[center_node_id]],
                )

        return transition_matrix

    def _handle_with_track_graph(self) -> np.ndarray:
        n_position_dims = self.environment.place_bin_centers_.shape[1]
        if n_position_dims != 1:
            raise NotImplementedError(
                "Random walk with track graph is only implemented for 1D environments"
            )

        place_bin_center_ind_to_node = np.asarray(
            self.environment.place_bin_centers_nodes_df_.node_id
        )
        return _random_walk_on_track_graph(
            self.environment.place_bin_centers_,
            self.movement_mean,
            self.movement_var,
            place_bin_center_ind_to_node,
            self.environment.distance_between_nodes_,
        )


@dataclass
class Uniform:
    """
    Attributes
    ----------
    environment_name : str, optional
        Name of first environment to fit
    environment_name2 : str, optional
        Name of second environment to fit if going from one environment to
        another
    """

    environment_name: str = ""
    environment2_name: str = None

    def make_state_transition(self, environments: Tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment1 = environments[environments.index(self.environment_name)]
        n_bins1 = self.environment1.place_bin_centers_.shape[0]
        is_track_interior1 = self.environment1.is_track_interior_.ravel()

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            self.environment2 = environments[environments.index(self.environment2_name)]
            n_bins2 = self.environment2.place_bin_centers_.shape[0]
            is_track_interior2 = self.environment2.is_track_interior_.ravel()

        transition_matrix = np.ones((n_bins1, n_bins2))

        transition_matrix[~is_track_interior1] = 0.0
        transition_matrix[:, ~is_track_interior2] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Identity:
    """A transition where the movement stays within a place bin

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    """

    environment_name: str = ""

    def make_state_transition(self, environments: Tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        n_bins = self.environment.place_bin_centers_.shape[0]

        transition_matrix = np.identity(n_bins)

        is_track_interior = self.environment.is_track_interior_.ravel()
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class EmpiricalMovement:
    """A transition matrix trained on the animal's actual movement

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    encoding_group : str, optional
        Name of encoding group to fit
    speedup : int, optional
        Used to make the empirical transition matrix "faster", means allowing for
        all the same transitions made by the animal but sped up by
        `speedup` times. So `speedup​=20` means 20x faster than the
        animal's movement.
    is_time_reversed : bool, optional
    """

    environment_name: str = ""
    encoding_group: str = 0
    speedup: int = 1
    is_time_reversed: bool = False

    def make_state_transition(
        self,
        environments: Tuple[Environment],
        position: np.ndarray,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
    ):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model
        position : np.ndarray
            Position of the animal
        is_training : np.ndarray, optional
            Boolean array that determines what data to train the place fields on, by default None
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, Some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]

        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            is_encoding = np.ones((n_time,), dtype=bool)
        else:
            is_encoding = encoding_group_labels == self.encoding_group

        if environment_labels is None:
            is_environment = np.ones((n_time,), dtype=bool)
        else:
            is_environment = environment_labels == self.environment_name

        position = position if position.ndim > 1 else position[:, np.newaxis]
        position = position[is_training & is_encoding & is_environment]

        if (
            len(self.environment.edges_) == 1
            and self.environment.track_graph is not None
        ):
            position = get_linearized_position(
                position=position,
                track_graph=self.environment.track_graph,
                edge_order=self.environment.edge_order,
                edge_spacing=self.environment.edge_spacing,
            ).linear_position.to_numpy()
            position = position if position.ndim > 1 else position[:, np.newaxis]

        if self.is_time_reversed:
            samples = np.concatenate((position[1:], position[:-1]), axis=1)
        else:
            samples = np.concatenate((position[:-1], position[1:]), axis=1)
        state_transition, _ = np.histogramdd(
            samples,
            bins=self.environment.edges_ * 2,
            range=self.environment.position_range,
        )
        original_shape = state_transition.shape
        n_position_dims = position.shape[1]
        shape_2d = np.product(original_shape[:n_position_dims])
        state_transition = _normalize_row_probability(
            state_transition.reshape((shape_2d, shape_2d))
        )

        return np.linalg.matrix_power(state_transition, self.speedup)


@dataclass
class RandomWalkDirection1:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: Tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.triu(random))


@dataclass
class RandomWalkDirection2:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: Tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.tril(random))


class Discrete:
    pass

    def make_state_transition(self, *args, **kwargs):
        """Creates a continuous transition matrix for a discrete state space.

        Essentially, there is no continuous state space, so the transition matrix is just an identity matrix.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        return np.ones((1, 1))
