"""Classes for constructing different types of movement models."""

from dataclasses import dataclass

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from scipy.stats import multivariate_normal  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

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
    # Handle cases where the sum is zero to avoid division by zero -> NaN
    row_sums = x.sum(axis=1, keepdims=True)
    # Use np.errstate to temporarily ignore invalid division warnings
    with np.errstate(invalid="ignore"):
        normalized_x = np.where(row_sums > 0, x / row_sums, 0.0)
    # Ensure any remaining NaNs (though unlikely with the above) are zero
    normalized_x[np.isnan(normalized_x)] = 0.0
    return normalized_x


def estimate_movement_var(position: np.ndarray) -> np.ndarray:
    """Estimates the movement variance based on position differences.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dim)
        Position of the animal

    Returns
    -------
    movement_var : np.ndarray, shape (n_position_dim,)
        Variance of the movement per time bin

    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    return np.cov(np.diff(position, axis=0), rowvar=False)


def _random_walk_on_track_graph(
    place_bin_centers: np.ndarray,
    movement_mean: float,
    movement_var: float,
    place_bin_center_ind_to_node: np.ndarray,
    distance_between_nodes: dict[int, dict[int, float]],
) -> np.ndarray:
    """Estimates the random walk probabilities based on the spatial
    topology given by the track graph.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins,)
        Center positions of the bins (assumed 1D for track graph).
    movement_mean : float
        Mean displacement for the random walk.
    movement_var : float
        Variance of the displacement for the random walk.
    place_bin_center_ind_to_node : np.ndarray, shape (n_position_bins,)
        Mapping from the place bin center index to the corresponding node ID
        in the track graph. Bins not on the track might map to a sentinel value like -1.
    distance_between_nodes : dict[dict]
        Nested dictionary where `distance_between_nodes[node1][node2]` gives the
        shortest path distance between `node1` and `node2` on the track graph.

    Returns
    -------
    random_walk : np.ndarray, shape (n_position_bins, n_position_bins)
        Transition probabilities between place bins based on graph distance.
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


def _euclidean_random_walk(
    environment: Environment, movement_mean: float, movement_var: float
) -> np.ndarray:
    """Estimates the random walk probabilities based on the Euclidean distance

    Parameters
    ----------
    environment : Environment
    movement_mean : float
    movement_var : float

    Returns
    -------
    transition : np.ndarray, shape (n_position_bins, n_position_bins)

    """
    if environment.place_bin_centers_ is not None:
        place_bin_centers = environment.place_bin_centers_
    else:
        raise ValueError("Environment must have defined place bin centers")

    return np.stack(
        [
            multivariate_normal(mean=center + movement_mean, cov=movement_var).pdf(
                place_bin_centers
            )
            for center in place_bin_centers
        ],
        axis=1,
    )


@dataclass
class RandomWalk:
    """A transition where the movement stays locally close in space.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit, defaults to "".
    movement_var : float, optional
        Variance of the displacement in one time step. Defaults to 6.0.
    movement_mean : float, optional
        Mean displacement in one time step. Defaults to 0.0.
    use_manifold_distance : bool, optional
        Whether to use graph distance (if available) or Euclidean distance.
        Defaults to False (Euclidean).
    direction : ("inward", "outward"), optional
        If using manifold distance on a 2D track, constrain movement direction
        relative to the track center. Defaults to None (no constraint).

    """

    environment_name: str = ""
    movement_var: float = 6.0
    movement_mean: float = 0.0
    use_manifold_distance: bool = False
    direction: str | None = None

    def make_state_transition(self, environments: tuple[Environment, ...]) -> np.ndarray:
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized transition probability matrix.
        """
        self.environment = environments[environments.index(self.environment_name)]
        if self.environment.track_graph is None:
            transition_matrix = self._handle_no_track_graph()
        else:
            # Linearized position
            transition_matrix = self._handle_with_track_graph()
        if self.environment.is_track_interior_ is not None:
            is_track_interior = self.environment.is_track_interior_.ravel()
        else:
            is_track_interior = np.ones(
                len(self.environment.place_bin_centers_), dtype=bool
            )
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0
        return _normalize_row_probability(transition_matrix)

    def _handle_no_track_graph(self) -> np.ndarray:
        """Calculate transition for environments without a defined track graph."""
        if not self.use_manifold_distance:
            transition_matrix = _euclidean_random_walk(
                self.environment, self.movement_mean, self.movement_var
            )
        else:
            if (
                isinstance(self.environment.distance_between_nodes_, np.ndarray)
                and self.environment.distance_between_nodes_ is not None
            ):
                distance_data = self.environment.distance_between_nodes_
                transition_matrix = (
                    multivariate_normal(mean=self.movement_mean, cov=self.movement_var)
                    .pdf(distance_data.flat)
                    .reshape(distance_data.shape)
                )
            else:
                # Fallback for dict or None case
                transition_matrix = _euclidean_random_walk(
                    self.environment, self.movement_mean, self.movement_var
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
        """Calculate transition for environments with a defined track graph (typically 1D)."""
        if self.environment.place_bin_centers_ is not None:
            n_position_dims = self.environment.place_bin_centers_.shape[1]
        else:
            raise ValueError("Environment must have defined place bin centers")
        if n_position_dims != 1:
            raise NotImplementedError(
                "Random walk with track graph is only implemented for 1D environments"
            )

        if self.environment.place_bin_centers_nodes_df_ is not None:
            place_bin_center_ind_to_node = np.asarray(
                self.environment.place_bin_centers_nodes_df_.node_id
            )
        else:
            raise ValueError(
                "Environment must have defined place_bin_centers_nodes_df_ for track graph"
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
    """Transition where all valid destination bins are equally likely.

    Attributes
    ----------
    environment_name : str, optional
        Name of the source environment. Defaults to "".
    environment2_name : str, optional
        Name of the destination environment if different from the source.
        If None, assumes transition within the same environment. Defaults to None.
    """

    environment_name: str = ""
    environment2_name: str = None

    def make_state_transition(self, environments: tuple[Environment, ...]) -> np.ndarray:
        """Creates a uniform transition matrix between environments.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_bins1, n_bins2)
            Row-normalized uniform transition probability matrix.
        """
        self.environment1 = environments[environments.index(self.environment_name)]
        if self.environment1.place_bin_centers_ is not None:
            n_bins1 = self.environment1.place_bin_centers_.shape[0]
        else:
            raise ValueError("Environment must have defined place bin centers")

        if self.environment1.is_track_interior_ is not None:
            is_track_interior1 = self.environment1.is_track_interior_.ravel()
        else:
            is_track_interior1 = np.ones(n_bins1, dtype=bool)

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            self.environment2 = environments[environments.index(self.environment2_name)]
            if self.environment2.place_bin_centers_ is not None:
                n_bins2 = self.environment2.place_bin_centers_.shape[0]
            else:
                raise ValueError("Environment must have defined place bin centers")

            if self.environment2.is_track_interior_ is not None:
                is_track_interior2 = self.environment2.is_track_interior_.ravel()
            else:
                is_track_interior2 = np.ones(n_bins2, dtype=bool)

        transition_matrix = np.ones((n_bins1, n_bins2))

        transition_matrix[~is_track_interior1] = 0.0
        transition_matrix[:, ~is_track_interior2] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Identity:
    """A transition where the movement must stay within the same place bin.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    """

    environment_name: str = ""

    def make_state_transition(self, environments: tuple[Environment, ...]) -> np.ndarray:
        """Creates an identity transition matrix for a given environment.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Identity matrix where invalid bins have zero probability.
        """
        self.environment = environments[environments.index(self.environment_name)]
        if self.environment.place_bin_centers_ is not None:
            n_bins = self.environment.place_bin_centers_.shape[0]
        else:
            raise ValueError("Environment must have defined place bin centers")

        transition_matrix = np.identity(n_bins)

        if self.environment.is_track_interior_ is not None:
            is_track_interior = self.environment.is_track_interior_.ravel()
        else:
            is_track_interior = np.ones(n_bins, dtype=bool)
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class EmpiricalMovement:
    """A transition matrix estimated from the animal's actual movement patterns.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    encoding_group : str or int, optional
        Name of encoding group to filter data by. Defaults to 0.
    speedup : int, optional
        Factor by which to speed up the estimated transitions. Corresponds to
        taking matrix powers of the one-step transition matrix. Defaults to 1.
    is_time_reversed : bool, optional
        If True, estimate transitions based on p(z_{t-1} | z_t) instead of p(z_{t+1} | z_t).
        Defaults to False.
    """

    environment_name: str = ""
    encoding_group: str = 0
    speedup: int = 1
    is_time_reversed: bool = False

    def make_state_transition(
        self,
        environments: tuple[Environment, ...],
        position: np.ndarray,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Creates a transition matrix based on observed animal movement.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.
        position : np.ndarray, shape (n_time, n_dims)
            Position of the animal.
        is_training : np.ndarray, shape (n_time,), optional
            Boolean mask to select time points for training. Defaults to all points.
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            Labels for encoding groups (e.g., inbound/outbound). Defaults to group 0.
        environment_labels : np.ndarray, shape (n_time,), optional
            Labels for environments. Defaults to the first environment.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized transition probability matrix, potentially powered by `speedup`.
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
        shape_2d = np.prod(original_shape[:n_position_dims])
        state_transition = _normalize_row_probability(
            state_transition.reshape((shape_2d, shape_2d))
        )

        return np.linalg.matrix_power(state_transition, self.speedup)


@dataclass
class RandomWalkDirection1:
    """A Gaussian random walk constrained to move in one direction (increasing index).

    Only valid for 1D environments.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    movement_var : float, optional
        Variance of the displacement. Defaults to 6.0.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment, ...]) -> np.ndarray:
        """Creates a unidirectional transition matrix.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized upper triangular transition matrix.
        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.triu(random))


@dataclass
class RandomWalkDirection2:
    """A Gaussian random walk constrained to move in one direction (decreasing index).

    Only valid for 1D environments.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    movement_var : float, optional
        Variance of the displacement. Defaults to 6.0.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment, ...]) -> np.ndarray:
        """Creates a unidirectional transition matrix.

        Parameters
        ----------
        environments : Tuple[Environment, ...]
            Tuple of available environments in the model.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized lower triangular transition matrix.
        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.tril(random))


class Discrete:
    """Creates a continuous transition matrix for a discrete state space.

    Essentially, there is no continuous state space, so the transition matrix
    is just an identity matrix.
    """

    pass

    def make_state_transition(self, *args, **kwargs) -> np.ndarray:
        """Creates a continuous transition matrix for a discrete state space.

        Since there is no continuous spatial component, the transition matrix
        is effectively a 1x1 identity matrix representing staying within the
        single discrete "location".

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (1, 1)
        """
        return np.ones((1, 1))
