import copy
import pickle
from logging import getLogger

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xarray as xr
from sklearn.base import BaseEstimator

from non_local_detector.continuous_state_transitions import EmpiricalMovement
from non_local_detector.core import (
    check_converged,
    convert_to_state_probability,
    forward,
    smoother,
)
from non_local_detector.discrete_state_transitions import _estimate_discrete_transition
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.models.non_local_model_defaults import (
    _DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    _DEFAULT_CONTINUOUS_INITIAL_CONDITIONS,
    _DEFAULT_CONTINUOUS_TRANSITIONS,
    _DEFAULT_DISCRETE_INITIAL_CONDITIONS,
    _DEFAULT_DISCRETE_TRANSITION_STICKINESS,
    _DEFAULT_DISCRETE_TRANSITION_TYPE,
    _DEFAULT_ENVIRONMENT,
    _DEFAULT_OBSERVATION_MODELS,
    _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    _DEFAULT_STATE_NAMES,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    Environments,
    ContinuousTransitions,
    Observations,
    ContinuousInitialConditions,
    Stickiness,
    DiscreteTransitions,
)

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)
np.seterr(divide="ignore", invalid="ignore")


class _DetectorBase(BaseEstimator):
    """Base class for detector objects."""

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = _DEFAULT_DISCRETE_INITIAL_CONDITIONS,
        continuous_initial_conditions_types: ContinuousInitialConditions = _DEFAULT_CONTINUOUS_INITIAL_CONDITIONS,
        discrete_transition_type: DiscreteTransitions = _DEFAULT_DISCRETE_TRANSITION_TYPE,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = _DEFAULT_DISCRETE_TRANSITION_STICKINESS,
        discrete_transition_regularization: float = 1e-3,
        continuous_transition_types: ContinuousTransitions = _DEFAULT_CONTINUOUS_TRANSITIONS,
        observation_models: Observations = _DEFAULT_OBSERVATION_MODELS,
        environments: Environments = _DEFAULT_ENVIRONMENT,
        infer_track_interior: bool = True,
        state_names: list[str] | None = _DEFAULT_STATE_NAMES,
    ):
        if len(discrete_initial_conditions) != len(continuous_initial_conditions_types):
            raise ValueError(
                "Number of discrete initial conditions must match number of continuous initial conditions."
            )
        if len(discrete_initial_conditions) != len(continuous_transition_types):
            raise ValueError(
                "Number of discrete initial conditions must match number of continuous transition types"
            )
        if len(discrete_initial_conditions) != len(
            discrete_transition_stickiness
        ) and not isinstance(discrete_transition_stickiness, float):
            raise ValueError(
                "Discrete transition stickiness must be set for all states or a float"
            )

        # Initial conditions parameters
        self.discrete_initial_conditions = discrete_initial_conditions
        self.continuous_initial_conditions_types = continuous_initial_conditions_types

        # Discrete state transition parameters
        self.discrete_transition_concentration = discrete_transition_concentration
        self.discrete_transition_stickiness = discrete_transition_stickiness
        self.discrete_transition_regularization = discrete_transition_regularization
        self.discrete_transition_type = discrete_transition_type

        # Continuous state transition parameters
        self.continuous_transition_types = continuous_transition_types

        # Environment parameters
        if environments is None:
            environments = (_DEFAULT_ENVIRONMENT,)
        if isinstance(environments, Environment):
            environments = (environments,)
        self.environments = environments
        self.infer_track_interior = infer_track_interior

        # Observation model parameters
        if observation_models is None:
            n_states = len(continuous_transition_types)
            env_name = environments[0].environment_name
            observation_models = (ObservationModel(env_name),) * n_states
        elif isinstance(observation_models, ObservationModel):
            observation_models = (observation_models,) * n_states

        self.observation_models = observation_models

        # State names
        if state_names is None:
            state_names = [
                f"state {state_ind}"
                for state_ind in range(len(discrete_initial_conditions))
            ]
        if len(state_names) != len(discrete_initial_conditions):
            raise ValueError("Number of state names must match number of states.")
        self.state_names = state_names

    def initialize_environments(
        self, position: np.ndarray, environment_labels: None | np.ndarray = None
    ) -> None:
        """Fits the Environment class on the position data to get information about the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None

        """
        if position.ndim == 1:
            position = position[:, np.newaxis]

        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name
            environment.fit_place_grid(
                position[is_environment], infer_track_interior=self.infer_track_interior
            )

    def initialize_state_index(self):
        self.n_discrete_states_ = len(self.state_names)
        bin_sizes = []
        state_ind = []
        for ind, obs in enumerate(self.observation_models):
            if obs.is_local or obs.is_no_spike:
                bin_sizes.append(1)
                state_ind.append(ind * np.ones((1,), dtype=int))
            else:
                environment = self.environments[
                    self.environments.index(obs.environment_name)
                ]
                bin_sizes.append(environment.place_bin_centers_.shape[0])
                state_ind.append(ind * np.ones((bin_sizes[-1],), dtype=int))

        self.state_ind_ = np.concatenate(state_ind)
        self.n_state_bins_ = len(self.state_ind_)
        self.bin_sizes_ = np.array(bin_sizes)

    def initialize_initial_conditions(self):
        """Constructs the initial probability for the state and each spatial bin."""
        logger.info("Fitting initial conditions...")
        self.continuous_initial_conditions_ = np.concatenate(
            [
                cont_ic.make_initial_conditions(obs, self.environments)
                for obs, cont_ic in zip(
                    self.observation_models,
                    self.continuous_initial_conditions_types,
                )
            ]
        )
        self.continuous_initial_conditions_ /= self.continuous_initial_conditions_.sum()
        self.initial_conditions_ = (
            self.continuous_initial_conditions_
            * self.discrete_initial_conditions[self.state_ind_]
        )

    def initialize_continuous_state_transition(
        self,
        continuous_transition_types: ContinuousTransitions = _DEFAULT_CONTINUOUS_TRANSITIONS,
        position: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
    ) -> None:
        """Constructs the transition matrices for the continuous states.

        Parameters
        ----------
        continuous_transition_types : list of list of transition matrix instances, optional
            Types of transition models, by default _DEFAULT_CONTINUOUS_TRANSITIONS
        position : np.ndarray, optional
            Position of the animal in the environment, by default None
        is_training : np.ndarray, optional
            Boolean array that determines what data to train the place fields on, by default None
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, Some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None

        """
        logger.info("Fitting continuous state transition...")

        self.continuous_transition_types = continuous_transition_types

        n_total_bins = len(self.state_ind_)
        self.continuous_state_transitions_ = np.zeros((n_total_bins, n_total_bins))

        for from_state, row in enumerate(self.continuous_transition_types):
            for to_state, transition in enumerate(row):
                inds = np.ix_(
                    self.state_ind_ == from_state, self.state_ind_ == to_state
                )

                if isinstance(transition, EmpiricalMovement):
                    if is_training is None:
                        n_time = position.shape[0]
                        is_training = np.ones((n_time,), dtype=bool)

                    if encoding_group_labels is None:
                        n_time = position.shape[0]
                        encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

                    is_training = np.asarray(is_training).squeeze()
                    self.continuous_state_transitions_[
                        inds
                    ] = transition.make_state_transition(
                        self.environments,
                        position,
                        is_training,
                        encoding_group_labels,
                        environment_labels,
                    )
                else:
                    n_row_bins = np.max(inds[0].shape)
                    n_col_bins = np.max(inds[1].shape)
                    self.continuous_state_transitions_[
                        inds
                    ] = transition.make_state_transition(self.environments)[
                        :n_row_bins, :n_col_bins
                    ]

    def initialize_discrete_state_transition(
        self, covariate_data: pd.DataFrame | dict | None = None
    ):
        """Constructs the transition matrix for the discrete states."""
        logger.info("Fitting discrete state transition")
        (
            self.discrete_state_transitions_,
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
        ) = self.discrete_transition_type.make_state_transition(covariate_data)

    def plot_discrete_state_transition(
        self,
        state_names: list[str] | None = None,
        cmap: str = "Oranges",
        ax: matplotlib.axes.Axes | None = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
    ) -> None:
        """Plot heatmap of discrete transition matrix.

        Parameters
        ----------
        state_names : list[str], optional
            Names corresponding to each discrete state, by default None
        cmap : str, optional
            matplotlib colormap, by default "Oranges"
        ax : matplotlib.axes.Axes, optional
            Plotting axis, by default plots to current axis
        convert_to_seconds : bool, optional
            Convert the probabilities of state to expected duration of state, by default False
        sampling_frequency : int, optional
            Number of samples per second, by default 1

        """

        if self.discrete_state_transitions_.ndim == 2:
            if ax is None:
                ax = plt.gca()

            if state_names is None:
                state_names = [
                    f"state {ind + 1}"
                    for ind in range(self.discrete_state_transitions_.shape[0])
                ]

            if convert_to_seconds:
                discrete_state_transition = (
                    1 / (1 - self.discrete_state_transitions_)
                ) / sampling_frequency
                vmin, vmax, fmt = 0.0, None, "0.03f"
                label = "Seconds"
            else:
                discrete_state_transition = self.discrete_state_transitions_
                vmin, vmax, fmt = 0.0, 1.0, "0.03f"
                label = "Probability"

            sns.heatmap(
                data=discrete_state_transition,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                xticklabels=state_names,
                yticklabels=state_names,
                ax=ax,
                cbar_kws={"label": label},
            )
            ax.set_ylabel("Previous State", fontsize=12)
            ax.set_xlabel("Current State", fontsize=12)
            ax.set_title("Discrete State Transition", fontsize=16)
        else:
            raise NotImplementedError

    def fit(
        self,
        position=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
        self.initialize_environments(
            position=position, environment_labels=environment_labels
        )
        self.initialize_state_index()
        self.initialize_initial_conditions()
        self.initialize_discrete_state_transition(
            covariate_data=discrete_transition_covariate_data
        )
        self.initialize_continuous_state_transition(
            position=position,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )

        return self

    def predict(self):
        (
            causal_posterior,
            predictive_distribution,
            marginal_log_likelihood,
        ) = forward(
            self.initial_conditions_,
            self.log_likelihood_,
            self.discrete_state_transitions_,
            self.continuous_state_transitions_,
            self.state_ind_,
        )
        acausal_posterior = smoother(
            causal_posterior,
            predictive_distribution,
            self.discrete_state_transitions_,
            self.continuous_state_transitions_,
            self.state_ind_,
        )

        (
            causal_state_probabilities,
            acausal_state_probabilities,
            predictive_state_probabilities,
        ) = convert_to_state_probability(
            causal_posterior,
            acausal_posterior,
            predictive_distribution,
            self.state_ind_,
        )

        return (
            causal_posterior,
            acausal_posterior,
            predictive_distribution,
            causal_state_probabilities,
            acausal_state_probabilities,
            predictive_state_probabilities,
            marginal_log_likelihood,
        )

    def fit_predict(self):
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def estimate_parameters(
        self,
        log_likelihood,
        time=None,
        estimate_inital_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
    ):
        marginal_log_likelihoods = []
        n_iter = 0
        converged = False

        while not converged and (n_iter < max_iter):
            # Expectation step
            print("Expectation Step")
            (
                causal_posterior,
                predictive_distribution,
                marginal_log_likelihood,
            ) = forward(
                self.initial_conditions_,
                log_likelihood,
                self.discrete_state_transitions_,
                self.continuous_state_transitions_,
                self.state_ind_,
            )
            acausal_posterior = smoother(
                causal_posterior,
                predictive_distribution,
                self.discrete_state_transitions_,
                self.continuous_state_transitions_,
                self.state_ind_,
            )
            # Maximization step
            print("Maximization Step")
            (
                causal_state_probabilities,
                acausal_state_probabilities,
                predictive_state_probabilities,
            ) = convert_to_state_probability(
                causal_posterior,
                acausal_posterior,
                predictive_distribution,
                self.state_ind_,
            )

            if estimate_discrete_transition:
                (
                    self.discrete_state_transitions_,
                    self.discrete_transition_coefficients_,
                ) = _estimate_discrete_transition(
                    causal_state_probabilities,
                    predictive_state_probabilities,
                    acausal_state_probabilities,
                    self.discrete_state_transitions_,
                    self.discrete_transition_coefficients_,
                    self.discrete_transition_design_matrix_,
                    self.discrete_transition_concentration,
                    self.discrete_transition_stickiness,
                    self.discrete_transition_regularization,
                )

                if estimate_inital_conditions:
                    self.initial_conditions_ = acausal_posterior[0]
                    self.discrete_initial_conditions_ = acausal_state_probabilities[0]

                    expanded_discrete_ic = acausal_state_probabilities[0][
                        self.state_ind_
                    ]
                    self.continuous_initial_conditions_ = np.where(
                        np.isclose(expanded_discrete_ic, 0.0),
                        0.0,
                        acausal_posterior[0] / expanded_discrete_ic,
                    )

                # Stats
                print("Stats")
                n_iter += 1

                marginal_log_likelihoods.append(marginal_log_likelihood)
                if n_iter > 1:
                    log_likelihood_change = (
                        marginal_log_likelihoods[-1] - marginal_log_likelihoods[-2]
                    )
                    converged, _ = check_converged(
                        marginal_log_likelihoods[-1],
                        marginal_log_likelihoods[-2],
                        tolerance,
                    )

                    print(
                        f"iteration {n_iter}, "
                        f"likelihood: {marginal_log_likelihoods[-1]}, "
                        f"change: {log_likelihood_change}"
                    )
                else:
                    print(
                        f"iteration {n_iter}, "
                        f"likelihood: {marginal_log_likelihoods[-1]}"
                    )

        return (
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihoods,
        )

    def save_model(self, filename: str = "model.pkl"):
        """Save the detector to a pickled file.

        Parameters
        ----------
        filename : str, optional

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str = "model.pkl"):
        """Load the detector from a file.

        Parameters
        ----------
        filename : str, optional

        Returns
        -------
        detector instance

        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def copy(self):
        """Makes a copy of the detector"""
        return copy.deepcopy(self)

    def _convert_results_to_xarray(
        self,
        time,
        acausal_posterior,
        acausal_state_probabilities,
        marginal_log_likelihoods,
    ):
        if time is None:
            time = np.arange(acausal_posterior.shape[0])

        data_vars = {
            "acausal_posterior": (("time", "state_bins"), acausal_posterior),
            "acausal_state_probabilities": (
                ("time", "states"),
                acausal_state_probabilities,
            ),
        }
        environment_names = [obs.environment_name for obs in self.observation_models]
        encoding_group_names = [obs.encoding_group for obs in self.observation_models]
        coords = {
            "time": time,
            "state_bins": self.state_ind_,
            "states": np.asarray(self.state_names),
            "environments": ("states", environment_names),
            "encoding_groups": ("states", encoding_group_names),
        }

        position = []
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]
        for obs in self.observation_models:
            if obs.is_local or obs.is_no_spike:
                nan_array = np.array([np.nan] * n_position_dims)
                if n_position_dims == 1:
                    nan_array = nan_array[:, np.newaxis]
                position.append(nan_array)
            else:
                environment = self.environments[
                    self.environments.index(obs.environment_name)
                ]
                position.append(environment.place_bin_centers_)
        position = np.concatenate(position, axis=0)

        coords["position"] = (("state_bins", "position_dims"), position)

        attrs = {"marginal_log_likelihoods": marginal_log_likelihoods}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )


class ClusterlessDetector(_DetectorBase):
    def __init__(
        self,
        clusterless_algorithm: str = "multiunit_likelihood_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_MODEL_KWARGS,
        discrete_initial_conditions: np.ndarray = _DEFAULT_DISCRETE_INITIAL_CONDITIONS,
        continuous_initial_conditions_types: ContinuousInitialConditions = _DEFAULT_CONTINUOUS_INITIAL_CONDITIONS,
        discrete_transition_type: DiscreteTransitions = _DEFAULT_DISCRETE_TRANSITION_TYPE,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = _DEFAULT_DISCRETE_TRANSITION_STICKINESS,
        discrete_transition_regularization: float = 0.001,
        continuous_transition_types: ContinuousTransitions = _DEFAULT_CONTINUOUS_TRANSITIONS,
        observation_models: Observations = _DEFAULT_OBSERVATION_MODELS,
        environments: Environments = _DEFAULT_ENVIRONMENT,
        infer_track_interior: bool = True,
        state_names: list[str] | None = _DEFAULT_STATE_NAMES,
    ):
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            infer_track_interior,
            state_names,
        )
        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_marks(
        self,
        position,
        marks,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        pass

    def fit(
        self,
        position,
        marks,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
        super().fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        return self

    def compute_log_likelihood(self, marks, is_missing=None):
        pass

    def predict(self, marks, time=None, is_missing=None):
        self.log_likelihood_ = self.compute_log_likelihood(marks, is_missing)

        (
            causal_posterior,
            acausal_posterior,
            predictive_distribution,
            causal_state_probabilities,
            acausal_state_probabilities,
            predictive_state_probabilities,
            marginal_log_likelihood,
        ) = super().predict()

        return (
            causal_posterior,
            acausal_posterior,
            predictive_distribution,
            causal_state_probabilities,
            acausal_state_probabilities,
            predictive_state_probabilities,
            marginal_log_likelihood,
        )


class SortedSpikesDetector(_DetectorBase):
    def __init__(
        self,
        sorted_spikes_algorithm: str = "sorted_spikes_kde_jax",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
        discrete_initial_conditions: np.ndarray = _DEFAULT_DISCRETE_INITIAL_CONDITIONS,
        continuous_initial_conditions_types: ContinuousInitialConditions = _DEFAULT_CONTINUOUS_INITIAL_CONDITIONS,
        discrete_transition_type: DiscreteTransitions = _DEFAULT_DISCRETE_TRANSITION_TYPE,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = _DEFAULT_DISCRETE_TRANSITION_STICKINESS,
        discrete_transition_regularization: float = 0.001,
        continuous_transition_types: ContinuousTransitions = _DEFAULT_CONTINUOUS_TRANSITIONS,
        observation_models: Observations = _DEFAULT_OBSERVATION_MODELS,
        environments: Environments = _DEFAULT_ENVIRONMENT,
        infer_track_interior: bool = True,
        state_names: list[str] | None = _DEFAULT_STATE_NAMES,
    ):
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            infer_track_interior,
            state_names,
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    def fit_place_fields(
        self,
        position,
        spikes,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        logger.info("Fitting place fields...")
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()

        kwargs = self.sorted_spikes_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}
        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)
            ]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
            )
            encoding_algorithm, _ = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ]
            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position=position[is_training & is_encoding & is_environment],
                spikes=spikes[is_training & is_encoding & is_environment],
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
                edges=environment.edges_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
                **kwargs,
            )

    def fit(
        self,
        position,
        spikes,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
        super().fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_place_fields(
            position,
            spikes,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(self, position, spikes, is_missing=None):
        if is_missing is None:
            is_missing = np.zeros((spikes.shape[0],), dtype=bool)

        n_time, n_state_bins = spikes.shape[0], self.n_state_bins_
        log_likelihood = np.zeros((n_time, n_state_bins), dtype=np.float32)

        _, likelihood_func = _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]
        _, no_spike_likelihood_func = _SORTED_SPIKES_ALGORITHMS["no_spike"]
        computed_keys = []

        for state_id, obs in enumerate(self.observation_models):
            is_state_bin = self.state_ind_ == state_id
            obs_key = (
                obs.environment_name,
                obs.encoding_group,
                obs.is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                log_likelihood[:, is_state_bin] = no_spike_likelihood_func(spikes)
            elif not obs_key in computed_keys:
                log_likelihood[:, is_state_bin] = likelihood_func(
                    position,
                    spikes,
                    **self.encoding_model_[obs_key[:2]],
                    is_local=obs.is_local,
                )
            else:
                previously_computed_bins = self.state_ind_ == computed_keys.index(
                    obs_key
                )
                log_likelihood[:, is_state_bin] = log_likelihood[
                    :, previously_computed_bins
                ]

            computed_keys.append(obs_key)

        return log_likelihood

    def predict(self, position, spikes, time=None, is_missing=None):
        self.log_likelihood_ = self.compute_log_likelihood(position, spikes, is_missing)
        (
            _,
            acausal_posterior,
            _,
            _,
            acausal_state_probabilities,
            _,
            marginal_log_likelihood,
        ) = super().predict()

        return super()._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
        )

    def estimate_parameters(
        self,
        position,
        spikes,
        time=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        estimate_inital_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        max_iter: int = 20,
        tolerance: float = 0.0001,
    ):
        self.fit(
            position,
            spikes,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )
        self.log_likelihood_ = self.compute_log_likelihood(position, spikes)
        return super().estimate_parameters(
            self.log_likelihood_,
            time,
            estimate_inital_conditions,
            estimate_discrete_transition,
            max_iter,
            tolerance,
        )
