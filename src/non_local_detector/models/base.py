import copy
import pickle
from collections import namedtuple
from functools import partial
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
    get_transition_matrix,
    hmm_smoother,
)
from non_local_detector.discrete_state_transitions import _estimate_discrete_transition
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    predict_no_spike_log_likelihood,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    Observations,
    Stickiness,
)

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)
np.seterr(divide="ignore", invalid="ignore")

_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS = {
    "waveform_std": 24.0,
    "position_std": 6.0,
    "block_size": 100,
}
_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS = {
    "position_std": 6.0,
    "use_diffusion": False,
    "block_size": 100,
}

State = namedtuple("state", ("environment_name", "encoding_group"))


class _DetectorBase(BaseEstimator):
    """Base class for detector objects."""

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        infer_track_interior: bool = True,
        state_names: list[str] | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
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
            environments = (Environment(),)
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

        self.sampling_frequency = sampling_frequency
        self.no_spike_rate = no_spike_rate

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
        self.initial_conditions_ = (
            self.continuous_initial_conditions_
            * self.discrete_initial_conditions[self.state_ind_]
        )

    def initialize_continuous_state_transition(
        self,
        continuous_transition_types: ContinuousTransitions,
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
                xticklabels=self.state_names,
                yticklabels=self.state_names,
                ax=ax,
                cbar_kws={"label": label},
            )
            ax.set_ylabel("Previous State", fontsize=12)
            ax.set_xlabel("Current State", fontsize=12)
            ax.set_title("Discrete State Transition", fontsize=16)
        else:
            raise NotImplementedError

    def plot_continuous_state_transition(self, figsize_scaling=1.5, vmax=0.3):
        GOLDEN_RATIO = 1.618

        fig, axes = plt.subplots(
            self.n_discrete_states_,
            self.n_discrete_states_,
            gridspec_kw=dict(
                width_ratios=self.bin_sizes_, height_ratios=self.bin_sizes_
            ),
            constrained_layout=True,
            figsize=(
                self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO,
                (self.n_discrete_states_ * figsize_scaling),
            ),
        )

        try:
            for from_state, ax_row in enumerate(axes):
                for to_state, ax in enumerate(ax_row):
                    ind = np.ix_(
                        self.state_ind_ == from_state, self.state_ind_ == to_state
                    )
                    ax.pcolormesh(
                        self.continuous_state_transitions_[ind], vmin=0.0, vmax=vmax
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if to_state == 0:
                        ax.set_ylabel(
                            self.state_names[from_state],
                            rotation=0,
                            ha="right",
                            va="center",
                        )

                    if from_state == self.n_discrete_states_ - 1:
                        ax.set_xlabel(
                            self.state_names[to_state],
                            rotation=45,
                            ha="right",
                            va="top",
                            labelpad=1.0,
                        )
            fig.supylabel("From State")
            fig.supxlabel("To State")
        except TypeError:
            axes.pcolormesh(self.continuous_state_transitions_, vmin=0.0, vmax=vmax)
            axes.set_xticks([])
            axes.set_yticks([])

    def plot_initial_conditions(self, figsize_scaling=1.5, vmax=0.3):
        GOLDEN_RATIO = 1.618
        fig, axes = plt.subplots(
            1,
            self.n_discrete_states_,
            gridspec_kw=dict(width_ratios=self.bin_sizes_),
            constrained_layout=True,
            figsize=(self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO, 1.1),
        )

        try:
            for state, ax in enumerate(axes):
                ind = self.state_ind_ == state
                ax.pcolormesh(
                    self.initial_conditions_[ind][:, np.newaxis], vmin=0.0, vmax=0.3
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(
                    self.state_names[state],
                    rotation=45,
                    ha="right",
                    va="top",
                    labelpad=0.0,
                )
        except TypeError:
            axes.pcolormesh(self.initial_conditions_[:, np.newaxis], vmin=0.0, vmax=0.3)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel(
                self.state_names[0],
                rotation=45,
                ha="right",
                va="top",
                labelpad=0.0,
            )

        fig.suptitle("Initial Conditions")

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
            continuous_transition_types=self.continuous_transition_types,
            position=position,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )

        return self

    def predict(self):
        logger.info("Computing posterior...")
        transition_fn = partial(
            get_transition_matrix,
            self.continuous_state_transitions_,
            self.discrete_state_transitions_,
            self.state_ind_,
        )

        (
            marginal_log_likelihood,
            causal_posterior,
            predictive_distribution,
            acausal_posterior,
        ) = hmm_smoother(
            self.initial_conditions_,
            None,
            self.log_likelihood_,
            transition_fn=transition_fn,
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
        transition_fn = partial(
            get_transition_matrix,
            self.continuous_state_transitions_,
            self.discrete_state_transitions_,
            self.state_ind_,
        )

        while not converged and (n_iter < max_iter):
            # Expectation step
            logger.info("Expectation step...")

            (
                marginal_log_likelihood,
                causal_posterior,
                predictive_distribution,
                acausal_posterior,
            ) = hmm_smoother(
                self.initial_conditions_,
                None,
                self.log_likelihood_,
                transition_fn=transition_fn,
            )
            # Maximization step
            logger.info("Maximization step..")
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
                logger.info("Computing stats..")
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

                    logger.info(
                        f"iteration {n_iter}, "
                        f"likelihood: {marginal_log_likelihoods[-1]}, "
                        f"change: {log_likelihood_change}"
                    )
                else:
                    logger.info(
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
        causal_posterior,
        acausal_posterior,
        acausal_state_probabilities,
        marginal_log_likelihoods,
    ):
        if time is None:
            time = np.arange(acausal_posterior.shape[0]) / self.sampling_frequency

        data_vars = {
            "causal_posterior": (("time", "state_bins"), causal_posterior),
            "acausal_posterior": (("time", "state_bins"), acausal_posterior),
            "acausal_state_probabilities": (
                ("time", "states"),
                acausal_state_probabilities,
            ),
        }
        environment_names = [obs.environment_name for obs in self.observation_models]
        encoding_group_names = [obs.encoding_group for obs in self.observation_models]

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

        states = np.asarray(self.state_names)

        n_position_dims = position.shape[1]

        if n_position_dims == 1:
            state_bins = pd.MultiIndex.from_arrays(
                (
                    (
                        states[self.state_ind_],
                        position[:, 0],
                    )
                ),
                names=("state", "position"),
            )
        elif n_position_dims == 2:
            state_bins = pd.MultiIndex.from_arrays(
                (
                    (
                        states[self.state_ind_],
                        position[:, 0],
                        position[:, 1],
                    )
                ),
                names=("state", "x_position", "y_position"),
            )
        elif n_position_dims == 3:
            state_bins = pd.MultiIndex.from_arrays(
                (
                    (
                        states[self.state_ind_],
                        position[:, 0],
                        position[:, 1],
                        position[:, 2],
                    )
                ),
                names=("state", "x_position", "y_position", "z_position"),
            )
        else:
            raise NotImplementedError

        coords = {
            "time": time,
            "state_bins": state_bins,
            "state_ind": self.state_ind_,
            "states": states,
            "environments": ("states", environment_names),
            "encoding_groups": ("states", encoding_group_names),
        }

        attrs = {"marginal_log_likelihoods": marginal_log_likelihoods}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        ).squeeze()


class ClusterlessDetector(_DetectorBase):
    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        clusterless_algorithm: str = "clusterless_kde_jax",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: list[str] | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
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
            sampling_frequency,
            no_spike_rate,
        )
        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_encoding_model(
        self,
        position,
        multiunits,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        logger.info("Fitting multiunits...")
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

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)
            ]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            likelihood_name = State(
                environment_name=obs.environment_name, encoding_group=obs.encoding_group
            )
            likelihood_name = State(
                environment_name=obs.environment_name, encoding_group=obs.encoding_group
            )

            encoding_algorithm, _ = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
            is_group = is_training & is_encoding & is_environment

            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position=position[is_group],
                multiunits=multiunits[is_group],
                place_bin_centers=environment.place_bin_centers_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
                edges=environment.edges_,
                **kwargs,
            )

    def fit(
        self,
        position,
        multiunits,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
        position = position[:, np.newaxis] if position.ndim == 1 else position
        super().fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_encoding_model(
            position,
            multiunits,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(self, position, multiunits, is_missing=None):
        logger.info("Computing log likelihood...")
        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
        if position is None and np.any(
            [obs.is_local for obs in self.observation_models]
        ):
            raise ValueError("Position must be provided for local observations.")

        n_time = multiunits.shape[0]
        if is_missing is None:
            is_missing = np.zeros((n_time,), dtype=bool)

        log_likelihood = np.zeros((n_time, self.n_state_bins_), dtype=np.float32)

        _, likelihood_func = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
        computed_likelihoods = []

        for state_id, obs in enumerate(self.observation_models):
            is_state_bin = self.state_ind_ == state_id
            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                obs.is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                # Likelihood of no spike times
                is_spike = np.any(np.isnan(multiunits), axis=1)
                log_likelihood[:, is_state_bin] = predict_no_spike_log_likelihood(
                    is_spike, self.no_spike_rate, self.sampling_frequency
                )
            elif likelihood_name not in computed_likelihoods:
                log_likelihood[:, is_state_bin] = likelihood_func(
                    position,
                    multiunits,
                    **self.encoding_model_[likelihood_name[:2]],
                    is_local=obs.is_local,
                )
            else:
                # Use previously computed likelihoods
                previously_computed_bins = (
                    self.state_ind_ == computed_likelihoods.index(likelihood_name)
                )
                log_likelihood[:, is_state_bin] = log_likelihood[
                    :, previously_computed_bins
                ]

            computed_likelihoods.append(likelihood_name)

        # missing data should be 1.0 because there is no information
        log_likelihood[is_missing] = 1.0

        return log_likelihood

    def predict(self, multiunits, position=None, time=None, is_missing=None):
        self.log_likelihood_ = self.compute_log_likelihood(
            position, multiunits, is_missing
        )

        (
            causal_posterior,
            acausal_posterior,
            _,
            _,
            acausal_state_probabilities,
            _,
            marginal_log_likelihood,
        ) = super().predict()

        return super()._convert_results_to_xarray(
            time,
            causal_posterior,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
        )

    def estimate_parameters(
        self,
        multiunits,
        position=None,
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
            multiunits,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )
        self.log_likelihood_ = self.compute_log_likelihood(position, multiunits)
        return super().estimate_parameters(
            self.log_likelihood_,
            time,
            estimate_inital_conditions,
            estimate_discrete_transition,
            max_iter,
            tolerance,
        )


class SortedSpikesDetector(_DetectorBase):
    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        sorted_spikes_algorithm: str = "sorted_spikes_kde_jax",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: list[str] | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
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
            sampling_frequency,
            no_spike_rate,
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
            likelihood_name = State(
                environment_name=obs.environment_name, encoding_group=obs.encoding_group
            )
            encoding_algorithm, _ = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ]
            is_group = is_training & is_encoding & is_environment

            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position=position[is_group],
                spikes=spikes[is_group],
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
        position = position[:, np.newaxis] if position.ndim == 1 else position
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
        logger.info("Computing log likelihood...")
        n_time = spikes.shape[0]

        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
        if position is None and np.any(
            [obs.is_local for obs in self.observation_models]
        ):
            raise ValueError("Position must be provided for local observations.")

        if is_missing is None:
            is_missing = np.zeros((n_time,), dtype=bool)

        log_likelihood = np.zeros((n_time, self.n_state_bins_), dtype=np.float32)

        _, likelihood_func = _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]
        computed_likelihoods = []

        for state_id, obs in enumerate(self.observation_models):
            is_state_bin = self.state_ind_ == state_id
            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                obs.is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                # Likelihood of no spike times
                log_likelihood[:, is_state_bin] = predict_no_spike_log_likelihood(
                    spikes, self.no_spike_rate, self.sampling_frequency
                )
            elif likelihood_name not in computed_likelihoods:
                log_likelihood[:, is_state_bin] = likelihood_func(
                    position,
                    spikes,
                    **self.encoding_model_[likelihood_name[:2]],
                    is_local=obs.is_local,
                )
            else:
                # Use previously computed likelihoods
                previously_computed_bins = (
                    self.state_ind_ == computed_likelihoods.index(likelihood_name)
                )
                log_likelihood[:, is_state_bin] = log_likelihood[
                    :, previously_computed_bins
                ]

            computed_likelihoods.append(likelihood_name)

        # missing data should be 1.0 because there is no information
        log_likelihood[is_missing] = 1.0

        return log_likelihood

    def predict(self, spikes, position=None, time=None, is_missing=None):
        self.log_likelihood_ = self.compute_log_likelihood(position, spikes, is_missing)
        (
            causal_posterior,
            acausal_posterior,
            _,
            _,
            acausal_state_probabilities,
            _,
            marginal_log_likelihood,
        ) = super().predict()

        return super()._convert_results_to_xarray(
            time,
            causal_posterior,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
        )

    def estimate_parameters(
        self,
        spikes,
        position=None,
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
