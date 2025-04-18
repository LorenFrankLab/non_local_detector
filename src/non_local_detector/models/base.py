import copy
import pickle
from logging import getLogger
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
import sklearn
import xarray as xr
from patsy import build_design_matrices
from sklearn.base import BaseEstimator
from track_linearization import get_linearized_position

from non_local_detector.continuous_state_transitions import EmpiricalMovement
from non_local_detector.core import (
    check_converged,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
)
from non_local_detector.discrete_state_transitions import (
    _estimate_discrete_transition,
    centered_softmax_forward,
    predict_discrete_state_transitions,
)
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
    StateNames,
    Stickiness,
)

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)
np.seterr(divide="ignore", invalid="ignore")

_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS = {
    "waveform_std": 24.0,
    "position_std": 6.0,
    "block_size": 10_000,
}
_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS = {
    "position_std": 6.0,
    "block_size": 10_000,
}


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
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ) -> None:
        """
        Initialize the _DetectorBase class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        """
        self._validate_initial_conditions(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            continuous_transition_types,
            discrete_transition_stickiness,
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
        self.environments = self._initialize_environments(environments)
        self.infer_track_interior = infer_track_interior

        # Observation model parameters
        self.observation_models = self._initialize_observation_models(
            observation_models, continuous_transition_types, environments
        )

        # State names
        self.state_names = self._initialize_state_names(
            state_names, discrete_initial_conditions
        )

        self.sampling_frequency = sampling_frequency
        self.no_spike_rate = no_spike_rate

    def _validate_initial_conditions(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        continuous_transition_types: ContinuousTransitions,
        discrete_transition_stickiness: Stickiness,
    ) -> None:
        """
        Validate the initial conditions.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.

        Raises
        ------
        ValueError
            If the number of discrete initial conditions does not match the number of continuous initial conditions or transition types.
        """
        if len(discrete_initial_conditions) != len(continuous_initial_conditions_types):
            raise ValueError(
                "Number of discrete initial conditions must match number of continuous initial conditions."
            )
        if len(discrete_initial_conditions) != len(continuous_transition_types):
            raise ValueError(
                "Number of discrete initial conditions must match number of continuous transition types."
            )
        if len(discrete_initial_conditions) != len(
            discrete_transition_stickiness
        ) and not isinstance(discrete_transition_stickiness, float):
            raise ValueError(
                "Discrete transition stickiness must be set for all states or a float"
            )

    def _initialize_environments(self, environments: Environments) -> Environments:
        """
        Initialize environments.

        Parameters
        ----------
        environments : Environments
            Environments in which the detector operates.

        Returns
        -------
        Environments
            Initialized environments.
        """
        if environments is None:
            environments = (Environment(),)
        if not hasattr(environments, "__iter__"):
            environments = (environments,)
        return environments

    def _initialize_observation_models(
        self,
        observation_models: Observations,
        continuous_transition_types: ContinuousTransitions,
        environments: Environments,
    ) -> Observations:
        """
        Initialize observation models.

        Parameters
        ----------
        observation_models : Observations
            Observation models for the detector.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        environments : Environments
            Environments in which the detector operates.

        Returns
        -------
        Observations
            Initialized observation models.
        """
        if observation_models is None:
            n_states = len(continuous_transition_types)
            env_name = environments[0].environment_name
            observation_models = (ObservationModel(env_name),) * n_states
        elif isinstance(observation_models, ObservationModel):
            observation_models = (observation_models,) * len(
                continuous_transition_types
            )
        return observation_models

    def _initialize_state_names(
        self, state_names: StateNames, discrete_initial_conditions: np.ndarray
    ) -> list[str]:
        """
        Initialize state names.

        Parameters
        ----------
        state_names : StateNames, optional
            Names of the states.
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.

        Returns
        -------
        state_names : list[str]

        Raises
        ------
        ValueError
            If the number of state names does not match the number of discrete initial conditions.
        """
        if state_names is None:
            state_names = [
                f"state {state_ind}"
                for state_ind in range(len(discrete_initial_conditions))
            ]
        if len(state_names) != len(discrete_initial_conditions):
            raise ValueError("Number of state names must match number of states.")
        return state_names

    def initialize_environments(
        self, position: np.ndarray, environment_labels: Optional[np.ndarray] = None
    ) -> None:
        """
        Fits the Environment class on the position data to get information about the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None
        """
        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name

            env_position = position[is_environment]
            if environment.track_graph is not None:
                # convert to 1D
                env_position = get_linearized_position(
                    env_position,
                    environment.track_graph,
                    edge_order=environment.edge_order,
                    edge_spacing=environment.edge_spacing,
                ).linear_position.to_numpy()

            environment.fit_place_grid(
                env_position, infer_track_interior=self.infer_track_interior
            )

    def initialize_state_index(self) -> None:
        """Initialize indices and parameters related to the combined state space.

        Determines the total number of bins across all discrete states (spatial
        bins for continuous states, 1 for discrete states like 'Local' or
        'No-Spike') and creates mappings between these combined bins and the
        original discrete states. Also identifies which combined bins correspond
        to the track interior.

        Attributes
        ----------
        n_discrete_states_ : int
            Total number of discrete states defined in the model.
        state_ind_ : np.ndarray, shape (n_total_bins,)
            Index mapping each combined bin to its corresponding discrete state index.
        n_state_bins_ : int
            Total number of bins across all states (sum of spatial bins for
            continuous states and 1 for discrete states). Referred to as
            `n_total_bins` in the shape description here for clarity.
        bin_sizes_ : np.ndarray, shape (n_discrete_states_,)
             Number of bins associated with each discrete state (e.g., number
             of place bins for spatial states, 1 for non-spatial states).
        is_track_interior_state_bins_ : np.ndarray, shape (n_total_bins,)
             Boolean array indicating if a combined state bin corresponds to the
             track interior. For non-spatial states, this is typically True.

        """
        self.n_discrete_states_ = len(self.state_names)
        bin_sizes = []
        state_ind = []
        is_track_interior = []
        for ind, obs in enumerate(self.observation_models):
            if obs.is_local or obs.is_no_spike:
                bin_sizes.append(1)
                state_ind.append(ind * np.ones((1,), dtype=int))
                is_track_interior.append(np.ones((1,), dtype=bool))
            else:
                environment = self.environments[
                    self.environments.index(obs.environment_name)
                ]
                bin_sizes.append(environment.place_bin_centers_.shape[0])
                state_ind.append(ind * np.ones((bin_sizes[-1],), dtype=int))
                is_track_interior.append(environment.is_track_interior_.ravel())

        self.state_ind_ = np.concatenate(state_ind)
        self.n_state_bins_ = len(self.state_ind_)
        self.bin_sizes_ = np.array(bin_sizes)
        self.is_track_interior_state_bins_ = np.concatenate(is_track_interior)

    def initialize_initial_conditions(self) -> None:
        """Constructs the initial probability for the state and each spatial bin.

        Attributes
        ----------
        continuous_initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Initial probability distribution over the bins within each state.
        initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Overall initial probability distribution across all state bins.
        """
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
        position: Optional[np.ndarray] = None,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Constructs the transition matrices for the continuous states.

        Parameters
        ----------
        continuous_transition_types : ContinuousTransitions
            Types of transition models.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None.

        Attributes
        ----------
        continuous_state_transitions_ : np.ndarray, shape (n_state_bins, n_state_bins)
            Probability of transitioning between bins, assuming a transition between the corresponding discrete states occurs.
        continuous_transition_types : ContinuousTransitions
            Stores the continuous transition types used.
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
                    self.continuous_state_transitions_[inds] = (
                        transition.make_state_transition(
                            self.environments,
                            position,
                            is_training,
                            encoding_group_labels,
                            environment_labels,
                        )
                    )
                else:
                    n_row_bins = np.max(inds[0].shape)
                    n_col_bins = np.max(inds[1].shape)

                    if np.logical_and(n_row_bins == 1, n_col_bins > 1):
                        # transition from discrete to continuous
                        # ASSUME uniform for now
                        environment = self.environments[
                            self.environments.index(transition.environment_name)
                        ]
                        self.continuous_state_transitions_[inds] = (
                            environment.is_track_interior_.ravel()
                            / environment.is_track_interior_.sum()
                        ).astype(float)
                    else:
                        self.continuous_state_transitions_[inds] = (
                            transition.make_state_transition(self.environments)
                        )

    def initialize_discrete_state_transition(
        self, covariate_data: Union[pd.DataFrame, dict, None] = None
    ) -> None:
        """
        Constructs the transition matrix for the discrete states.

        Parameters
        ----------
        covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Attributes
        ----------
        discrete_state_transitions_ : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
            Probability of transitioning between discrete states. Shape is (n_states, n_states) if no covariates, otherwise depends on covariate data.
        discrete_transition_coefficients_ : np.ndarray or None
            Fitted coefficients for covariate-dependent transitions.
        discrete_transition_design_matrix_ : patsy.DesignMatrix or None
            Design matrix information for covariate-dependent transitions.
        """
        logger.info("Fitting discrete state transition")
        (
            self.discrete_state_transitions_,
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
        ) = self.discrete_transition_type.make_state_transition(covariate_data)

    def plot_discrete_state_transition(
        self,
        cmap: str = "Oranges",
        ax: Optional[matplotlib.axes.Axes] = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
        covariate_data: Union[pd.DataFrame, dict, None] = None,
    ) -> None:
        """
        Plot heatmap of discrete transition matrix.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap, by default "Oranges".
        ax : matplotlib.axes.Axes, optional
            Plotting axis, by default plots to current axis.
        convert_to_seconds : bool, optional
            Convert the probabilities of state to expected duration of state, by default False.
        sampling_frequency : int, optional
            Number of samples per second, by default 1.
        covariate_data: dict or pd.DataFrame, optional
            Dictionary or DataFrame of covariate data, by default None.
            Keys are covariate names and values are 1D arrays.
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
            discrete_transition_design_matrix = self.discrete_transition_design_matrix_
            discrete_transition_coefficients = self.discrete_transition_coefficients_
            state_names = self.state_names

            predict_matrix = build_design_matrices(
                [discrete_transition_design_matrix.design_info], covariate_data
            )[0]

            n_states = len(state_names)

            for covariate in covariate_data:
                fig, axes = plt.subplots(
                    1, n_states, sharex=True, constrained_layout=True, figsize=(10, 5)
                )

                for from_state_ind, (ax, from_state) in enumerate(
                    zip(axes.flat, state_names)
                ):
                    from_local_transition = centered_softmax_forward(
                        predict_matrix
                        @ discrete_transition_coefficients[:, from_state_ind]
                    )

                    ax.plot(covariate_data[covariate], from_local_transition)
                    ax.set_xlabel(covariate)
                    ax.set_ylabel("Prob.")
                    if from_state_ind == n_states - 1:
                        ax.legend(state_names)
                    ax.set_title(f"From {from_state}")
                fig.suptitle(f"Predicted transitions: {covariate}")

    def plot_continuous_state_transition(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of continuous state transition matrices.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
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

    def plot_initial_conditions(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of initial conditions.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
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
                    self.initial_conditions_[ind][:, np.newaxis], vmin=0.0, vmax=vmax
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
            axes.pcolormesh(
                self.initial_conditions_[:, np.newaxis], vmin=0.0, vmax=vmax
            )
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

    def _fit(
        self,
        position: Optional[np.ndarray] = None,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
    ) -> "_DetectorBase":
        """
        Fit the model to the data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, optional, shape (n_time,)
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, optional, shape (n_time,)
            If there are multiple environments, label each time point with the environment name, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        _DetectorBase
            Fitted model.
        """
        position = position[:, np.newaxis] if position.ndim == 1 else position
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

    def compute_log_likelihood(self):
        """Compute the log likelihood. To be implemented by inheriting class."""
        raise NotImplementedError

    def _predict(
        self,
        time: Optional[np.ndarray] = None,
        log_likelihood_args: tuple = (),
        is_missing: Optional[np.ndarray] = None,
        log_likelihoods: Optional[np.ndarray] = None,
        cache_likelihood: bool = True,
        n_chunks: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the posterior probabilities.

        Parameters
        ----------
        time : np.ndarray, optional
            Time points for decoding, by default None
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default ()
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None
        log_likelihoods : np.ndarray, optional
            Precomputed log likelihoods, by default None
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default True
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1

        Returns
        -------
        acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        marginal_log_likelihood : float
        causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        """
        logger.info("Computing posterior...")
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]

        if self.discrete_state_transitions_.ndim == 2:
            return chunked_filter_smoother(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * self.discrete_state_transitions_[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
            )
        else:
            return chunked_filter_smoother_covariate_dependent(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=self.discrete_state_transitions_,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
            )

    def fit_predict(self) -> xr.Dataset:
        """Fit the model and predict the posterior probabilities. To be implemented by inheriting class."""
        raise NotImplementedError

    def fit_encoding_model(self):
        """Fit the encoding model. To be implemented by inheriting class."""
        raise NotImplementedError

    def estimate_parameters(
        self,
        time: Optional[np.ndarray] = None,
        log_likelihood_args: Optional[tuple] = None,
        is_missing: Optional[np.ndarray] = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        save_log_likelihood_to_results: bool = False,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        time : np.ndarray, optional, shape (n_time,)
            Time points for decoding, by default None.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True.
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, by default 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        save_log_likelihood_to_results : bool, optional
            Whether to save the log likelihood to the results, by default False.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding, including posterior probabilities and marginal log likelihoods.
        """
        marginal_log_likelihoods = []
        n_iter = 0
        converged = False

        while not converged and (n_iter < max_iter):
            # Expectation step
            logger.info("Expectation step...")
            (
                acausal_posterior,
                acausal_state_probabilities,
                marginal_log_likelihood,
                causal_state_probabilities,
                predictive_state_probabilities,
                log_likelihood,
            ) = self._predict(
                time=time,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                cache_likelihood=cache_likelihood,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )
            # Maximization step
            logger.info("Maximization step...")

            if estimate_encoding_model:
                try:
                    local_state_index = self.state_names.index("Local")
                except ValueError:
                    # Handle case where 'Local' state might not exist or has different name
                    local_state_index = None  # Or raise error

                if local_state_index is not None:
                    logger.info("Estimating encoding model...")
                    local_state_weights = acausal_state_probabilities[
                        :, local_state_index
                    ]
                    # Ensure local_state_weights are not zero
                    local_state_weights = np.clip(
                        local_state_weights, a_min=1e-15, a_max=1.0
                    )
                    # Re-fit the encoding model using the posterior weights
                    self.fit_encoding_model(
                        **self._encoding_model_data,
                        weights=local_state_weights,
                    )
                    if cache_likelihood:
                        try:
                            del self.log_likelihood_
                        except AttributeError:
                            pass

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

            if estimate_initial_conditions:
                self.initial_conditions_[self.is_track_interior_state_bins_] = (
                    acausal_posterior[0]
                )
                self.discrete_initial_conditions = acausal_state_probabilities[0]

                expanded_discrete_ic = acausal_state_probabilities[0][self.state_ind_]
                self.continuous_initial_conditions_ = np.where(
                    np.isclose(expanded_discrete_ic, 0.0),
                    0.0,
                    self.initial_conditions_ / expanded_discrete_ic,
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

        if store_log_likelihood:
            self.log_likelihood_ = log_likelihood

        if hasattr(self, "encoding_model_data_"):
            del self.encoding_model_data_

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihoods,
            log_likelihood=log_likelihood if save_log_likelihood_to_results else None,
        )

    def most_likely_sequence(
        self,
        time: np.ndarray,
        log_likelihood_args: Optional[tuple] = None,
        is_missing: Optional[np.ndarray] = None,
        n_chunks: int = 1,
    ) -> np.ndarray:
        """Find the most likely sequence of states.

        Returns
        -------
        pd.DataFrame, shape (n_time, n_columns)
            DataFrame containing the most likely sequence of states
            and corresponding positions/metadata at each time step.

        """
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]
        if self.discrete_state_transitions_.ndim == 2:
            sequence_ind = most_likely_sequence(
                time=time,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * self.discrete_state_transitions_[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )
        else:
            sequence_ind = most_likely_sequence_covariate_dependent(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=self.discrete_state_transitions_,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )

        return self._convert_seq_to_df(sequence_ind, time)

    def calculate_time_bins(self, time_range: np.ndarray) -> np.ndarray:
        """
        Calculate time bins based on the provided time range.

        Parameters
        ----------
        time_range : np.ndarray, shape (2,)
            Array specifying the range of time.

        Returns
        -------
        time : np.ndarray, shape (n_time_bins,)
            Array of time bins.
        """
        n_time_bins = int(
            np.ceil((time_range[-1] - time_range[0]) * self.sampling_frequency)
        )
        return time_range[0] + np.arange(n_time_bins) / self.sampling_frequency

    def save_model(self, filename: str = "model.pkl") -> None:
        """
        Save the detector to a pickled file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the model, by default "model.pkl".
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str = "model.pkl") -> "_DetectorBase":
        """
        Load the detector from a file.

        Parameters
        ----------
        filename : str, optional
            Filename to load the model from, by default "model.pkl"

        Returns
        -------
        _DetectorBase
            Loaded detector instance
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_results(results: xr.Dataset, filename: str = "results.nc") -> None:
        """
        Save the results to a netcdf file.

        `state_bins`is a multiindex, which is not supported by netcdf so
        it is converted before saving.

        Parameters
        ----------
        results : xr.Dataset
            Decoding results
        filename : str, optional
            Name to save, by default "results.nc"
        """
        results.reset_index("state_bins").to_netcdf(filename)

    @staticmethod
    def load_results(filename: str = "results.nc") -> xr.Dataset:
        """
        Loads the results from a netcdf file and converts the
        index back to a multiindex.

        Parameters
        ----------
        filename : str, optional
            File containing results, by default "results.nc"

        Returns
        -------
        xr.Dataset
            Decoding results
        """
        results = xr.open_dataset(filename)
        coord_names = list(results["state_bins"].coords)
        return results.set_index(state_bins=coord_names)

    def copy(self) -> "_DetectorBase":
        """
        Makes a copy of the detector.

        Returns
        -------
        _DetectorBase
            Deep copy of the detector instance.
        """
        return copy.deepcopy(self)

    def _convert_results_to_xarray(
        self,
        time: np.ndarray,
        acausal_posterior: np.ndarray,
        acausal_state_probabilities: np.ndarray,
        marginal_log_likelihoods: list[float],
        log_likelihood: Optional[np.ndarray] = None,
    ) -> xr.Dataset:
        """
        Convert the results to an xarray Dataset.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
            Acausal posterior probabilities.
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
            Acausal state probabilities.
        marginal_log_likelihoods : list of float
            Marginal log likelihoods for each iteration.
        log_likelihood : np.ndarray, optional, shape (n_time, n_state_bins)
            Log likelihoods, by default None.

        Returns
        -------
        xr.Dataset
            Decoding results in an xarray Dataset.
        """
        is_track_interior = self.is_track_interior_state_bins_

        environment_names = [obs.environment_name for obs in self.observation_models]
        encoding_group_names = [obs.encoding_group for obs in self.observation_models]

        position = []
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]
        for obs in self.observation_models:
            if obs.is_local or obs.is_no_spike:
                position.append(np.full((1, n_position_dims), np.nan))
            else:
                environment = self.environments[
                    self.environments.index(obs.environment_name)
                ]
                position.append(environment.place_bin_centers_)
        position = np.concatenate(position, axis=0)

        states = np.asarray(self.state_names)

        if n_position_dims == 1:
            position_names = ["position"]
        else:
            position_names = [
                f"{name}_position" for name, _ in zip(["x", "y", "z", "w"], position.T)
            ]
        state_bins = pd.MultiIndex.from_arrays(
            ((states[self.state_ind_], *[pos for pos in position.T])),
            names=("state", *position_names),
        )

        coords = {
            "time": time,
            "state_bins": state_bins,
            "state_ind": self.state_ind_,
            "states": states,
            "environments": ("states", environment_names),
            "encoding_groups": ("states", encoding_group_names),
        }

        attrs = {"marginal_log_likelihoods": np.asarray(marginal_log_likelihoods)}

        posterior_shape = (acausal_posterior.shape[0], len(is_track_interior))

        results = xr.Dataset(
            data_vars={
                "acausal_posterior": (
                    ("time", "state_bins"),
                    np.full(posterior_shape, np.nan, dtype=np.float32),
                ),
                "acausal_state_probabilities": (
                    ("time", "states"),
                    acausal_state_probabilities,
                ),
            },
            coords=coords,
            attrs=attrs,
        )

        results["acausal_posterior"][:, is_track_interior] = acausal_posterior

        if log_likelihood is not None:
            results["log_likelihood"] = (
                ("time", "state_bins"),
                np.full(posterior_shape, np.nan, dtype=np.float32),
            )
            results["log_likelihood"][:, is_track_interior] = log_likelihood

        return results.squeeze()

    def _convert_seq_to_df(
        self, sequence_ind: np.ndarray, time: np.ndarray
    ) -> pd.DataFrame:
        """Converts the sequence indices to a DataFrame.

        Parameters
        ----------
        sequence_ind : np.ndarray, shape (n_time,)
            Most likely sequence indices.
        time : np.ndarray, shape (n_time,)

        Returns
        -------
        results : pd.DataFrame, shape (n_time, n_cols)
        """
        position = []
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]
        environment_names = []
        encoding_group_names = []
        for obs in self.observation_models:
            if obs.is_local or obs.is_no_spike:
                position.append(np.full((1, n_position_dims), np.nan))
                environment_names.append([None])
                encoding_group_names.append([None])
            else:
                environment = self.environments[
                    self.environments.index(obs.environment_name)
                ]
                position.append(environment.place_bin_centers_)
                environment_names.append(
                    [obs.environment_name] * environment.place_bin_centers_.shape[0]
                )
                encoding_group_names.append(
                    [obs.encoding_group] * environment.place_bin_centers_.shape[0]
                )

        position = np.concatenate(position, axis=0)
        environment_names = np.concatenate(environment_names, axis=0)
        encoding_group_names = np.concatenate(encoding_group_names, axis=0)

        states = np.asarray(self.state_names)
        if n_position_dims == 1:
            position_names = ["position"]
        else:
            position_names = [
                f"{name}_position" for name, _ in zip(["x", "y", "z", "w"], position.T)
            ]
        state_bins = pd.DataFrame(
            {
                "state": states[self.state_ind_],
                **{name: pos for name, pos in zip(position_names, position.T)},
                "environment": environment_names,
                "encoding_group_names": encoding_group_names,
            }
        ).iloc[self.is_track_interior_state_bins_]

        return state_bins.iloc[sequence_ind].set_index(pd.Index(time, name="time"))


class ClusterlessDetector(_DetectorBase):
    """
    Detector class for clusterless spikes.
    """

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
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ) -> None:
        """
        Initialize the ClusterlessDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        clusterless_algorithm : str, optional
            Algorithm for clusterless spikes, by default "clusterless_kde".
        clusterless_algorithm_params : dict, optional
            Parameters for the clusterless algorithm, by default _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        """
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

    def _get_group_spike_data(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_group: np.ndarray,
        position_time: np.ndarray,
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Get group spike data based on is_group mask.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_group : np.ndarray, shape (n_time_position,)
            Boolean mask indicating group membership.
        position_time : np.ndarray
            Time points for position data.

        Returns
        -------
        group_spike_times : list of np.ndarray
        group_spike_waveform_features : list of np.ndarray
        """
        # get consecutive runs in each group
        group_labels, n_groups = scipy.ndimage.label(is_group)

        time_delta = position_time[1] - position_time[0]

        group_spike_times = []
        group_spike_waveform_features = []
        for electrode_spike_times, electrode_spike_waveform_features in zip(
            spike_times, spike_waveform_features
        ):
            group_electrode_spike_times = []
            group_electrode_waveform_features = []
            # get spike times for each run
            for group in range(1, n_groups + 1):
                start_time, stop_time = position_time[group_labels == group][[0, -1]]
                # Add half a time bin to the start and stop times
                # to ensure that the spike times are within the group
                start_time -= time_delta
                stop_time += time_delta
                is_valid_spike_time = np.logical_and(
                    electrode_spike_times >= start_time,
                    electrode_spike_times <= stop_time,
                )
                group_electrode_spike_times.append(
                    electrode_spike_times[is_valid_spike_time]
                )
                group_electrode_waveform_features.append(
                    electrode_spike_waveform_features[is_valid_spike_time]
                )
            group_spike_times.append(np.concatenate(group_electrode_spike_times))
            group_spike_waveform_features.append(
                np.concatenate(group_electrode_waveform_features, axis=0)
            )

        return group_spike_times, group_spike_waveform_features

    def fit_encoding_model(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the encoding model to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array or weights indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,) optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, optional
            Environment labels, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models for each unique observation model
            configuration (environment, encoding group).
            The values depend on the chosen `clusterless_algorithm`.
        """
        logger.info("Fitting clusterless spikes...")
        n_time = position.shape[0]
        position = position if position.ndim > 1 else position[:, np.newaxis]

        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()

        is_nan = np.any(np.isnan(position), axis=1)
        position = position[~is_nan]
        position_time = position_time[~is_nan]
        is_training = is_training[~is_nan]
        encoding_group_labels = encoding_group_labels[~is_nan]
        environment_labels = environment_labels[~is_nan]

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
            likelihood_name = (obs.environment_name, obs.encoding_group)

            encoding_algorithm, _ = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
            is_group = is_training & is_encoding & is_environment
            (
                group_spike_times,
                group_spike_waveform_features,
            ) = self._get_group_spike_data(
                spike_times, spike_waveform_features, is_group, position_time
            )
            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position_time[is_group],
                position[is_group],
                group_spike_times,
                group_spike_waveform_features,
                environment,
                self.sampling_frequency,
                **kwargs,
            )

    def fit(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
    ) -> "ClusterlessDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        ClusterlessDetector
            Fitted detector instance.
        """
        self._fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_encoding_model(
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(
        self,
        time: np.ndarray,
        position_time: np.ndarray,
        position: Optional[np.ndarray],
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_missing: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the log likelihood for the given data.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.

        Returns
        -------
        log_likelihood : jnp.ndarray, shape (n_time, n_state_bins)
        """
        logger.info("Computing log likelihood...")
        if position is None and np.any(
            [obs.is_local for obs in self.observation_models]
        ):
            raise ValueError("Position must be provided for local observations.")

        n_time = len(time)
        if is_missing is None:
            is_missing = jnp.zeros((n_time,), dtype=bool)

        log_likelihood = jnp.zeros(
            (n_time, self.is_track_interior_state_bins_.sum()), dtype=np.float32
        )

        _, likelihood_func = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
        computed_likelihoods = []

        for state_id, obs in enumerate(self.observation_models):
            is_state_bin = (
                self.state_ind_[self.is_track_interior_state_bins_] == state_id
            )
            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                obs.is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                # Likelihood of no spike times
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    predict_no_spike_log_likelihood(
                        time, spike_times, self.no_spike_rate
                    )
                )
            elif likelihood_name not in computed_likelihoods:
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    likelihood_func(
                        time,
                        position_time,
                        position,
                        spike_times,
                        spike_waveform_features,
                        **self.encoding_model_[likelihood_name[:2]],
                        is_local=obs.is_local,
                    )
                )
            else:
                # Use previously computed likelihoods
                previously_computed_bins = self.state_ind_[
                    self.is_track_interior_state_bins_
                ] == computed_likelihoods.index(likelihood_name)
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    log_likelihood[:, previously_computed_bins]
                )

            computed_likelihoods.append(likelihood_name)

        # missing data should be 0.0 because there is no information
        return jnp.where(is_missing[:, jnp.newaxis], 0.0, log_likelihood)

    def predict(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        position: Optional[np.ndarray] = None,
        position_time: Optional[np.ndarray] = None,
        is_missing: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        save_log_likelihood_to_results: bool = False,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : np.ndarray, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        save_log_likelihood_to_results : bool, optional
            Whether to save the log likelihood to the results, by default False.

        Returns
        -------
        xr.Dataset
            Predicted posterior probabilities.
        """
        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
            nan_position = np.any(np.isnan(position), axis=1)
            if np.any(nan_position) and is_missing is None:
                is_missing = nan_position
            elif np.any(nan_position) and is_missing is not None:
                is_missing = np.logical_or(is_missing, nan_position)

        if is_missing is not None and len(is_missing) != len(time):
            raise ValueError(
                f"Length of is_missing must match length of time. Time is n_samples: {len(time)}"
            )

        if discrete_transition_covariate_data is not None:
            self.discrete_state_transitions_ = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )
        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            _,
            _,
            log_likelihood,
        ) = self._predict(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
        )

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            log_likelihood if save_log_likelihood_to_results else None,
        )

    def estimate_parameters(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        is_missing: Optional[np.ndarray] = None,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        save_log_likelihood_to_results: bool = False,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, by default 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        save_log_likelihood_to_results : bool, optional
            Whether to save the log likelihood to the results, by default False.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding.
        """
        self._encoding_model_data = {
            "position_time": position_time,
            "position": position,
            "spike_times": spike_times,
            "spike_waveform_features": spike_waveform_features,
            "is_training": None,
            "encoding_group_labels": encoding_group_labels,
            "environment_labels": environment_labels,
        }
        self.fit(
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
        )

        return super().estimate_parameters(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
        )

    def most_likely_sequence(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        is_missing: Optional[np.ndarray] = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_time, n_cols)
        """
        return super().most_likely_sequence(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            n_chunks=n_chunks,
        )


class SortedSpikesDetector(_DetectorBase):
    """
    Detector class for sorted spikes.
    """

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
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ) -> None:
        """
        Initialize the SortedSpikesDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        sorted_spikes_algorithm : str, optional
            Algorithm for sorted spikes, by default "sorted_spikes_kde".
        sorted_spikes_algorithm_params : dict, optional
            Parameters for the sorted spikes algorithm, by default _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        """
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

    @staticmethod
    def _get_group_spikes(
        spike_times: list[np.ndarray], is_group: np.ndarray, position_time: np.ndarray
    ) -> list[np.ndarray]:
        """
        Get group spike times based on is_group mask.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_group : np.ndarray
            Boolean mask indicating group membership.
        position_time : np.ndarray
            Time points for position data.

        Returns
        -------
        list of np.ndarray
            Grouped spike times.
        """
        # get consecutive runs in each group
        group_labels, n_groups = scipy.ndimage.label(is_group)

        time_delta = position_time[1] - position_time[0]

        group_spike_times = []
        for neuron_spike_times in spike_times:
            group_neuron_spike_times = []
            # get spike times for each run
            for group in range(1, n_groups + 1):
                start_time, stop_time = position_time[group_labels == group][[0, -1]]
                # Add half a time bin to the start and stop times
                # to ensure that the spike times are within the group
                start_time -= time_delta
                stop_time += time_delta
                group_neuron_spike_times.append(
                    neuron_spike_times[
                        np.logical_and(
                            neuron_spike_times >= start_time,
                            neuron_spike_times <= stop_time,
                        )
                    ]
                )
            group_spike_times.append(np.concatenate(group_neuron_spike_times))

        return group_spike_times

    def fit_encoding_model(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit place fields to the data.

        Parameters
        ----------
        position_time : np.ndarray
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_training : np.ndarray, optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, optional
            Environment labels, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models (place fields) for each unique
            observation model configuration (environment, encoding group).
            The values depend on the chosen `sorted_spikes_algorithm`.
        """
        logger.info("Fitting place fields...")
        n_time = position.shape[0]
        position = position if position.ndim > 1 else position[:, np.newaxis]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()
        is_nan = np.any(np.isnan(position), axis=1)
        position = position[~is_nan]
        position_time = position_time[~is_nan]
        is_training = is_training[~is_nan]
        encoding_group_labels = encoding_group_labels[~is_nan]
        environment_labels = environment_labels[~is_nan]

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
            likelihood_name = (obs.environment_name, obs.encoding_group)
            encoding_algorithm, _ = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ]
            is_group = is_training & is_encoding & is_environment
            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position_time=position_time,
                position=position,
                spike_times=self._get_group_spikes(
                    spike_times, is_group, position_time
                ),
                environment=environment,
                sampling_frequency=self.sampling_frequency,
                weights=weights[is_group] if weights is not None else None,
                **kwargs,
            )

    def fit(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
    ) -> "SortedSpikesDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        SortedSpikesDetector
            Fitted detector instance.
        """
        self._fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_encoding_model(
            position_time,
            position,
            spike_times,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(
        self,
        time: np.ndarray,
        position_time: np.ndarray,
        position: Optional[np.ndarray],
        spike_times: list[np.ndarray],
        is_missing: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the log likelihood for the given data.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.

        Returns
        -------
        log_likelihood : jnp.ndarray, shape (n_time, n_state_bins)
        """
        logger.info("Computing log likelihood...")
        n_time = len(time)

        if position is None and np.any(
            [obs.is_local for obs in self.observation_models]
        ):
            raise ValueError("Position must be provided for local observations.")

        if is_missing is None:
            is_missing = np.zeros((n_time,), dtype=bool)

        log_likelihood = jnp.zeros((n_time, self.is_track_interior_state_bins_.sum()))

        _, likelihood_func = _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]
        computed_likelihoods = []

        for state_id, obs in enumerate(self.observation_models):
            is_state_bin = (
                self.state_ind_[self.is_track_interior_state_bins_] == state_id
            )
            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                obs.is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                # Likelihood of no spike times
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    predict_no_spike_log_likelihood(
                        time, spike_times, self.no_spike_rate
                    )
                )
            elif likelihood_name not in computed_likelihoods:
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    likelihood_func(
                        time,
                        position_time,
                        position,
                        spike_times,
                        **self.encoding_model_[likelihood_name[:2]],
                        is_local=obs.is_local,
                    )
                )
            else:
                # Use previously computed likelihoods
                previously_computed_bins = self.state_ind_[
                    self.is_track_interior_state_bins_
                ] == computed_likelihoods.index(likelihood_name)
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    log_likelihood[:, previously_computed_bins]
                )

            computed_likelihoods.append(likelihood_name)

        # missing data should be 0.0 because there is no information
        return jnp.where(is_missing[:, jnp.newaxis], 0.0, log_likelihood)

    def predict(
        self,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        position: Optional[np.ndarray] = None,
        position_time: Optional[np.ndarray] = None,
        is_missing: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        save_log_likelihood_to_results: bool = False,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : np.ndarray, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        save_log_likelihood_to_results : bool, optional
            Whether to save the log likelihood to the results, by default False.

        Returns
        -------
        xr.Dataset
            Predicted posterior probabilities.
        """
        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
            nan_position = np.any(np.isnan(position), axis=1)
            if np.any(nan_position) and is_missing is None:
                is_missing = nan_position
            elif np.any(nan_position) and is_missing is not None:
                is_missing = np.logical_or(is_missing, nan_position)

        if is_missing is not None and len(is_missing) != len(time):
            raise ValueError(
                f"Length of is_missing must match length of time. Time is n_samples: {len(time)}"
            )

        if discrete_transition_covariate_data is not None:
            self.discrete_state_transitions_ = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )

        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            _,
            _,
            log_likelihood,
        ) = self._predict(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
        )

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            log_likelihood=log_likelihood if save_log_likelihood_to_results else None,
        )

    def estimate_parameters(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        is_missing: Optional[np.ndarray] = None,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
        discrete_transition_covariate_data: Union[pd.DataFrame, dict, None] = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        save_log_likelihood_to_results: bool = False,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities
         using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            Each element of the list is an array of spike times for a neuron
        time : np.ndarray, shape (n_time,)
            Decoding time points
        is_missing : np.ndarray, shape (n_time,), optional
            Denote missing samples, None includes all samples, by default None
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array where True values include the sample in estimating the firing rate by
            position, None includes all samples, by default None
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Labels denoting which environment the sample corresponds to, by default None
        discrete_transition_covariate_data : dict, optional
            Covariate data for a covariate dependent discrete transition.
            A dict-like object that can be used to look up variables, by default None
        estimate_initial_conditions : bool, optional
            Estimate the initial conditions, by default True
        estimate_discrete_transition : bool, optional
            Estimate the discrete transition matrix, by default True
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True.
        max_iter : int, optional
            Maximuim number of EM iterations, by default 20
        tolerance : float, optional
            Convergence tolerance for EM, by default 0.0001
        cache_likelihood : bool, optional
            Store the likelihood for faster iterations, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Number of chunks for processing, by default 1
        save_log_likelihood_to_results : bool, optional
            Whether to save the log likelihood to the results, by default False.

        Returns
        -------
        xr.Dataset
            Results of the decoding
        """
        position = position[:, np.newaxis] if position.ndim == 1 else position
        self._encoding_model_data = {
            "position_time": position_time,
            "position": position,
            "spike_times": spike_times,
            "is_training": None,
            "encoding_group_labels": encoding_group_labels,
            "environment_labels": environment_labels,
        }
        self.fit(
            position_time,
            position,
            spike_times,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
        )

        return super().estimate_parameters(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
        )

    def most_likely_sequence(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        is_missing: Optional[np.ndarray] = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            Each element of the list is an array of spike times for a neuron
        time : np.ndarray, shape (n_time,)
            Decoding time points
        is_missing : np.ndarray, shape (n_time,), optional
            Denote missing samples, None includes all samples, by default None
        n_chunks : int, optional
            Number of chunks for processing, by default 1

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_time, n_cols)
        """
        return super().most_likely_sequence(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            n_chunks=n_chunks,
        )
