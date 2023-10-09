import copy
import pickle
from functools import partial
from logging import getLogger
from typing import Union, Optional

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
    convert_to_state_probability,
    hmm_smoother,
)
from non_local_detector.discrete_state_transitions import (
    _estimate_discrete_transition,
    centered_softmax_forward,
    non_stationary_discrete_transition_fn,
    predict_discrete_state_transitions,
    stationary_discrete_transition_fn,
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
        if not hasattr(environments, "__iter__"):
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
        self, position: np.ndarray, environment_labels: Union[None, np.ndarray] = None
    ) -> None:
        """Fits the Environment class on the position data to get information about the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None

        """
        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name

            if environment.track_graph is not None:
                # convert to 1D
                position = get_linearized_position(
                    position,
                    environment.track_graph,
                    edge_order=environment.edge_order,
                    edge_spacing=environment.edge_spacing,
                ).linear_position.to_numpy()

            environment.fit_place_grid(
                position[is_environment], infer_track_interior=self.infer_track_interior
            )

    def initialize_state_index(self):
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
        position: Optional[np.ndarray] = None,
        is_training: Optional[np.ndarray] = None,
        encoding_group_labels: Optional[np.ndarray] = None,
        environment_labels: Optional[np.ndarray] = None,
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
                        self.continuous_state_transitions_[
                            inds
                        ] = transition.make_state_transition(self.environments)

    def initialize_discrete_state_transition(
        self, covariate_data: Union[pd.DataFrame, dict, None] = None
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
        ax: Union[matplotlib.axes.Axes, None] = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
        covariate_data: Union[dict, None] = None,
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
        covariate_data: dict, optional
            Dictionary of covariate data, by default None. Keys are covariate names and values are 1D arrays.

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

    def _fit(
        self,
        position=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
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

    def _predict(self):
        logger.info("Computing posterior...")
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)

        transition_fn = (
            stationary_discrete_transition_fn
            if self.discrete_state_transitions_.ndim == 2
            else non_stationary_discrete_transition_fn
        )
        transition_fn = partial(
            transition_fn,
            jnp.asarray(self.continuous_state_transitions_[cross_is_track_interior]),
            jnp.asarray(self.discrete_state_transitions_),
            jnp.asarray(self.state_ind_[is_track_interior]),
        )

        (
            marginal_log_likelihood,
            causal_posterior,
            predictive_distribution,
            acausal_posterior,
        ) = hmm_smoother(
            self.initial_conditions_[is_track_interior],
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
            self.state_ind_[is_track_interior],
        )
        logger.info("Finished computing posterior...")
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
        time=None,
        estimate_inital_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        return_causal_posterior: bool = False,
    ):
        marginal_log_likelihoods = []
        n_iter = 0
        converged = False

        while not converged and (n_iter < max_iter):
            # Expectation step
            logger.info("Expectation step...")
            (
                causal_posterior,
                acausal_posterior,
                _,
                causal_state_probabilities,
                acausal_state_probabilities,
                predictive_state_probabilities,
                marginal_log_likelihood,
            ) = self._predict()
            # Maximization step
            logger.info("Maximization step...")

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
                    self.initial_conditions_[
                        self.is_track_interior_state_bins_
                    ] = acausal_posterior[0]
                    self.discrete_initial_conditions = acausal_state_probabilities[0]

                    expanded_discrete_ic = acausal_state_probabilities[0][
                        self.state_ind_
                    ]
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

        return self._convert_results_to_xarray(
            time,
            causal_posterior,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihoods,
            return_causal_posterior,
        )

    def calculate_time_bins(self, time_range):
        n_time_bins = int(
            np.ceil((time_range[-1] - time_range[0]) * self.sampling_frequency)
        )
        return time_range[0] + np.arange(n_time_bins) / self.sampling_frequency

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

    @staticmethod
    def save_results(results: xr.Dataset, filename: str = "results.nc"):
        """Save the results to a netcdf file.

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
        """Loads the results from a netcdf file and converts the
        index back to a multiindex.

        Parameters
        ----------
        filename : str, optional
            File containing results, by default "results.nc"

        Returns
        -------
        results : xr.Dataset
            Decoding results
        """
        results = xr.open_dataset(filename)
        coord_names = list(results["state_bins"].coords)
        return results.set_index(state_bins=coord_names)

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
        return_causal_posterior: bool = False,
    ):
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
        if return_causal_posterior:
            results["causal_posterior"] = (
                ("time", "state_bins"),
                np.full(posterior_shape, np.nan, dtype=np.float32),
            )
            results["causal_posterior"][:, is_track_interior] = causal_posterior

        return results.squeeze()


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
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
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

    def _get_group_spike_data(
        self, spike_times, spike_waveform_features, is_group, position_time
    ):
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
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
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
        position_time: jnp.ndarray,
        position: jnp.ndarray,
        spike_times: list[jnp.ndarray],
        spike_waveform_features: list[jnp.ndarray],
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
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
        time,
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        is_missing=None,
    ):
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

        # missing data should be 1.0 because there is no information
        return jnp.where(is_missing[:, jnp.newaxis], 1.0, log_likelihood)

    def predict(
        self,
        spike_times,
        spike_waveform_features,
        time,
        position=None,
        position_time=None,
        is_missing=None,
        discrete_transition_covariate_data=None,
        return_causal_posterior: bool = False,
    ):
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

        self.log_likelihood_ = self.compute_log_likelihood(
            time,
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_missing,
        )

        (
            causal_posterior,
            acausal_posterior,
            _,
            _,
            acausal_state_probabilities,
            _,
            marginal_log_likelihood,
        ) = self._predict()

        return self._convert_results_to_xarray(
            time,
            causal_posterior,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            return_causal_posterior,
        )

    def estimate_parameters(
        self,
        position_time,
        position,
        spike_times,
        spike_waveform_features,
        time,
        is_missing=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
        estimate_inital_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        max_iter: int = 20,
        tolerance: float = 0.0001,
    ):
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
        self.log_likelihood_ = self.compute_log_likelihood(
            time,
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_missing,
        )
        return super().estimate_parameters(
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
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
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

    @staticmethod
    def _get_group_spikes(spike_times, is_group, position_time):
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

    def fit_place_fields(
        self,
        position_time,
        position,
        spike_times,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
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
                **kwargs,
            )

    def fit(
        self,
        position_time,
        position,
        spike_times,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
    ):
        self._fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_place_fields(
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
        time,
        position_time,
        position,
        spike_times,
        is_missing=None,
    ):
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

        # missing data should be 1.0 because there is no information
        return jnp.where(is_missing[:, jnp.newaxis], 1.0, log_likelihood)

    def predict(
        self,
        spike_times,
        time,
        position=None,
        position_time=None,
        is_missing=None,
        discrete_transition_covariate_data=None,
        return_causal_posterior: bool = False,
    ):
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

        self.log_likelihood_ = self.compute_log_likelihood(
            time, position_time, position, spike_times, is_missing
        )
        (
            causal_posterior,
            acausal_posterior,
            _,
            _,
            acausal_state_probabilities,
            _,
            marginal_log_likelihood,
        ) = self._predict()

        return self._convert_results_to_xarray(
            time,
            causal_posterior,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            return_causal_posterior,
        )

    def estimate_parameters(
        self,
        position_time,
        position,
        spike_times,
        time,
        is_missing=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
        discrete_transition_covariate_data=None,
        estimate_inital_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        max_iter: int = 20,
        tolerance: float = 0.0001,
        return_causal_posterior: bool = False,
    ):
        position = position[:, np.newaxis] if position.ndim == 1 else position

        self.fit(
            position_time,
            position,
            spike_times,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
        )
        self.log_likelihood_ = self.compute_log_likelihood(
            time, position_time, position, spike_times, is_missing
        )
        return super().estimate_parameters(
            time,
            estimate_inital_conditions,
            estimate_discrete_transition,
            max_iter,
            tolerance,
            return_causal_posterior,
        )
