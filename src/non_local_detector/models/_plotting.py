"""Plotting of a detector's fitted transitions and initial conditions.

``_DetectorBase`` inherits the methods of ``_PlottingMixin``: heatmaps of
the discrete and continuous state transitions and of the initial
conditions.

Methods that read the detector's attributes annotate ``self`` as
``_DetectorBase``, which mypy only accepts with ``type: ignore[misc]`` on a
mixin.
"""

from itertools import pairwise
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices

from non_local_detector.discrete_state_transitions import centered_softmax_forward

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class _PlottingMixin:
    """Plotting methods of ``_DetectorBase``."""

    def plot_discrete_state_transition(  # type: ignore[misc]
        self: "_DetectorBase",
        cmap: str = "Oranges",
        ax: mpl.axes.Axes | None = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
        covariate_data: pd.DataFrame | dict[str, Any] | None = None,
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
        covariate_data : dict or pd.DataFrame, optional
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
                vmin, vmax = 0.0, None
                label = "Seconds"
            else:
                discrete_state_transition = self.discrete_state_transitions_
                vmin, vmax = 0.0, 1.0
                label = "Probability"

            n_states = len(self.state_names)
            mesh = ax.pcolormesh(
                discrete_state_transition,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            # Row 0 at the top and tick labels centered on the cells.
            ax.set_xlim(0, n_states)
            ax.set_ylim(n_states, 0)
            ticks = np.arange(n_states) + 0.5
            ax.set_xticks(ticks, labels=self.state_names)
            ax.set_yticks(ticks, labels=self.state_names)
            cbar = ax.figure.colorbar(mesh, ax=ax, label=label)
            plt.setp(cbar.outline, visible=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Rotate tick labels that would overlap (as seaborn's heatmap did).
            ax.figure.canvas.draw()
            for axis in (ax.xaxis, ax.yaxis):
                boxes = [label.get_window_extent() for label in axis.get_ticklabels()]
                if any(a.overlaps(b) for a, b in pairwise(boxes)):
                    plt.setp(axis.get_ticklabels(), rotation="vertical")
            # Dark text on light cells, white text on dark cells.
            for (row, col), value, rgba in zip(
                np.ndindex(discrete_state_transition.shape),
                discrete_state_transition.ravel(),
                mesh.to_rgba(discrete_state_transition.ravel()),
                strict=True,
            ):
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    format(value, ".03f"),
                    ha="center",
                    va="center",
                    color=".15" if luminance > 0.408 else "w",
                )
            ax.set_ylabel("Previous State", fontsize=12)
            ax.set_xlabel("Current State", fontsize=12)
            ax.set_title("Discrete State Transition", fontsize=16)
        else:
            discrete_transition_design_matrix = self.discrete_transition_design_matrix_
            discrete_transition_coefficients = self.discrete_transition_coefficients_
            state_names = self.state_names

            if discrete_transition_design_matrix is None:
                msg = "discrete_transition_design_matrix_ is None for covariate-dependent prediction"
                raise ValueError(msg)

            predict_matrix = build_design_matrices(
                [discrete_transition_design_matrix.design_info],
                covariate_data,
            )[0]

            n_states = len(state_names)

            # The coefficients are set together with the design matrix.
            assert discrete_transition_coefficients is not None

            for covariate in covariate_data:  # type: ignore[union-attr]
                fig, axes = plt.subplots(
                    1, n_states, sharex=True, constrained_layout=True, figsize=(10, 5)
                )

                for from_state_ind, (ax, from_state) in enumerate(
                    zip(axes.flat, state_names, strict=False)
                ):
                    from_local_transition = centered_softmax_forward(
                        predict_matrix
                        @ discrete_transition_coefficients[:, from_state_ind]
                    )

                    ax.plot(covariate_data[covariate], from_local_transition)  # type: ignore[index]  # covariate_data is required on this path
                    ax.set_xlabel(covariate)
                    ax.set_ylabel("Prob.")
                    if from_state_ind == n_states - 1:
                        ax.legend(state_names)
                    ax.set_title(f"From {from_state}")
                fig.suptitle(f"Predicted transitions: {covariate}")

    def plot_continuous_state_transition(  # type: ignore[misc]
        self: "_DetectorBase", figsize_scaling: float = 1.5, vmax: float = 0.3
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

        if self.n_discrete_states_ == 1:
            # A single state has one panel, which needs room for its tick labels.
            _, ax = plt.subplots(
                constrained_layout=True,
                figsize=(4.0 * figsize_scaling, 3.5 * figsize_scaling),
            )
            ax.pcolormesh(self.continuous_state_transitions_, vmin=0.0, vmax=vmax)
            ax.set_xlabel("To position bin")
            ax.set_ylabel("From position bin")
            ax.set_title(self.state_names[0])
            return

        fig, axes = plt.subplots(
            self.n_discrete_states_,
            self.n_discrete_states_,
            gridspec_kw={
                "width_ratios": self.bin_sizes_,
                "height_ratios": self.bin_sizes_,
            },
            constrained_layout=True,
            figsize=(
                self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO,
                (self.n_discrete_states_ * figsize_scaling),
            ),
        )

        for from_state, ax_row in enumerate(axes):
            for to_state, ax in enumerate(ax_row):
                ind = np.ix_(self.state_ind_ == from_state, self.state_ind_ == to_state)
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

    def plot_initial_conditions(  # type: ignore[misc]
        self: "_DetectorBase", figsize_scaling: float = 1.5, vmax: float = 0.3
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
            gridspec_kw={"width_ratios": self.bin_sizes_},
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
