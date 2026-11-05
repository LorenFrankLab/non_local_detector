"""Build decoding results as xarray Datasets.

The detectors' ``predict`` and ``estimate_parameters`` return results built
here, from in-memory arrays (dense inference, ``convert_results_to_xarray``)
or from a checkpointed result store (``label_checkpointed_result``).
``non_local_detector.result_store`` writes and lazily reads that store.
``most_likely_sequence`` returns a DataFrame built by ``convert_seq_to_df``.

The layout of these results (dimensions, coordinates, data variables and
attributes, by model type, ``return_outputs`` option and inference mode) is
documented once, in ``docs/result_layout.md``;
``tests/models/test_result_schema.py`` checks its tables against the results
built here.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from non_local_detector._position_dims import get_position_dim_names
from non_local_detector.exceptions import ConfigurationError
from non_local_detector.result_store import pad_state_bins

try:
    from non_local_detector._version import __version__ as _PACKAGE_VERSION
except ImportError:
    _PACKAGE_VERSION = "unknown"

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase

# Rows of the full posterior are filled and scattered in blocks of about this
# many bytes. On a 2 cm 2-D grid (20,178 state bins, 3,298 interior; 177,330
# rows, 14.3 GB) filling with NaN and then scattering the whole array took
# 8.9 s; 1 MiB blocks took 3.8 s, 4-16 MiB 4.4 s and 64 MiB 5.7 s.
_MASKED_POSTERIOR_BLOCK_BYTES = 2**20


def state_bin_positions(detector: "_DetectorBase") -> list[np.ndarray]:
    """Position of each state's bins, state by state as in ``state_ind_``.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector.

    Returns
    -------
    positions : list of np.ndarray, each shape (n_bins, n_position_dims)
        One NaN row for a state without place bins (see
        ``StateSpace.has_place_bins``), otherwise its environment's place bin
        centers.

    Raises
    ------
    ValueError
        If a state's environment has no place bin centers.
    """
    n_position_dims = detector.environments_[0].place_bin_centers_.shape[1]
    state_space = detector.state_space_
    positions: list[np.ndarray] = []
    for state, obs in enumerate(state_space.observation_models):
        if not state_space.has_place_bins(state):
            positions.append(np.full((1, n_position_dims), np.nan))
            continue
        environment = detector._get_environment_by_name(obs.environment_name)
        positions.append(environment.place_bin_centers_)
    return positions


def unify_marginal_log_likelihoods(attrs: dict[str, Any]) -> None:
    """Hold results' marginal log likelihoods in their one form, in place.

    ``marginal_log_likelihoods`` becomes a one-dimensional float64 array. A
    checkpointed store records the final decode's under the singular
    ``marginal_log_likelihood`` (EM's stores the per-E-step list as well),
    and results saved by earlier versions held a decode's as a scalar, beside
    that singular alias for checkpointed results; the alias is removed.

    Parameters
    ----------
    attrs : dict
        The results' attributes.
    """
    final = attrs.pop("marginal_log_likelihood", None)
    values = attrs.get("marginal_log_likelihoods", final)
    if values is not None:
        attrs["marginal_log_likelihoods"] = np.atleast_1d(
            np.asarray(values, dtype=np.float64)
        )


def label_checkpointed_result(
    dataset: xr.Dataset, metadata: dict[str, Any]
) -> xr.Dataset:
    """Attach native labels and a bounded lazy padded view to a stored result."""
    attrs = dataset.attrs
    unify_marginal_log_likelihoods(attrs)
    # The JSON metadata of a store EM wrote holds its traces as lists.
    for attr_name in ("em_objectives", "em_previous_objectives"):
        if attr_name in attrs:
            attrs[attr_name] = np.asarray(attrs[attr_name], dtype=np.float64)
    if "time_bin_end_inclusive" not in dataset.coords:
        dataset = dataset.assign_coords(
            time_bin_end_inclusive=(
                "time",
                np.asarray(dataset.source_row)
                == int(dataset.attrs["recording_rows"]) - 1,
            )
        )
    states = np.asarray(metadata["state_names"])
    dataset = dataset.assign_coords(
        states=states,
        environments=("states", metadata["environment_names"]),
        encoding_groups=("states", metadata["encoding_groups"]),
    )
    if "state_bins" in dataset.dims:
        interior = np.asarray(metadata["interior_mask"], dtype=bool)
        variables = {}
        for name, variable in dataset.data_vars.items():
            variables[name] = (
                pad_state_bins(
                    variable, interior, max_read_bytes=metadata["max_read_bytes"]
                )
                if "state_bins" in variable.dims
                else variable.variable
            )
        coordinates = {
            name: coord
            for name, coord in dataset.coords.items()
            if "state_bins" not in coord.dims
        }
        positions = np.asarray(metadata["position"])
        state_ind = np.asarray(metadata["state_ind"], dtype=int)
        index = pd.MultiIndex.from_arrays(
            (states[state_ind], *positions.T),
            names=("state", *metadata["position_names"]),
        )
        coordinates.update(xr.Coordinates.from_pandas_multiindex(index, "state_bins"))
        coordinates["state_ind"] = ("state_bins", state_ind)
        dataset = xr.Dataset(variables, coords=coordinates, attrs=dataset.attrs)
    if len(states) == 1:
        dataset = dataset.squeeze("states")
    return dataset


def create_masked_posterior(
    data: np.ndarray, is_track_interior: np.ndarray, n_total_bins: int
) -> np.ndarray:
    """
    Create full posterior array with NaN for non-interior bins.

    Parameters
    ----------
    data : np.ndarray, shape (n_time, n_interior_bins)
        Posterior data for interior bins only.
    is_track_interior : np.ndarray, shape (n_total_bins,)
        Boolean mask indicating which bins are track interior.
    n_total_bins : int
        Total number of bins including non-interior.

    Returns
    -------
    full_array : np.ndarray, shape (n_time, n_total_bins)
        Full array with NaN for non-interior bins.
    """
    n_time = data.shape[0]
    columns = np.flatnonzero(is_track_interior)
    rows = max(1, _MASKED_POSTERIOR_BLOCK_BYTES // max(4 * n_total_bins, 1))
    full_array = np.empty((n_time, n_total_bins), dtype=np.float32)
    # Fill and scatter one cache-sized block at a time, so each output
    # byte is written to memory once instead of twice.
    for start in range(0, n_time, rows):
        block = full_array[start : start + rows]
        block.fill(np.nan)
        block[:, columns] = data[start : start + rows]
    return full_array


def convert_results_to_xarray(
    detector: "_DetectorBase",
    time: np.ndarray,
    acausal_posterior: np.ndarray,
    acausal_state_probabilities: np.ndarray,
    marginal_log_likelihoods: list[float] | float,
    log_likelihood: np.ndarray | None = None,
    causal_posterior: np.ndarray | None = None,
    causal_state_probabilities: np.ndarray | None = None,
    predictive_state_probabilities: np.ndarray | None = None,
    predictive_posterior: np.ndarray | None = None,
    *,
    time_edges: np.ndarray,
    is_missing: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Convert the results to an xarray Dataset.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector whose state space labels the results.
    time : np.ndarray, shape (n_time,)
        Decode bin centers, the ``time`` coordinate of the results.
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Acausal posterior probabilities.
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Acausal state probabilities.
    marginal_log_likelihoods : list of float or float
        Marginal log likelihoods for each iteration, or the one of a
        decode; stored as a one-dimensional float64 array.
    log_likelihood : np.ndarray, optional, shape (n_time, n_state_bins)
        Log likelihoods, by default None.
    causal_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
        Causal (filtered) posterior probabilities, by default None.
    causal_state_probabilities : np.ndarray, optional, shape (n_time, n_states)
        Causal state probabilities, by default None.
    predictive_state_probabilities : np.ndarray, optional, shape (n_time, n_states)
        One-step-ahead predicted state probabilities, by default None.
    predictive_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
        One-step-ahead predicted posterior probabilities over state bins, by default None.
    time_edges : np.ndarray, shape (n_time + 1,)
        Time bin edges; stored as the ``time_bin_start`` / ``time_bin_end`` /
        ``time_bin_width`` coordinates.
    is_missing : np.ndarray, shape (n_time,), optional
        Boolean mask of bins with missing data, stored as a coordinate. By
        default None, which marks no bin as missing.

    Returns
    -------
    xr.Dataset
        Decoding results in an xarray Dataset.
    """
    is_track_interior = detector.is_track_interior_state_bins_

    # Extract environment and encoding group names in single pass
    observation_models = detector.state_space_.observation_models
    environment_names = [obs.environment_name for obs in observation_models]
    encoding_group_names = [obs.encoding_group for obs in observation_models]

    # Get position dimensionality
    if not getattr(detector, "environments_", ()):
        msg = "No environments found in model"
        raise ConfigurationError(
            msg,
            hint="Environments are set up during fit(). Either fit the model first, or check that position data was provided",
        )
    position = np.concatenate(state_bin_positions(detector), axis=0)
    n_position_dims = position.shape[1]

    states = np.asarray(detector.state_space_.state_names)

    # Generate position dimension names (canonical convention)
    position_names = get_position_dim_names(n_position_dims)
    # Create MultiIndex for state_bins coordinate
    state_bins_mindex = pd.MultiIndex.from_arrays(
        ((states[detector.state_ind_], *position.T)),
        names=("state", *position_names),
    )

    # Returned provenance must survive reuse or mutation of caller inputs.
    bin_edges = np.array(time_edges, copy=True)
    saved_missing = (
        np.zeros(len(time), dtype=bool)
        if is_missing is None
        else np.array(is_missing, dtype=bool, copy=True)
    )
    coords = {
        "time": time,
        "time_bin_start": ("time", bin_edges[:-1]),
        "time_bin_end": ("time", bin_edges[1:]),
        "time_bin_width": ("time", np.diff(bin_edges)),
        "time_bin_end_inclusive": ("time", np.arange(len(time)) == len(time) - 1),
        "is_missing": (
            "time",
            saved_missing,
        ),
        "state_ind": ("state_bins", detector.state_ind_),
        "states": states,
        "environments": ("states", environment_names),
        "encoding_groups": ("states", encoding_group_names),
    }

    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        state_bins_mindex, "state_bins"
    )

    attrs = {
        "marginal_log_likelihoods": np.atleast_1d(
            np.asarray(marginal_log_likelihoods, dtype=np.float64)
        ),
        "time_bin_convention": "left_closed_right_open_final_closed",
        "non_local_detector_version": _PACKAGE_VERSION,
        "time_units": "seconds",
    }

    # Build data_vars dict with masked posteriors
    n_total_bins = len(is_track_interior)
    data_vars = {
        "acausal_posterior": (
            ("time", "state_bins"),
            create_masked_posterior(acausal_posterior, is_track_interior, n_total_bins),
        ),
        "acausal_state_probabilities": (
            ("time", "states"),
            acausal_state_probabilities,
        ),
    }

    if log_likelihood is not None:
        data_vars["log_likelihood"] = (
            ("time", "state_bins"),
            create_masked_posterior(log_likelihood, is_track_interior, n_total_bins),
        )

    if causal_posterior is not None:
        data_vars["causal_posterior"] = (
            ("time", "state_bins"),
            create_masked_posterior(causal_posterior, is_track_interior, n_total_bins),
        )

    if causal_state_probabilities is not None:
        data_vars["causal_state_probabilities"] = (
            ("time", "states"),
            causal_state_probabilities,
        )

    if predictive_state_probabilities is not None:
        data_vars["predictive_state_probabilities"] = (
            ("time", "states"),
            predictive_state_probabilities,
        )

    if predictive_posterior is not None:
        data_vars["predictive_posterior"] = (
            ("time", "state_bins"),
            create_masked_posterior(
                predictive_posterior, is_track_interior, n_total_bins
            ),
        )

    # Create Dataset with MultiIndex coordinates
    results = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    results = results.assign_coords(mindex_coords)

    return results.squeeze(
        dim=[
            dim
            for dim, size in results.sizes.items()
            if size == 1 and dim not in {"time", "state_bins"}
        ]
    )


def convert_seq_to_df(
    detector: "_DetectorBase", sequence_ind: np.ndarray, time: np.ndarray
) -> pd.DataFrame:
    """Convert the sequence indices to a DataFrame.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector whose state space labels the sequence.
    sequence_ind : np.ndarray, shape (n_time,)
        Most likely sequence indices.
    time : np.ndarray, shape (n_time,)
        Time of each sequence index.

    Returns
    -------
    results : pd.DataFrame, shape (n_time, n_cols)
    """
    position_blocks = state_bin_positions(detector)
    environment_names_blocks: list[Any] = []
    encoding_group_names_blocks: list[Any] = []
    state_space = detector.state_space_
    for state, (obs, block) in enumerate(
        zip(state_space.observation_models, position_blocks, strict=True)
    ):
        placed = state_space.has_place_bins(state)
        environment_names_blocks.append(
            [obs.environment_name if placed else None] * len(block)
        )
        encoding_group_names_blocks.append(
            [obs.encoding_group if placed else None] * len(block)
        )

    position = np.concatenate(position_blocks, axis=0)
    environment_names = np.concatenate(environment_names_blocks, axis=0)
    encoding_group_names = np.concatenate(encoding_group_names_blocks, axis=0)

    states = np.asarray(detector.state_space_.state_names)
    position_names = get_position_dim_names(position.shape[1])
    state_bins = pd.DataFrame(
        {
            "state": states[detector.state_ind_],
            **dict(zip(position_names, position.T, strict=False)),
            "environment": environment_names,
            "encoding_group_names": encoding_group_names,
        }
    ).iloc[detector.is_track_interior_state_bins_]

    return state_bins.iloc[sequence_ind].set_index(pd.Index(time, name="time"))
