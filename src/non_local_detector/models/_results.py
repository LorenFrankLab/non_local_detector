"""Build decoding results as xarray Datasets.

The detectors' ``predict`` and ``estimate_parameters`` return results built
here, from in-memory arrays (dense inference, ``convert_results_to_xarray``)
or from a checkpointed result store (``label_checkpointed_result``).
``non_local_detector.result_store`` writes and lazily reads that store.
``most_likely_sequence`` returns a DataFrame built by ``convert_seq_to_df``.

Result layout (user docs: ``docs/result_layout.md``; keep the two in step,
``tests/models/test_result_schema.py`` checks both name every field)

Dimensions
    ``time``: returned decode bins. ``state_bins``: every (state, position
    bin) pair, absent with checkpointed ``output_mode="compact"``.
    ``states``: discrete states, squeezed away for single-state models
    (decoders), whose ``(time, states)`` variables are then ``(time,)`` and
    whose ``states``, ``environments`` and ``encoding_groups`` are scalars.

Coordinates on ``time``
    ``time`` (bin centers, s), ``time_bin_start``, ``time_bin_end``,
    ``time_bin_width`` (end - start), ``time_bin_end_inclusive`` (True only for
    the final bin of the decode grid) and ``is_missing``. Checkpointed results
    also carry ``source_row``, each row's index in the full decode grid.

Coordinates on ``states``
    ``states`` (names), ``environments`` and ``encoding_groups``.

Coordinates on ``state_bins``
    ``state_bins``, a MultiIndex with levels ``state`` and the position
    dimensions (``position``; ``x_position``, ``y_position``; ...); states with
    no position (No-Spike, Local without ``local_position_std``) have one bin
    with NaN position. ``state_ind``, each bin's index into ``states``
    (``detector.state_ind_``).

Data variables (``state_bins`` variables are float32, NaN off the track interior)
    ``acausal_posterior`` (time, state_bins) and
    ``acausal_state_probabilities`` (time, states) always, except that compact
    results have no ``state_bins`` variables. ``return_outputs`` adds
    ``causal_state_probabilities`` and ``causal_posterior`` ("filter"),
    ``predictive_state_probabilities`` ("predictive") and
    ``predictive_posterior`` ("predictive" or "predictive_posterior"), and
    ``log_likelihood`` (time, state_bins) ("log_likelihood"); "all" adds every
    one. In a list, ``"predictive"`` adds only the state probabilities.

Attributes
    Always ``marginal_log_likelihoods`` (array), ``time_bin_convention``,
    ``time_units`` and ``non_local_detector_version``. ``estimate_parameters``
    adds ``em_transition_representation``, ``em_objectives``,
    ``em_previous_objectives`` and ``em_n_chunks`` (dense EM) or
    ``em_chunk_size`` and ``em_replay_cache_bytes`` (structured EM).
    Checkpointed results add ``marginal_log_likelihood``, ``inference_mode``,
    ``output_mode``, ``conditioning``, ``whole_recording_conditioning``,
    ``recording_rows``, ``n_recording_bins``, ``transition_representation``,
    ``dtype``, ``state_probability_dtype``, ``evidence_accumulation``,
    ``evidence_dtype``, the diagnostics ``n_degenerate``, ``n_nan``,
    ``degenerate_row_mask_hex``, ``nan_row_mask_hex``,
    ``diagnostic_row_mask_encoding``, ``diagnostics_are_global``,
    ``max_chunk_rows``, ``likelihood_evaluations``, ``checkpoint_bytes``,
    ``checkpoint_cache_entries``, ``replay_cache_bytes`` and
    ``global_gradients_supported``, and the internal ``_native_metadata``
    (labels for reopening the store; ``save_results`` drops it).
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


def label_checkpointed_result(
    dataset: xr.Dataset, metadata: dict[str, Any]
) -> xr.Dataset:
    """Attach native labels and a bounded lazy padded view to a stored result."""
    attrs = dataset.attrs
    # The JSON metadata of a store EM wrote holds its traces as lists.
    for attr_name in (
        "marginal_log_likelihoods",
        "em_objectives",
        "em_previous_objectives",
    ):
        if attr_name in attrs:
            attrs[attr_name] = np.asarray(attrs[attr_name], dtype=float)
    attrs.setdefault("marginal_log_likelihoods", attrs["marginal_log_likelihood"])
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
        Marginal log likelihoods for each iteration.
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

    Returns
    -------
    xr.Dataset
        Decoding results in an xarray Dataset.
    """
    is_track_interior = detector.is_track_interior_state_bins_

    # Extract environment and encoding group names in single pass
    environment_names = [obs.environment_name for obs in detector.observation_models]
    encoding_group_names = [obs.encoding_group for obs in detector.observation_models]

    # Create environment lookup dict for O(1) access
    env_dict = {env.environment_name: env for env in detector.environments}

    # Get position dimensionality
    if not detector.environments:
        msg = "No environments found in model"
        raise ConfigurationError(
            msg,
            hint="Environments are set up during fit(). Either fit the model first, or check that position data was provided",
        )
    n_position_dims = detector.environments[0].place_bin_centers_.shape[1]  # type: ignore[union-attr]

    # Build position array
    position_blocks: list[np.ndarray] = []
    for obs in detector.observation_models:
        if obs.is_no_spike or (obs.is_local and detector.local_position_std is None):
            position_blocks.append(np.full((1, n_position_dims), np.nan))
        else:
            environment = env_dict.get(obs.environment_name)
            if environment is None:
                msg = f"Environment '{obs.environment_name}' not found in environments list"
                raise ValueError(msg)
            if environment.place_bin_centers_ is None:
                msg = (
                    f"place_bin_centers_ is None for environment {obs.environment_name}"
                )
                raise ValueError(msg)
            position_blocks.append(environment.place_bin_centers_)
    position = np.concatenate(position_blocks, axis=0)

    states = np.asarray(detector.state_names)

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
        "marginal_log_likelihoods": np.asarray(marginal_log_likelihoods),
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
    """Converts the sequence indices to a DataFrame.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector whose state space labels the sequence.
    sequence_ind : np.ndarray, shape (n_time,)
        Most likely sequence indices.
    time : np.ndarray, shape (n_time,)

    Returns
    -------
    results : pd.DataFrame, shape (n_time, n_cols)
    """
    position_blocks: list[np.ndarray] = []
    n_position_dims = detector.environments[0].place_bin_centers_.shape[1]  # type: ignore[union-attr]
    environment_names_blocks: list[Any] = []
    encoding_group_names_blocks: list[Any] = []
    for obs in detector.observation_models:
        if obs.is_no_spike or (obs.is_local and detector.local_position_std is None):
            position_blocks.append(np.full((1, n_position_dims), np.nan))
            environment_names_blocks.append([None])
            encoding_group_names_blocks.append([None])
        else:
            environment = detector._get_environment_by_name(obs.environment_name)
            if environment.place_bin_centers_ is None:
                msg = (
                    f"place_bin_centers_ is None for environment {obs.environment_name}"
                )
                raise ValueError(msg)
            position_blocks.append(environment.place_bin_centers_)
            environment_names_blocks.append(
                [obs.environment_name] * environment.place_bin_centers_.shape[0]
            )
            encoding_group_names_blocks.append(
                [obs.encoding_group] * environment.place_bin_centers_.shape[0]
            )

    position = np.concatenate(position_blocks, axis=0)
    environment_names = np.concatenate(environment_names_blocks, axis=0)
    encoding_group_names = np.concatenate(encoding_group_names_blocks, axis=0)

    states = np.asarray(detector.state_names)
    position_names = get_position_dim_names(n_position_dims)
    state_bins = pd.DataFrame(
        {
            "state": states[detector.state_ind_],
            **dict(zip(position_names, position.T, strict=False)),
            "environment": environment_names,
            "encoding_group_names": encoding_group_names,
        }
    ).iloc[detector.is_track_interior_state_bins_]

    return state_bins.iloc[sequence_ind].set_index(pd.Index(time, name="time"))
