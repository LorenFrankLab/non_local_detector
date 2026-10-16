"""Shared helpers for the ``row_slice`` likelihood tests.

Import the helpers explicitly, as the root ``conftest`` is imported elsewhere:

    from non_local_detector.tests.likelihoods.conftest import parity_kwargs
"""

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import non_local_detector.likelihoods.common as common
from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_MARK_REFERENCE_SCALES,
    _SORTED_SPIKES_ALGORITHMS,
    fit_mark_references,
)
from non_local_detector.likelihoods.common import LOG_EPS
from non_local_detector.likelihoods.mark_reference import MarkReference

ALGORITHMS = sorted(_SORTED_SPIKES_ALGORITHMS) + sorted(_CLUSTERLESS_ALGORITHMS)

# Tolerances by prediction path.
#
# The sorted-spikes backends and the no-spike model accumulate integer spike
# COUNTS per row. Most then use fixed fitted rate maps, so a row range is
# bit-identical to the corresponding full-time rows. The local GLM instead
# evaluates exp(spline_matrix @ coefficients) for each requested row range;
# changing the matrix shape can change float32 rounding on some CPU backends.
# Linux x86 CI measured 7.63e-6 absolute / 2.41e-7 relative likelihood differences,
# including a final row with no spikes. Its counts must still match exactly.
#
# The clusterless backends scatter-add one float32 row per selected spike into
# the output rows (``jax.ops.segment_sum`` / ``.at[].add``). A row range hands
# XLA a different number of input rows and a different ``num_segments``, so it
# may group that reduction differently; the same happens when the decoding
# spikes arrive in a different order. ``clusterless_kde`` additionally
# re-blocks its stabilized log-sum over the decoding spikes. Those regroupings
# are float32 rounding, not a different answer: measured at one to two spikes
# per row, ``clusterless_kde`` moves by 7.6e-6 abs, ``clusterless_gmm`` by 2.0
# abs on values of order 2e7 and ``clusterless_diffusion`` by 3.8e-6 abs -- at
# most 1.2e-7 RELATIVE in every case. The tolerance therefore follows the reduction kind rather than any one
# fixture's spike density; a real regression (a lost or double-counted spike)
# moves a log likelihood by O(1) and is caught either way.
EXACT: dict[str, float] = {"rtol": 0.0, "atol": 0.0}
FLOAT32_ROUNDING: dict[str, float] = {"rtol": 1e-6, "atol": 1e-5}


@pytest.fixture(params=["sequential", "segmented_scan"])
def reduction_path(request, monkeypatch):
    """Run on CPU's sequential scatter and on the off-CPU segmented scan."""
    if request.param == "segmented_scan":
        monkeypatch.setattr(common, "_reduces_sequentially", lambda values: False)
        jax.clear_caches()
        yield request.param
        jax.clear_caches()
    else:
        yield request.param


def tolerance(x64: bool) -> dict[str, float]:
    """Float64 references at 1e-10; float32 at the rounding bound above."""
    return {"rtol": 1e-10, "atol": 1e-10} if x64 else FLOAT32_ROUNDING


def parity_kwargs(algorithm: str, *, is_local: bool = False) -> dict[str, float]:
    """Likelihood parity: fixed-rate sorted paths are exact; others may round."""
    if algorithm in _SORTED_SPIKES_ALGORITHMS and not (
        algorithm == "sorted_spikes_glm" and is_local
    ):
        return EXACT
    return FLOAT32_ROUNDING


def training_mark_references(
    position_time,
    spike_times,
    spike_waveform_features,
    kernel_scale,
    weights=None,
    position=None,
    *,
    encoding_time_range=None,
    valid_position_intervals=None,
):
    """Mark references of an encoding fit's training spikes, as a detector fits them.

    Each spike is weighted by ``weights`` (per position sample; uniform by
    default) interpolated at its time over the fit's encoding support, so
    spikes outside the support carry zero weight. Without ``position`` the
    support is that of ``position_time`` with every position finite.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
    spike_times : list of array_like, each shape (n_spikes,)
    spike_waveform_features : list of array_like, each shape (n_spikes, n_features)
    kernel_scale : float or array_like, shape (n_features,)
        The backend's ``mark_reference_scale`` of its fit parameters.
    weights : array_like, shape (n_time_position,), optional
    position : array_like, shape (n_time_position, n_position_dims), optional
    encoding_time_range, valid_position_intervals : optional
        The fit's encoding-support arguments.

    Returns
    -------
    mark_reference : list of MarkReference or None
    """
    position_time = np.asarray(position_time)
    if position is None:
        position = np.zeros((position_time.shape[0], 1))
    support, sample_weights, _, _ = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
    )
    spike_weights = [
        common.interpolate_weights_at_spike_times(
            np.asarray(times), position_time, sample_weights, encoding_support=support
        )
        for times in spike_times
    ]
    return fit_mark_references(
        [np.asarray(features) for features in spike_waveform_features],
        kernel_scale,
        spike_weights,
    )


def fit_registered_backends(
    data: dict, fit_params: dict[str, dict] | None = None
) -> Iterator[tuple[str, Callable, dict, bool]]:
    """Fit every registered backend on one encoding data set.

    Parameters
    ----------
    data : dict
        Carries ``position_time``, ``position``, ``environment``,
        ``encoding_spike_times`` and ``encoding_features``.
    fit_params : dict[str, dict] | None, optional
        Extra fit keyword arguments per backend name.

    Yields
    ------
    name : str
    predict_func : Callable
    encoding_model : dict
        A clusterless model also carries ``mark_reference``, fitted from the
        encoding features, so it can be unpacked into ``predict_func``.
    is_clusterless : bool
        Whether ``predict_func`` also takes waveform features.
    """
    fit_params = fit_params or {}
    environment = data["environment"]

    for name, (fit_func, predict_func) in _SORTED_SPIKES_ALGORITHMS.items():
        geometry = {}
        if name == "sorted_spikes_glm":
            geometry = {
                "place_bin_edges": environment.place_bin_edges_,
                "edges": environment.edges_,
                "is_track_interior": environment.is_track_interior_,
                "is_track_boundary": environment.is_track_boundary_,
            }
        encoding_model = fit_func(
            position_time=data["position_time"],
            position=data["position"],
            spike_times=data["encoding_spike_times"],
            environment=environment,
            **geometry,
            **fit_params.get(name, {}),
        )
        yield name, predict_func, encoding_model, False

    for name, (fit_func, predict_func) in _CLUSTERLESS_ALGORITHMS.items():
        encoding_model = fit_func(
            position_time=data["position_time"],
            position=data["position"],
            spike_times=data["encoding_spike_times"],
            spike_waveform_features=data["encoding_features"],
            environment=environment,
            **fit_params.get(name, {}),
        )
        encoding_model["mark_reference"] = training_mark_references(
            data["position_time"],
            data["encoding_spike_times"],
            data["encoding_features"],
            _CLUSTERLESS_MARK_REFERENCE_SCALES[name](**fit_params.get(name, {})),
            fit_params.get(name, {}).get("weights"),
            data["position"],
            encoding_time_range=fit_params.get(name, {}).get("encoding_time_range"),
            valid_position_intervals=fit_params.get(name, {}).get(
                "valid_position_intervals"
            ),
        )
        yield name, predict_func, encoding_model, True


def realistic_log_background(
    decoding_features: np.ndarray | jnp.ndarray, log_rate: float = LOG_EPS
) -> jnp.ndarray:
    """Deterministic per-spike No-Spike background for reduction parity tests.

    A diagonal-Gaussian mark reference centred on the decoding marks' mean,
    with scale ``std + 1`` so constant features stay finite, gives each spike a
    background of the size a fitted reference would. Parity tests pass the
    same background to every path they compare.

    Parameters
    ----------
    decoding_features : np.ndarray or jnp.ndarray, shape (n_decoding_spikes, n_features)
    log_rate : float, optional
        Log background rate in the caller's intensity units: ``LOG_EPS`` for
        the KDE backends, ``LOG_RATE_EPS_HZ`` for GMM.

    Returns
    -------
    log_background : jnp.ndarray, shape (n_decoding_spikes,)
    """
    features = np.asarray(decoding_features, dtype=np.float64)
    if features.shape[0] == 0:
        return jnp.zeros(0)
    reference = MarkReference(features.mean(axis=0), features.std(axis=0) + 1.0)
    return jnp.asarray(log_rate + reference.log_density(features))
