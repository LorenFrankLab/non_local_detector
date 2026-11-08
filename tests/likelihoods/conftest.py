"""Shared helpers for the ``row_slice`` likelihood tests.

Import the helpers explicitly, as the root ``conftest`` is imported elsewhere:

    from tests.likelihoods.conftest import parity_kwargs
"""

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import non_local_detector.likelihoods.common as common
import non_local_detector.likelihoods.common.reductions as reductions
from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    fit_mark_references,
)
from non_local_detector.likelihoods.common import LOG_EPS
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
)
from tests.likelihoods._sorted_spikes_helpers import make_2d_env

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


class _PatchedKernels:
    """The patch, if any, that the JAX caches may hold kernels traced under.

    A patch that changes a trace-time decision reaches only kernels traced
    after it, and a kernel traced under it must never run once it is undone.
    """

    key: object = None


_SEGMENTED_SCAN = "segmented_scan"


def _retrace_under(key: object) -> None:
    """Clear the JAX caches unless they hold only kernels traced under ``key``."""
    if _PatchedKernels.key != key:
        jax.clear_caches()
        _PatchedKernels.key = key


def _shared_patch(item: pytest.Item) -> object:
    """The patch that consecutive tests may share: the segmented scan, or none."""
    callspec = getattr(item, "callspec", None)
    if (
        callspec is not None
        and callspec.params.get("reduction_path") == _SEGMENTED_SCAN
    ):
        return _SEGMENTED_SCAN
    return None


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Drop the patched kernels unless the next test reuses them as is.

    Kept only when the next test is also on the segmented scan, in the same
    module or class, and requests no fixture this test did not: then nothing
    runs unpatched in between (no shared fixture is built) to trace kernels on
    the sequential path or to run the segmented ones. Every other test, here or
    in any later module, starts without them. Kernels traced under a
    ``patch_and_retrace`` patch are always dropped.
    """
    if _PatchedKernels.key is None:
        return
    if (
        nextitem is not None
        and _shared_patch(nextitem) == _PatchedKernels.key
        and nextitem.parent is item.parent
        and set(nextitem.fixturenames) <= set(item.fixturenames)
    ):
        return
    jax.clear_caches()
    _PatchedKernels.key = None


@pytest.fixture
def patch_and_retrace(monkeypatch):
    """Patch a trace-time decision of the compiled kernels, then retrace them.

    Returns ``patch(target, name, value)``: ``monkeypatch.setattr`` followed by
    clearing the JAX caches, so no kernel traced before it bypasses the patch.
    The caches are cleared again after the test, so no kernel traced under the
    patch outlives it.
    """

    def patch(target: object, name: str, value: object) -> None:
        monkeypatch.setattr(target, name, value)
        _retrace_under(object())

    return patch


@pytest.fixture(params=["sequential", _SEGMENTED_SCAN])
def reduction_path(request, monkeypatch):
    """Run on CPU's sequential scatter and on the off-CPU segmented scan.

    Consecutive segmented-scan cases share their compiled kernels (see
    ``pytest_runtest_teardown``) instead of each clearing every JAX cache.
    """
    if request.param == _SEGMENTED_SCAN:
        monkeypatch.setattr(reductions, "_reduces_sequentially", lambda values: False)
        _retrace_under(_SEGMENTED_SCAN)
    return request.param


def tolerance(
    x64: bool, float32: dict[str, float] = FLOAT32_ROUNDING
) -> dict[str, float]:
    """Float64 references at 1e-10; float32 at ``float32`` (the bound above)."""
    return {"rtol": 1e-10, "atol": 1e-10} if x64 else float32


def spike_counts_by_edge_comparison(spike_times, edges, rows=None):
    """Spikes per decode bin, compared with each bin's edges directly.

    Bin ``i`` owns ``edges[i] <= t < edges[i + 1]``, and the final bin also owns
    ``t == edges[-1]``. Every bin is counted over the whole recording and
    ``rows`` slices the result, so the counts do not go through the package's
    row-range spike selection, which the likelihoods under test use.

    Parameters
    ----------
    spike_times : array_like, shape (n_spikes,)
    edges : np.ndarray, shape (n_bins + 1,)
    rows : slice, optional
        Bins to return; all bins by default.

    Returns
    -------
    counts : np.ndarray, shape (n_rows,)
    """
    times = np.asarray(spike_times)[:, None]
    edges = np.asarray(edges)
    owned = (times >= edges[:-1]) & (times < edges[1:])
    owned[:, -1] |= times[:, 0] == edges[-1]
    counts = owned.sum(axis=0)
    return counts if rows is None else counts[rows]


def log_duration_per_spike(spike_times, edges, rows=None, time_scale=1.0):
    """Each bin's ``sum over spikes of log(duration / time_scale)``.

    The ``N_bin * log(dt)`` term of a marked point process, from
    ``spike_counts_by_edge_comparison`` rather than the package's own helper.

    Parameters
    ----------
    spike_times : list of array_like, each shape (n_spikes,)
        One spike train per electrode.
    edges : np.ndarray, shape (n_bins + 1,)
    rows : slice, optional
        Bins to return; all bins by default.
    time_scale : float, optional
        Time unit of the caller's intensities, in seconds.

    Returns
    -------
    log_duration : np.ndarray, shape (n_rows,)
    """
    counts = sum(spike_counts_by_edge_comparison(times, edges) for times in spike_times)
    log_duration = counts * np.log(np.diff(edges) / time_scale)
    return log_duration if rows is None else log_duration[rows]


@pytest.fixture(scope="module")
def env_2d():
    """The 2D open field, built once per module.

    Fits cache eigenbases on the environment, so tests that count eigensolves or
    inspect those caches build their own environment instead.
    """
    return make_2d_env()


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

    for name, record in _SORTED_SPIKES_ALGORITHMS.items():
        encoding_model = record.fit(
            position_time=data["position_time"],
            position=data["position"],
            spike_times=data["encoding_spike_times"],
            environment=environment,
            **fit_params.get(name, {}),
        )
        yield name, record.predict, encoding_model, False

    for name, record in _CLUSTERLESS_ALGORITHMS.items():
        encoding_model = record.fit(
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
            record.mark_reference_scale(**fit_params.get(name, {})),
            fit_params.get(name, {}).get("weights"),
            data["position"],
            encoding_time_range=fit_params.get(name, {}).get("encoding_time_range"),
            valid_position_intervals=fit_params.get(name, {}).get(
                "valid_position_intervals"
            ),
        )
        yield name, record.predict, encoding_model, True


def realistic_log_background(
    decoding_features: np.ndarray | jax.Array, log_rate: float = LOG_EPS
) -> jax.Array:
    """Deterministic per-spike No-Spike background for reduction parity tests.

    A mark reference centred on the decoding marks' mean, with scale
    ``std + 1`` so constant features stay finite, gives each spike a
    background of the size a fitted reference would. Parity tests pass the
    same background to every path they compare.

    Parameters
    ----------
    decoding_features : np.ndarray or jax.Array, shape (n_decoding_spikes, n_features)
    log_rate : float, optional
        Log background rate in the caller's intensity units: ``LOG_EPS`` for
        the KDE backends, ``LOG_RATE_EPS_HZ`` for GMM.

    Returns
    -------
    log_background : jax.Array, shape (n_decoding_spikes,)
    """
    features = np.asarray(decoding_features, dtype=np.float64)
    if features.shape[0] == 0:
        return jnp.zeros(0)
    reference = MarkReference(features.mean(axis=0), features.std(axis=0) + 1.0)
    return jnp.asarray(background_log_intensity(reference, features, log_rate))


@pytest.fixture(scope="session")
def line_backends() -> tuple[dict, dict[str, tuple[Callable, dict, bool]]]:
    """Every registered backend fit once on a two-unit 0-100 cm line.

    The animal oscillates over 5-95 cm for 20 s at 100 Hz on ten 10 cm bins;
    unit 0 fires 300 spikes within 10 cm of 25 cm and unit 1 within 10 cm of
    75 cm, each 1 ms after a position sample. Tests must not mutate the
    fitted models.

    Returns
    -------
    encoding_data : dict
        The ``fit_registered_backends`` input.
    fitted : dict[str, tuple[Callable, dict, bool]]
        ``(predict_func, encoding_model, is_clusterless)`` per backend name.
    """
    from non_local_detector.environment import Environment

    rng = np.random.default_rng(3)
    position_time = np.arange(0.0, 20.0, 0.01)
    position = (50.0 + 45.0 * np.sin(2 * np.pi * position_time / 5.0))[:, None]
    environment = Environment(
        environment_name="line", place_bin_size=10.0, position_range=((0.0, 100.0),)
    ).fit_place_grid(position=position, infer_track_interior=False)
    spike_times, features = [], []
    for center in (25.0, 75.0):
        near = np.abs(position[:, 0] - center) < 10.0
        times = np.sort(rng.choice(position_time[near], 300, replace=False))
        spike_times.append(times + 0.001)
        features.append(rng.standard_normal((300, 2)) * 3.0 + center / 5.0)
    encoding_data = {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": spike_times,
        "encoding_features": features,
    }
    fitted = {
        name: (predict, model, is_clusterless)
        for name, predict, model, is_clusterless in fit_registered_backends(
            encoding_data
        )
    }
    return encoding_data, fitted
