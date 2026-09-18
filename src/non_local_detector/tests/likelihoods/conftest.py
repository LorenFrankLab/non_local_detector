"""Shared helpers for the ``row_slice`` likelihood tests.

Import the helpers explicitly, as the root ``conftest`` is imported elsewhere:

    from non_local_detector.tests.likelihoods.conftest import parity_kwargs
"""

from collections.abc import Callable, Iterator

from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)

ALGORITHMS = sorted(_SORTED_SPIKES_ALGORITHMS) + sorted(_CLUSTERLESS_ALGORITHMS)

# Tolerances by reduction kind.
#
# The sorted-spikes backends and the no-spike model accumulate integer spike
# COUNTS per row and only then do float work, so a row range is bit-identical to
# the corresponding full-time rows.
#
# The clusterless backends scatter-add one float32 row per selected spike into
# the output rows (``jax.ops.segment_sum`` / ``.at[].add``). A row range hands
# XLA a different number of input rows and a different ``num_segments``, so it
# may group that reduction differently; the same happens when the decoding
# spikes arrive in a different order. ``clusterless_kde_log`` additionally
# re-blocks its stabilized log-sum over the decoding spikes. Those regroupings
# are float32 rounding, not a different answer: measured at one to two spikes
# per row, ``clusterless_kde`` moves by 1.5e-5 abs, ``clusterless_gmm`` by 2.0
# abs on values of order 2e7, ``clusterless_diffusion`` by 3.8e-6 abs and
# ``clusterless_kde_log`` by 7.6e-6 abs -- at most 1.2e-7 RELATIVE in every
# case. The tolerance therefore follows the reduction kind rather than any one
# fixture's spike density; a real regression (a lost or double-counted spike)
# moves a log likelihood by O(1) and is caught either way.
EXACT: dict[str, float] = {"rtol": 0.0, "atol": 0.0}
FLOAT32_REDUCTION: dict[str, float] = {"rtol": 1e-6, "atol": 1e-5}


def parity_kwargs(algorithm: str) -> dict[str, float]:
    """Tolerance for one backend: exact counts, or float32 reduction rounding."""
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        return EXACT
    return FLOAT32_REDUCTION


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
        yield name, predict_func, encoding_model, True
