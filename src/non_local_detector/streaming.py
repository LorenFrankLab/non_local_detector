"""Memory-aware streaming helpers for HMM predict.

The detector's ``predict()`` materialises an ``(n_time, n_state_bins)``
log-likelihood array by default.  For chronic recordings (1 hour+) with
large state spaces (multiple discrete states × many position bins),
that array can exceed GPU memory — tens to hundreds of GB.

When chunked (``n_chunks > 1``), the filter consumes each chunk's
likelihood slab immediately and discards it, so peak memory scales
with ``chunk_size × n_state_bins`` instead of ``n_time × n_state_bins``.

:func:`_resolve_n_chunks` picks a chunk count that keeps each
chunk's likelihood slab under a fraction of device memory, letting
users pass ``n_chunks="auto"`` and not think about it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

_DEFAULT_SAFETY_FRACTION = 0.40


def _query_device_memory_bytes() -> int | None:
    """Return ``bytes_limit`` from the active JAX device, or ``None`` on CPU.

    Imports JAX lazily so module import stays cheap.
    """
    try:
        import jax
    except ImportError:
        return None
    try:
        device = jax.devices()[0]
        stats = device.memory_stats() or {}
        limit = int(stats.get("bytes_limit", 0))
        return limit or None
    except (AttributeError, NotImplementedError, RuntimeError, IndexError):
        return None


def _resolve_n_chunks(
    n_chunks: int | Literal["auto"],
    *,
    n_time: int,
    n_state_bins: int,
    memory_budget_bytes: int | None = None,
    dtype_bytes: int = 4,
    safety_fraction: float = _DEFAULT_SAFETY_FRACTION,
) -> int:
    """Resolve ``n_chunks`` to a concrete positive int.

    When ``n_chunks`` is an int, pass through unchanged (validates
    positivity).  When ``n_chunks == "auto"``, pick the smallest
    ``n_chunks`` such that each chunk's likelihood slab
    ``(ceil(n_time / n_chunks), n_state_bins)`` fits in
    ``safety_fraction * memory_budget_bytes``.

    Parameters
    ----------
    n_chunks
        ``"auto"`` (memory-aware) or an explicit positive int.
    n_time
        Number of decoding time bins.
    n_state_bins
        Number of state bins the likelihood is computed over
        (typically ``is_track_interior_state_bins_.sum()``).
    memory_budget_bytes, optional
        Override the device-memory query.  When ``None`` and
        ``n_chunks == "auto"``, queries
        ``jax.devices()[0].memory_stats()["bytes_limit"]``; if the
        query fails (CPU, platform without stats), falls back to
        ``n_chunks=1`` (no chunking).
    dtype_bytes
        Bytes per likelihood entry.  Defaults to 4 (fp32).
    safety_fraction
        Fraction of the budget the per-chunk slab is allowed to
        occupy.  Default 0.40 leaves room for the filter state, the
        transition matrix, the encoding model, and runtime scratch.

    Returns
    -------
    int
        Concrete ``n_chunks >= 1``.

    Raises
    ------
    ValueError
        If ``n_chunks`` isn't a positive int or ``"auto"``, or if any
        size argument is non-positive.
    """
    if n_time <= 0:
        raise ValueError(f"n_time must be positive; got {n_time}")
    if n_state_bins <= 0:
        raise ValueError(f"n_state_bins must be positive; got {n_state_bins}")

    # Reject bool before the int check — bool is an int subclass in Python
    # and would otherwise be silently coerced to 0/1.
    if isinstance(n_chunks, bool) or not isinstance(n_chunks, int | str):
        raise ValueError(
            f"n_chunks must be a positive int or the string 'auto'; "
            f"got {type(n_chunks).__name__} {n_chunks!r}"
        )
    if isinstance(n_chunks, int):
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be >= 1 when an int; got {n_chunks}")
        return n_chunks
    if n_chunks != "auto":
        raise ValueError(
            f"n_chunks string must be 'auto' (got {n_chunks!r}); "
            f"pass an int for a fixed chunk count."
        )

    # --- 'auto' resolution ---
    if memory_budget_bytes is None:
        memory_budget_bytes = _query_device_memory_bytes()
    if memory_budget_bytes is None:
        # No device info available (e.g. CPU).  Safest fallback is no
        # chunking — the user gets the same behavior as today.
        return 1
    if memory_budget_bytes <= 0:
        raise ValueError(
            f"memory_budget_bytes must be positive; got {memory_budget_bytes}"
        )

    # Per-chunk byte budget after leaving headroom.
    effective_budget = int(memory_budget_bytes * safety_fraction)
    # Bytes per time bin in the likelihood slab.
    per_time_bytes = n_state_bins * dtype_bytes
    # Largest chunk size that fits; at least 1 to guarantee forward progress.
    chunk_size = max(1, effective_budget // per_time_bytes)
    n_chunks_auto = max(1, math.ceil(n_time / chunk_size))
    return int(n_chunks_auto)


# ---------------------------------------------------------------------------
# Multi-knob memory-knob selectors (Task 5)
# ---------------------------------------------------------------------------

_MULTI_KNOB_SAFETY_FRACTION = 0.90


def _chunk_ladder(n_time: int) -> list[int]:
    """Chunk counts to try, smallest-first.  Bounded by ``n_time``."""
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return [c for c in candidates if c <= max(1, n_time)]


def _block_ladder() -> list[int]:
    """Decoding-spike block sizes to try, largest-first.

    Smaller than 100 is launch-overhead-bound on GPU and gains little
    compared to the 2× memory cut; stop at 100.
    """
    return [10_000, 5_000, 2_000, 1_000, 500, 200, 100]


def _enc_tile_ladder(n_encoding_spikes_max: int) -> list[int]:
    """Encoding-spike tile sizes, largest-first.

    The top of the ladder is ``n_encoding_spikes_max`` rounded up to a
    power of 2; descend from there.  Anything smaller than 256 is
    pathological; stop there.
    """
    if n_encoding_spikes_max <= 0:
        return []
    top = 1 << (n_encoding_spikes_max - 1).bit_length()  # round up to pow2
    candidates = []
    size = top
    while size >= 256:
        if size <= n_encoding_spikes_max:
            candidates.append(size)
        size //= 2
    return candidates or [min(n_encoding_spikes_max, 256)]


def _pos_tile_ladder(n_pos: int) -> list[int]:
    """Position tile sizes, largest-first.  Stop at 64 (small-tile
    overhead)."""
    if n_pos <= 0:
        return []
    top = 1 << (n_pos - 1).bit_length()
    candidates = []
    size = top
    while size >= 64:
        if size <= n_pos:
            candidates.append(size)
        size //= 2
    return candidates or [min(n_pos, 64)]


def _fit_block_ladder() -> list[int]:
    """Fit-time KDE block sizes, largest-first."""
    return [100_000, 50_000, 20_000, 10_000, 5_000, 2_000, 1_000, 500]


def auto_select_predict_memory_knobs(
    *,
    peak_estimator: Callable[..., int],
    workload: dict,
    memory_budget_bytes: int,
    safety_fraction: float = _MULTI_KNOB_SAFETY_FRACTION,
) -> dict:
    """Pick ``{n_chunks, block_size, enc_tile_size, pos_tile_size}``
    such that ``peak_estimator(**workload, **knobs) <= budget * safety``.

    Greedy search, preference-ordered (fastest → slowest config):

    1. Stage 1: vary ``n_chunks`` and ``block_size``; no tiling.
    2. Stage 2: enable ``enc_tile_size``.
    3. Stage 3: enable ``pos_tile_size``.

    Returns the first config in that preference order whose estimated
    peak fits the budget.

    Parameters
    ----------
    peak_estimator
        A callable with signature ``(**workload, **knobs) -> int bytes``.
        Typically one of the algorithm-specific
        ``_estimate_predict_peak_bytes`` functions in
        ``non_local_detector.likelihoods.*``.
    workload
        Dict of workload shape args the estimator expects (n_time,
        n_state_bins, n_pos, n_encoding_spikes_max, etc.).
    memory_budget_bytes
        Device memory budget.  Typically queried via
        :func:`_query_device_memory_bytes`.
    safety_fraction
        Fraction of the budget the estimated peak is allowed to use.
        Default 0.90 — the per-algorithm estimators include their own
        2× multiplicative safety factor, so this is just for
        model imprecision, not for hidden multipliers.

    Returns
    -------
    dict
        ``{n_chunks: int, block_size: int, enc_tile_size: int | None,
        pos_tile_size: int | None}``.

    Raises
    ------
    RuntimeError
        If no knob combination fits the budget.
    """
    if memory_budget_bytes <= 0:
        raise ValueError(
            f"memory_budget_bytes must be positive; got {memory_budget_bytes}"
        )
    effective_budget = int(memory_budget_bytes * safety_fraction)
    n_time = int(workload.get("n_time", 1))
    n_enc = int(workload.get("n_encoding_spikes_max", 0))
    n_pos = int(workload.get("n_pos", 1))

    def _fits(knobs: dict) -> bool:
        peak = peak_estimator(**workload, **knobs)
        return peak <= effective_budget

    # Stage 1: no tiling.
    for n_chunks in _chunk_ladder(n_time):
        for block_size in _block_ladder():
            knobs = {
                "n_chunks": n_chunks,
                "block_size": block_size,
                "enc_tile_size": None,
                "pos_tile_size": None,
            }
            if _fits(knobs):
                return knobs

    # Stage 2: enable encoding tiling.  Only meaningful when the
    # workload actually has encoding spikes (clusterless).
    if n_enc > 0:
        for n_chunks in _chunk_ladder(n_time):
            for enc_tile in _enc_tile_ladder(n_enc):
                knobs = {
                    "n_chunks": n_chunks,
                    "block_size": 1_000,
                    "enc_tile_size": enc_tile,
                    "pos_tile_size": None,
                }
                if _fits(knobs):
                    return knobs

    # Stage 3: enable position tiling.
    for n_chunks in _chunk_ladder(n_time):
        for pos_tile in _pos_tile_ladder(n_pos):
            knobs = {
                "n_chunks": n_chunks,
                "block_size": 1_000,
                "enc_tile_size": (
                    min(_enc_tile_ladder(n_enc)) if n_enc > 0 else None
                ),
                "pos_tile_size": pos_tile,
            }
            if _fits(knobs):
                return knobs

    raise RuntimeError(
        f"Workload does not fit in {memory_budget_bytes / 2**30:.1f} GB even "
        f"with maximum chunking + tiling.  Reduce n_state_bins / n_pos / "
        f"encoding cloud size, or increase the memory budget."
    )


def auto_select_fit_memory_knobs(
    *,
    peak_estimator: Callable[..., int],
    workload: dict,
    memory_budget_bytes: int,
    safety_fraction: float = _MULTI_KNOB_SAFETY_FRACTION,
) -> dict:
    """Pick ``{fit_block_size}`` such that the fit peak fits the budget.

    Only one fit-side knob today.  Greedy search over
    :func:`_fit_block_ladder` largest-first.

    Returns
    -------
    dict
        ``{fit_block_size: int}``.

    Raises
    ------
    RuntimeError
        If no fit_block_size fits the budget (common for GLM fits,
        where the design matrix isn't chunked).
    """
    if memory_budget_bytes <= 0:
        raise ValueError(
            f"memory_budget_bytes must be positive; got {memory_budget_bytes}"
        )
    effective_budget = int(memory_budget_bytes * safety_fraction)

    for fit_block_size in _fit_block_ladder():
        peak = peak_estimator(**workload, fit_block_size=fit_block_size)
        if peak <= effective_budget:
            return {"fit_block_size": fit_block_size}

    raise RuntimeError(
        f"Fit does not fit in {memory_budget_bytes / 2**30:.1f} GB even with "
        f"smallest fit_block_size.  This typically means the workload has "
        f"fixed allocations (GLM design matrix, occupancy output) too big for "
        f"the budget; reduce n_pos / n_time_pos / n_coefficients."
    )
