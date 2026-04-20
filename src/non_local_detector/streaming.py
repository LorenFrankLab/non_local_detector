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
