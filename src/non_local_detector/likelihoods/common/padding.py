"""Padded sizes that let electrodes and chunks of similar size share kernels."""

import numpy as np
from numpy.typing import ArrayLike


def _octave_size(n: int, log2_sizes: int) -> int:
    """Smallest of ``2**log2_sizes`` sizes per octave at least ``n`` (``n >= 1``)."""
    step = 1 << max((n - 1).bit_length() - 1 - log2_sizes, 0)
    return -(-n // step) * step


def _quarter_octave(n: int) -> int:
    """Smallest of four sizes per octave that is at least ``n`` (``n >= 1``)."""
    return _octave_size(n, 2)


def padded_sample_count(n_samples: int) -> int:
    """Encoding samples per electrode: four sizes per octave, at most 25% padding."""
    return 16 if n_samples <= 16 else _quarter_octave(n_samples)


def padded_occupancy_count(n_samples: int) -> int:
    """Occupancy KDE samples: 64 sizes per octave, at most 1/64 padding.

    Occupancy samples grow with the recording and each one is evaluated at
    every spike, so they are padded more finely than encoding samples. EM
    refits them from decode bins, one fewer than the position samples of the
    first fit; both counts then usually share a size, so the local kernels
    do not recompile.
    """
    return _octave_size(n_samples, 6)


def padded_spike_count(n_spikes: int, block_size: int, *, fine: bool = False) -> int:
    """Decoding spikes per electrode kernel call, from a few sizes.

    Up to one block: powers of two (at least 16), or with ``fine`` four sizes
    per octave (at most 25% padding) for kernels whose per-spike cost is
    high. Above one block: whole blocks, their count padded to four sizes
    per octave, so a large request grows by at most a quarter plus one
    block. Chunks with varying spike counts then reuse few compiled kernels.

    Parameters
    ----------
    n_spikes : int
    block_size : int
    fine : bool, optional

    Returns
    -------
    n_padded : int
    """
    if n_spikes <= block_size:
        size = (
            padded_sample_count(n_spikes)
            if fine
            else 1 << max(n_spikes - 1, 15).bit_length()
        )
        return min(block_size, size)
    return block_size * _quarter_octave(-(-n_spikes // block_size))


def pad_rows(array: ArrayLike, n_rows: int) -> np.ndarray:
    """Zero rows appended on the host to reach ``n_rows``."""
    array = np.asarray(array)
    padding = np.zeros((n_rows - array.shape[0], *array.shape[1:]), array.dtype)
    return np.concatenate([array, padding])


def padded_index(index: np.ndarray) -> np.ndarray:
    """Nonempty ``index`` padded to four sizes per octave (at least 16).

    Padding repeats the last entry, so a scatter through the padded index
    writes only indexed rows. Kernels evaluated on the indexed subset then
    compile for few sizes; gathering callers keep the first ``index.size``
    results.
    """
    return np.pad(
        index, (0, _quarter_octave(max(index.size, 16)) - index.size), mode="edge"
    )
