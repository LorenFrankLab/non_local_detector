from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.model.state_spec import StateSpec

from ..base import ContinuousTransition, Kernel


@dataclass
class BlockTransition(ContinuousTransition):
    """
    Assemble a full transition matrix from a mapping of (source_state, dest_state) to Kernel blocks,
    and a sequence of StateSpec defining the names and number of bins per state.

    State bins are laid out in the order provided by `state_specs`. The slice indices for each state
    are determined by summing `n_bins` for previous specs.

    For any (src, dst) pair not in `state_map`, a uniform block is used (unless overridden by a zero-
    probability user block). All rows are renormalized to ensure row-stochasticity.
    """

    state_map: Dict[tuple[str, str], Kernel]
    state_specs: Sequence[StateSpec]

    # Internal attributes (populated in __post_init__)
    n_bins_total: int = 0
    _slice_of: dict[str, slice] = (
        None  # mapping state name -> slice in flattened matrix
    )
    _env_of: dict[str, Optional[Environment]] = (
        None  # state name -> environment or None
    )

    def __post_init__(self):
        # Ensure state_spec names are unique
        names = [spec.name for spec in self.state_specs]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate StateSpec names found: {sorted(duplicates)}")

        # Build a mapping from each state name to its bin-slice and record environment
        self._slice_of = {}
        self._env_of = {}
        start = 0
        for spec in self.state_specs:
            n = spec.n_bins
            if n < 1:
                raise ValueError(
                    f"State '{spec.name}' has invalid number of bins: {n}. Must be >= 1."
                )
            end = start + n
            self._slice_of[spec.name] = slice(start, end)
            self._env_of[spec.name] = spec.env
            start = end
        self.n_bins_total = start

    def matrix(self, covariates: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build the full transition matrix of shape (n_bins_total, n_bins_total).

        - For each (src, dst) in state_map, obtain block = kernel.block(src_env, dst_env, covariates).
          Validate its shape matches (n_src_bins, n_dst_bins).
        - For any (src, dst) not provided in state_map, fill a uniform block.
        - After placing all blocks, renormalize each row to sum to 1.

        Returns:
            A dense (n_bins_total x n_bins_total) row-stochastic matrix.
        """
        total = self.n_bins_total
        flat = np.zeros((total, total), dtype=float)

        # Track which (src_name, dst_name) pairs have an explicit block
        provided_pairs = set(self.state_map.keys())

        # Place user-provided kernel blocks
        for (src_name, dst_name), kernel in self.state_map.items():
            if src_name not in self._slice_of or dst_name not in self._slice_of:
                raise KeyError(
                    f"Unknown state in state_map key: ({src_name}, {dst_name})"
                )
            s_slice = self._slice_of[src_name]
            d_slice = self._slice_of[dst_name]
            src_env = self._env_of[src_name]
            dst_env = self._env_of[dst_name]

            # Compute the block from the kernel
            block = kernel.block(
                src_env=src_env, dst_env=dst_env, covariates=covariates
            )
            n_src = s_slice.stop - s_slice.start
            n_dst = d_slice.stop - d_slice.start
            expected_shape = (n_src, n_dst)
            if block.shape != expected_shape:
                raise ValueError(
                    f"Kernel {kernel.__class__.__name__} for {src_name}->{dst_name} returned shape {block.shape}, "
                    f"expected {expected_shape}."
                )
            flat[s_slice, d_slice] = block

        # Fill missing blocks uniformly
        for src_name, src_slice in self._slice_of.items():
            n_src = src_slice.stop - src_slice.start
            for dst_name, dst_slice in self._slice_of.items():
                if (src_name, dst_name) not in provided_pairs:
                    n_dst = dst_slice.stop - dst_slice.start
                    if n_dst < 1:
                        raise ValueError(
                            f"Destination state '{dst_name}' has zero bins."
                        )
                    uniform_block = np.full((n_src, n_dst), 1.0 / n_dst, dtype=float)
                    flat[src_slice, dst_slice] = uniform_block

        # Re-normalize rows to ensure row-stochasticity
        row_sums = flat.sum(axis=1, keepdims=True)
        # Prevent division by zero: if a row sums to zero, fill it uniformly across all bins
        zero_rows = row_sums.flatten() == 0.0
        if np.any(zero_rows):
            flat[zero_rows, :] = 1.0 / total
            row_sums = flat.sum(axis=1, keepdims=True)

        flat = flat / row_sums
        return flat

    @classmethod
    def from_state_map(
        cls, state_map: Dict[tuple[str, str], Kernel], state_specs: Sequence[StateSpec]
    ) -> BlockTransition:
        """
        Alternate constructor alias for clarity.
        """
        return cls(state_map=state_map, state_specs=state_specs)

    def __repr__(self):
        state_names = ", ".join(self._slice_of.keys())
        return f"<BlockTransition size={self.n_bins_total} states=({state_names})>"
