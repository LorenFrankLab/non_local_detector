"""
non_local_detector.transitions.continuous.block
==============================================

`BlockTransition` stitches together per-state *Kernel* objects into the large
``(n_bins_total, n_bins_total)`` matrix required by the core HMM code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ....environment import Environment
from ....model.state_spec import StateSpec
from ...continuous.base import Array, ContinuousTransition, Covariates, Kernel


@dataclass
class BlockTransition(ContinuousTransition):
    """
    Combine per-(src, dst) Kernels into one flattened continuous-transition
    matrix.

    Parameters
    ----------
    state_map
        Mapping ``(src_state_name, dst_state_name) → Kernel``.
        Omitted pairs default to *row-uniform* entry into the destination
        state's bins.
    state_specs
        List of `StateSpec` objects (same list you pass to `HHMMDetector`).
    """

    state_map: Mapping[Tuple[str, str], Kernel]
    state_specs: list[StateSpec]

    # Filled in __post_init__
    _slice_of: Dict[str, slice] = field(init=False, repr=False, default_factory=dict)
    _env_of: Dict[str, Optional[Environment]] = field(
        init=False, repr=False, default_factory=dict
    )
    n_bins_total: int = field(init=False)

    # ------------------------------------------------------------------ #
    #  Dataclass initialisation                                          #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        cursor = 0
        # Order of state specs dictates the order of slices in the matrix.
        # Check if all state names are unique.
        state_names = {spec.name for spec in self.state_specs}
        if len(state_names) != len(self.state_specs):
            raise ValueError("State names must be unique.")

        for spec in self.state_specs:
            self._slice_of[spec.name] = slice(cursor, cursor + spec.n_bins)
            self._env_of[spec.name] = spec.env
            cursor += spec.n_bins
        self.n_bins_total = cursor

    # ------------------------------------------------------------------ #
    #  ContinuousTransition API                                          #
    # ------------------------------------------------------------------ #
    def matrix(
        self,
        *,
        covariates: Optional[Covariates] = None,
    ) -> Array:
        """
        Assemble and return the row-stochastic flattened matrix.
        """
        flat = np.zeros((self.n_bins_total, self.n_bins_total), dtype=float)

        # 1) Place all user-provided kernels
        for (src_name, dst_name), kernel in self.state_map.items():
            src_slice = self._slice_of[src_name]
            dst_slice = self._slice_of[dst_name]
            flat[src_slice, dst_slice] = kernel.block(
                src_env=self._env_of[src_name],
                dst_env=self._env_of[dst_name],
                covariates=covariates,
            )

        # 2) Fill missing blocks with uniform entry
        filled_pairs = set(self.state_map.keys())
        for src_name, src_slice in self._slice_of.items():
            for dst_name, dst_slice in self._slice_of.items():
                if (src_name, dst_name) not in filled_pairs:
                    dst_bins = dst_slice.stop - dst_slice.start
                    flat[src_slice, dst_slice] = 1.0 / dst_bins

        # 3) Ensure every row sums to 1.0 (safe re-normalisation)
        row_sums = flat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        flat /= row_sums

        return flat

    # ------------------------------------------------------------------ #
    #  Alternate constructor                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_state_map(
        cls,
        state_map: Mapping[Tuple[str, str], Kernel],
        state_specs: list[StateSpec],
    ) -> "BlockTransition":
        return cls(state_map=state_map, state_specs=state_specs)

    # ------------------------------------------------------------------ #
    #  Debug representation                                              #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        state_names = ", ".join(self._slice_of)
        return (
            f"BlockTransition(n_bins_total={self.n_bins_total}, states=[{state_names}])"
        )
