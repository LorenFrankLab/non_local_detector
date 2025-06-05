from ...models import StateSpec
from .base import Kernel
from .kernels import EuclideanRandomWalkKernel
from .orchestrators.block import BlockTransition


def uniform_entry(state_specs: list[StateSpec], var: float = 6.0) -> BlockTransition:
    """Uniform jump whenever the discrete state changes."""
    mapping: dict[tuple[str, str], Kernel] = {}
    for spec in state_specs:
        mapping[(spec.name, spec.name)] = EuclideanRandomWalkKernel(mean=0.0, var=var)
    return BlockTransition(mapping, state_specs)
