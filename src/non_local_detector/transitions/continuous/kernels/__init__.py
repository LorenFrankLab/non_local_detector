"""
non_local_detector.transitions.continuous.kernels
=================================================

This package contains various `Kernel` implementations that define specific
local motion rules for transitions between states.

Each kernel is automatically registered with the central kernel registry
(see `non_local_detector.transitions.continuous.registry`) upon import of
its module, allowing them to be instantiated by name. This `__init__.py`
file ensures all standard kernels are imported and thus registered.
"""

from .diffusion_random_walk import DiffusionRandomWalkKernel
from .dirac_current import DiracToCurrentPosition
from .euclidean_random_walk import EuclideanRandomWalkKernel
from .geodesic_random_walk import GeodesicRandomWalkKernel
from .identity import IdentityKernel
from .uniform import UniformKernel

__all__ = [
    "DiffusionRandomWalkKernel",
    "DiracToCurrentPosition",
    "EuclideanRandomWalkKernel",
    "GeodesicRandomWalkKernel",
    "IdentityKernel",
    "UniformKernel",
]
