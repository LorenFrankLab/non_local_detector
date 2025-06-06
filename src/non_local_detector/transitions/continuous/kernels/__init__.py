"""
non_local_detector.transitions.continuous.kernels
=================================================

This package contains various `Kernel` implementations that define specific
local motion rules for transitions between continuous states.

"""

from .diffusion_random_walk import DiffusionRandomWalkKernel
from .dirac_current import DiracToCurrentSample
from .empirical import EmpiricalKernel
from .euclidean_random_walk import EuclideanRandomWalkKernel
from .geodesic_random_walk import GeodesicRandomWalkKernel
from .identity import IdentityKernel
from .uniform import UniformKernel

__all__ = [
    "DiffusionRandomWalkKernel",
    "DiracToCurrentSample",
    "EmpiricalKernel",
    "EuclideanRandomWalkKernel",
    "GeodesicRandomWalkKernel",
    "IdentityKernel",
    "UniformKernel",
]
