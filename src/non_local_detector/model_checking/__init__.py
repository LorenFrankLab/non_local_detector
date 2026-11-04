"""Goodness-of-fit checks for decoding models and their posteriors."""

from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from non_local_detector.model_checking.sorted_spikes import TimeRescaling

__all__ = [
    "TimeRescaling",
    "get_HPD_spatial_coverage",
    "get_highest_posterior_threshold",
]
