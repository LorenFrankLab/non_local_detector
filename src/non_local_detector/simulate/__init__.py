"""Simulators for position, sorted spikes, and clusterless marks.

The shared utility functions (``simulate_time``, ``simulate_poisson_spikes``,
``simulate_place_field_firing_rate``, ``simulate_neuron_with_place_field``,
``get_trajectory_direction``) live in ``_common`` and are re-exported here for
convenience. Module-specific helpers stay in their respective simulator
modules and can be imported from there directly.
"""

from non_local_detector.simulate._common import (
    get_trajectory_direction,
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
    simulate_poisson_spikes,
    simulate_time,
)
from non_local_detector.simulate.simulate import (
    simulate_multiunit_with_place_fields,
    simulate_position,
    simulate_position_with_pauses,
)

__all__ = [
    "get_trajectory_direction",
    "simulate_multiunit_with_place_fields",
    "simulate_neuron_with_place_field",
    "simulate_place_field_firing_rate",
    "simulate_poisson_spikes",
    "simulate_position",
    "simulate_position_with_pauses",
    "simulate_time",
]
