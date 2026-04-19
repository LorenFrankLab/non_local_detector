"""Tests that _compute_non_local_position_penalty respects track topology.

The non-local penalty is a negative Gaussian centered on the animal's
position. For multi-arm tracks (or N-D environments with obstacles), the
penalty should use graph shortest-path distance so that two bins separated
by a junction or wall are correctly far apart in penalty space, even if
they are close in Euclidean/linearized coordinates.
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment


def _make_two_arm_track():
    """Two-segment track with a gap between arms (edge_spacing > 0)."""
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(50.0, 0.0))
    g.add_node(2, pos=(60.0, 0.0))
    g.add_node(3, pos=(110.0, 0.0))
    g.add_edge(0, 1, distance=50.0, edge_id=0)
    g.add_edge(2, 3, distance=50.0, edge_id=1)

    env = Environment(
        environment_name="",
        place_bin_size=5.0,
        track_graph=g,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
    )
    return env.fit_place_grid(position_1d, infer_track_interior=True)


@pytest.mark.unit
class TestNonLocalPenaltyTopology:
    """Verify penalty uses topology-aware distances."""

    def test_penalty_shape(self):
        """Penalty returns (n_time, n_interior_bins)."""
        env = _make_two_arm_track()
        detector = NonLocalSortedSpikesDetector(
            environments=[env],
            non_local_position_penalty=1.0,
            non_local_penalty_sigma=5.0,
        )

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        # Animal near start of arm 0
        animal_position = np.array([[10.0], [10.0]])

        penalty = detector._compute_non_local_position_penalty(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        n_interior = int(env.is_track_interior_.sum())
        assert penalty.shape == (1, n_interior)

    def test_penalty_uses_graph_distance_on_two_arm_track(self):
        """On a two-arm track, penalty respects graph distance, not linearized.

        With the animal on arm 0 (position 25, firmly inside an
        interior bin) and the penalty sigma narrow (5 cm), bins across
        the gap on arm 1 should be far in graph space so the penalty
        there is weak compared to the animal's own bin.
        """
        env = _make_two_arm_track()
        detector = NonLocalSortedSpikesDetector(
            environments=[env],
            non_local_position_penalty=10.0,
            non_local_penalty_sigma=5.0,
        )

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        # Animal firmly on arm 0 → maps to bin center 27.5 (interior)
        animal_position = np.array([[25.0], [25.0]])

        penalty = detector._compute_non_local_position_penalty(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )
        penalty_np = np.asarray(penalty)

        # Penalty is indexed over interior bins. The animal maps to the
        # interior bin with center 27.5; find its index and the index
        # of a bin on arm 1 (e.g., center 62.5).
        is_interior = env.is_track_interior_.ravel()
        interior_centers = env.place_bin_centers_[is_interior].ravel()

        animal_bin_idx = np.argmin(np.abs(interior_centers - 27.5))
        arm1_bin_idx = np.argmin(np.abs(interior_centers - 62.5))

        assert penalty_np[0, animal_bin_idx] < -5.0, (
            "Penalty should be strong near the animal's bin, got "
            f"{penalty_np[0, animal_bin_idx]:.3f}"
        )
        # Arm-1 bin is >35 cm from animal in graph distance (≥ 5 cm to
        # end of arm 0 + gap + into arm 1 ≥ 35cm), so penalty should be
        # essentially zero compared to the near-animal magnitude.
        assert abs(penalty_np[0, arm1_bin_idx]) < 0.1 * abs(
            penalty_np[0, animal_bin_idx]
        ), "Penalty at arm-1 bin should be much weaker than at animal bin"

    def test_penalty_finite_when_animal_off_track(self):
        """Off-track animal position produces finite (zero) penalty, not NaN."""
        env = _make_two_arm_track()
        detector = NonLocalSortedSpikesDetector(
            environments=[env],
            non_local_position_penalty=10.0,
            non_local_penalty_sigma=5.0,
        )

        # Find a gap-bin center to use as the "off-track" animal position
        is_interior = env.is_track_interior_.ravel()
        gap_bins = np.where(~is_interior)[0]
        assert len(gap_bins) > 0, "Environment should have gap bins"
        gap_center = env.place_bin_centers_[gap_bins[0], 0]

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[gap_center], [gap_center]])

        penalty = detector._compute_non_local_position_penalty(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )
        penalty_np = np.asarray(penalty)

        # No NaN or inf
        assert np.all(np.isfinite(penalty_np)), (
            "Off-track animal should not produce NaN/inf penalty"
        )
        # All zero (exp(-inf) == 0)
        np.testing.assert_allclose(penalty_np, 0.0, atol=1e-10)

    def test_penalty_zero_when_disabled(self):
        """With non_local_position_penalty=0, penalty is identically zero."""
        env = _make_two_arm_track()
        detector = NonLocalSortedSpikesDetector(
            environments=[env],
            non_local_position_penalty=0.0,
            non_local_penalty_sigma=5.0,
        )

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[10.0], [10.0]])

        penalty = detector._compute_non_local_position_penalty(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )
        penalty_np = np.asarray(penalty)

        np.testing.assert_allclose(penalty_np, 0.0, atol=1e-10)

    def test_penalty_monotone_in_graph_distance(self):
        """Penalty magnitude monotonically decreases with graph distance.

        Within a single arm of the track, the penalty at the animal's bin
        is the strongest and monotonically weakens as distance grows.
        """
        env = _make_two_arm_track()
        detector = NonLocalSortedSpikesDetector(
            environments=[env],
            non_local_position_penalty=10.0,
            non_local_penalty_sigma=8.0,
        )

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        # Animal at position 25 maps to interior bin with center 27.5.
        # Graph distances are measured from THAT bin center, not 25.
        animal_position = np.array([[25.0], [25.0]])
        animal_bin_center = 27.5

        penalty = detector._compute_non_local_position_penalty(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )
        penalty_np = np.asarray(penalty[0])

        is_interior = env.is_track_interior_.ravel()
        interior_centers = env.place_bin_centers_[is_interior].ravel()

        # Restrict to arm 0 interior bins (centers <= 50). Graph distance
        # along a linear arm equals |bin_center - animal_bin_center|.
        arm0_mask = interior_centers <= 50.0
        arm0_dist = np.abs(interior_centers[arm0_mask] - animal_bin_center)
        arm0_penalty_mag = -penalty_np[arm0_mask]

        # Sort by graph distance and assert the sorted penalty
        # magnitudes are monotonically non-increasing. Use a small
        # atol for float32 precision in the exp computation.
        order = np.argsort(arm0_dist)
        sorted_mag = arm0_penalty_mag[order]
        assert np.all(np.diff(sorted_mag) <= 1e-5), (
            "Penalty magnitude should be monotonically non-increasing "
            f"with graph distance within a single arm. Sorted mags: "
            f"{sorted_mag}"
        )
