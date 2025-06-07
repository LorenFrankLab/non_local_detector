import non_local_detector.transitions.continuous.registry as cont_registry

# Force‐clear any previous registrations so we can import safely.
for name in list(cont_registry._KERNELS.keys()):
    cont_registry._KERNELS.pop(name)


import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.model.state_spec import StateSpec
from non_local_detector.observations.base import ObservationModel
from non_local_detector.transitions.continuous.kernels import (
    EuclideanRandomWalkKernel,
    UniformKernel,
)
from non_local_detector.transitions.continuous.orchestrators.block import (
    BlockTransition,
)


@pytest.fixture
def simple_environments():
    """
    Create two small 2D environments (A and B) using sample clusters.
    Each returns:
      - env_A, env_B: the two Environment instances
      - spec_A, spec_B: corresponding StateSpec objects
    """
    # Define clusters for each environment
    cluster_A = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    cluster_B = np.array([[5.0, 5.0], [5.0, 6.0], [6.0, 5.0], [6.0, 6.0]], dtype=float)

    # Build environments with RegularGrid layout and bin_size=2.0
    env_A = Environment.from_samples(
        data_samples=cluster_A,
        name="StateA",
        layout_kind="RegularGrid",
        bin_size=2.0,
        infer_active_bins=True,
        bin_count_threshold=0,
    )
    env_B = Environment.from_samples(
        data_samples=cluster_B,
        name="StateB",
        layout_kind="RegularGrid",
        bin_size=2.0,
        infer_active_bins=True,
        bin_count_threshold=0,
    )

    class ShamObservationModel(ObservationModel):
        required_sources = ("x",)

        @property
        def n_bins(self):
            return 1

        def log_likelihood(self, batch):
            return 1.0  # Placeholder for actual log-likelihood computation

    spec_A = StateSpec(name="A", env=env_A, obs_model=ShamObservationModel())
    spec_B = StateSpec(name="B", env=env_B, obs_model=ShamObservationModel())

    return env_A, env_B, spec_A, spec_B


def test_environment_bin_counts(simple_environments):
    env_A, env_B, _, _ = simple_environments
    nA, nB = env_A.n_bins, env_B.n_bins

    # Each cluster of four points should map to at least 1 bin;
    # with bin_size=2, these 4 points likely form a 2x2 grid -> 4 bins each.
    assert nA == 4, f"Expected 4 bins for environment A, got {nA}"
    assert nB == 4, f"Expected 4 bins for environment B, got {nB}"


def test_euclidean_kernel_row_stochastic(simple_environments):
    env_A, env_B, _, _ = simple_environments
    kernel_A = EuclideanRandomWalkKernel(mean=np.zeros(env_A.n_dims), var=1.0)

    # Generate the within-state block for A→A
    block_AA = kernel_A.block(src_env=env_A, dst_env=env_A, covariates={})

    # Check shape and row-stochasticity
    assert block_AA.shape == (env_A.n_bins, env_A.n_bins)
    row_sums = block_AA.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "Rows of A→A block not summing to 1"


def test_uniform_kernel_block(simple_environments):
    env_A, env_B, _, _ = simple_environments
    kernel_uniform = UniformKernel()

    # Within-state uniform for A→A
    uniform_AA = kernel_uniform.block(src_env=env_A, dst_env=env_A, covariates={})
    expected_AA = np.full((env_A.n_bins, env_A.n_bins), 1.0 / env_A.n_bins)
    assert np.allclose(uniform_AA, expected_AA), "UniformKernel A→A block is incorrect"

    # Cross-state uniform for A→B
    uniform_AB = kernel_uniform.block(src_env=env_A, dst_env=env_B, covariates={})
    expected_AB = np.full((env_A.n_bins, env_B.n_bins), 1.0 / env_B.n_bins)
    assert np.allclose(uniform_AB, expected_AB), "UniformKernel A→B block is incorrect"


def test_block_transition_matrix_properties(simple_environments):
    env_A, env_B, spec_A, spec_B = simple_environments

    # Define Gaussian kernels for diagonal blocks
    kernel_A = EuclideanRandomWalkKernel(mean=np.zeros(env_A.n_dims), var=1.0)
    kernel_B = EuclideanRandomWalkKernel(mean=np.zeros(env_B.n_dims), var=1.0)

    # Build state_map with only diagonal entries
    state_map = {
        ("A", "A"): kernel_A,
        ("B", "B"): kernel_B,
    }

    # Create BlockTransition and compute the full transition matrix
    block_trans = BlockTransition(state_map=state_map, state_specs=[spec_A, spec_B])
    full_matrix = block_trans.matrix(covariates={})

    # Check overall shape
    total_bins = env_A.n_bins + env_B.n_bins
    assert full_matrix.shape == (
        total_bins,
        total_bins,
    ), "Full matrix has incorrect shape"

    # Check row-stochasticity of full matrix
    row_sums = full_matrix.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-8
    ), "Rows of full transition matrix do not sum to 1"

    # Check block structure: top-left (A→A), top-right (A→B), bottom-left (B→A), bottom-right (B→B)
    nA, nB = env_A.n_bins, env_B.n_bins
    AA = full_matrix[:nA, :nA]
    AB = full_matrix[:nA, nA:]
    BA = full_matrix[nA:, :nA]
    BB = full_matrix[nA:, nA:]

    # 1) A→A and B→B should each be row-stochastic sub-blocks before mixing:
    assert np.allclose(
        AA.sum(axis=1) + AB.sum(axis=1), 1.0, atol=1e-8
    ), "A→rows don't sum to 1"
    assert np.allclose(
        BB.sum(axis=1) + BA.sum(axis=1), 1.0, atol=1e-8
    ), "B→rows don't sum to 1"

    # 2) Since AB and BA were filled uniformly, test that all entries in AB are equal,
    #    and all entries in BA are equal.
    assert np.allclose(AB, AB[0, 0]), "A→B block is not uniformly filled"
    assert np.allclose(BA, BA[0, 0]), "B→A block is not uniformly filled"

    # 3) Check that the diagonal Gaussian sub-blocks have some variability (i.e., not uniform).
    #    For instance, A→A block should not have all entries identical.
    assert not np.allclose(
        AA, AA[0, 0]
    ), "A→A Gaussian block appears uniform; expected variation"
    assert not np.allclose(
        BB, BB[0, 0]
    ), "B→B Gaussian block appears uniform; expected variation"
