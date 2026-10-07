# Phase 7b exact transition capabilities

The default dense representation keeps the existing constructor and public
matrix. Structured and auto are opt-in. Products implement the same padded
conditional blocks, discrete weights, interior restriction, and zero rows;
restriction does not renormalize a block. There is no kernel truncation,
diffusion approximation, or change to the movement model.

| Configuration | Structured behavior | Auto behavior |
| --- | --- | --- |
| Uniform, Identity, single-bin Discrete | Exact masked products | Same |
| Scalar/spatial overrides and multi-bin Local Discrete upgrades | Existing builder semantics, including rectangular state blocks | Same |
| Cartesian Euclidean RandomWalk, finite positive scalar/diagonal covariance and scalar mean | Exact separable axis products with the original masks and row normalization | Same |
| Full covariance, graph/manifold, directional, EmpiricalMovement | Explicit unsupported capability | Original dense block with fit context and cumulative byte preflight |
| Subnormal, overflowed or ambiguous original float64 Gaussian PDF; unsafe scaled terms/normalization | Explicit unsupported numerical capability | Original dense block under budget |
| Opaque custom descriptor with deferred graph distances | Explicit unsupported capability | Explicit unsupported: no verified legacy distance-view contract |
| Custom descriptor with legacy eager environments, including slotted objects | Explicit unsupported capability | Explicit original dense constructor under the retained-block budget; arbitrary constructor workspace is not bounded |

All original float64 all-zero Gaussian rows stay zero. Positive per-axis row
scaling cancels in row normalization and allows the float32 runtime to retain
ordinary float64-normalized models whose raw density factors are small.
Closest valid destinations are located with a bounded spatial tree; capability
checks never form a grid-by-grid matrix. Explicit unsupported corners fall back
only when requested and within budget. Standalone legacy transition primitives
are unchanged.
Backward products rescale any nonzero input magnitude before contraction so
tiny finite vectors retain the same normalized result. A float32-normal sum
does not establish safe factors: the closest valid scaled term must itself be
normal, preventing contraction hardware from flushing all its contributors.

The default nonlocal four-state layouts, nonuniform priors, time-varying
discrete weights, masks with holes, singleton sequences, neutral emissions,
impossible rows and NaNs have dense-product/filter/smoother/evidence controls.
Actual x64 gradients through priors, likelihoods and discrete weights match the
dense reference, including impossible and zero-support boundary controls.
These are mathematical and CPU numerical controls, not production GPU evidence.

## Distance setup and compatibility

`Environment.fit_place_grid(compute_all_pairs_distances=False)` keeps an exact
CSR graph-distance view for Cartesian environments. Paired and `np.ix_` reads
run bounded SciPy Dijkstra source-row batches and return exact shortest paths,
including infinity across disconnected components. Topology penalties and
local kernels can request their existing graph distances. The legacy eager
default remains unchanged. Explicit 1D track graphs retain the existing
dictionary setup and have no bounded setup-memory claim.

Legacy manifold RandomWalk constructors select an ndarray path. Auto supplies
temporary copied environment views with budgeted dense distances, then restores
descriptor references to the original sparse environment. The preflight includes
the sum of all retained dense fallback blocks plus simultaneously materialized
distance matrices. EmpiricalMovement receives the original position, training,
group and environment labels; omitted labels retain the high-level defaults.
Opaque custom constructors can have arbitrary internal workspace; legacy dense
fallback is not a bounded scientific capability claim for them.
They are rejected before construction when any environment has deferred graph
distances, and therefore never receive or retain temporary dense environment
views. This limit applies equally to slotted custom descriptors. Slotted custom
descriptors remain usable with explicitly selected legacy eager environments.

`LazyDenseTransition` preserves read-only shape/dtype, basic and paired reads,
`np.ix_`, explicit dense conversion and pickle semantics. Full conversion checks
the requested output size before allocating. It stores the operator, not an
N-by-N cache. Large EM/Viterbi/plotting operations that require dense data fail
with a resource message unless a feasible budget is explicitly selected.
Standalone NumPy consumers should explicitly convert a bounded selection or
matrix; this view does not implement every ndarray operation.

## Measurement and precision

`scripts/benchmark_phase7b_operators.py` records environment/operator setup,
compile latency, at least five synchronized interleaved warm product timings,
retained/pickle bytes, compiler memory estimates, RSS and actual devices/dtype.
Dense same-input products run only under the preflight reference budget. A large
structured-only run does not establish dense numerical parity or full-pipeline
speed. Root pipeline qualification covers the complete fit/predict hour and
output storage separately.

JAX contractions explicitly request `Precision.HIGHEST`. An A100 speed gain
from TF32 is not accepted as equivalent arithmetic. CPU float32 and actual
float64 parity are covered. All 72 operator controls pass on the A100;
full-pipeline hour/device-memory qualification is recorded separately in the
[validation record](../../../../docs/performance_validation.md). Pytrees use
explicit `register_pytree_node`, avoiding newer inferred-dataclass APIs. The
older tested Python 3.11/JAX 0.6.2 suite passes 415 checks with x64 enabled,
including these operator controls; existing references and tolerances remain
unchanged.
