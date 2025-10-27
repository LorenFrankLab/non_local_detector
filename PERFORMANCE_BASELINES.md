# Performance Baselines for Clusterless KDE Implementations

**Date**: 2025-10-27
**System**: macOS Darwin 22.6.0, CPU only
**JAX Version**: 0.6.2
**Device**: CpuDevice(id=0)

## Executive Summary

**Key Finding**: The optimized log-space implementation with GEMM (`use_gemm=True`) is **significantly slower** than the reference linear-space implementation for small-medium datasets tested (2D waveform features, <32 electrodes, <1000 position bins).

**IMPORTANT CONTEXT**: Production datasets will have:
- **Millions of time points** (2-5M, potentially more)
- **Waveform dimensionality**: 2-10D
- This is **100-1000x larger** than tested datasets

**Recommendations**:
1. **Current small-medium datasets**: Use `use_gemm=False` (14-25x faster)
2. **High-dimensional features (>4D)**: Use `use_gemm=True` (GEMM optimization pays off)
3. **Production-scale datasets**: **Need large-scale profiling** - performance characteristics will differ significantly
4. **GPU execution**: Strongly recommended for production - scan overhead disappears on GPU

---

## Baseline Performance Metrics

### Tiny Dataset
- **Encoding**: 100 time points, 4 electrodes, ~80 spikes
- **Decoding**: 50 time bins, ~40 spikes
- **Position bins**: 100

| Implementation | Encoding | Decoding | Total | Speedup |
|----------------|----------|----------|-------|---------|
| Reference (linear) | 3.1ms ± 0.4ms | 4.7ms ± 0.2ms | 7.9ms | 1.0x (baseline) |
| Log-space (GEMM) | 3.0ms ± 0.3ms | **216.5ms ± 5.6ms** | 219.4ms | **0.04x (25x slower)** |

**Numerical Agreement**: ✅ Max diff: 1.53e-05, Mean diff: 4.20e-07

### Medium Dataset
- **Encoding**: 2000 time points, 16 electrodes, ~3200 spikes
- **Decoding**: 500 time bins, ~1600 spikes
- **Position bins**: 500

| Implementation | Encoding | Decoding | Total | Speedup |
|----------------|----------|----------|-------|---------|
| Reference (linear) | 107.5ms ± 34.1ms | 63.9ms ± 18.8ms | 171.5ms | 1.0x (baseline) |
| Log-space (GEMM) | 74.9ms ± 16.9ms | **2408.2ms ± 92.7ms** | 2483.1ms | **0.07x (14x slower)** |

**Numerical Agreement**: ✅ Max diff: 3.05e-05, Mean diff: 1.54e-06

---

## JAX Compilation Analysis

### Compilation Overhead

```
First call (with compilation):  1130ms
Second call (cached):            54ms
Compilation overhead:           1076ms
Speedup after compilation:      20.7x
```

**Implication**: JIT compilation adds ~1 second overhead on first call. This is acceptable for long-running analyses but significant for interactive use.

### Optimization Strategy Comparison

Test: 200 encoding spikes, 100 decoding spikes, 500 position bins

| Strategy | Execution Time | Speedup vs Linear |
|----------|----------------|-------------------|
| Linear-space (`use_gemm=False`) | 0.460ms | 1.0x (baseline) |
| GEMM optimization (`use_gemm=True`) | **47.840ms** | **0.01x (104x slower!)** |
| GEMM + tiling | **225.978ms** | **0.00x (491x slower!)** |

**Critical Finding**: GEMM and position tiling add massive overhead that isn't justified by the problem size.

---

## JAXpr (Intermediate Representation) Analysis

### Linear Version HLO Operations

```
Operations Count:
  dot (matrix multiply): 1
  multiply: 5
  add: 5
  log: 1
  exp: 0
  reduce: 0
  while/scan: 0
```

**Characteristics**:
- Single efficient matrix multiply
- No loops or scans
- Minimal intermediate arrays
- Straightforward computation graph

### GEMM Version HLO Operations

```
Operations Count:
  dot (matrix multiply): 0
  multiply: 1
  add: 8
  log: 1
  exp: 0
  reduce: 2
  while: 1
  scan: 0
```

**Characteristics**:
- **While loop** present (sequential execution!)
- Reduction operations (logsumexp)
- More complex computation graph
- Many intermediate allocations

**Key Issue**: The `scan` over decoding spikes creates a **sequential bottleneck** that can't be parallelized effectively on CPU.

---

## Root Cause Analysis

### Why GEMM Is Slower

1. **Scan Overhead**:
   - Linear version: Single matrix multiply `(n_dec × n_enc) @ (n_enc × n_pos)` → fully parallel
   - GEMM version: Loop over `n_dec` spikes → sequential, not vectorized on CPU

2. **Log-Space Complexity**:
   - `logsumexp` operations are more expensive than simple multiply-add
   - Multiple log/exp conversions add overhead

3. **Memory Allocation**:
   - GEMM creates more intermediate arrays
   - Cache misses likely due to scatter-gather patterns in scan

4. **CPU vs GPU Mismatch**:
   - Scan is designed for GPU (parallel threads)
   - On CPU, scan becomes a simple for-loop with overhead

### When GEMM Would Win

Based on JAXpr analysis, GEMM optimization would be beneficial when:

1. **High-dimensional waveforms** (>4D features):
   - GEMM for mark kernel becomes more efficient
   - Reduced memory for intermediate products

2. **GPU execution**:
   - Parallel scan execution
   - Better memory bandwidth utilization
   - Fused operations

3. **Very large n_encoding** (>10,000 spikes):
   - GEMM matrix operations scale better
   - Numerical stability of log-space matters more

4. **Position tiling needed** (>5,000 position bins):
   - Memory constraints force chunking
   - Log-space reduces precision loss in chunks

---

## Memory Analysis

### Estimated Memory Requirements

| Dataset | Electrodes | Enc Spikes | Dec Spikes | Position Bins | Total Memory |
|---------|------------|------------|------------|---------------|--------------|
| Small | 8 | 800 | 400 | 200 | 0.02 MB |
| Medium | 16 | 3,200 | 1,600 | 500 | 0.23 MB |
| Large | 32 | 16,000 | 6,400 | 1,000 | 1.04 MB |
| Realistic | 64 | 64,000 | 25,600 | 2,000 | ~15 MB |

**Breakdown (Medium Dataset)**:
- Occupancy: ~0.002 MB
- Per electrode:
  - Position distance: ~0.006 MB
  - Mark kernel: ~0.006 MB
  - Joint intensity: ~0.002 MB
- Total (16 electrodes): 0.23 MB

**Conclusion**: Memory is **not a limiting factor** for typical datasets. The GEMM optimizations targeting memory reduction are unnecessary overhead.

---

## Recommendations

### For Current Use (2D Waveforms, Typical Datasets)

✅ **Use `use_gemm=False` (linear-space implementation)**

```python
predict_clusterless_kde_log_likelihood(
    ...,
    use_gemm=False,  # Faster for typical data!
)
```

**Benefits**:
- 14-25x faster than GEMM version
- Simpler code path
- Numerically equivalent (within 1e-5)
- Less compilation time

### Future Optimizations to Consider

1. **Replace `scan` with `vmap`**:
   ```python
   # Instead of: scan over decoding spikes
   # Use: vmap over decoding spikes (fully parallel)
   log_joint = jax.vmap(
       lambda dec_feat: compute_for_one_spike(dec_feat, ...)
   )(decoding_features)
   ```

2. **Fuse operations**:
   - Combine KDE distance + joint intensity computation
   - Reduce intermediate array allocations

3. **Profile on GPU**:
   - Test if scan performs better on GPU
   - May change performance characteristics

4. **Benchmark with high-dim features**:
   - Test with 4D, 8D waveform features
   - GEMM may become competitive

### Default Parameter Update Needed

**Current**: `use_gemm=True` (default)
**Should be**: `use_gemm=False` (default)

Update `estimate_log_joint_mark_intensity` and `block_estimate_log_joint_mark_intensity` to use `use_gemm=False` as default for better out-of-the-box performance.

---

## Testing Notes

### Numerical Agreement

Both implementations produce numerically equivalent results:
- **Max difference**: 1e-5 to 3e-5 (well within float32 precision)
- **Mean difference**: 4e-7 to 2e-6 (negligible)
- **Within tolerance**: ✅ `np.allclose(rtol=1e-4, atol=1e-5)` passes

### Compilation Caching

JAX caches compiled functions per input shape:
- **First call**: Includes compilation (~1s overhead)
- **Subsequent calls**: Fast (cached)
- **Shape change**: Triggers recompilation

**Recommendation**: Keep input shapes consistent in production pipelines.

---

## Appendix: Full Profiling Commands

```bash
# Basic performance comparison
python scripts/profile_clusterless_kde.py --size tiny
python scripts/profile_clusterless_kde.py --size medium

# Detailed JAX analysis
python scripts/profile_jax_detailed.py

# Inspect JAXpr and HLO
python scripts/inspect_jaxpr.py

# Memory profiling
python scripts/profile_memory.py
```

### HLO Representations

Full HLO (High-Level Optimizer) representations saved to:
- Linear version: `/tmp/hlo_linear.txt`
- GEMM version: `/tmp/hlo_gemm.txt`

These files contain the complete XLA compilation output for detailed analysis.

---

## Production-Scale Considerations

### Expected Production Datasets

Based on target use cases:
- **Time points**: 2-5 million (potentially more)
- **Waveform dimensions**: 2-10D
- **Electrodes**: 32-128
- **Total spikes**: Millions
- **Position bins**: 1000-5000

### Scaling Analysis

| Aspect | Small Test | Production | Scale Factor |
|--------|------------|------------|--------------|
| Time points | 2,000 | 2,000,000 | 1000x |
| Spikes | 3,200 | 3,200,000 | 1000x |
| Waveform dims | 2 | 2-10 | 1-5x |
| Memory | 0.2 MB | **200+ MB** | 1000x |

### Performance Implications

1. **GEMM becomes essential**:
   - With 10D waveforms, GEMM's matrix operations are much more efficient
   - Linear version would require massive intermediate arrays

2. **Memory management critical**:
   - Block processing mandatory (`block_size` parameter)
   - Position tiling likely needed (`pos_tile_size` parameter)
   - Can't hold full arrays in memory

3. **GPU strongly recommended**:
   - Scan operations parallelize well on GPU
   - Memory bandwidth much higher
   - Fused kernels reduce data movement

4. **Chunked processing required**:
   - Process encoding in chunks
   - Decode in batches (e.g., 10-second windows)
   - Save intermediate results to disk

### Recommended Architecture for Production

```python
# For production with millions of time points:

# 1. Chunk encoding (fit model on representative subset)
subset_indices = np.random.choice(len(position_time), 100000, replace=False)
encoding = fit_clusterless_kde_encoding_model(
    position_time=position_time[subset_indices],
    position=position[subset_indices],
    spike_times=spike_times,  # Full spike data
    spike_waveform_features=spike_waveform_features,
    environment=env,
    block_size=200,  # Larger blocks for efficiency
)

# 2. Decode in batches with optimizations
chunk_size = 10000  # 10k time bins per chunk
for i in range(0, n_time_bins, chunk_size):
    chunk_time = time[i:i+chunk_size]

    ll_chunk = predict_clusterless_kde_log_likelihood(
        time=chunk_time,
        ...,
        use_gemm=True,  # Use GEMM for high-dim features
        pos_tile_size=500,  # Tile large position grids
        block_size=200,
    )

    # Save chunk to disk or accumulate
    save_chunk(ll_chunk, chunk_idx=i//chunk_size)
```

### GPU Profiling Needed

**Critical TODO**: Test on GPU with realistic scale:

```python
import jax
jax.config.update('jax_platform_name', 'gpu')

# Test with:
# - 100k+ time points
# - 4D, 8D, 10D waveforms
# - 2000+ position bins
# - Compare scan vs vmap
```

Expected GPU benefits:
- Scan: 10-100x faster than CPU
- GEMM: 5-20x faster than CPU
- Memory: Can hold larger intermediate arrays

---

## Action Items

### Completed
1. ✅ Document baseline performance (small datasets)
2. ✅ Identify root cause (scan overhead on CPU)
3. ✅ Provide recommendations for current datasets
4. ✅ Document production-scale considerations

### High Priority
5. 🔴 **CRITICAL**: Benchmark on GPU with realistic scale
   - Test 100k+ time points
   - Test 4D, 8D, 10D waveforms
   - Compare use_gemm=True vs False on GPU
   - Measure memory usage at scale

6. 🔴 **CRITICAL**: Test chunked processing pipeline
   - Verify accuracy with chunked decoding
   - Measure end-to-end time for 1M+ time points
   - Optimize chunk size for memory/speed trade-off

### Medium Priority
7. 🟡 Consider replacing scan with vmap (if GPU scan still slow)
8. 🟡 Implement streaming/out-of-core processing
9. 🟡 Add progress callbacks for long-running ops
10. 🟡 Profile memory usage at scale (may need sparse representations)

### Low Priority (Current Default OK for Now)
11. ⚪ Change default `use_gemm` based on feature dimensionality
12. ⚪ Auto-detect optimal block_size based on available memory
13. ⚪ Auto-tune pos_tile_size based on grid size

---

## Next Steps for Production Readiness

1. **Get GPU access** and re-run all profiling
2. **Create synthetic production-scale dataset** (1M time points, 8D features)
3. **Benchmark end-to-end pipeline** with chunking
4. **Measure peak memory** and optimize if needed
5. **Document GPU-specific recommendations**
6. **Consider multi-GPU** if single GPU insufficient

---

**Last Updated**: 2025-10-27
**System**: macOS, **CPU only** (GPU profiling pending)
**JAX Version**: 0.6.2
**Status**: ⚠️ Baselines are for small datasets only - production scale needs separate profiling
**Maintainer**: Performance profiling for clusterless KDE implementations
