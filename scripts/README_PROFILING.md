# Profiling Clusterless KDE Implementations

This directory contains profiling scripts to analyze and compare the performance of `clusterless_kde.py` (reference) and `clusterless_kde_log.py` (optimized) implementations.

## Quick Start

```bash
# Basic performance comparison
python scripts/profile_clusterless_kde.py --size medium

# Detailed JAX profiling
python scripts/profile_jax_detailed.py

# Memory profiling
python scripts/profile_memory.py
```

## Scripts Overview

### 1. `profile_clusterless_kde.py` - Main Performance Profiling

**Purpose**: Compare execution time between reference and optimized implementations.

**Usage**:
```bash
# Quick test
python scripts/profile_clusterless_kde.py --size tiny

# Medium dataset (recommended)
python scripts/profile_clusterless_kde.py --size medium

# Large realistic dataset
python scripts/profile_clusterless_kde.py --size realistic

# Use GPU if available
python scripts/profile_clusterless_kde.py --size medium --device gpu
```

**Dataset Sizes**:
- `tiny`: Fast test (100 time points, 4 electrodes)
- `small`: Small experiment (500 time points, 8 electrodes)
- `medium`: **Recommended** (2000 time points, 16 electrodes)
- `large`: Large experiment (5000 time points, 32 electrodes)
- `realistic`: Full-scale (10000 time points, 64 electrodes)

**Outputs**:
- Encoding time (model fitting)
- Decoding time (likelihood prediction)
- Total time and speedup
- Numerical agreement verification

### 2. `profile_jax_detailed.py` - JAX-Specific Analysis

**Purpose**: Analyze JAX compilation, caching, and optimization strategies.

**Usage**:
```bash
python scripts/profile_jax_detailed.py
```

**Analyses**:
- **Compilation vs Execution**: Shows JIT compilation overhead
- **Memory Usage**: Estimates for different dataset sizes
- **Optimization Strategies**: Compares GEMM, tiling, etc.
- **Compilation Caching**: Demonstrates JAX function caching

**Key Insights**:
- First call includes compilation (~100-1000ms)
- Subsequent calls are cached (~1-10ms)
- Changing input shapes triggers recompilation
- Keep consistent shapes for best performance

### 3. `profile_memory.py` - Memory Usage Analysis

**Purpose**: Track memory consumption over time.

**Requirements**:
```bash
pip install memory_profiler matplotlib
```

**Usage**:
```bash
# Basic profiling
python scripts/profile_memory.py

# Line-by-line profiling
python -m memory_profiler scripts/profile_memory.py

# Generate plot
mprof run scripts/profile_memory.py
mprof plot
```

**Outputs**:
- Memory requirements estimation table
- Detailed component breakdown
- Memory optimization tips

## Performance Optimization Guide

### Best Practices

1. **JIT Compilation Warmup**
   ```python
   # Warmup: compile functions with representative inputs
   for _ in range(2):
       result = predict_func(...)
       result.block_until_ready()

   # Now time actual runs
   start = time.perf_counter()
   result = predict_func(...)
   result.block_until_ready()
   elapsed = time.perf_counter() - start
   ```

2. **Keep Input Shapes Consistent**
   - JAX compiles functions per unique input shape
   - Changing shapes = recompilation overhead
   - Batch process data with same dimensions

3. **Use Optimizations Appropriately**
   ```python
   # For small datasets or debugging: use_gemm=False
   result = predict_clusterless_kde_log_likelihood(..., use_gemm=False)

   # For production (multi-dim features): use_gemm=True (default)
   result = predict_clusterless_kde_log_likelihood(..., use_gemm=True)

   # For very large grids: add position tiling
   result = predict_clusterless_kde_log_likelihood(..., pos_tile_size=500)
   ```

4. **Block Size Tuning**
   ```python
   # Memory-constrained: smaller blocks
   block_size = 50

   # Memory-abundant: larger blocks (faster)
   block_size = 200
   ```

### When to Use Each Implementation

| Scenario | Recommendation |
|----------|---------------|
| Development/Testing | `clusterless_kde.py` (reference) |
| Debugging numerical issues | `use_gemm=False` |
| Production (2D+ features) | `use_gemm=True` (default) |
| Very large grids (>2000 bins) | `use_gemm=True, pos_tile_size=500` |
| GPU available | Enable with `jax.config.update('jax_platform_name', 'gpu')` |

### Expected Performance

Typical speedups (optimized vs reference):

| Dataset Size | Speedup | Notes |
|--------------|---------|-------|
| Tiny (4 elec, 100 bins) | 0.8-1.2x | Overhead dominates |
| Medium (16 elec, 500 bins) | 1.5-2.5x | Sweet spot |
| Large (32 elec, 1000 bins) | 2-4x | GEMM shines |
| Realistic (64 elec, 2000 bins) | 3-5x | Full optimization |

*Note: Speedups vary by hardware (CPU vs GPU) and dataset characteristics.*

## Advanced Profiling

### TensorBoard Profiling

```python
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/tensorboard")

# Run code
result = predict_clusterless_kde_log_likelihood(...)

# Stop profiling
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

### HLO Analysis

```python
from jax import jit

# Get HLO (High-Level Optimizer) representation
jitted_func = jit(estimate_log_joint_mark_intensity)
lowered = jitted_func.lower(
    decoding_features,
    encoding_features,
    waveform_std,
    occupancy,
    mean_rate,
    position_distance,
)

# Print HLO
print(lowered.as_text())

# Compile and inspect
compiled = lowered.compile()
print(compiled.as_text())
```

### GPU Profiling

```python
import jax

# Configure for GPU
jax.config.update('jax_platform_name', 'gpu')

# Check devices
print(jax.devices())  # Should show GPU

# Profile GPU memory
memory_stats = jax.profiler.device_memory_profile()
for device, stats in memory_stats.items():
    print(f"{device}: {stats.bytes_in_use / 1024**2:.1f} MB")
```

## Troubleshooting

### Out of Memory Errors

1. Reduce `block_size`: `block_size=50`
2. Enable position tiling: `pos_tile_size=200`
3. Process electrodes in batches
4. Reduce position grid resolution if acceptable

### Slow Compilation

1. Warm up functions once at startup
2. Keep input shapes consistent
3. Use static arguments where possible
4. Consider `@partial(jit, static_argnums=(...))`

### Numerical Differences

Expected differences between implementations:
- Float32 precision: ~1e-7
- Use `np.allclose(result1, result2, rtol=1e-4, atol=1e-5)` for comparison
- Log-space vs linear-space: minor rounding differences are normal

## Further Reading

- [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)
- [memory_profiler Documentation](https://pypi.org/project/memory-profiler/)
- [JAX Performance Tips](https://jax.readthedocs.io/en/latest/faq.html#performance-tips)
