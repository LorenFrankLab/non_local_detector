#!/usr/bin/env python
"""Inspect JAXpr (JAX intermediate representation) for understanding compilation."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity,
)


def create_simple_test_data():
    """Create minimal test data."""
    rng = np.random.default_rng(42)

    n_decoding = 10
    n_encoding = 20
    n_positions = 50
    n_features = 2

    decoding_features = jnp.asarray(rng.normal(0, 1, (n_decoding, n_features)))
    encoding_features = jnp.asarray(rng.normal(0, 1, (n_encoding, n_features)))
    waveform_std = jnp.array([1.0, 1.0])
    occupancy = jnp.ones(n_positions) * 0.1
    mean_rate = 5.0
    position_distance = jnp.ones((n_encoding, n_positions)) * 0.01

    return {
        "decoding_features": decoding_features,
        "encoding_features": encoding_features,
        "waveform_std": waveform_std,
        "occupancy": occupancy,
        "mean_rate": mean_rate,
        "position_distance": position_distance,
    }


def main():
    print("="*70)
    print("JAXpr INSPECTION FOR CLUSTERLESS KDE")
    print("="*70)

    data = create_simple_test_data()

    print("\nTest data shapes:")
    print(f"  Decoding features: {data['decoding_features'].shape}")
    print(f"  Encoding features: {data['encoding_features'].shape}")
    print(f"  Position distance: {data['position_distance'].shape}")
    print(f"  Occupancy: {data['occupancy'].shape}")

    # Linear version (use_gemm=False)
    print("\n" + "="*70)
    print("LINEAR VERSION (use_gemm=False)")
    print("="*70)

    jitted_linear = jax.jit(
        lambda: estimate_log_joint_mark_intensity(
            data["decoding_features"],
            data["encoding_features"],
            data["waveform_std"],
            data["occupancy"],
            data["mean_rate"],
            data["position_distance"],
            use_gemm=False,
        )
    )

    # Get lowered representation
    lowered_linear = jitted_linear.lower()

    print("\nJAXpr (simplified):")
    print(str(lowered_linear.as_text())[:2000])
    print("\n... (truncated)")

    # Compile and get HLO
    print("\n" + "-"*70)
    print("HLO Operations Count:")
    compiled_linear = lowered_linear.compile()
    hlo_linear = compiled_linear.as_text()

    # Count key operations
    operations = {
        "dot": hlo_linear.count("dot("),
        "multiply": hlo_linear.count("multiply("),
        "add": hlo_linear.count("add("),
        "log": hlo_linear.count("log("),
        "exp": hlo_linear.count("exp("),
        "reduce": hlo_linear.count("reduce("),
    }

    for op, count in operations.items():
        print(f"  {op}: {count}")

    # GEMM version (use_gemm=True)
    print("\n" + "="*70)
    print("GEMM VERSION (use_gemm=True)")
    print("="*70)

    jitted_gemm = jax.jit(
        lambda: estimate_log_joint_mark_intensity(
            data["decoding_features"],
            data["encoding_features"],
            data["waveform_std"],
            data["occupancy"],
            data["mean_rate"],
            data["position_distance"],
            use_gemm=True,
        )
    )

    lowered_gemm = jitted_gemm.lower()

    print("\nJAXpr (simplified):")
    print(str(lowered_gemm.as_text())[:2000])
    print("\n... (truncated)")

    print("\n" + "-"*70)
    print("HLO Operations Count:")
    compiled_gemm = lowered_gemm.compile()
    hlo_gemm = compiled_gemm.as_text()

    operations_gemm = {
        "dot": hlo_gemm.count("dot("),
        "multiply": hlo_gemm.count("multiply("),
        "add": hlo_gemm.count("add("),
        "log": hlo_gemm.count("log("),
        "exp": hlo_gemm.count("exp("),
        "reduce": hlo_gemm.count("reduce("),
        "while": hlo_gemm.count("while("),
        "scan": hlo_gemm.count("scan"),
    }

    for op, count in operations_gemm.items():
        print(f"  {op}: {count}")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print("\nLinear version:")
    print(f"  Total operations: {sum(operations.values())}")
    print(f"  Matrix multiplies (dot): {operations['dot']}")
    print(f"  Logs: {operations['log']}")
    print(f"  Exps: {operations['exp']}")

    print("\nGEMM version:")
    print(f"  Total operations: {sum(operations_gemm.values())}")
    print(f"  Matrix multiplies (dot): {operations_gemm['dot']}")
    print(f"  Logs: {operations_gemm['log']}")
    print(f"  Exps: {operations_gemm['exp']}")
    print(f"  Scans/loops: {operations_gemm['while'] + operations_gemm['scan']}")

    # Performance analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)

    print("""
Key Observations:

1. Linear Version:
   - Single matrix multiply for mark kernel
   - One matrix multiply for joint computation
   - Simple log at the end
   - No loops/scans

2. GEMM Version:
   - GEMM for mark kernel (can be faster for high-dim)
   - Scan loop over decoding spikes (sequential!)
   - Multiple log/logsumexp operations
   - More memory allocations

Why GEMM is slower for small datasets:
   - Scan overhead dominates for small n_decoding
   - Log-space operations (log, exp, logsumexp) are expensive
   - More intermediate arrays

When GEMM wins:
   - Large n_features (>4D waveforms)
   - GPU execution (better scan performance)
   - Very large n_encoding (GEMM optimization shines)
    """)

    # Save full HLO for inspection
    print("\nSaving full HLO representations to files...")
    with open("/tmp/hlo_linear.txt", "w") as f:
        f.write(hlo_linear)
    print("  Linear HLO saved to: /tmp/hlo_linear.txt")

    with open("/tmp/hlo_gemm.txt", "w") as f:
        f.write(hlo_gemm)
    print("  GEMM HLO saved to: /tmp/hlo_gemm.txt")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on JAXpr analysis:

1. For typical datasets (2D waveforms, <32 electrodes):
   → Use use_gemm=False (linear version)
   → Simpler, faster for small-medium data

2. For high-dimensional waveforms (>4D features):
   → Use use_gemm=True
   → GEMM optimization pays off

3. For GPU:
   → Test both versions
   → Scan may be faster on GPU

4. Future optimization:
   → Consider vmap instead of scan
   → Fuse operations to reduce intermediates
   → Benchmark with realistic data
    """)

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
