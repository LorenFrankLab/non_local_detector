"""Inspect jaxprs / optimized HLO for the ordering and barrier changes.

Usage: [JAX_ENABLE_X64=1] uv run python inspect_jaxprs.py PREFIX_COMMON_PY
"""

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import non_local_detector.likelihoods.common as common

X64 = jax.config.x64_enabled
print(f"jax {jax.__version__}  backend={jax.default_backend()}  x64={X64}")

prefix = Path(sys.argv[1]).read_text()
ns = dict(common.__dict__)
for name in ("_poisson_nonlocal_log_likelihood",):
    node = next(n for n in ast.parse(prefix).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(ast.get_source_segment(prefix, node), ns)
prefix_matrix = jax.jit(ns["_poisson_nonlocal_log_likelihood"])


@jax.jit
def log_gaussian_pdf_nobarrier(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


def kernel_matrix(gauss):
    def f(eval_points, samples, std):
        log_kernel = jnp.zeros((samples.shape[0], eval_points.shape[0]))
        for de, ds, sd in zip(eval_points.T, samples.T, std, strict=True):
            log_kernel += gauss(jnp.expand_dims(de, 0), jnp.expand_dims(ds, 1), sd)
        return log_kernel
    return jax.jit(f)



def _subjaxprs(eqn):
    for v in eqn.params.values():
        for item in (v if isinstance(v, (tuple, list)) else (v,)):
            if hasattr(item, "jaxpr") and hasattr(item.jaxpr, "eqns"):
                yield item.jaxpr
            elif hasattr(item, "eqns"):
                yield item

def prims(jaxpr, out=None):
    out = Counter() if out is None else out
    for eqn in jaxpr.eqns:
        out[eqn.primitive.name] += 1
        for sub in _subjaxprs(eqn):
            prims(sub, out)
    return out


def barrier_operands(jaxpr, found=None):
    found = [] if found is None else found
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "optimization_barrier":
            found += [f"{v.aval.dtype}{list(v.aval.shape)}" for v in eqn.invars]
        for sub in _subjaxprs(eqn):
            barrier_operands(sub, found)
    return found


def hlo_ops(compiled_text, names=("divide", "multiply", "while", "scatter", "opt-barrier", "dynamic-update-slice", "dot", "fusion")):
    return {n: len(re.findall(rf"\b{n}\(", compiled_text)) for n in names}


f32 = jnp.float32
print("\n### 1. log_gaussian_pdf barrier inside _log_kernel_matrix (2 mark dims)")
cases = {
    "tile (64 eval x 2000 samples)": (64, 2000),
    "singleton sample tail (64 x 1)": (64, 1),
    "big (1000 eval x 20000 samples)": (1000, 20000),
}
for label, (n_eval, n_samp) in cases.items():
    ev = jnp.ones((n_eval, 2), f32)
    sa = jnp.ones((n_samp, 2), f32)
    sd = jnp.ones(2, f32)
    for gname, g in (("current", common.log_gaussian_pdf), ("no-barrier", log_gaussian_pdf_nobarrier)):
        fn = kernel_matrix(g)
        jx = jax.make_jaxpr(fn)(ev, sa, sd)
        comp = fn.lower(ev, sa, sd).compile()
        mem = comp.memory_analysis()
        temp = getattr(mem, "temp_size_in_bytes", None)
        out = getattr(mem, "output_size_in_bytes", None)
        print(f"  {label:34s} {gname:10s} barrier operands={barrier_operands(jx.jaxpr) or '-'}"
              f"  temp={temp/1e6 if temp is not None else '?':.1f}MB out={out/1e6:.1f}MB"
              f"  HLO={hlo_ops(comp.as_text(), ('divide', 'multiply', 'opt-barrier', 'fusion'))}")

print("\n  Optimized HLO of x/sigma (no barrier), non-singleton vs singleton:")
for shape_x, shape_m in (((1, 64), (2000, 1)), ((1, 64), (1, 1))):
    fn = jax.jit(lambda x, m, s: (x - m) / s)
    txt = fn.lower(jnp.ones(shape_x, f32), jnp.ones(shape_m, f32), f32(0.7)).compile().as_text()
    lines = [l.strip() for l in txt.splitlines() if re.search(r"\b(divide|multiply)\(", l)]
    print(f"   x{shape_x} m{shape_m}:", lines or "no divide/multiply")

print("\n### 2. Spike row sums: ordered loop vs segment_sum (values (n_spikes, 500))")
for n in (50, 20000):
    vals = jnp.ones((n, 500), f32)
    idx = jnp.asarray(np.sort(np.random.default_rng(0).integers(0, 256, n)).astype(np.int32))
    ordered = jax.jit(lambda v, i: common._ordered_spike_row_sum(v, i, 256))
    seg = jax.jit(lambda v, i: jax.ops.segment_sum(v, i, num_segments=256, indices_are_sorted=True))
    for name, fn in (("ordered", ordered), ("segment_sum", seg)):
        jx = jax.make_jaxpr(fn)(vals, idx)
        c = prims(jx.jaxpr)
        trips = [str(e.params.get("cond_jaxpr", "")) and e.params for e in jx.jaxpr.eqns if e.primitive.name == "while"]
        comp = fn.lower(vals, idx).compile()
        print(f"  n_spikes={n:6d} {name:11s} prims={dict(c.most_common(8))}  HLO={hlo_ops(comp.as_text(), ('while', 'scatter', 'dynamic-update-slice', 'dynamic-slice'))}")
    if n == 50:
        print("  ordered jaxpr:")
        print("   " + str(jax.make_jaxpr(ordered)(vals, idx)).replace("\n", "\n   ")[:3000])

print("\n### 3. Sorted emissions: current ordered packed scan vs pre-fix matrix")
n_rows, n_neurons, n_bins = 256, 96, 1000
rng = np.random.default_rng(1)
counts = (rng.random((n_rows, n_neurons)) < 0.05).astype(np.int32)
rates = jnp.asarray(rng.uniform(0.1, 20, (n_neurons, n_bins)), f32)
dur = jnp.full(n_rows, 0.002, f32)
summed = rates.sum(0)
cur = jax.make_jaxpr(lambda c, r, d, s: common._poisson_nonlocal_log_likelihood(c, r, d, s))
active = np.count_nonzero(counts, 1)
cap = min(1 << (int(active.max()) - 1).bit_length(), n_neurons)
sel_r, sel_n = np.nonzero(counts)
slots = np.arange(len(sel_r)) - (np.cumsum(active) - active)[sel_r]
ids = np.full((n_rows, cap), n_neurons, np.int32); ev = np.zeros((n_rows, cap), np.int32)
ids[sel_r, slots] = sel_n; ev[sel_r, slots] = counts[sel_r, sel_n]
packed = common._poisson_packed_log_likelihood
jp = jax.make_jaxpr(packed)(ids, ev, rates, dur, summed)
jm = jax.make_jaxpr(prefix_matrix)(jnp.asarray(counts), rates, dur, summed)
print(f"  capacity={cap}")
print(f"  ordered packed prims={dict(prims(jp.jaxpr).most_common(12))}")
print(f"  pre-fix matrix prims={dict(prims(jm.jaxpr).most_common(12))}")
for name, fn, args in (("ordered packed", packed, (ids, ev, rates, dur, summed)), ("pre-fix matrix", prefix_matrix, (jnp.asarray(counts), rates, dur, summed))):
    comp = fn.lower(*args).compile()
    mem = comp.memory_analysis()
    print(f"  {name:15s} HLO={hlo_ops(comp.as_text(), ('while', 'dot', 'opt-barrier', 'fusion', 'log'))} temp={mem.temp_size_in_bytes/1e6:.2f}MB")
