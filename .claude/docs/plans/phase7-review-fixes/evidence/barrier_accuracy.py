"""Streamed-fit CI case: true division vs explicit reciprocal vs float64 oracle.

Reproduces test_streamed_fit_and_local_nonlocal_rows_preserve_original_support
[False-uniform-False] quantities with two log_gaussian_pdf variants:
- division:   (x - mean) / sigma           (what arm64 XLA emits, with or without barrier)
- reciprocal: (x - mean) * (1 / sigma)      (the rewrite the barrier comment attributes to x86)
and a float64 oracle computed from the same float32 inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np

import non_local_detector.likelihoods.clusterless_kde_log as kde_log
import non_local_detector.likelihoods.common as common
from non_local_detector import Environment
from non_local_detector.tests.likelihoods.test_clusterless_kde_streaming import data, fit

env = Environment(environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),))
env = env.fit_place_grid(position=np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False)


def division(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


def reciprocal(x, mean, sigma):
    return -0.5 * ((x - mean) * (1 / sigma)) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


KEYS = ["occupancy", "summed_ground_process_intensity"]


def run(variant, x64):
    jax.config.update("jax_enable_x64", x64)
    g = jax.jit(variant)
    common.log_gaussian_pdf = g
    kde_log.log_gaussian_pdf = g
    jax.clear_caches()
    values = data(np.float32, "uniform", False)
    if x64:
        values = {k: (v.astype(np.float64) if isinstance(v, np.ndarray) and v.dtype == np.float32
                      else [a.astype(np.float64) for a in v] if isinstance(v, list) else v)
                  for k, v in values.items()}
    ref = fit(env, values)
    streamed = fit(env, values, encoding_block_size=4, position_block_size=5)
    return {k: (np.asarray(ref[k], np.float64), np.asarray(streamed[k], np.float64)) for k in KEYS}


oracle = run(division, True)
for k in KEYS:
    assert np.allclose(oracle[k][0], oracle[k][1], rtol=1e-12, atol=1e-12)
out = {name: run(fn, False) for name, fn in (("division", division), ("reciprocal", reciprocal))}

for k in KEYS:
    truth = oracle[k][0]
    print(f"== {k}  (oracle range {truth.min():.4g}..{truth.max():.4g})")
    for name, res in out.items():
        for path, arr in zip(("reference", "streamed"), res[k]):
            rel = np.abs(arr - truth) / np.abs(truth)
            print(f"  {name:10s} {path:9s} max rel err vs float64 = {np.nanmax(rel):.2e}  (worst bin {int(np.nanargmax(rel))})")
        d = np.abs(res[k][1] - res[k][0])
        print(f"  {name:10s} streamed-vs-reference max abs = {d.max():.3e}, rel = {np.nanmax(d / np.abs(res[k][0])):.2e}")
    cross = np.abs(out["division"][k][0] - out["reciprocal"][k][1])
    print(f"  division-reference vs reciprocal-streamed (the x86 CI mix): max abs = {cross.max():.3e}, "
          f"rel = {np.nanmax(cross / np.abs(out['division'][k][0])):.2e}; "
          f"test allows atol 1e-5 + rtol 1e-6*|x|")
