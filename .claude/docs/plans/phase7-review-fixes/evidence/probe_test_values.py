"""Print bin values of the streamed CI case under production, plain division, explicit reciprocal."""
import os, sys
import jax, jax.numpy as jnp, numpy as np
import non_local_detector.likelihoods.common as common
import non_local_detector.likelihoods.clusterless_kde_log as kde_log
from non_local_detector import Environment
from non_local_detector.tests.likelihoods.test_clusterless_kde_streaming import data, fit

env = Environment(environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),))
env = env.fit_place_grid(position=np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False)
mode = sys.argv[1]
if mode == "division":
    g = jax.jit(lambda x, m, s: -0.5 * ((x - m) / s) ** 2 - jnp.log(s * jnp.sqrt(2.0 * jnp.pi)))
elif mode == "reciprocal":
    g = jax.jit(lambda x, m, s: -0.5 * ((x - m) * (1 / s)) ** 2 - jnp.log(s * jnp.sqrt(2.0 * jnp.pi)))
if mode != "production":
    common.log_gaussian_pdf = g; kde_log.log_gaussian_pdf = g
v = data(np.float32, "uniform", False)
ref = fit(env, v); st = fit(env, v, encoding_block_size=4, position_block_size=5)
k = "summed_ground_process_intensity"
r, s = np.asarray(ref[k]), np.asarray(st[k])
print(f"{mode:10s} ref[9]={r[9]!r} streamed[9]={s[9]!r} ok={np.allclose(s, r, rtol=1e-6, atol=1e-5)} maxabs={np.abs(s-r).max():.3e}")
