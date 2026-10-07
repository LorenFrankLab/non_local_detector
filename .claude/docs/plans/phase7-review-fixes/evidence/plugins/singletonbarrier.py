"""pytest plugin: keep the barrier only for singleton tiles (drop the mixed-precision trigger)."""
import jax
import jax.numpy as jnp

import non_local_detector.likelihoods.clusterless_kde_log as kde_log
import non_local_detector.likelihoods.common as common


@jax.jit
def log_gaussian_pdf(x, mean, sigma):
    divisor = sigma
    if x.size == 1 or mean.size == 1:
        shape = jnp.broadcast_shapes(x.shape, mean.shape, jnp.shape(sigma))
        divisor = jax.lax.optimization_barrier(jnp.broadcast_to(sigma, shape))
    return -0.5 * ((x - mean) / divisor) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


common.log_gaussian_pdf = log_gaussian_pdf
kde_log.log_gaussian_pdf = log_gaussian_pdf
print(f"[singletonbarrier] jax {jax.__version__}")
