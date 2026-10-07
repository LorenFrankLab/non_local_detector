"""pytest plugin: restore the pre-79156376 log_gaussian_pdf (no optimization_barrier)."""
import jax
import jax.numpy as jnp

import non_local_detector.likelihoods.clusterless_kde_log as kde_log
import non_local_detector.likelihoods.common as common


@jax.jit
def log_gaussian_pdf(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


common.log_gaussian_pdf = log_gaussian_pdf
kde_log.log_gaussian_pdf = log_gaussian_pdf
print(f"[nobarrier] jax {jax.__version__}")
