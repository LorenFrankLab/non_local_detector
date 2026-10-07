import re, jax, jax.numpy as jnp
import non_local_detector.likelihoods.common as common

@jax.jit
def nobarrier(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))

def km(g):
    def f(ev, sa, sd):
        lk = jnp.zeros((sa.shape[0], ev.shape[0]))
        for de, ds, s in zip(ev.T, sa.T, sd, strict=True):
            lk += g(jnp.expand_dims(de, 0), jnp.expand_dims(ds, 1), s)
        return lk
    return jax.jit(f)

f32 = jnp.float32
print("x64", jax.config.x64_enabled)
for label, ne, ns in (("non-singleton", 64, 2000), ("singleton-sample", 64, 1)):
    for name, g in (("current", common.log_gaussian_pdf), ("no-barrier", nobarrier)):
        txt = km(g).lower(jnp.ones((ne, 2), f32), jnp.ones((ns, 2), f32), jnp.ones(2, f32)).compile().as_text()
        divs = sorted({re.search(r"= (f\d+\[[\d,]*\])", l).group(1) for l in txt.splitlines() if " divide(" in l})
        print(f"  {label:17s} {name:10s} divide result shapes: {divs}")
