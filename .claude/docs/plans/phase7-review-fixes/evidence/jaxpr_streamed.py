import jax, jax.numpy as jnp
from non_local_detector.likelihoods.streamed_kde import _joint_core

def subs(eqn):
    for v in eqn.params.values():
        for it in (v if isinstance(v, (tuple, list)) else (v,)):
            if hasattr(it, "jaxpr") and hasattr(it.jaxpr, "eqns"): yield it.jaxpr
            elif hasattr(it, "eqns"): yield it

def walk(jx, path=()):
    for e in jx.eqns:
        name = e.primitive.name
        if name in ("scan", "while"):
            trip = e.params.get("length", "?")
            p = path + (f"{name}[{trip}]",)
        else:
            p = path
        if name in ("exp", "dot_general"):
            shapes = [list(v.aval.shape) for v in e.invars]
            print(f"  {' > '.join(p) or '(top)':45s} {name:11s} in={shapes}")
        for s in subs(e):
            walk(s, p)

f32 = jnp.float32
n_dec, n_enc, n_pos, n_marks = 64, 400, 2000, 4
args = (jnp.ones((n_dec, n_marks), f32), jnp.ones((n_enc, n_marks), f32), jnp.ones((n_enc, 1), f32),
        jnp.ones((n_pos, 1), f32), jnp.ones(n_marks, f32), jnp.ones(1, f32), jnp.ones(n_pos, f32),
        f32(1.0), jnp.ones(n_enc, f32))
print(f"n_dec={n_dec} n_enc={n_enc} n_pos={n_pos}; tiles: encoding=100 position=64 decoding=32")
jx = jax.make_jaxpr(lambda *a: _joint_core(*a, encoding_tile_size=100, position_tile_size=64, decoding_tile_size=32))(*args)
walk(jx.jaxpr)
