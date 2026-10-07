"""Is the ordered sorted-spike emission closer to the exact sum than the matrix one?

Reproduces the CI-failing scenario (96 neurons all firing in rows 100 and 512),
float32 arithmetic, and compares three emission implementations against a
float64 oracle built from the *same* float32 inputs:

- ordered: current production (_poisson_nonlocal_log_likelihood)
- matrix:  pre-fix 73813467 matmul accumulation
- oracle:  numpy float64 xlogy sum, rounded once to float32

Usage: uv run python order_accuracy.py PREFIX_COMMON_PY [nonlocal 0|1]
"""

import ast
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import xlogy

import non_local_detector.likelihoods.common as common
import non_local_detector.likelihoods.sorted_spikes_kde as skde
from non_local_detector import Environment, NonLocalSortedSpikesDetector, SortedSpikesDecoder
from non_local_detector.continuous_state_transitions import Uniform

assert not jax.config.x64_enabled

prefix_path, nonlocal_model = Path(sys.argv[1]), bool(int(sys.argv[2]))
source = prefix_path.read_text()
node = next(
    n for n in ast.parse(source).body
    if isinstance(n, ast.FunctionDef) and n.name == "_poisson_nonlocal_log_likelihood"
)
namespace = dict(common.__dict__)
exec(ast.get_source_segment(source, node), namespace)
matrix_fn = jax.jit(namespace["_poisson_nonlocal_log_likelihood"])
ordered_fn = common._poisson_nonlocal_log_likelihood

captured = {}


def oracle_fn(counts, rates, durations, summed_rates):
    c = np.asarray(counts, np.float64)
    r = np.asarray(rates, np.float64)
    d = np.asarray(durations, np.float64)
    s = np.asarray(summed_rates, np.float64)
    ll = np.zeros((c.shape[0], r.shape[1]))
    for n in range(c.shape[1]):
        ll += xlogy(c[:, n][:, None], r[n][None, :] * d[:, None])
    ll -= d[:, None] * s
    return jnp.asarray(ll.astype(np.float32))


def recording(name, fn):
    def wrapped(counts, rates, durations, summed_rates):
        out = fn(counts, rates, durations, summed_rates)
        captured.setdefault(name, []).append(
            (np.asarray(counts), np.asarray(rates), np.asarray(durations),
             np.asarray(summed_rates), np.asarray(out))
        )
        return out
    return wrapped


# Scenario copied from test_many_neurons_match_native_posterior_and_evidence.
rng = np.random.default_rng(7303)
position_time = np.linspace(0, 4, 501)
position = (10 + 8 * np.sin(position_time * 2))[:, None]
cls = NonLocalSortedSpikesDetector if nonlocal_model else SortedSpikesDecoder
model = cls(
    environments=Environment(place_bin_size=2, position_range=((0, 20),)),
    continuous_transition_types=None if nonlocal_model else [[Uniform()]],
    infer_track_interior=False,
    sorted_spikes_algorithm_params={"position_std": 3, "block_size": 100, "disable_progress_bar": True},
)
training_spikes = [np.sort(rng.uniform(0, 4, 12)) for _ in range(96)]
model.fit(position_time, position, training_spikes)
edges = np.arange(514) * 0.002
spikes = [np.sort(np.r_[rng.uniform(0, edges[-1], 4), edges[100], edges[-1]]) for _ in range(96)]
missing = np.zeros(513, bool)
missing[240:250] = True
args = dict(time_edges=edges, position_time=position_time, position=position,
            is_missing=missing, return_outputs="all")

results = {}
for name, fn in [("oracle", oracle_fn), ("ordered", ordered_fn), ("matrix", matrix_fn)]:
    skde._poisson_nonlocal_log_likelihood = recording(name, fn)
    results[name] = model.predict(spikes, **args)
skde._poisson_nonlocal_log_likelihood = ordered_fn

# 1. Emission error vs float64 sum of the same float32 inputs (unrounded oracle).
print(f"== nonlocal_model={nonlocal_model}")
print("emission calls captured:", {k: len(v) for k, v in captured.items()})
for name in ("ordered", "matrix", "oracle"):
    errs, ulps = [], []
    for counts, rates, d, s, out in captured[name]:
        c, r = counts.astype(np.float64), rates.astype(np.float64)
        dd = d.astype(np.float64)
        exact = sum(xlogy(c[:, n][:, None], r[n][None, :] * dd[:, None]) for n in range(c.shape[1]))
        exact = exact - dd[:, None] * s.astype(np.float64)
        err = np.abs(out.astype(np.float64) - exact)
        errs.append(err)
        ulps.append(err / np.spacing(np.abs(exact).astype(np.float32)).astype(np.float64))
    err = np.concatenate([e.ravel() for e in errs])
    ulp = np.concatenate([u.ravel() for u in ulps])
    print(f"  LL  {name:8s} max|err|={err.max():.3e}  mean|err|={err.mean():.3e}  max ulp={ulp.max():.1f}  mean ulp={ulp.mean():.3f}")

# 2. Downstream: posterior/state output differences vs the oracle-emission run.
for name in ("ordered", "matrix"):
    worst = {}
    for var in results["oracle"].data_vars:
        a = np.asarray(results[name][var], np.float64)
        b = np.asarray(results["oracle"][var], np.float64)
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            worst[var] = np.nanmax(np.abs(a - b))
    evid = abs(results[name].attrs["marginal_log_likelihoods"] - results["oracle"].attrs["marginal_log_likelihoods"])
    print(f"  OUT {name:8s} vs oracle: " + ", ".join(f"{k}={v:.2e}" for k, v in worst.items()) + f", evidence={np.max(evid):.2e}")
a = results["ordered"]; b = results["matrix"]
print("  OUT ordered vs matrix: " + ", ".join(
    f"{k}={np.nanmax(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64))):.2e}" for k in a.data_vars))

# 3. Reproduce CI's assert_allclose(actual, expected, rtol=1e-6, atol=1e-6) report,
# with actual=oracle-emission run and expected=ordered (== per-neuron order) run.
for actual_name, expected_name in [("oracle", "ordered"), ("matrix", "ordered")]:
    for var in results["oracle"].data_vars:
        act = np.asarray(results[actual_name][var]); exp = np.asarray(results[expected_name][var])
        bad = ~np.isclose(act, exp, rtol=1e-6, atol=1e-6, equal_nan=True)
        if bad.any():
            print(f"  CI-style {actual_name} vs {expected_name}: first failing var={var} "
                  f"mismatched {bad.sum()} / {bad.size}, max abs among violations="
                  f"{np.abs(act - exp)[bad].max():.8e}, max rel={np.max(np.abs(act-exp)[bad]/np.abs(exp)[bad]):.8e}")
            break
    else:
        print(f"  CI-style {actual_name} vs {expected_name}: all vars pass")
