# %% [markdown]
# # EM Stability: Soft Position Exclusion (`non_local_position_penalty`)
#
# Test whether penalizing non-local likelihood near the animal's current
# position prevents non-local states from "stealing" local spikes.
#
# A Gaussian penalty is applied to non-local state log-likelihoods:
# `penalty = -non_local_position_penalty * exp(-0.5 * dist^2 / sigma^2)`
#
# This is the most principled approach: non-local activity by definition
# represents positions OTHER than where the animal currently is.

# %%
import time as time_module

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment

# %% [markdown]
# ## Load Data

# %%
DATA_PATH = "/Users/edeno/Downloads/"

animal_date_epoch = "j1620210710_02_r1"
position_info = pd.read_pickle(DATA_PATH + f"{animal_date_epoch}_position_info.pkl")
spike_times = joblib.load(DATA_PATH + f"{animal_date_epoch}_HPC_spike_times.pkl")
track_graph = joblib.load(DATA_PATH + f"{animal_date_epoch}_track_graph.pkl")
edge_order = joblib.load(DATA_PATH + f"{animal_date_epoch}_linear_edge_order.pkl")
edge_spacing = joblib.load(DATA_PATH + f"{animal_date_epoch}_linear_edge_spacing.pkl")

time = position_info.index
if hasattr(time, "to_numpy"):
    time = time.to_numpy()
if np.issubdtype(np.asarray(time).dtype, np.timedelta64):
    time = np.asarray(time) / np.timedelta64(1, "s")
time = np.asarray(time, dtype=float)

speed = position_info["head_speed"].to_numpy()
position2D = position_info[["head_position_x", "head_position_y"]].to_numpy()

print(f"Time: {time.shape}, Position: {position2D.shape}, Neurons: {len(spike_times)}")

# %% [markdown]
# ## Configure

# %%
env = Environment(
    environment_name="",
    track_graph=track_graph,
    edge_order=edge_order,
    edge_spacing=edge_spacing,
)

COMMON_PARAMS = dict(
    environments=[env],
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={
        "position_std": np.sqrt(12.5),
        "block_size": int(2**12),
    },
)

MAX_ITER = 10
N_TIME = min(len(time), 50000)

time_subset = time[:N_TIME]
position_subset = position2D[:N_TIME]

# %% [markdown]
# ## Run Experiments

# %%
experiments = {
    "baseline (no penalty)": dict(),
    "penalty=2, sigma=2": dict(non_local_position_penalty=2.0, non_local_penalty_sigma=2.0),
    "penalty=5, sigma=3": dict(non_local_position_penalty=5.0, non_local_penalty_sigma=3.0),
    "penalty=10, sigma=3": dict(non_local_position_penalty=10.0, non_local_penalty_sigma=3.0),
    "penalty=10, sigma=5": dict(non_local_position_penalty=10.0, non_local_penalty_sigma=5.0),
}

results = {}

for name, params in experiments.items():
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    detector = NonLocalSortedSpikesDetector(**COMMON_PARAMS, **params)

    t0 = time_module.time()
    try:
        result = detector.estimate_parameters(
            position_time=time_subset,
            position=position_subset,
            spike_times=spike_times,
            time=time_subset,
            max_iter=MAX_ITER,
            store_log_likelihood=True,
        )
        elapsed = time_module.time() - t0
        print(f"Completed in {elapsed:.1f}s")

        results[name] = {
            "result": result,
            "detector": detector,
            "marginal_log_likelihoods": result.attrs.get("marginal_log_likelihoods", []),
            "elapsed": elapsed,
        }

        pf_norms = {}
        for key, em in detector.encoding_model_.items():
            if "place_fields" in em:
                pf_norms[key] = float(np.linalg.norm(em["place_fields"]))
        results[name]["place_field_norms"] = pf_norms

        state_probs = result.acausal_state_probabilities
        if "Local" in state_probs.state:
            local_prob = state_probs.sel(state="Local").values
            results[name]["local_prob_mean"] = float(np.mean(local_prob))
            results[name]["local_prob_median"] = float(np.median(local_prob))
            results[name]["local_prob_min"] = float(np.min(local_prob))
            print(f"  Local: mean={np.mean(local_prob):.4f}, median={np.median(local_prob):.4f}")

    except Exception as e:
        elapsed = time_module.time() - t0
        print(f"FAILED after {elapsed:.1f}s: {e}")
        results[name] = {"error": str(e), "elapsed": elapsed}

# %% [markdown]
# ## Compare Results

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EM Stability: Non-Local Position Penalty", fontsize=14)

ax = axes[0, 0]
for name, res in results.items():
    if "marginal_log_likelihoods" in res:
        mlls = res["marginal_log_likelihoods"]
        if len(mlls) > 0:
            ax.plot(mlls, "o-", label=name, markersize=4)
ax.set_xlabel("EM Iteration")
ax.set_ylabel("Marginal Log Likelihood")
ax.set_title("Convergence")
ax.legend(fontsize=8)

ax = axes[0, 1]
names_list = [n for n, r in results.items() if "local_prob_mean" in r]
means_list = [r["local_prob_mean"] for r in results.values() if "local_prob_mean" in r]
ax.barh(names_list, means_list)
ax.set_xlabel("Mean Local State Probability")
ax.set_title("Local State Dominance")

ax = axes[1, 0]
for name, res in results.items():
    if "place_field_norms" in res:
        for key, norm in res["place_field_norms"].items():
            ax.bar(name, norm)
ax.set_ylabel("Place Field Norm")
ax.set_title("Place Field Magnitude (after EM)")
ax.tick_params(axis="x", rotation=30)

ax = axes[1, 1]
runtimes = {n: r["elapsed"] for n, r in results.items() if "elapsed" in r}
ax.barh(list(runtimes.keys()), list(runtimes.values()))
ax.set_xlabel("Runtime (s)")
ax.set_title("Computational Cost")

plt.tight_layout()
plt.savefig("em_stability_position_exclusion.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## State Probabilities Over Time

# %%
fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
if len(results) == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    if "result" not in res:
        ax.set_title(f"{name}: FAILED")
        continue
    state_probs = res["result"].acausal_state_probabilities
    for state in state_probs.state.values:
        ax.plot(state_probs.sel(state=state).values, label=state, alpha=0.7, linewidth=0.5)
    ax.set_ylabel("P(state)")
    ax.set_title(name)
    ax.legend(loc="upper right", fontsize=7)

axes[-1].set_xlabel("Time bin")
plt.tight_layout()
plt.savefig("em_state_probs_position_exclusion.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary

# %%
print(f"\n{'Experiment':<30} {'Local Mean':>12} {'Local Med':>12} {'PF Norm':>12} {'MLL Final':>14} {'Time':>8}")
print("-" * 90)
for name, res in results.items():
    if "error" in res:
        print(f"{name:<30} {'FAILED':>12}")
        continue
    local_mean = res.get("local_prob_mean", float("nan"))
    local_med = res.get("local_prob_median", float("nan"))
    pf_norm = list(res.get("place_field_norms", {}).values())
    pf_norm = pf_norm[0] if pf_norm else float("nan")
    mlls = res.get("marginal_log_likelihoods", [])
    mll_final = mlls[-1] if mlls else float("nan")
    elapsed = res.get("elapsed", float("nan"))
    print(f"{name:<30} {local_mean:>12.4f} {local_med:>12.4f} {pf_norm:>12.2f} {mll_final:>14.2f} {elapsed:>8.1f}")
