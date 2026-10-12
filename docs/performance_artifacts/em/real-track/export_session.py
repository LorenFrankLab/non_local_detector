"""Export a state-space-playground session pickle to a plain recording .npz.

Run with the playground venv. Output keys (all NumPy, no pickled objects):
position_time (n,), position (n, 2) head x/y in cm, spike_times_flat + spike_offsets
(per unit/tetrode, seconds), optional features_flat (n_spikes, d) + same offsets,
track nodes/edges, edge_order, edge_spacing, and source metadata.
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

source, out = Path(sys.argv[1]), Path(sys.argv[2])
with open(source, "rb") as f:
    d = pickle.load(f)
pi = d["position_info"]
time = pi.index.to_numpy().astype(np.float64)
position = pi[["head_position_x", "head_position_y"]].to_numpy().astype(np.float64)
spikes = [np.asarray(s, np.float64) for s in d["spike_times"]["HPC"]]
spikes = [np.sort(s[(s >= time[0]) & (s <= time[-1])]) for s in spikes]
offsets = np.cumsum([0] + [len(s) for s in spikes])
arrays = dict(
    position_time=time,
    position=position,
    spike_times_flat=np.concatenate(spikes),
    spike_offsets=offsets,
)
features = d.get("spike_waveform_features", {}).get("HPC")
if features is not None:
    kept = []
    for s_all, f_all in zip(d["spike_times"]["HPC"], features, strict=True):
        s_all = np.asarray(s_all, np.float64)
        keep = (s_all >= time[0]) & (s_all <= time[-1])
        order = np.argsort(s_all[keep], kind="stable")
        kept.append(np.asarray(f_all)[keep][order])
    arrays["features_flat"] = np.concatenate(kept).astype(np.float32)
graph = d["track_graph"]
nodes = sorted(graph.nodes)
arrays["track_node_ids"] = np.asarray(nodes)
arrays["track_node_pos"] = np.asarray([graph.nodes[n]["pos"] for n in nodes], float)
arrays["track_edges"] = np.asarray([(u, v) for u, v in graph.edges])
arrays["track_edge_distance"] = np.asarray([graph.edges[e]["distance"] for e in graph.edges], float)
arrays["track_edge_id"] = np.asarray([graph.edges[e]["edge_id"] for e in graph.edges])
arrays["edge_order"] = np.asarray(d["linear_edge_order"])
arrays["edge_spacing"] = np.asarray(d["linear_edge_spacing"], float)
arrays["source"] = np.asarray(str(source))
arrays["exporter_sha256"] = np.asarray(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
np.savez(out, **arrays)
print(out, {k: v.shape for k, v in arrays.items()})
