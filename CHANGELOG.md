# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `analysis.distance1D.get_map_speed`: the trailing boundary speed was inserted before the last array element instead of appended; for every chunk with three or more time bins, the final two speed samples were misordered.
- `models.base._DetectorBase.estimate_parameters`: post-EM cleanup of the encoding-model data did nothing because the attribute name in the `hasattr` check did not match the leading-underscore form used at write sites; spike-time and waveform-feature arrays were retained for the lifetime of every fitted model.
- `models.cont_frag_model.ContFragSortedSpikesClassifier.get_posterior` and `ContFragClusterlessClassifier.get_posterior`: raised an exception for 2D environments because the implementation summed over the dimension name `"position"`, which only exists for 1D environments. Now collapses every dimension whose name matches `"position"` or ends with `"_position"`, supporting 1D, 2D, and higher-D environments. The returned dim is still named `state` (singular), so existing `state_probs.sel(state="Continuous")` callers continue to work.
- `environment.Environment.fit_place_grid`: a tuple `place_bin_size` provided with a `track_graph` previously dereferenced a dead local variable and reached a confusing `TypeError` deep inside `np.linspace`. Now raises a clear `ValidationError` explaining the constraint and showing an example.
- `continuous_state_transitions.RandomWalk`: `direction`-aware transitions on an environment lacking an N-D track graph and array-typed inter-node distances would crash with `AttributeError` deep inside `networkx`. Now raises a clear `ConfigurationError` with a hint pointing to the two valid configurations.
- `visualization.figurl_2D.process_decoded_data`: linearized bin indices stored as `uint16` would silently wrap modulo 65535 for grids with more than 65535 cells (e.g., 256×256), corrupting the rendered decoded-position view. The function now raises a clear `ValueError` at the boundary; the `uint16` dtype is retained because the downstream `sortingview.views.franklab.DecodedPositionData` schema receives these indices as `uint16`.
- `simulate.clusterless_simulation.make_simulated_run_data`: hardcoded `mark_spacing=10` while the module's replay generators (`make_continuous_replay`, `make_hover_replay`, `make_fragmented_replay`) default to `MARK_SPACING = 5`. Test marks landed at `{0, 5, 10, 15}` while encoding marks were at `{0, 10, 20, 30}`, silently invalidating any test that mixed the two. All generators now use the module-level `MARK_SPACING` constant.
- `model_checking.highest_posterior_density.get_HPD_spatial_coverage`: only computed a 1D bin width even though its docstring advertised 2D support; for 2D posteriors it either crashed (no `position` dim) or returned wrong units. Now branches on the posterior's dim names and uses the 2D cell area (`bin_width_x * bin_width_y`) when `x_position` and `y_position` are present.

### Changed

- `model_checking.posterior_consistency.posterior_consistency_hpd_overlap`: docstring expanded with a Notes block explaining that the metric is the containment coefficient (`|A∩B| / min(|A|, |B|)`) — asymmetric — rather than Sørensen-Dice or Jaccard. The implementation is unchanged.

### Removed

- `environment.order_boundary` and `environment.get_track_boundary_points`: both called `nx.from_scipy_sparse_matrix`, which was removed in NetworkX 3.0. Neither was used anywhere outside their own module. The now-unused `from sklearn.neighbors import NearestNeighbors` import was removed alongside them.
