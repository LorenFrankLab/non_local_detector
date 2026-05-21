# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `_DetectorBase.estimate_parameters` (and all detector subclasses that inherit it) now sets three new attributes when EM finishes: `converged_` (`bool` — True if EM met its tolerance before `max_iter`), `n_iter_` (`int` — number of EM iterations actually performed), and `em_monotonicity_violations_` (`list[int]` — 1-based iteration indices where the marginal log-likelihood decreased, empty when EM is well-behaved).
- `likelihoods.gmm.GaussianMixtureModel.converged_` now reflects the actual EM lower-bound convergence flag from `_em_fit_while_loop`, replacing the prior heuristic `n_iter < max_iter`. A `UserWarning` is emitted at the end of `fit` when the model did not converge.
- `likelihoods.sorted_spikes_glm.fit_poisson_regression` now emits a `UserWarning` when SciPy's BFGS optimizer reports `success=False`, including the BFGS message, iteration count, and final loss. Place-field coefficients may be unreliable in that case.
- `core._condition_on` now falls back to the predicted distribution at degenerate timesteps where every state has `-inf` log-likelihood (previously emitted an invalid all-zero posterior). `log_norm` is set to `-inf` for those steps so the marginal log-likelihood reflects the impossible-data step. The host-side `filter`, `filter_covariate_dependent`, `chunked_filter_smoother`, and `chunked_filter_smoother_covariate_dependent` functions now log a single `logger.warning` summarizing the count of degenerate timesteps when any are found.

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

- EM iterations where the marginal log-likelihood decreased (previously silent) now log a `logger.warning` describing the iteration index, before/after log-likelihoods, and change magnitude. The iteration index is also appended to `em_monotonicity_violations_`.
- EM that exits via `max_iter` without converging (previously silent) now emits a `UserWarning` describing the final log-likelihood change, the tolerance, and that fitted parameters may be biased.
- The final E-step now checks that the post-M-step log-likelihood did not decrease by more than `tolerance`; a `UserWarning` flags inconsistencies between the E-step and the M-step output.
- **Note on previously-silent issues now surfaced:** several pre-existing tests of the sorted-spikes GLM encoding path emit the new `UserWarning` for non-convergent BFGS exits. The fitted coefficients still satisfy the existing test assertions, but the warning indicates these tests have been silently fitting non-converged models. The `test_sorted_spikes_glm_encoding_runs_end_to_end` integration test additionally trips the new max-iter and final-E-step warnings under default parameters. None are regressions from these changes; they are previously-silent issues now made visible. Investigation of GLM EM convergence behavior is tracked as a follow-up.
- `model_checking.posterior_consistency.posterior_consistency_hpd_overlap`: docstring expanded with a Notes block explaining that the metric is the containment coefficient (`|A∩B| / min(|A|, |B|)`) — asymmetric — rather than Sørensen-Dice or Jaccard. The implementation is unchanged.

### Removed

- `environment.order_boundary` and `environment.get_track_boundary_points`: both called `nx.from_scipy_sparse_matrix`, which was removed in NetworkX 3.0. Neither was used anywhere outside their own module. The now-unused `from sklearn.neighbors import NearestNeighbors` import was removed alongside them.
