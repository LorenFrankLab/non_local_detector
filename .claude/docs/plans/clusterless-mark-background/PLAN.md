# Clusterless Mark Background Implementation Plan

**Status:** Not started. Step 1 (No-Spike mark reference) is committed on branch `fix/no-spike-mark-reference` as `7882f2ee`; this plan is step 2 on the same branch, same PR.

**Why it matters (measured 2026-10-15):**
- On `7882f2ee`, simulated marks with 8 or 16 µV features at `waveform_std=24` have 100% of single-spike bins floored to `LOG_EPS` at every position. The spikes then carry zero position information; at D = 4 the spread is 16 nats.
- The same floor drives main's No-Spike to 100% of bins at D ≥ 8.
- `LOG_EPS` is a number in rate × 2 ms × mark-density units, so where it bites depends on the mark units and dimension.

**Goal:** Replace every dimensionful per-spike floor on clusterless marked intensities with a coherent background component, so that every clusterless state's odds are invariant to waveform units and an outlier mark is explained by the shared background rather than by a units-dependent constant.

**Architecture:** Each active clusterless state's marked intensity becomes `λ'(x, m) = λ(x, m) + β q(m)` with compensator `Λ'(x) = Λ(x) + β` per electrode. `q` is the same per-electrode `MarkReference` that No-Spike already uses (step 1), and `β = RATE_EPS_HZ = 5e-13 Hz`. The per-spike term `log λ` is replaced by `logaddexp(log λ, b)` with `b = log β + log q(m)` in the backend's own units. The change happens inside the four existing backends at the sites where they currently floor. No parallel predictor is added; this deliberately differs from PR #77, whose 1,311-line `background_marks.py` is not reused. `β` must stay below `no_spike_rate` (default 1e-10 Hz), so an outlier spike favours No-Spike by `log(no_spike_rate / β)` instead of counting against silence. This matches the No-Spike state's purpose: capturing extended population silence.

**Tech stack:** JAX (jit, lax.map/fori_loop/cond), NumPy host-side selection, existing helpers in `likelihoods/common.py` and `likelihoods/streamed_kde.py`.

**Out of scope:**
- Changing `q`'s family (it stays a diagonal Gaussian).
- Fitting `β`.
- Sorted-spikes likelihoods.
- PR #75 (trajectory exposure).
- Performance work beyond holding the regression budget in the Validation slice.

## Inputs to read first

**Step 1 code:**
- `src/non_local_detector/likelihoods/mark_reference.py`: `MarkReference.log_density` (host float64) and `fit_mark_reference`.
- `src/non_local_detector/models/base.py`, `ClusterlessDetector._fit_mark_reference` and `fit_encoding_model`, where `mark_reference_` is assigned next to `encoding_model_`. The No-Spike term is added in `compute_log_likelihood` (`mark_reference_log_likelihood`).

**Constants:** `src/non_local_detector/likelihoods/common.py:32-38`, `EPS`, `LOG_EPS`, `RATE_REFERENCE_SECONDS`, `LOG_RATE_EPS_HZ`, `RATE_EPS_HZ`.
- KDE and log-KDE per-spike values are `log(λ · RATE_REFERENCE_SECONDS)`, corrected later by `log_bin_duration_evidence(..., intensity_time_scale=RATE_REFERENCE_SECONDS)` (`clusterless_kde.py:1085-1095`).
- GMM and diffusion per-spike values are `log λ` in Hz.
- So `b_KDE = LOG_EPS + log q(m)` and `b_GMM/DIFF = LOG_RATE_EPS_HZ + log q(m)`.

**Floor sites to replace** (verified on `7882f2ee`):
- **KDE, non-local dense:**
  - `clusterless_kde.py:268-275` (`safe_log` of a probability-space intensity).
  - `clusterless_kde.py:276-300`, the occupancy-tail fallback through `clusterless_kde_log.estimate_log_joint_mark_intensity`.
  - `clusterless_kde.py:309` (`jnp.clip(output, min=LOG_EPS)`).
- **KDE, non-local streamed:**
  - `streamed_kde.py:528-536` (`safe_log` plus clip).
  - `streamed_kde.py:561-566` (tail `jnp.maximum(..., LOG_EPS)`).
- **KDE, local:**
  - `clusterless_kde.py:520-528` (`safe_log`).
  - `clusterless_kde.py:535-552` (tail `maximum(..., LOG_EPS)`).
  - Streamed local: `clusterless_kde.py:1369-1423`.
- **log-KDE:**
  - `clusterless_kde_log.py:793-795` (`maximum(..., LOG_EPS)`).
  - `clusterless_kde_log.py:288-296` (`_log_joint_from_log_marginal` floors -inf and zero occupancy to `LOG_EPS`).
  - `clusterless_kde_log.py:82-102`, `_log_marginal_floor`, the exactness threshold tied to `LOG_EPS`. It is used at the `needs_exact_tails` gate and in the tile-floor logic of `_estimate_with_enc_chunking`.
  - Local: `clusterless_kde_log.py:1307-1310`.
- **GMM:**
  - Per-spike values are already unfloored log-space: `clusterless_gmm.py:213-221` (non-local dense), `:134` (bin-tiled), `:271` (local).
  - Zero-rate electrodes add `LOG_RATE_EPS_HZ` per spike: `clusterless_gmm.py:980-985` (non-local) and `:1172-1180` (local).
- **Diffusion:**
  - `clusterless_diffusion.py:645-647` (non-local `safe_log(..., eps=RATE_EPS_HZ)`).
  - `:735-737` (local).
  - Zero-rate: `:948-952` (local) and `:1071-1076` (non-local).
- **Compensator floors, each applied after summing over electrodes:**
  - `clusterless_kde.py:773-775` (fit) and `:1444-1446` (local).
  - `clusterless_kde_log.py:1611-1613` and `:2154-2156`.
  - `clusterless_gmm.py:764-766` and `:1219-1221`.
  - `clusterless_diffusion.py:482-484` (fit; the local path reuses it).
- **Predict signatures:** see the plumbing task. KDE (`clusterless_kde.py:804`) and log-KDE (`clusterless_kde_log.py:1639`) have no `**kwargs`. GMM absorbs unknown keys. Diffusion reads from `**encoding_model`.

## Design rules (apply at every site)

1. **Per-spike term.** `term = logaddexp(log_lambda_unfloored, b)`, where `b` is a per-spike vector broadcast across position bins and `log_lambda_unfloored` may be `-inf` (zero rate, zero mass, underflow).
   - Use a log that maps 0 to `-inf` with zero gradient: `jnp.where(x > 0, jnp.log(jnp.where(x > 0, x, 1.0)), -jnp.inf)`. Do not use `safe_log`.
   - Keep the existing NaN propagation: a NaN λ stays NaN and must not be absorbed by `logaddexp`. Add `jnp.where(jnp.isnan(log_lambda), jnp.nan, term)` if `logaddexp` would mask it.
2. **Exactness threshold.** In float32, `logaddexp(x, b) == b` whenever `x < b + log(finfo(float32).eps)`, which is about `b − 15.9`. Every exactness or tail decision that currently compares against `LOG_EPS`-derived thresholds must compare against the per-spike `b + log(eps)` instead:
   - log-KDE: `_log_marginal_floor` and its uses.
   - KDE: the tail routing at `clusterless_kde.py:965` and `:1299`.
   - Streamed: `streamed_kde.py:541`.

   Values below the threshold may be inexact; values above must be exact to the existing tolerances.
3. **Probability-space underflow (KDE and streamed KDE).** A spike whose required range reaches below float32 underflow must be routed through the existing log-space estimator. This is the same mechanism as the occupancy-tail fallback, extended with a per-spike condition: `b + log(eps) − log(mean_rate) + log(occupancy) < log(finfo(float32).tiny) + margin`. Decide the routing on the host per electrode and chunk, as the current tail routing is.
4. **Diffusion resolution floor** (user decision 2026-10-15, after the Task 1 study).
   - The heat-kernel field resolves only about `resolution = max(finfo(dtype).eps, heat-kernel rank tol 1e-6)` relative to each spike's column maximum. Use `max(finfo(float32).eps, _HEAT_KERNEL_RANK_TOL)` when the basis is truncated, and `finfo(float32).eps` at full rank.
   - Per decode spike, raise field values below `resolution · max_bins(P[:, j])` to that level, then apply rule 1.
   - This is unit-consistent, because the floor scales with the spike's own peak. It keeps about 16 nats of peak-versus-tail contrast per spike, and its errors are deep-tail overestimates: 2.5% of real pairs exceed 1 nat.
   - Stay in float32; there is no float64 apply.
   - Diffusion's occupancy floor `EPS` (`clusterless_diffusion.py:400`) is also dimensionful. It did not bind on the real 3.5 cm grid; note it in the design doc but leave it unchanged here.
5. **Zero-rate electrodes.**
   - With a reference: per-spike term `= b`.
   - Without a reference (`mark_reference_[e] is None`, no training spikes anywhere): per-spike term `= log β` in the backend's units. This equals today's `LOG_RATE_EPS_HZ` / `LOG_EPS`. It is unit-consistent because No-Spike also omits `q` for that electrode.
6. **Compensator.** Replace `clip(sum_e Λ_e, min=RATE_EPS_HZ)` with `sum_e Λ_e + n_electrodes · β`. Count every electrode, including zero-rate ones, to match No-Spike's per-electrode `no_spike_rate`. Remove the clips.

## Tasks

### 1. Diffusion accuracy study (decision gate; do first)

- **Measure.** On the simulated run (`make_simulated_run_data`) and on the real session, compare float32 `_diffused_mark_field` against a float64 evaluation of the same operator.
  - Use `jax.experimental.enable_x64` in a scratch script.
  - Real session: `/cumulus/edeno/nld-markref-validation/j16_02_r1_hpc.npz` on breeze; see the memory note `gpu-hosts-breeze-zephyr`.
- **Report:**
  - the relative error as a function of `P / max(P)`;
  - the level `noise_rel` below which the float32 values are dominated by error;
  - the fraction of real decode (spike, bin) pairs whose background threshold `b + log(eps)` lies below `log(noise_rel · max P)`.
- **Gate.**
  - If that fraction is under 1e-3 of spike-bin pairs that matter (posterior mass), apply rule 4.
  - Otherwise stop and report to the user before implementing diffusion.

**Task 1 result (2026-10-15): gate failed. The user chose (B): a float32 resolution floor (rule 4).**
- **Script:** `scratchpad/diffusion_floor_study.py`, copied to breeze `/cumulus/edeno/nld-markref-validation/`.
- **Setting:** real 2-D session, 723 bins, full rank, so the errors are round-off, not truncation.
- **Precision:** the float32 field is accurate to about 1e-6 relative down to 1e-4 of each spike's peak; values below about 1e-8 of the peak are noise or exactly 0. A field computed with a float64 basis is accurate down to about 1e-13 of the peak.
- **Required range:** each spike needs λ from its peak down to `b + log eps`, a median of 66 nats (p90 69, max 165), beyond float64's ~36.
- **Pairs off by more than 1 nat, by option:**
  - float32 with rule 4 at 1e-6: 4.6%;
  - float32, no clamp: 1.5%;
  - float32, (B) per-spike floor at 1e-7 of the column peak: 2.5%, all overestimates, p99 5 nats;
  - float64, no clamp: 0.09%, p99 8e-7 nats; the failures are tail bins at least 36 nats below the spike's peak.
- **Decision needed between:**
  - (B) a unit-consistent per-spike resolution floor at the operator's accuracy;
  - (A) a float64 heat-kernel apply, plus (B) at float64 resolution; costs about 2× the apply time, still to be measured;
  - excluding diffusion.

### 2. Plumbing: per-electrode references reach every predict function

- **Fit functions.** Each clusterless fit function fits its own references from its training marks and weights with `fit_mark_reference`:
  - `fit_clusterless_kde_encoding_model`, `clusterless_kde_log`'s fit, `fit_clusterless_gmm_encoding_model`, `fit_clusterless_diffusion_encoding_model`.
  - Fallback scale: `waveform_std`, or `sqrt(gmm_reg_covar)` for GMM.
  - Each returns them under the key `"mark_reference"` (list per electrode), so standalone callers get self-consistent background terms. There is no floored legacy path to keep.
- **Detector.** `ClusterlessDetector.fit_encoding_model` overwrites every `encoding_model[key]["mark_reference"]` with the shared `self.mark_reference_`, so all states and No-Spike share one `q`.
  - The EM refit path keeps `_refit_mark_reference=False` and must still overwrite with the frozen reference.
- **Predict functions.** Add an explicit `mark_reference` parameter to the KDE and log-KDE predict functions. GMM and diffusion read it from their kwargs or encoding dict.
  - Compute `log q` on the host for each electrode's selected decode marks (`MarkReference.log_density`), cast to the backend dtype, and pad exactly as the features are padded. Pad values are irrelevant because padded row ids are dropped.
- **Saved models.** Saved encoding dicts lacking `"mark_reference"` raise `ValidationError("refit")`, as `_validate_time_contract` already does for No-Spike. Extend that check to all clusterless detectors.

### 3. GMM (smallest; proves the helper and tests)

- Add a shared helper `log_intensity_with_background(log_lambda, background)` in `likelihoods/common.py`, implementing rule 1. Unit-test its forward values, NaN propagation, `-inf` input and gradients.
- Apply it at `clusterless_gmm.py:213-221`, `:134` and `:271`.
- Replace the zero-rate branches at `:980-985` and `:1172-1180` with rule 5. These need `select_spike_rows(elect_feats, selection)` to evaluate `q`.
- Apply rule 6 at `:764-766` and `:1219-1221`.

### 4. log-KDE

- Thread a per-spike `background` array `(n_dec,)` into `estimate_log_joint_mark_intensity` and `_estimate_with_enc_chunking`.
- Replace `_log_marginal_floor` (`clusterless_kde_log.py:82-102`) with a per-spike, per-bin threshold derived from `background + log(eps)` (rule 2). Update its callers: the `needs_exact_tails` gate and the tile floor.
- Remove the `LOG_EPS` floors at `:288-296` and `:793-795`. `_log_joint_from_log_marginal` returns unfloored `-inf` for zero occupancy or zero mass, and the caller applies rule 1.
- Apply the local path at `:1307-1310`. Apply rule 6 at `:1611-1613` and `:2154-2156`.

### 5. KDE (probability space) and streamed KDE

- Apply rule 1 at every site listed under "KDE" above. Apply rule 3's routing for spikes whose required range underflows. The routing reuses the log-KDE estimator with the same `background` argument; it no longer uses its floored output.
- Apply rule 6 at `clusterless_kde.py:773-775` and `:1444-1446`.
- Delete the now-unused floor-only helpers if nothing else calls them. `clusterless_kde.estimate_log_joint_mark_intensity` / `block_estimate_log_joint_mark_intensity` (`:175-364`) were reported unused by predict; confirm with grep before deleting.

### 6. Diffusion

- Apply rules 1, 4 and 5 at `clusterless_diffusion.py:645-647`, `:735-737`, `:948-952` and `:1071-1076`.
- Apply rule 6 at `:482-484`.

### 7. Tests that pin old floors

- **Update** these to the new contract. Each update states the new expected value from the formula, not from re-running the code:
  - `tests/test_clusterless_gmm_optimization.py:745-870`
  - `tests/likelihoods/test_clusterless_gmm_padding.py:115-174`
  - `test_clusterless_kde_padding.py:185-247`
  - `test_clusterless_kde_log_padding.py:122-182`
  - `test_clusterless_diffusion.py:412-414, 604-609, 705-707, 899-904`
  - `test_clusterless_kde_log_optimization.py:306, 443-446`
  - `test_streaming_integration.py:615-621`
  - `test_clusterless_kde_log_zero_encoding_spike.py:43, 120`
  - `test_kde_log_weight_normalization.py:35`
  - `test_streamed_kde.py:135` and `:189-195` (the zero-gradient-below-floor expectation becomes zero gradient below `b + log(eps)`)
  - `test_marked_kde_blocks.py:139-150`
  - `test_kde_empty_tiles.py:71-73`
- **Keep** `test_likelihood_edge_cases.py:667` (GMM per-spike value unfloored); extend it to the background form.

### 8. Documentation

- **CHANGELOG** (`[Unreleased]` → "Unit-independent clusterless No-Spike odds"):
  - Replace the closing sentence ("The fixed lower bounds … unchanged here") with the background contract.
  - State that the β < `no_spike_rate` relationship is what lets outlier spikes not break No-Spike.
  - State that all saved clusterless models must be refit.
  - Add the real-data numbers from the Validation slice.
- **Docstrings:** document `mark_reference` on each public predict and fit function.
- **Design note:** `docs/likelihood_mark_background.md`, with the model, rules 1–6, the exactness argument and the diffusion study result.

### 9. Golden data (approved by the user 2026-10-15, after step 2 only)

- Regenerate `src/non_local_detector/tests/golden_data/nonlocal_detector_golden.pkl` once, after tasks 3–6, with the four-part analysis from `.claude/skills/numerical-validation/SKILL.md`.
- Any other golden or snapshot file that changes needs separate explicit approval.

## Deliberately not in this plan

- A heavier-tailed or richer `q`: a separate statistical change, validated on held-out data.
- Fitting `β` as a background rate: β stays a unit-consistent numerical floor.
- Batching `log q` evaluation across electrodes into one reduction (review item from step 1). Revisit only if the benchmark below shows the No-Spike or background term costing more than 5%.
- PR #75 (trajectory exposure): next PR, from main after this merges.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_log_intensity_with_background` (unit) | Equals `logaddexp`; `-inf` λ gives `b`; NaN λ stays NaN; gradient w.r.t. λ is `exp(λ − out)`, finite, 0 at `-inf` |
| Unit invariance, all four algorithms (integration, extend `tests/models/test_no_spike_mark_reference.py`) | With marks `m' = A m + c` and bandwidths transformed alike (GMM: `gmm_reg_covar` scaled), the full state posterior (all states, incl. Non-Local) is unchanged within float32 tolerance (`atol 1e-4` on probabilities; justify if looser) |
| Outlier mark favours No-Spike | A single decode spike with a mark 50σ from every training mark, in an otherwise silent bin: No-Spike − Local log-odds ≈ `log(no_spike_rate/β)` (+5.3) within 1e-3 |
| Zero-rate electrode with/without reference | Per-spike term equals `b` / `log β` exactly |
| Compensator | Silent-bin score per active state = `−dt·(Σ_e Λ_e(x) + n_e·β)` |
| log-KDE exactness | Per-spike results above `b + log(eps)` match a float64 oracle within existing tolerances; adversarial far-tail cases (copy PR #77's "200 and 1457 bandwidths away" pattern) |
| KDE underflow routing | A spike whose required range underflows probability space matches the log-KDE value |
| Diffusion | Per task 1's gate; field values below `noise_rel·max P` treated as zero; invariance test passes |
| Chunk parity | `row_slice`/`n_chunks`/checkpointed predict give identical active and No-Spike columns (existing pattern in `test_no_spike_mark_reference.py`) |
| Full suite (slow) | `uv run pytest` passes except approved golden changes |
| Floor no longer erases spike evidence (integration) | Simulated run with marks padded to D = 8 and 16 µV features (extra N(0, 30²) channels, `waveform_std=24`): the fraction of single-spike bins whose non-local spike evidence is flat across all positions drops from 1.000 (`7882f2ee`, script `scratchpad/floor_hits.py` pattern) to ~0; the No-Spike fraction stays ~2–3% for D = 4–32, raw and z-scored (`dim_sweep.py` pattern) |
| Performance vs main (slow; user request 2026-10-15) | Same inputs, fresh processes, warm-up excluded, all four algorithms, CPU and breeze A100. **Speed:** warm and first-call `predict` (non-local and local), median of ≥5. **Memory:** peak device memory (`memory_stats()["peak_bytes_in_use"]` on GPU; host RSS on CPU) and compiled temp size (`memory_analysis().temp_size_in_bytes`) of the changed kernels. **Accuracy:** per-spike terms against a float64 oracle (`logaddexp(log λ, b)`), the invariance identity, and real-data state agreement. Report regressions over 10% with their cause |
| Real data (slow, breeze A100) | `/cumulus/edeno/nld-markref-validation/real_markref.py` + `compare_real.py`: branch ×1 vs ×10 winning-state flips drop from 304 toward 0 (report the residual and explain it); No-Spike bouts tolerate occasional outlier spikes again (inside-bout rate > 0 like main's 1.6 Hz); warm `predict` within 10% of main (main 2.07 s, step 1 2.16 s) |

## Fixtures

- **Simulated:** `make_simulated_run_data(n_runs=1, seed=0)`, as in the `run_data` fixture of `tests/models/test_no_spike_mark_reference.py`. Reuse that module fixture.
- **Real:** `j16_02_r1_hpc.npz` (22 tetrodes, 4-D µV amplitudes, 709k samples at 500 Hz), exported by `/cumulus/edeno/nld-markref-validation/export_session.py`.
  - Remote worktrees `/tmp/nld-markref-main` (origin/main) and `/tmp/nld-markref-branch` on breeze. Run with `PYTHONPATH=<worktree>/src /cumulus/edeno/non_local_detector/.venv/bin/python`.
  - Copy changed files with `scp`; nothing is pushed.

## Review

Before opening the PR, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task is implemented as specified; nothing in "Deliberately not in this plan" was added.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial: they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan.
