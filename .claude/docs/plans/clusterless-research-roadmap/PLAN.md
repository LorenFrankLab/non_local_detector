# Clusterless decoding research roadmap

**Status (2026-10-22):** items 3b, 3c, 3d and 3e done. PRs: #105 (compression, including the low-rank merge), #106 (3b recompiles) and #107 (3c KDE step; merge after #105). Everything else is not started. Builds on the 2026-10-19/20 results summarized below.

**Goal:** answer six questions about clusterless decoding with `non_local_detector`, each with evidence from held-out real data:

1. Are there better ways to estimate the clusterless likelihood from a given set of mark features?
2. Which waveform features, or which use of the full waveform, and which biophysical priors improve that estimate?
3. For real-time use, where can we trade a little accuracy for speed?
4. How do we scale to longer recordings and larger state spaces, and can the state-space model itself improve?
5. Which parameters we currently assume can be estimated from data instead?
6. How should high-density probes (polymer probes, Neuropixels) be handled?

**Shape:** a research roadmap, not a single PR. Each numbered item below is a unit of work that ends in a report, and sometimes a package change on its own feature branch. Items can run in parallel unless a dependency is listed.

**Out of scope:** clusterless spike sorting; non-spatial decoding targets (only position is decoded here); changes to spyglass itself. Spyglass is read-only here, but findings may become recommendations for it.

## How to work with this plan

- Each item states a goal, when it is done, fixed decisions with their reasons, a suggested approach, validation and what is out of scope. The approach is a suggestion: deviate when the evidence says so, and record why in the decision log.
- Thresholds are marked **hard** (invariants, regressions, exact reproductions: a miss blocks the item) or **target** (statistical: a miss is reported, never loosened after the fact).
- Package code goes on a local feature branch from `main`, in its own git worktree, committed as small separate commits. Commits on feature branches are pre-approved. Pushing, opening PRs, updating snapshots or golden data, and changing numerical tolerances need the user's approval.
- Harnesses and results live on `/cumulus/edeno/lowocc/<item>/`, with a README in each folder: what runs where, how to check progress, how to requeue, how to aggregate. Jobs run in named tmux sessions and must finish and aggregate on the host without a local watcher (the user's VPN drops overnight).
- Don't reference this plan, item numbers or `.claude/docs/plans/` in code, docstrings, test names or module names.
- Update the decision log and the results snapshot when an item finishes.
- `literature-review.md` (in this folder) reviews the relevant work of Eden, Arai, Yousefi and Paninski. Items below cite it as "review, Area N.k" for the evidence and test design; the review is the only home for those details.

## Shared evaluation protocol

All items that compare likelihood models use this protocol, so results stay comparable.

- **Primary score:** `info_vs_flat_per_s`, the held-out log likelihood of the spikes minus that of the same model refit without position, in nats/s. The position-free refit must use the same fitted settings as the spatial model (for example, the MDN's chosen EM iteration count). A null built separately leaves non-positional structure in the score: the 2026-10-19 null checks showed 9–60 nats/s of it on RS58. Also report `bits_per_spike` from the Local-state log likelihood.
- **Secondary, descriptive only:** HPD coverage (50/95%), median MAP error while running, non-local fraction while running. Theta sequences make the decoded position legitimately sweep ahead of the animal while it runs, so these aren't error measures during running.
- **Folds:** 5 blocked (contiguous-time) folds. Report each comparison paired by fold: mean ± SD of the difference, folds better, and the baseline's fold-to-fold SD.
- **Exception, time-discounted encoding (item 6b):** it needs forward-chaining folds (train on earlier blocks, test on the next). The user approved this on 2026-10-20. Say so in that item's report.
- **Tuning parameters** (bandwidths, priors, component counts, iteration counts) are chosen inside each fold's training data, by blocked inner validation (item 5b). They are never chosen on a test fold. The 2026-10-20 KDE grid chose its bandwidth on fold 0, which made fold 0 in-sample; don't repeat that.
- **Promotion bar for a new likelihood or default:**
  - It beats the current default by more than the fold-to-fold spread on at least 3 of 4 recordings, or it matches the default within noise at clearly lower cost.
  - Smaller, consistent paired gains (every fold better) are reported as such. They can justify an opt-in option, not a default change.
- **Baseline model:** `clusterless_mark_glm` (`markglm:32`) on amplitude marks, `main` at or after `f66c42dc`.
- **Hosts:** zephyr and breeze, 10 × A100 each.
  - Plan sweeps across both hosts, putting heavy jobs on breeze first. Zephyr hosts the lab database VM.
  - Run with `nice -n 10` and at most 8 threads per job. Check `nvidia-smi` first, pin `CUDA_VISIBLE_DEVICES`, and set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - Size each sweep before launching; about 100 jobs has been enough per round. Wrap every job in a timeout so one slow job can't block a queue.
  - Never kill a tmux window that has a job running in it; that kills the job.
- **Database:** read-only. Never insert, populate or delete.
  - `SpikeSortingRecording.get_recording` silently regenerates a missing analysis file, which is a write. Check that the file exists first.
  - Re-extracting from raw NWB into `/cumulus` scratch is fine; this is how j16 was rebuilt.

## Data

**Current panel** (harnesses in the folders named below; details in `.claude/docs/plans/clusterless-learned-models.md`, "Data, harnesses and infrastructure"):

| Recording | Probe | Region | Notes |
|---|---|---|---|
| j1620210710 (j16) | 22 tetrodes | CA1 | W-track; waveforms re-extracted to `/cumulus/edeno/lowocc/waveforms/j16/` |
| RS5820251203 epoch 6 (RS58) | polymer, 11 sort groups of 26–32 channels | CA1 | W-track; waveforms, PCA and noise covariance in `/cumulus/edeno/lowocc/waveforms/` |
| Jacob20250618 | polymer | CA1 | W-track; no waveforms extracted |
| L1420240604 | polymer | CA1 | linear track; no waveforms extracted |

Channels per sort group are recorded only for RS58. Record them for every recording in the panel, because item 1a's main hypothesis depends on it.

**Expansion candidates.** From the read-only survey in `/cumulus/edeno/lowocc/dbsurvey/`: `broad_sessions.csv`, `electrode_groups.csv`, `wf_entries.csv`, `wf_hpc_sortgroups.csv` and the `pos_meta_*.csv` files. Verify each before use: that the waveform-feature entries cover the region's sort groups, that position exists for the run epochs, and that the analysis files exist.
- **Hippocampus, polymer probes, with waveform features and position:**
  - Charlie20260223 (W-track)
  - Embry20260604 (fork track)
  - Marcus20260803 (fork maze)
  - RS3420250223 (fork)
  - RS4520250729, RS6020260415
  - SC10020250519 (fork track, delayed alternation)
  - SC8520241110 and SC8520241111
  - SC7920240910, SC8520241202
  - Toby20250318 (hex maze)
- **Hippocampus, 256-channel silicon probe:** IM-1478_20220725 and IM-1594_20230725 (32 groups × 8 channels, hex maze). These give a different channel layout.
- **PFC, polymer probes:** about 120 sessions have PFC electrode groups. Prefer sessions that also have sorted PFC units, so clusterless can be compared with sorted decoding on the same data (item 1c). From `pos_meta_ssg.csv`, `wf_entries.csv` and the `pos_meta_*` files, these have a PFC sorted-spikes group plus waveform-feature entries:
  - RS6420260703: `PFC_manual_curated`, trodes position. Also RS6420260704 and RS6420260707.
  - SC8520241202: `PFC_spikes_accepted`, trodes position.
  - Charlie20260223: `single_mPFC_probe`, DLC and trodes.
  - Marcus20260803: `mPFC`, DLC.
  - Embry20260604: `mPFC_test_shank6`, DLC.
  - SFA0320260902–0905: `PFC_manual_curated`, but no position in the survey. Use them only if position turns up; PFC sessions need position.

  The survey counts waveform entries per session, not per region. Confirm the entries include the PFC sort groups; extract them if not. Other PFC sessions with waveform features but no sorted units (SC7920240910, SC10020250519, SC8520241110/11) are a fallback.
- **Neuropixels:** not in spyglass. Public datasets (DANDI) would need raw AP-band data, position and a threshold-detection plus snippet-extraction pipeline built for them (item 6d).

## Results snapshot (2026-10-20)

Source files are listed per row, so this table is the only summary needed. All numbers are paired by fold over 5 blocked folds, in nats/s, unless noted.

| Finding | Evidence | Source |
|---|---|---|
| The mark GLM beats the tuned KDE on 3 of 4 recordings (+5–10%) and loses on L14 (−6%) | PR #104 | memory `clusterless-mark-glm` |
| MDN beats the mark GLM on RS58 only: amplitudes +26.0 ± 4.1, PCA-12 +18.5, peakwin +18.0, all 5/5 folds. On j16, Jacob and L14 its validation chooses 0 EM iterations (= the mark GLM). The 50-iteration cap binds on RS58. Fit cost 6–10× | branch `feat/clusterless-mdn` (`28759591`) | `/cumulus/edeno/lowocc/mdn/`, `wffeat/RESULTS.txt` |
| Density-ratio model: RS58 +7.0 ± 5.9, j16 −6.8 ± 17.7; parked | branch `feat/clusterless-ratio` (`612b7d21`) | `/cumulus/edeno/lowocc/ratio/` |
| PCA-12 of snippets beats amplitudes for the mark GLM: RS58 +5.8 ± 1.3, j16 +6.8 ± 3.4, both 5/5. Peak-relative amplitudes (−6) and full snippets (−6) lose. Snippet MDN exceeds 4 h per fold | harness only | `/cumulus/edeno/lowocc/wffeat/RESULTS.txt` |
| Noise-covariance mark floor adds about +1 beyond `reg=0.1` on RS58 (4/5 folds). Amplitude scaling hurts. `reg=0.1` alone adds +0.02 to +1.07 on all four recordings (not a default change) | branch `feat/mark-glm-biophysical` (`0a484924`, `561a67a7`) | `/cumulus/edeno/lowocc/biophys/` |
| KDE diagonal merge (PR #105) is near-free on tetrodes (−0.5 at 26% of kernels) but costs −157 on RS58 at 23% | PR #105 | `/cumulus/edeno/lowocc/compress/sweep_summary.txt` |
| Low-rank merge, full likelihood step, RS58 fold 0, 10 streams, warm, like-for-like evaluator: r=8 at threshold 3 is 4.1× faster than uncompressed for −5.8 nats/s; r=8 at 2.5 is 1.8× faster for +13.8; r=8 at 4 is 11.7× faster for −74. The diagonal merge is 3.9× faster at −162. Cold (compile) time is barely changed by compression. Merge fitting takes about 45 s per stream at r=8 vs 4–5 s diagonal | offline, accurate to 1.9e-4 nats vs float64 | `/cumulus/edeno/lowocc/compress/fullcov/LRTIME_RESULTS.txt` |
| The package's KDE likelihood step is slower than a simpler like-for-like evaluator: uncompressed RS58 9.2 s warm at `block_size=100`, 3.1 s at 1024, vs 1.25 s. Cold: 137 s (about 13 s per stream of compile) | `lrtime` | `LRTIME_RESULTS.txt` |
| Low-rank merge through the detector (`feat/kde-lowrank-merge`, `e81ddee2`; 5 blocked folds, info against the uncompressed KDE's flat refit, paired vs the full KDE): RS58 rank 8 at threshold 2.5: +16.2 ± 1.4 (5/5) at 23% of kernels; rank 8 at threshold 3: −1.3 ± 1.5 at 10%, 2.0× faster warm. j16 rank 2 at threshold 2: +2.7 ± 0.5 (5/5) at 6.5%, 2.2× faster. The diagonal merge on RS58 gives −157 / −283 at the same sizes. Cold (compile) time unchanged; fit about 8× the diagonal merge | item 3d | `/cumulus/edeno/lowocc/rt3d/LOWRANK_DETECTOR_RESULTS.txt` |
| Real-time per-call compiles: the mark GLM recompiled for every new per-electrode spike count; fixed on `perf/streaming-recompiles` (`bf2e0bf7`), with CPU p99 per 1-bin chunk 12.7 → 6.7 ms. Calls with a varying number of bins still recompile (filter and likelihoods) | item 3b | `perf/streaming-recompiles` CHANGELOG |
| Package KDE step speed-ups on `perf/kde-predict-step`: branch-free mark kernel (bit-identical) and grouped exact repair. RS58 warm 3.96 → 2.79 s and compile 211 → 100 s at block 10,000 | item 3c | `/cumulus/edeno/lowocc/rt3c/README.md` |
| Item 3e (memory): KDE peak memory is set by stored spikes × bins (without streaming), decoding spikes × bins and time bins × bins per call, not by `block_size` (about 86 MB at 10,000). `pos_tile_size` never lowered the peak. The fixed default (block 10,000, `pos_tile` None) is within 10% of the fastest setting that fits in all 68 measured cases where it fits. It runs out of memory only on 2-D grids of 2,756 bins or more with 100k–300k stored spikes on 8–24 GB, where streaming (7–12% slower, up to 5.8× less memory) or time chunking fits. Formulas: median held-out error 7% without streaming, 2.4% with. Leave 30% headroom | item 3e | `/cumulus/edeno/lowocc/rt3e/SUMMARY.txt` |
| Real-time KDE latency per call scales with stored spikes × bins, not spikes in the bin: one 5 ms bin, one electrode, takes 5.4 ms at 10k stored spikes and 33 ms at 300k (1-D). The stored-spike position kernel is recomputed every call. The mark GLM's `spike_block_size` 10,000 is 3.4× faster than its default 1,000 at 2-D. `fit_place_grid` takes 509 s at 100 × 100 bins | item 3e | `/cumulus/edeno/lowocc/rt3e/RESULTS.txt` |
| JAX's default matmul precision on A100 is TF32. It caused 20-nat errors in an early low-rank evaluator and made its timings look faster. Benchmarks must use full float32 matmul precision, as the package does | `lrtime` | `LRTIME_RESULTS.txt` |

## Items

### 0. Shared foundation

#### 0a. Expand the recording panel

- **Goal:** test every later item on enough recordings to separate "many channels per group" from "this one animal". That means at least 3 more hippocampal polymer-probe recordings, the 256-channel silicon probe, and at least 2 PFC recordings.
- **Done when:**
  - Each added recording has a harness entry producing the shared protocol's metrics for `markglm:32` and the tuned KDE. Where curated sorted units exist, it also produces them for `sorted_spikes_glm` on those units (needed by item 1c).
  - Channels per sort group are recorded for every recording, existing ones included.
  - Its amplitude marks, and PCA-12 where waveforms were extracted, are saved with a README in `/cumulus/edeno/lowocc/`.
- **Fixed decisions:**
  - Read-only database access; re-extraction goes to scratch (see the database rules above).
  - The spike sets must match the waveform-feature entries exactly, including any artifact or coincidence filter, so every model sees the same spikes. That's how RS58 and j16 were done.
- **Suggested approach:**
  - Start from `/cumulus/edeno/lowocc/rs58/` and the j16 re-extraction scripts in `/cumulus/edeno/lowocc/waveforms/j16/`.
  - Pick hippocampal sessions with clean position (the `pos_quality_*.csv` files) and a linearizable track (`pos_meta_linv1.csv`).
  - For PFC, decode position on the same track. PFC spatial information is expected to be lower, so the comparison is between models on the same data, not across regions.
- **Validation:**
  - **Hard:** extracted snippet centre samples equal the cached marks (RS58 matched exactly; j16 to within one int16 step at save-chunk edges).
  - **Hard:** the harness reproduces any baseline that already exists for that recording.
- **Out of scope:** Neuropixels (item 6d).

#### 0b. Blocked inner-validation tuning

- **Goal:** one harness utility that picks a model's tuning parameters inside the training data. Items 1–6 use it instead of hand tuning or test-fold selection.
- **Done when:**
  - Given a model and a parameter grid, it returns the chosen parameters and the inner validation curve.
  - It reproduces the MDN's leave-one-block-out choice on j16 fold 2 (0 iterations).
  - It recovers known-good KDE bandwidths on RS58 and j16 to within one grid step.
- **Fixed decisions:**
  - Contiguous blocks of equal weighted exposure. Interleaved validation failed on j16 fold 2: it kept rising while the test fold fell.
  - Pooled held-out log likelihood against the matched position-free refit as the criterion. Raw log likelihood misleads; see memory `low-occupancy-real-data-study`.
- **Suggested approach:** generalize the MDN's validation (`clusterless_mdn.py` on `feat/clusterless-mdn`, the `em_validation_blocks` path) into a harness function first. Move it into the package only if item 5b shows it should be a fit option.
- **Validation:** the two reproductions above (**hard**).

### 1. Better likelihood estimators

#### 1a. Test whether the richer models need many channels

- **Goal:** find out whether the MDN's RS58 gain is about channels per group, about RS58, or about this hippocampal recording, using the panel from 0a.
- **Done when:**
  - There's a table of MDN, mark GLM and KDE on every panel recording with waveform features (amplitudes and PCA-12).
  - It's plotted against channels per sort group and spikes per group.
  - The report says which explanation the data support.
- **Fixed decisions:** MDN tuned by its own blocked validation; position-free refit matched to its iteration count. The decision log has why.
- **Suggested approach:** reuse the `wffeat` harness pattern (`/cumulus/edeno/lowocc/wffeat/`); package copy from `feat/clusterless-mdn`.
- **Validation:**
  - **Target:** MDN − mark GLM > fold-to-fold spread on at least 3 recordings with ≥ 16 channels per group would justify item 1b and a PR. Otherwise the MDN stays parked on its branch.
- **Also report:** goodness-of-fit diagnostics from `model_checking/clusterless.py` for the mark GLM, MDN and KDE, to show whether the MDN's RS58 gain comes from a better rate fit or a better mark-conditional fit (review, Area 1.2). This is descriptive; it guides which model to build next.
- **Out of scope:** new model families until this answers the channel question.

#### 1b. Make the MDN cheaper and better (only if 1a supports it)

- **Goal:** bring the MDN's fit cost near the mark GLM's and see whether the iteration cap or fixed component means are limiting it.
- **Done when:** RS58 and the high-channel panel recordings are rerun with:
  - (i) warm-started M-step refits;
  - (ii) a higher iteration cap;
  - (iii) optionally, position-dependent component means.

  Each is reported paired against the current MDN.
- **Fixed decisions:** iteration 0 must stay exactly the mark GLM. That's the fallback that makes tetrodes safe.
- **Context:**
  - memory `em-refit-warm-start-followup`;
  - RS58 fold 2 at a fixed 100 iterations scored 207.7 vs 203.2 at 50.
- **Validation:**
  - **Hard:** iteration-0 equality with the mark GLM (existing test).
  - **Target:** fit time ≤ 3× the mark GLM on RS58.

#### 1c. Clusterless vs sorted on the same recordings

- **Goal:** measure how much spatial information clusterless decoding captures relative to decoding from sorted units, in hippocampus and in PFC, and whether better clusterless likelihoods (1a) or features (2a) close any gap.
- **Done when:** on every panel recording with curated sorted units, the report compares:
  - `sorted_spikes_glm`, plus the KDE if useful, on the curated units;
  - the mark GLM and MDN on clusterless marks from the same sort groups.

  Both are scored on the same time bins and blocked folds, with information per second and per spike and the descriptive decoding metrics. Sorted-unit candidates include RS58 (6 sorted-spikes groups), the Jacob sessions, and the PFC sessions listed under Data.
- **Fixed decisions:**
  - Sorted and clusterless scores are compared on the same time bins, decoded position grid and folds. Each model's information is still against its own position-free refit, so the scores are comparable as information, not as raw log likelihood.
  - Restrict the clusterless spikes to the sort groups the curated units come from. Otherwise the two decoders see different electrodes.
- **Context:** sorted-spikes harness work from the low-occupancy study (memory `low-occupancy-real-data-study`) and the GLM fixes in memory `glm-track-graph-topology-bug`.
- **Validation:**
  - **Hard:** both decoders use identical time bins, folds and position.
  - **Target:** report the clusterless/sorted information ratio per recording. No pass/fail threshold; this is a measurement.
- **Also report:** each clusterless model's held-out information split between spikes assigned to curated units and the rest, scored incrementally under the causal filter (review, Area 1.3). On tetrodes, Ghahari & Eden 2025 found that spikes discarded by sorting carry 48% of the incremental information.

#### 1d. Density-ratio model: parked

It stays on `feat/clusterless-ratio`. Revisit only if 1a shows that richer models matter broadly, and then compare it with the MDN on the same panel. Its `base.py` change (fit the mark reference first, `ad07ca1d`) is useful regardless and could be proposed for main on its own.

#### 1e. Low-rank-plus-noise component covariances in the mark GLM

- **Goal:** test component covariances of the form measured noise covariance plus a rank-r term (Σ_k = Σ_noise + W_k W_kᵀ, from DARTsort; review, Area 1.1). This combines this round's noise-floor and low-rank results.
- **Done when:** RS58 and j16 on PCA-12, and RS58 on a higher-dimensional feature set (D ≥ 50), compare full covariances with Σ_noise + rank r for r ∈ {2, 4, 8}, r chosen by inner validation (0b). Report fit and predict time alongside.
- **Fixed decisions:** opt-in, on a branch from `feat/mark-glm-biophysical`, which has the noise covariance.
- **Validation:** **target:** the gain should grow with mark dimension. If it doesn't, drop the idea.

### 2. Waveform features and biophysical priors

#### 2a. PCA dimension and noise whitening

- **Goal:** find the best compact waveform feature for the mark GLM, and check that it holds beyond RS58 and j16.
- **Done when:** on RS58, j16, Jacob, L14 and at least two 0a recordings, the report covers:
  - PCA dimension 6, 12 and 24, including 24 on tetrodes (Weber 2018 found PCA kept improving up to 10 components on 24 tetrodes; review, Area 2.2);
  - PCA computed after whitening by the measured noise covariance;
  - PCA plus the noise-covariance floor;

  each paired by fold against amplitudes and against PCA-12.
- **Fixed decisions:**
  - PCA is fit on each fold's training spikes only, as in `build_features.py` in `/cumulus/edeno/lowocc/waveforms/`.
  - Jacob and L14 waveforms are extracted here; the user deferred that on 2026-10-19 until it's needed.
- **Suggested approach:** the noise-covariance option is on `feat/mark-glm-biophysical`. The noise covariance comes from spike-free samples, as in `rs58_noise_cov.npz`.
- **Validation:**
  - **Target:** a feature set that beats amplitudes on every fold of at least 4 recordings would become a recommendation for spyglass's waveform features. That's a recommendation, not a package change.
- **Out of scope:**
  - Amplitude scaling (it hurt on real data).
  - Raw full snippets in the mixture models (fits exceed 4 h per fold at 112–252 dimensions).
  - Learned feature encoders, until compact PCA is exhausted.

#### 2c. Detection-threshold sweep

- **Goal:** find how information and speed change as weak spikes are dropped (review, Area 2.1). Fewer spikes per bin is also a real-time speed knob (item 3a).
- **Done when:**
  - First, the current detection threshold t0 of each panel recording's waveform-feature entries is read from its spyglass sorter parameters (read-only).
  - Then spikes are kept at peak amplitude ≥ t0 + {0, 10, 20, 40, 60} µV, on j16, RS58, Jacob and L14. Each set is refit and scored against its own matched position-free refit, with spikes per 6 ms bin alongside.
- **Follow-up:** if information is still rising as the threshold falls to t0, re-detect from raw data below t0 for one recording. That's a recommendation for spyglass's thresholds, not a package change.

#### 2b. Biophysical priors on high-density probes

Fold into item 6a. Locality and geometry priors only make sense once the channel-grouping question is answered. The 8-feature localized set cost −4 but fitted 3× faster; keep it as a cost option for item 3.

### 3. Real-time speed/accuracy trade-offs

#### 3a. Per-bin latency benchmark

- **Goal:** measure what matters in real time: the latency of one causal-filter time step, likelihood plus filter update, after compile. Measure on GPU and CPU, for realistic spike counts per bin.
- **Done when:** a table gives latency (median and 99th percentile) per decode step for:
  - the mark GLM on amplitudes and on PCA-12;
  - the KDE at `block_size` 100 and 1024;
  - the diagonal-merge KDE;
  - the low-rank-merge KDE (if 3d proceeds);
  - an 8-feature localized mark GLM.

  The benchmark covers 2, 4 and 6 ms bins, on both CPU and GPU, because the real-time hardware isn't fixed. The same table shows each model's information score. Recordings:
  - j16 (tetrodes);
  - RS58 and the largest polymer-probe recording from 0a (most channel groups and the highest spike rate);
  - a Neuropixels recording once 6d delivers one.

  Before real Neuropixels data exist, extrapolate with a synthetic scaling sweep. Its axes are spikes per bin, number of channel groups, mark dimension and number of position bins, covering Neuropixels-scale values (hundreds of spikes per 4–6 ms bin across a probe). Latency is then stated as a function of those numbers, not only per recording.
- **Fixed decisions:**
  - Compile time is reported separately and never mixed into the per-step latency. It's 12–15 s cold vs under 0.1 s warm, so mixing them would hide everything.
  - Polymer probes and Neuropixels are both targets. Use the channel grouping chosen in item 6a for dense probes, since grouping sets both the number of electrodes and the mark dimension. Until 6a reports, benchmark the current whole-group setup and the 8- and 16-contact blocks.
- **Reference point:** RealtimeDecoder (Chu et al. 2025, `LorenFrankLab/realtime_decoder`; review, Area 3.1). On tetrodes it measured a 1.72 ms median posterior update with 6 ms bins and 41 position bins on a 32-core CPU, and its cost grew about 0.5 ms per 10,000 encoding spikes.
  - It's fine to install and run it in a separate environment. It is a reference, not a strict bar (user, 2026-10-20). Its implementation may have its own issues, so check its numbers before relying on them.
  - Rebuild its estimator in the harness as well, so it can also be scored for information like every other model.
  - Copy its stage-by-stage timing: transfer, likelihood and posterior update, each with median and 99th percentile.
- **Also measure** (review, Areas 3.2, 3.4 and 3.5):
  - time from event onset to a confident causal decision (filter-only non-local probability above 0.8), and its agreement with the offline smoother;
  - CPU latency as the thread count and the number of electrode groups vary, to confirm the work is parallelized across electrodes;
  - for mixtures with hundreds of components (whole-shank or Neuropixels), latency and information lost when each spike evaluates only its top-k components (**hard:** identical at k = K).
- **From item 3e:** cache the stored-spike position kernel across calls (it depends only on the fitted model and the position grid). That is the KDE's real-time bottleneck: per-call latency grows with stored spikes × bins. Also check the mark GLM's `spike_block_size` default (10,000 was 3.4× faster than 1,000 at 2-D).
- **Constraint for Neuropixels:** real-time preprocessing must be causal, so IBL's `destripe` (it filters forwards and backwards) can't be used (review, Area 3.6).
- **Validation:** **target:** for each of CPU and GPU, and for each probe type (tetrode, polymer, Neuropixels-scale), identify settings whose per-step latency (99th percentile) fits inside a 4–6 ms bin, within 5% of the best model's information. If none do, report the closest. Include host-to-device transfer in the GPU latency: at a few milliseconds per step it can dominate, which may make the CPU the better real-time choice.

#### 3b. Avoid recompiles

- **Goal:** keep decode calls on compiled code paths when spike counts change from bin to bin or chunk to chunk.
- **Done when:** a demonstration that streaming chunks of varying spike count don't recompile, measured by compile counts and cold-vs-warm timings. The likely approach is padding spike arrays to a small set of bucket sizes; padded entries carry zero weight.
- **Context:** PR #96 (`fix/kde-tiled-path-recompiles`, merged 2026-10-17) already pads the KDE's tiled and streamed predict shapes so they share compiled kernels, and `df79fe47` pads the occupancy KDE samples for EM refits. So start by measuring what still recompiles on `main` when spike counts vary per call, for the KDE, the mark GLM and the causal filter step. Fix only what's left.
- **Validation:**
  - **Hard:** outputs match the unpadded path bit for bit, or within float32 rounding, stated.
  - **Hard:** zero recompiles after warm-up across the test sequence.

#### 3c. KDE `block_size` default

- **Goal:** decide whether `block_size=100` (`clusterless_kde.py:1436`, `:1574`, `:1805` and `:2129` on `main`) should change, and why the package's step is slower than a simpler evaluator of the same quantity. On RS58 the package took 9.2 s warm at block 100 and 3.1 s at block 1024, against 1.25 s for the `lrtime` evaluator, which matches it to 1e-4 nats. Candidate causes to measure: block size, the exactness gate and repair rows, and per-tile overhead.
- **Done when:**
  - Warm time and peak GPU memory are measured over block sizes {100, 256, 1024, 4096} on all panel recordings, on GPU and CPU.
  - A default is proposed with the measurements, or the reason to keep 100 is given.
- **Validation:**
  - **Hard:** log likelihoods identical across block sizes, or within float32 rounding, stated.
  - **Hard:** no memory regression on the smallest GPU we support.
- **Out of scope:** changing the algorithm.

#### 3e. Memory-aware block and tile sizes (offline and real time)

- **Goal:** know the fastest `block_size` and `pos_tile_size` that fit a given GPU's memory, for offline decoding (long sessions, many electrodes, 2-D grids) as well as real time. Then decide whether the package should choose them automatically.
- **Context:**
  - On RS58 (158 bins) on an 80 GB A100, warm time plateaus between `block_size` 4,096 and 10,000 and gets slower at 30,000; peak memory is about 0.9 GB.
  - From the code (`clusterless_kde.py` on `perf/kde-predict-step`), the per-block working set grows with block_size × (2,048 encoding rows per tile + a few × n_position_bins). So 2-D grids and smaller GPUs are where memory binds.
  - That growth estimate is from reading the code, not measured.
- **Done when:**
  - Peak device memory and warm time are measured over `block_size` × `pos_tile_size` × n_position_bins (1-D tracks up to 2-D 100 × 100 grids) × encoding spikes × mark dimension.
  - A fitted formula predicts peak memory within 20% across that grid.
  - GPU memory limits of 8, 16, 24 and 80 GB are emulated with `XLA_PYTHON_CLIENT_MEM_FRACTION`, as phase 9b did for 24 GB (memory `phase-9b-em-decisions`).
  - Report the fastest setting that fits for each limit and problem size, and whether one fixed default is within 10% of that everywhere.
- **Fixed decisions:**
  - Measurement first, with no package change.
  - An automatic choice is added only if a fixed default is materially slower or runs out of memory in realistic cases. It would be opt-in (`block_size="auto"`), and its outputs must match the fixed path to float32 rounding (block sizes already agree to within 2.4e-4).
- **Validation:**
  - **Target:** the memory formula is within 20% on held-out problem sizes.
  - **Hard:** any automatic choice never exceeds the memory limit in the measured grid.

#### 3d. Low-rank merge in the package (depends on `lrtime`)

- **Goal:** if `LRTIME_RESULTS.txt` confirms a real warm speed-up at small accuracy cost, add a low-rank-plus-diagonal merge option to the KDE compression. Branch from `feat/kde-compression`, or from `main` if PR #105 has merged.
- **Done when:**
  - The option exists, off by default.
  - It has tests against an exact dense evaluation.
  - It's evaluated through the detector on RS58 and j16 with the shared protocol.
- **Fixed decisions:**
  - Use the form that keeps the top-r directions of each kernel's covariance in excess of the base bandwidth, built during the merge. Flooring the leftover diagonal at the base bandwidth adds variance and got worse with rank.
  - The fast evaluation must match the exact float64 evaluation to < 0.01 nats. The first GEMM version was off by up to 20 nats from float32 cancellation.
- **Context:** `/cumulus/edeno/lowocc/compress/fullcov/lowrank_lib.py`; unexplained: a few online runs at rank ≥ 27 kept very few kernels.
- **Also try:** choosing which kernels to merge by a symmetric KL (or Runnalls-type) cost instead of the fixed Mahalanobis threshold, at equal kernel budgets (23% and 10%) (review, Area 3.3). Rezaei et al. 2021 showed it for posterior mixtures; applying it to encoding kernels is untested.
- **Validation:**
  - **Hard:** the exact-match check above.
  - **Hard:** the r = D limit reproduces the full-covariance merge.
  - **Target:** through the detector, ≥ 2× warm speed-up within 5 nats/s of the uncompressed KDE on RS58. Offline, r=8 at threshold 3 gave 4.1× for −5.8 nats/s.
- **Carry over from `lrtime`:**
  - full float32 matmul precision, not JAX's TF32 default;
  - an exact recompute for spikes whose fast reduction underflows (about 0.5% of RS58 held-out spikes), as the package's exactness gate does;
  - position kept as its own per-kernel variance, outside the low-rank mark covariance (costs < 0.6 nats/s).
- **Cost to state in the docs:** fitting the low-rank merge takes about 10× longer than the diagonal merge (about 45 s per RS58 stream at r=8).

### 4. Scaling and state-space models

#### 4a. Inventory existing state-space work

- **Goal:** know what already exists before proposing anything.
- **Done when:** a short status note covering each of:
  - the velocity/momentum decoding plan (`.claude/docs/plans/velocity-momentum-decoding/` on branch `feat/velocity-decoding`, latest `12f8e6d6`);
  - the uniform time-edges branch (`origin/feat/time-edges-uniform-bins`);
  - checkpointed EM and structured transitions on `main`.

  For each: what works, what's measured, what's blocked.
- **Context:** memories `velocity-state-decoding-exploration`, `phase-6a-time-edges-status`, `phase-9b-em-decisions`.

#### 4b. Measure scaling

- **Goal:** find where time and memory blow up as recording length and state-space size grow.
- **Done when:** filter and smoother time and peak memory are measured against:
  - session length (10 min to several hours);
  - number of position bins (1-D track, and 2-D grids up to at least 100 × 100);
  - number of discrete states,

  on GPU and CPU. Each bottleneck is named: likelihood, transition step or memory.
- **Known already (item 3e):** `Environment.fit_place_grid` takes 509 s at 100 × 100 bins and 28 s at 50 × 50, so environment setup is itself a 2-D bottleneck.
- **Suggested approach:** synthetic spikes are fine for timing. For memory, use `memory_analysis()` as in memory `jax-vmap-vs-laxmap-peak-memory`.

#### 4c. Faster transitions for large grids (depends on 4b)

- **Goal:** if 4b shows the dense transition step dominating on 2-D grids, replace the random-walk step with a structured operator. The diffusion likelihood's spectral engine is a starting point (`clusterless_diffusion`).
- **Fallback only:** Gaussian-mixture posterior filters (Yousefi 2019, Rezaei 2021) report large speed-ups, but against naive grids. They ignore track topology and have variable per-step cost, which hurts a 99th-percentile real-time budget (review, Area 4.2). Consider them only if a structured operator fails on 2-D.
- **Validation:**
  - **Hard:** matches the dense transition within a stated tolerance on small grids.
  - **Hard:** transition rows stay stochastic.
  - **Hard:** posterior differences on a real decode below a stated tolerance, with numerical-validation sign-off.

#### 4d. Better state-space models

Theta-sequence and velocity states continue under the velocity/momentum plan, not here. This roadmap only records that the evaluation protocol's descriptive metrics (coverage, non-local fraction while running) are the ones those states should change.

#### 4e. LFP observations in the discrete-state model

- **Goal:** test whether adding a ripple-band LFP power observation per discrete state improves causal replay detection, as Zeng & Eden 2025 did (review, Area 4.1). The package has no LFP observation model. This matters mainly for real time.
- **Done when:** on j16 and RS58 (5 blocked folds), an opt-in LFP likelihood is compared with spikes only, on:
  - held-out marginal log likelihood of spikes plus LFP, paired by fold;
  - causal detection latency, and agreement with the offline smoother's non-local events.
- **Fixed decisions:** the shared protocol's `info_vs_flat_per_s` scores only the spike likelihood, so it doesn't apply here; use the two scores above.

### 5. Estimating assumed parameters

#### 5a. Random-walk variance by EM

- **Goal:** estimate each continuous state's random-walk variance from data, especially the non-local (replay) state, instead of assuming `movement_var=6.0` (`continuous_state_transitions.py:158` on `main`). `estimate_movement_var` (`:37`) derives it from behaviour, which doesn't describe replay. EM currently estimates initial conditions, discrete transitions and, optionally, the encoding model (`models/base.py:3491-3493`).
- **Done when:**
  - A variance estimate recovers known variances in simulation, including near track boundaries and on track graphs.
  - Two routes that avoid storing two-slice marginals are compared on small grids, where they should agree (review, Area 5.1):
    - EM with an expected transition-count matrix accumulated inside the backward pass, then a 1-D search over the variance (the grid version of COMPASS's M-step). The current smoother (`core.py`, `_smoother_impl`) accumulates no transition counts.
    - Gradient of the marginal log likelihood with respect to log variance, with `jax.checkpoint` on the filter scan. This needs the track-graph transition to be differentiable in the variance.
  - Real-data estimates are reported for the non-local state on the panel, with their effect on decoded replay speed.
- **Fixed decisions:** it's opt-in and off by default, like `estimate_encoding_model`.
- **Validation:**
  - **Hard:** EM log likelihood is non-decreasing in simulation.
  - **Target:** recovery within 10% of the true variance at realistic replay durations.
- **Risk:** the transition-count route needs O(n_bins²) memory per continuous state. That's fine in 1-D; check 2-D sizes against item 4b first.
- **Note:** none of the reviewed authors estimated replay dynamics; they fixed them by hand, so there's no published value to check against.

#### 5b. Encoding tuning parameters by inner validation

- **Goal:** use item 0b to choose KDE bandwidths, the mark GLM's log-rate prior and knot spacing, and `mark_reg_covar` per recording. Then decide whether any should become a fit-time option in the package.
- **Done when:** the chosen values, and their information against the current defaults, are reported on the panel. Include one arm that chooses the mark GLM's log-rate prior and knot spacing by marginal likelihood instead (review, Area 5.2). Expect little: the sorted-MRF version gained nothing on real data (memory `per-neuron-reml-not-adopted`).
- **Fixed decisions:** default changes need every-fold gains on at least 3 of 4 recordings (the promotion bar). The `reg=0.1` check (+0.02 to +1.07) didn't meet it.

#### 5c. EM with encoding refits

- **Goal:** fix the non-monotone EM seen on j16 when the encoding model is refit inside EM.
- **Context:** memory `em-refit-warm-start-followup`; overlaps with item 1b's warm starts.
- **Validation:** **hard:** EM log likelihood non-decreasing on the j16 case that failed.

### 6. High-density probes

#### 6a. Channel grouping on RS58, then the 256-channel probe

- **Goal:** decide how to turn a dense probe into clusterless "electrodes".
- **Done when:** the mark GLM and MDN are compared, on the shared protocol, across:
  - (i) whole sort group (current);
  - (ii) fixed blocks of 8 and 16 contacts, each spike assigned to one block by its peak channel;
  - (iii) whole shank with marks = depth + PCA shape of the peak-centred snippet;
  - (iv) overlapping blocks with each spike counted once;
  - (v) one mixture per shank over channel neighbourhoods, DARTsort-style (review, Area 6.1): each spike is observed on its peak-channel neighbourhood, with low-rank-plus-noise Gaussians on temporal-PCA features and the unobserved channels marginalized. It's the most promising option and the most work, so validate it on item 6c's simulation first.

  Run on RS58 first, then IM-1478 or IM-1594 and one other polymer recording from 0a.
- **Fixed decisions:** each spike is counted once in every grouping, because double counting inflates evidence.
- **Context:**
  - Peak-relative amplitudes plus depth lost information on RS58 (mark GLM −6, MDN −14).
  - Point-source localization features lost 25–35%.
  - So (iii) needs shape features, not just amplitudes.
- **Validation:** **target:** a grouping that matches or beats whole-group amplitudes on every fold, at lower dimension.

#### 6b. Drift

- **Goal:** measure how much marks drift over a session on polymer probes, and handle it if it matters.
- **Done when:**
  - Per-unit-cluster mark drift is measured across the session on RS58 and 0a recordings. RS58 sort group 10 was stable, but that's one group.
  - If drift is material, the report compares, in rising cost (review, Areas 6.2–6.4):
    - DREDge registration before feature extraction (SpikeInterface `correct_motion(preset="dredge")`);
    - time-discounted encoding (Arai), using the fit functions' existing `weights` argument with weights decaying with time before the test block, and the decay time chosen by inner validation. This uses forward-chaining folds (see the shared protocol);
    - a mixture whose component means follow a random walk over time (Calabrese & Paninski 2011), only if drift persists after registration.
- **Validation:** **target:** report the drift magnitude relative to cluster width, and the information gain from correction if any.

#### 6c. Simulated dense probe with known place fields

- **Goal:** ground truth for grouping (6a) and drift (6b): how close each method gets to the true likelihood.
- **Done when:** a simulated probe recording exists, with known units, place fields and an optional drift trajectory, in the RS58 geometry. The decoder harness runs on it, and the true-intensity information is reported as the ceiling.
- **Suggested approach:**
  - MEArec was blocked earlier (memory `clusterless-curse-of-dimensionality`).
  - Try spikeinterface's ground-truth recording generator.
  - The harness `mearec_gen_recording.py` and `shank_mark_sweep --recording-npz` (memory `clusterless-curse-real-recordings`) is a starting point.
- **Validation:**
  - **Hard:** the true-intensity likelihood scores at least as high as every fitted model (otherwise the simulation or the scoring is wrong).

#### 6d. Neuropixels from DANDI

- **Goal:** test the methods on a real Neuropixels hippocampal or PFC recording during navigation.
- **Done when:**
  - (i) A feasibility note: candidate DANDI datasets with raw AP-band data, tracked position on a track or maze, and hippocampal or PFC placement, with sizes and licences. Dataset IDs must be checked on DANDI, not recalled from memory.
  - (ii) For one dataset: threshold detection, snippet extraction, motion correction and features, using the same pipeline choices as 0a where possible.
  - (iii) The 1a/2a/6a comparisons run on it.
- **Fixed decisions:** waveform clips won't be in these datasets, so (ii) is a real extraction pipeline. Size it with a one-segment smoke test before processing a full session.
- **Validation:** **hard:** extraction reproducibility. Re-running on one segment gives identical spikes and snippets.

## Suggested order and dependencies

1. **Now:**
   - 0a (panel expansion) and 0b (inner-validation tuning): everything else needs them.
   - 3c (block size), which is cheap.
   - 4a (inventory), which is reading.
   - 3d: `lrtime` met the offline target, so it can start (branch from `feat/kde-compression`).
2. **Next:** 1a, 1c, 2a and 2c on the expanded panel, run together on the same jobs. Also 6a on RS58 and the 256-channel probe, 3a, and 4b.
3. **Then:**
   - 5a (needs 4b's memory numbers);
   - 5b (needs 0b);
   - 6b and 6c;
   - 1b only if 1a supports it;
   - 3b and 4c if 3a and 4b show the need;
   - 1e after 2a (it builds on the best PCA features);
   - 4e, which is independent and can run any time after 0a.
4. **Last:** 6d, the largest data-engineering cost, once the methods it would test are settled.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-19 | All new models and options are opt-in; defaults unchanged | user; protects existing analyses |
| 2026-10-19 | Score against the matched position-free refit, not a separately built null | null checks showed residual non-positional structure in separate nulls |
| 2026-10-19 | MDN and density-ratio model stay on their branches, no PRs | both miss the promotion bar; gains only on RS58 |
| 2026-10-19 | Commits on feature branches are pre-approved; push, PR and main need asking | user |
| 2026-10-19 | Jacob and L14 waveforms deferred until needed | user; needed now by item 2a |
| 2026-10-20 | Coverage and non-local fraction while running are descriptive only | user: theta sequences make non-local decoding during running legitimate |
| 2026-10-20 | `reg=0.1` not adopted as default | +0.02 to +1.07 nats/s on all four recordings, below the bar |
| 2026-10-20 | Amplitude scaling dropped; full-snippet mixture models dropped | hurt on real data; exceeded 4 h per fold |
| 2026-10-20 | Expand to more polymer-probe hippocampal recordings and to PFC; Neuropixels later via DANDI with its own extraction | user |
| 2026-10-20 | PFC recordings should also have sorted units, so clusterless can be compared with sorted decoding (item 1c) | user |
| 2026-10-20 | PFC sessions must have position for the decoded epochs | user |
| 2026-10-20 | No preferred Neuropixels dataset; item 6d picks one by its stated criteria | user |
| 2026-10-20 | Real-time target: 4–6 ms bins; hardware open (CPU or GPU, whichever works better in real time) | user |
| 2026-10-20 | Real-time must work on polymer probes and Neuropixels, not only tetrodes | user |
| 2026-10-20 | RealtimeDecoder is a reference point for real-time latency, not a strict bar; it may be installed in a separate environment | user; its implementation may have its own issues |
| 2026-10-20 | Forward-chaining folds allowed for time-discounted encoding | user |
| 2026-10-20 | Added items 1e, 2c and 4e and extra measurements from the literature review | `literature-review.md` |
| 2026-10-20 | Low-rank merge (r=8, threshold 3) meets item 3d's offline speed/accuracy target: 4.1× warm for −5.8 nats/s | `LRTIME_RESULTS.txt` |
| 2026-10-20 | Item 3d's target met: low-rank merge rank 8, threshold 3 on RS58 gives 2.0× warm for −1.3 nats/s. The package evaluates the low-rank term from direct differences (offset-safe), which halves lrtime's 4.1× | `LOWRANK_DETECTOR_RESULTS.txt` |
| 2026-10-22 | Keep fixed KDE `block_size`/`pos_tile_size` defaults; no `block_size="auto"`. Any automatic memory handling should choose streaming and time chunking from the measured formula | item 3e results |
| 2026-10-20 | KDE `block_size` default 100 → 10,000, matching what detectors already pass (`perf/kde-predict-step`, `2474c8aa`) | user approved; measurements in `/cumulus/edeno/lowocc/rt3c/` |
| 2026-10-22 | Low-rank merge keeps the exact, offset-safe evaluation (direct differences). The faster GEMM form isn't pursued | user agreed; the GEMM form loses accuracy when kernels are far from the origin, and 2.0× already meets the target |
| 2026-10-22 | Open PRs #106 (`perf/streaming-recompiles`) and #107 (`perf/kde-predict-step`); #107 merges after #105 and is rebased onto it | user |

## Open questions

- Which PFC sessions have position for the same epochs as their PFC waveform features and sorted units? The survey counts waveform entries per session, not per region. Item 0a answers this; sessions without position are dropped.
