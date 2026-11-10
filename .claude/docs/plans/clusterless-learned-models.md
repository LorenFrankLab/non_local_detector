# Clusterless observation models: compressed KDE, learned (neural) models, richer marks

Written 2026-10-19 at the end of the session that merged `clusterless_mark_glm` (PR #104) and
the GLM track-graph fixes (PR #103). It carries the context a fresh session needs. Nothing in
it has been started.

## Goal

Find observation models that give more position information per spike on high-channel-count
probe data than the current best (`clusterless_mark_glm`, then a tuned `clusterless_kde`), at a
cost that scales to Neuropixels-sized recordings. Along the way, answer three questions:

1. Can the KDE we trust be made cheap by subsampling or compressing its training spikes?
2. Do learned (neural) models beat the mark GLM?
3. Do richer marks (full waveforms instead of amplitudes) and biophysical structure
   (noise correlations, amplitude scaling, locality along the shank) make better models?

Everything new stays opt-in. The package default does not change.

## Decisions so far (user, 2026-10-19)

- **Do now:**
  - KDE subsampling: uniform and smarter (stratified).
  - Kloosterman-lab kernel compression (Sodkomkham et al. 2016), as a KDE option regardless of
    how it ranks.
  - The binary density-ratio model ("Backend A" below).
  - A neural density estimator.
  - Full-waveform marks and biophysical priors.
- **Not now:**
  - Re-extracting spikes at a lower detection threshold. It's the largest measured loss, but
    it's deferred.
  - Position-aware responsibilities in the mark GLM. Earlier the user pointed to issue #31,
    where the direct marked-intensity backend is deferred; don't build it unless asked. A
    mixture density network whose component means don't depend on position is closely related.
    Keep it as a comparison point, not a goal.
- **Skip:** mark quantization (it's a hard-assignment mark GLM), C3PO embeddings into the KDE,
  and the marked-point-process VAE for now.
- **Dependencies:** a neural training library (optax, possibly flax) goes in as an *optional*
  extra, for example `[neural]`. Base installs must import and run without it.
- **Data:** waveforms come from spyglass, read-only (details below). `/cumulus/edeno/clusterless_high_d`
  can be used and extended. The raw NWB files in `/stelmo/nwb/raw` also exist.
- **Code location:** model code on local feature branches from `main`, one git worktree each,
  for example `feat/kde-compression`, `feat/clusterless-ratio`, `feat/clusterless-mdn`. Data
  extraction and real-data harnesses stay as scratch scripts on zephyr or in
  `clusterless_high_d`. Nothing is pushed or opened as a PR until a model earns it.
- **Repository rules still apply** (CLAUDE.md): ask before committing, pushing, updating
  snapshots or changing tolerances; use `uv run`; use the scientific-tdd and
  numerical-validation skills for likelihood code; run targeted tests, not the full suite.

## Where things stand

Package (`main` at `f66c42dc`):

- `src/non_local_detector/likelihoods/clusterless_mark_glm.py`: the mark GLM. Each electrode
  gets a variational-Bayes Gaussian mixture over its marks alone, a spike-weighted scikit-learn
  subclass with weighted priors and bound. Each component gets a Poisson GLM rate map through
  `sorted_spikes_glm._fit_rate_maps`. The intensity is λ(x, m) = Σ_k λ_k(x) N(m; μ_k, Σ_k) + β·q(m).
  Prediction cost is components × bins and doesn't depend on the number of training spikes.
- `clusterless_kde.py` (`fit_clusterless_kde_encoding_model`, `predict_clusterless_kde_log_likelihood`):
  log-space KDE, the model we trust.
- `clusterless_diffusion.py`: topology-aware spatial smoothing, but it still compares mark
  kernels against every training spike.
- `mark_reference.py`: the background β·q(m) that every clusterless active state includes
  (from PR #79). New backends must use the same reference so likelihoods are comparable
  across states.
- Registry: `_CLUSTERLESS_ALGORITHMS` and `_CLUSTERLESS_MARK_REFERENCE_SCALES` in
  `likelihoods/__init__.py`. A new backend isn't reachable until it's registered there.
- Contract tests pick up registered backends automatically; some model tests use hard-coded
  algorithm lists. #104 added the mark GLM to those lists, which shows which ones.

Results to beat: flat-referenced held-out position information, nats/s, 5 blocked folds.

| Recording | Marks per electrode | Mark GLM | Tuned KDE | Notes |
|---|---|---|---|---|
| j1620210710 02_r1 (22 tetrodes, bandit maze) | 4 | +5.5 ± 7.9 over KDE | — | 1-D track graph |
| Jacob20250618 (7 probes, W-track) | 4–5 | +1.8 ± 1.7 over KDE | — | sorted units carry far more |
| L14 20240604 (4 probes) | 5–6 | −5.6 ± 5.7 vs KDE | — | deficit is spatial; the mark fit is better |
| RS5820251203 epoch 6 (11 CA1 shanks) | 26–32 | 177.9 | 169.2 (`waveform_std` 24) | W-track, 1.03M spikes in 20 min |

Further RS58 findings:

- **Compact marks:** centre-of-mass depth plus the 7 contacts around the peak gives the mark
  GLM 174.1 with 8 numbers. One- to three-number localization summaries lose 30–40%.
- **Components:** a cap of 64 gives 176.2 and 128 gives 166.6, so 32 is enough. At 31-D the
  variational pruning keeps every component the cap allows.
- **KDE bandwidth:** on z-scored marks it must widen with dimension (0.25 at 3–4-D, 0.5 at
  7–8-D).
- **Decoding:** the median MAP error while running is about 5–6 cm for both models. The KDE
  labels 53% of running time non-local and the mark GLM 34%, with the "non-local" posterior
  sitting on the animal.
- **Encoding-level evidence:** +0.31 nats per spike for the mark GLM, +0.26 for the KDE.
- **Detection:** the 100 µV threshold misses 8–76% of sorted spikes per shank (deferred, as
  above).

## Evaluation (same for every model)

- **Data:** the four recordings above, with the same 5 blocked time folds and transitions held
  fixed.
- **Settings choice:** any setting a model needs (bandwidth, dictionary size, network width,
  rank, training epochs) is chosen on validation blocks inside the training folds, never on the
  test fold.
- **Primary metric:** held-out marginal log likelihood of the full detector minus that of the
  same model refit with flat rate maps, per second.
  - Flat KDE: `position_std=1e4`. Flat GLM or mark GLM: `log_rate_prior_std=1e-3`. A new
    backend needs its own flat version, with spatial structure removed and everything else
    kept.
  - Before believing a big gain, check that the flat model's information is about zero. A
    hand-built null that left out β·q(m) once inflated 31-D results fourfold.
- **Also report:**
  - encoding-level bits per spike (Pillow-style): held-out log likelihood at the animal's true
    position minus the flat model, per spike, divided by ln 2;
  - median decoding error while running (speed > 10 cm/s), measured along the track graph;
  - 50% and 95% posterior coverage;
  - fraction of running time labelled non-local;
  - fit time, predict time and peak memory, and how they grow with spikes and mark dimension.
- **Baselines in every table:** tuned `clusterless_kde`, `clusterless_mark_glm`,
  `clusterless_diffusion`.
- **Promotion to a PR:** a model must beat the mark GLM by more than the fold-to-fold spread on
  at least 3 of 4 recordings, or match it at clearly lower cost. A miss gets reported, not
  argued around.

## Work items

These are roughly in order, but dependencies are loose: waveform extraction can run alongside
the model work.

### KDE subsampling and Kloosterman compression

**Goal:** curves of information and run time against dictionary size, to see whether a small
dictionary matches the full KDE.

**Variants:**
- uniform subsampling;
- stratified subsampling that keeps rare marks and rarely visited places;
- the Sodkomkham et al. 2016 merge, which combines kernels within a distance threshold and
  keeps their summed weights.

**Notes:**
- The lab's implementation (py-compressed-kde) is a C++ extension in a Bitbucket repository
  with no stated license. Reimplement from the paper; don't copy code.
- Keep the true exposure and mean rates: a smaller dictionary must not look like a shorter
  recording.
- The expected-rate (no-spike) term must come from the compressed model itself, so the
  likelihood stays coherent.
- This belongs as an option on `clusterless_kde`, not a new backend.
- Dictionary sizes to sweep: about 500 to 30,000 per electrode.

### Biophysical structure in mark models

Test these first in the mark GLM, because it's cheap.

- **Noise covariance:** measure the channel noise covariance from spike-free stretches, then
  whiten marks with it or use it as each component's base covariance. Neighbouring channels
  share noise.
- **Amplitude scaling:** give components extra variance along their own mean direction,
  because bursts shrink amplitudes.
- **Locality:** use peak-centred windows plus depth.
- **Same shape anywhere on the shank:** for neural models, share weights along the shank (a
  1-D convolution over channels) and give depth as a separate input.
- **Drift:** a slow, time-dependent depth offset. Optional and later.

Each idea is an ablation against the plain mark GLM with the same marks.

### Backend A: binary density ratio with a spatial log basis

- **Model:**
  - The score is s(x, m) = a_θ(m)·φ(x). φ is a fixed spatial basis that includes a constant;
    on track graphs, reuse the branch-spline basis from `sorted_spikes_glm`. a_θ is a mark
    encoder: linear first, then a small MLP, then the shank-convolution encoder.
  - Train a class-balanced logistic classifier. Positives are (position, mark) pairs from
    encoding spikes. Negatives pair a position drawn from *exposure-weighted occupancy* with an
    independent mark from the reference q(m). Don't shuffle spike positions: that samples
    spike-triggered position, which is a different distribution.
  - Calibrate the rate: log R(x, m) = s(x, m) + log Λ̂(x) − log E_q[exp s(x, ·)]. Λ̂ is a spatial
    rate map, for example the GLM's ground rate, and the expectation uses a weighted bank of
    reference marks. Cache the correction field.
- **Prediction:** per time bin, sum the encoder outputs of that bin's spikes, multiply by φ
  once, add the per-spike correction and the no-spike term. Use bounded row blocks, and honour
  `row_slice` and spike ownership like the existing backends.
- **Diagnostics to save:**
  - mismatch between the uncalibrated ground rate and the rate map;
  - convergence of the reference integral across bank sizes, and its effective sample size;
  - held-out spatial scores.
- **Reference measure:** the reference must be the same q(m) the background uses for this
  electrode. Otherwise local, non-local and no-spike states are compared on different
  measures.

### Neural density estimator: mixture density network for p(m | x)

- **Model:** λ(x, m) = Λ(x) · Σ_k π_k(x) N(m; μ_k(x), Σ_k), plus the background β·q(m).
  - Λ and π_k come from smooth functions of position, on the same spatial basis.
  - μ_k(x) is constant at first; a small position dependence is an optional check on
    overlapping units.
  - It's normalized exactly, so it needs no reference bank.
- **Cost:** per spike, K Gaussians × bins. Fine in 1-D; it needs tiling in 2-D.
- **Later, if mark shape turns out to matter:** a normalizing flow for p(m | x).
- **Relation to the mark GLM:** with constant means this is a jointly fitted mark GLM. Report
  it next to the mark GLM so the difference is attributable.
- **Match Backend A:** same marks, folds and basis, and a comparable capacity and tuning
  budget.

### Full-waveform marks through spyglass (read-only)

All 11 RS58 epoch-6 sort groups (0–6 and 8–11) have preprocessed recordings on disk, 8–10 GB
each, checked 2026-10-19.

**Steps:**
1. Load each recording and the clusterless-thresholder spike times with
   `SpikeSortingOutput().get_recording(merge_key)` and `get_sorting(merge_key)`.
2. Extract waveforms with `si.extract_waveforms`, as
   `spyglass/decoding/v1/waveform_features.py::_fetch_waveform` does. Spyglass's default
   window is ±0.5 ms. Keep every spike, and write to a scratch folder on `/cumulus`, not
   spyglass's temp folder.
3. Keep each spike's peak contact and its neighbours (about 1M spikes × about 30 samples ×
   about 9 channels, 1–2 GB).
4. Match the snippets to the cached amplitude marks by spike time, including the
   coincident-artifact filter, so every model sees the same spikes.

**Feature sets to compare:**
1. amplitudes;
2. peak window plus depth;
3. waveform PCA;
4. full snippets into the neural encoders.

**Sources:**
- Sort-group merge IDs, recording IDs and analysis file names:
  `/cumulus/edeno/lowocc/dbsurvey/wf_hpc_sortgroups.csv` (rows with `rec_interval == "06_r3"`).
- Probe geometry: `/cumulus/edeno/lowocc/rs58/rs58_geometry.json`.

**Cautions:**
- `SpikeSortingRecording.get_recording` silently *regenerates* a missing analysis file, which
  is a write. Check `AnalysisNwbfile.get_abs_path(...)` exists before reading.
- Never insert or populate in the database; zephyr also hosts the lab database VM.
- Confirm the j16 files the same way before extracting them.

## Data, harnesses and infrastructure

- **Hosts:** zephyr or breeze (10 × A100 80 GB, 96 cores). Zephyr hosts the lab database VM:
  renice jobs (`nice -n 10`), cap threads (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` = 6–8),
  run long jobs in tmux, and check `nvidia-smi`, load and memory first.
  - GPUs 0, 5 and 6 were occupied by others on 2026-10-19.
  - Pin `CUDA_VISIBLE_DEVICES` and set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or concurrent
    jobs run out of memory.
- **Package code on zephyr:** don't switch `/cumulus/edeno/non_local_detector`. Copy a commit
  with `git archive <sha> src | ssh zephyr 'tar -x -C /tmp/<name>'` and put it on
  `PYTHONPATH`. Also copy `_version.py` from an existing copy such as
  `/tmp/nld-mark-glm-f0cac0d5`; the copy matching `main`'s mark GLM is
  `/tmp/nld-mark-glm-f0cac0d5/src`.
- **RS58** (`/cumulus/edeno/lowocc/rs58/`):
  - `rs58_ep6.npz` holds spike times and 26–32-channel amplitude marks per sort group, after
    the coincident filter (1,034,677 spikes), plus position, speed and the W-track graph.
  - `rs58_ep6_feats.npz` and `rs58_ep6_feats2.npz` hold the compact features (depth1, loc2,
    com3, win7, comwin4, comwin6, comwin8). `rs58_ep6_loc.npz` holds the point-source fit.
  - Held-out example decodes are in `examples/fold{0..4}.npz`; the plotting and metrics
    scripts are in the scripts folder below.
  - Registry entry: `RS58_2025_12_03` in `/cumulus/edeno/clusterless_high_d/configs/datasets.yaml`.
- **Harness for RS58, Jacob and L14:** `/cumulus/edeno/lowocc/jacob/jacob_detsel_v2.py`, run
  from `/cumulus/edeno/clusterless_high_d` with `.venv/bin/python` and
  `PYTHONPATH=<nld src>:src`.
  - Candidates: `kde:<ws>:<ps>`, `markglm:<K>[:<knot>[:<prior>]]`.
  - Options: `--dataset RS58|RS58_feat:<name>|Jacob_2025_06_18|L14_2024_06_04`,
    `--zscore-marks`, `--save-posterior`.
  - It writes `info_vs_flat_per_s`.
- **Harness for j16:** `/cumulus/edeno/lowocc/lowocc_detsel_pkg.py`, run from
  `/cumulus/edeno/state-space-playground` with `.venv/bin/python` (`uv` isn't on the PATH
  inside tmux shells).
  - Its `pick_free_gpu()` overrides `CUDA_VISIBLE_DEVICES`; concurrent jobs then pile onto one
    GPU and run out of memory.
  - `/cumulus/edeno/lowocc/realcheck/lowocc_detsel_emenc.py` is a copy that honours a pinned
    GPU and adds `--em-encoding`.
- **Scripts from this session** (aggregators, decode and plot scripts, the DB survey):
  `/cumulus/edeno/lowocc/scripts_from_session_2026-10-19/`.
- **Per-GPU job queues:** `/cumulus/edeno/lowocc/realcheck/run_lines.sh <queue file>` runs
  each line as a shell command and touches `<queue>.done`.
- **Time grid:** 2 ms bins, with position interpolated to 500 Hz. Detector predictions need
  explicit `time_edges` (`time_edges_from_centers`).

## Known pitfalls

- **Use exposure-weighted occupancy, not spike-triggered positions,** wherever a model needs
  "positions at random". Getting this wrong biases the rate maps toward place-field centres.
- **Keep the same reference q(m) and the same bin-duration convention across all states in a
  comparison.** Never normalize one state's likelihood separately.
- **Weights:** spike weights and sample weights are dimensionless, while exposure is in
  seconds. EM refits pass posterior weights.
  - The mark-GLM weighting fixes give fewer-but-better-converged components under weights.
  - EM with encoding refits is non-monotone with or without them; see the memory note on the
    EM-refit follow-up.
- **Float32 overflow:** marks are evaluated in float32 on GPU. A mark that overflows must fall
  back to the background, not produce NaN (the mark GLM does this now).
- **Trend checks:** before reporting a trend, spot-check folds and sanity-check that each flat
  model's information is about zero.

## References

- Kloosterman et al. 2014, J Neurophysiol 111:217 — clusterless decoding; our `clusterless_kde`
  is this model.
- Sodkomkham et al. 2016, Knowledge-Based Systems 94:1 — kernel density compression.
  Implementation docs: https://py-compressed-kde.readthedocs.io (Bitbucket
  `kloostermannerflab/fklab-compressed-decoder`, no license stated).
- Hermans et al. 2020 (ICML) and Cranmer et al. 2015 — classifier-based ratio estimation.
- Izbicki & Lee 2017 — FlexCode. Considered, ranked below Backend A because of its per-spike
  spatial cost.
- Shibue & Iwata 2024, PLoS Comput Biol — marked-point-process VAE, deferred.
- Zhang et al. 2023 (NeurIPS), code yzhang511/density_decoding — mixture with
  behaviour-dependent weights. Not reusable here.
- Pillow et al. 2008 (Nature) and Williamson, Savin & Pillow 2015 — bits-per-spike metric.
