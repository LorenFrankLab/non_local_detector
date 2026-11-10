# Literature review: Eden, Arai, Yousefi and Paninski, for the clusterless roadmap

**Date:** 2026-10-20. **Scope:** published papers, preprints and public GitHub code by Uri Eden (Boston University), Kensuke Arai (worked with Eden and with Loren Frank), Ali Yousefi (WPI, formerly MGH/Harvard) and Liam Paninski (Columbia), read against the six areas of `PLAN.md` in this folder.

**How this was verified.** Every work listed was found and opened during this review: full text where available (PubMed Central, bioRxiv, arXiv HTML, the BU thesis PDF), otherwise the PubMed abstract. The table says which. Repositories were cloned with `git clone --depth` into `/tmp/lit-review/`, read but never executed, and checked for licence and last commit. Nothing was installed. Anything I could not verify is listed in "Could not verify" and is not used in the recommendations.

**Two kinds of statement.** "**Paper:**" marks what a work reports. "**Extrapolation:**" marks my judgement about what it implies for `non_local_detector`. The ranked recommendations in section (b) are all extrapolation, built on the cited evidence.

**Headline.** Almost all of the Eden, Arai and Yousefi work is on tetrodes, in MATLAB or NumPy/Cython, and either offline or with latency only projected. The one measured real-time clusterless system with Eden as co-author (RealtimeDecoder, Chu et al. 2025) gets a 1.7 ms median posterior update with 6 ms bins on a 32-core CPU, without a GPU. Their most transferable ideas are:

- the marked-point-process goodness-of-fit transforms, which the package already implements;
- PCA of the concatenated waveform beating geometric features (Weber 2018), which agrees with the roadmap's PCA-12 result;
- per-spike information analysis, which says low-amplitude spikes still carry information once earlier spikes are accounted for (Ghahari & Eden 2025);
- time-discounted (sequentially updated) encoding models (Arai et al. 2018);
- an EM update for the state-noise variance (Yousefi's COMPASS toolbox);
- adding LFP rhythms to the switching-state model for causal replay detection (Zeng & Eden 2025);
- the mixture-reduction criteria behind their Gaussian-mixture filters (Rezaei et al. 2021).

The Paninski section is in Part 2.

---

## Part 1. Eden, Arai and Yousefi

### (a) Table of works

Verification levels: **FT** = full text read, **Abs** = PubMed or arXiv abstract only, **Code** = repository read.

| # | Citation | Link | Core idea | Tested on | Real-time evidence | Areas |
|---|---|---|---|---|---|---|
| E1 | Deng, Liu, Kay, Frank, Eden (2015) *Neural Comput* 27:1438 (FT) | [10.1162/NECO_a_00744](https://doi.org/10.1162/NECO_a_00744) | Clusterless marked-point-process filter with a Gaussian KDE joint mark intensity | Simulation; 1 rat, 5 tetrodes, linear track | None measured; "suitable candidate for real-time" | 1, 2 |
| E2 | Deng, Liu, Karlsson, Frank, Eden (2016) *J Neurophysiol* 116:2221 (FT) | [10.1152/jn.00151.2016](https://doi.org/10.1152/jn.00151.2016) | Discrete replay-category state (in/outbound × forward/reverse) on top of the clusterless filter; classify early in the event | 1 rat, 18 tetrodes, 1 ms bins | Offline MATLAB, single thread: 320 ms for a 340 ms event; most events classified within 20 ms of onset | 3, 4 |
| E3 | Tao, Weber, Arai, Eden (2018) *J Comput Neurosci* 45:147 (Abs) + `popTRT` (Code) | [10.1007/s10827-018-0698-4](https://doi.org/10.1007/s10827-018-0698-4) · [repo](https://github.com/Eden-Kramer-Lab/popTRT) | Marked time-rescaling: each spike rescaled to uniform in time × mark space, for goodness of fit | Simulation; rat hippocampus | n/a | 1 (diagnostics) |
| E4 | Yousefi, Amidi, Nazari, Eden (2020) *Neural Comput* 32:2145 (Abs) + repo (Code) | [10.1162/neco_a_01321](https://doi.org/10.1162/neco_a_01321) · [repo](https://github.com/YousefiLab/Marked-PointProcess-Goodness-of-Fit) | Time and rate rescaling transforms giving uniform samples in hypercubes, scalable to many mark dimensions | Simulation | n/a | 1 (diagnostics) |
| E5 | Arai, Liu, Frank, Eden (2018) bioRxiv 438440, unpublished (FT) + `EnDe`, `sumojam` (Code) | [10.1101/438440](https://doi.org/10.1101/438440) · [EnDe](https://github.com/AraiKensuke/EnDe) · [sumojam](https://github.com/Eden-Kramer-Lab/sumojam) | "JOYFULMoGs": joint mark–position intensity as a mixture of (Gaussian place field × Gaussian mark) components, fit by Gibbs sampling; posterior of epoch n−1, widened by a forgetting half-life, becomes the prior for epoch n | Simulation (stationary and drifting fields); rat W-track, 12–15 tetrodes (the text is inconsistent) | None measured; the paper says it is "still not real-time for most datasets" | 1, 4, 6b |
| E6 | Rezaei, Arai, Frank, Eden, Yousefi (2021) *J Neurosci Methods* 348:109006 (FT) + `GMM_PointProcess` (Code) | [10.1016/j.jneumeth.2020.109006](https://doi.org/10.1016/j.jneumeth.2020.109006) · [repo](https://github.com/Eden-Kramer-Lab/GMM_PointProcess) | With MoG joint mark intensities and a random walk, the posterior stays a Gaussian mixture in closed form. Component growth is controlled by dropping and merging under a symmetric-KL criterion | Rat W-maze, 9 tetrodes, clusterless; 1-D (1 tetrode) and 2-D (9 tetrodes); 1 ms steps | MATLAB: 2-D about 3.0 ms per 1 ms step vs 11.9 s exact grid (~4,000×); 1-D ~22× faster than a 24.4 ms exact step; RMSE +6–9%, 95% HPD coverage −0.7 to +3.7 points | 3, 4 |
| E7 | Yousefi, Gillespie, Guidera, Karlsson, Frank, Eden (2019) *IEEE TBME* 66:3486 (FT) + `Multi-Dimensional-Decoder` (Code) | [10.1109/TBME.2019.2906640](https://doi.org/10.1109/TBME.2019.2906640) · [repo](https://github.com/Eden-Kramer-Lab/Multi-Dimensional-Decoder) | Gaussian-mixture posterior: Laplace-type updates on no-spike steps; at spike steps, particle sampling then EM/BIC refit of the mixture. Enables 4-D (position + velocity) decoding | **Sorted** place cells, 2 rats, W-maze; 2-D and 4-D | MATLAB, i7: 682 ms per step on average for 2-D (exact grid 7.5 s); 4-D 1.21 s per step. Not real time | 4 |
| E8 | Weber (2018) BU PhD thesis, ch. 3–4 (FT) | [hdl 2144/34778](https://hdl.handle.net/2144/34778) | Which waveform features carry decodable information: geometric features vs PCA of the concatenated 4×40-sample tetrode waveform; marked time-rescaling | 1 rat, 24 tetrodes, W-maze, 15 epochs | n/a | 2 |
| E9 | Ghahari, Eden (2025) arXiv 2512.06280 (FT) | [arXiv:2512.06280](https://arxiv.org/abs/2512.06280) | Information carried by each spike (entropy and error reduction), in isolation (flat prior) and incrementally (the filter prior from earlier spikes) | 2 rats, 10 tetrodes each, W-maze (CRCNS) | n/a | 1, 2, 3 |
| E10 | Chu, Coulter, Denovellis, …, Eden, Kemere, Frank (2025) *eNeuro* 12(12) (FT) + `realtime_decoder` (Code) | [10.1523/ENEURO.0252-24.2025](https://doi.org/10.1523/ENEURO.0252-24.2025) · [repo](https://github.com/LorenFrankLab/realtime_decoder) | Python/MPI real-time clusterless decoder. One process per electrode group; Gaussian kernel in mark space, histogram over 41 position bins; ripple detection | Live rat experiments, 40–49 tetrodes used | **Measured:** 6 ms bins; median posterior update 1.722 ms; mark-intensity cost +0.5 ms per 10,000 encoding spikes; median latency <50 ms; 32-core Threadripper or Epyc; no GPU | 3 |
| E11 | Coulter, Gillespie, Chu, …, Eden, Kemere, Frank (2025) *Neuron* 113:949 (Abs) | [10.1016/j.neuron.2024.12.023](https://doi.org/10.1016/j.neuron.2024.12.023) | Closed-loop neurofeedback on decoded remote representations, using E10 | Rats, live | Shows real-time clusterless decoding works in vivo | 3 |
| E12 | Zeng, Eden (2026) *IEEE TBME* 73:153 (FT) | [10.1109/TBME.2025.3578583](https://doi.org/10.1109/TBME.2025.3578583) | Causal switching state-space model: latent LFP oscillators (separate sets for replay and non-replay), a represented-space latent, clusterless tetrode marks and MUA power; mixture Kalman filter with 50 particles | Simulation (218 events); 2 real datasets (20 and 22 tetrodes) | Detection latency reported as histograms, "within 30 ms" of the offline onset; no compute times | 3, 4 |
| E13 | Denovellis, Gillespie, …, Eden, Frank (2021) *eLife* 10:e64505 (Abs) | [10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505) | Mixture of movement-speed dynamics as discrete states; the model `non_local_detector` descends from | Rat tetrodes | Offline | 4 |
| E14 | Eden, Frank, Barbieri, Solo, Brown (2004) *Neural Comput* 16:971 (Abs) | [10.1162/089976604773135069](https://doi.org/10.1162/089976604773135069) | Point-process adaptive filters (Kalman-, RLS- and steepest-descent-type) that track receptive-field change on a millisecond timescale | Simulation | n/a | 4, 6b |
| E15 | Ergün, Barbieri, Eden, Wilson, Brown (2007) *IEEE TBME* 54:419 (Abs) | [10.1109/TBME.2006.888821](https://doi.org/10.1109/TBME.2006.888821) | Sequential Monte Carlo point-process filters using Gaussian-approximation proposals | Simulation; one rat place cell | n/a | 4 |
| E16 | COMPASS: Yousefi et al. (2019) *Front Neurosci* 12:957 (Abs) + `StateSpaceMarkedPointProcess` (Code) | [10.3389/fnins.2018.00957](https://doi.org/10.3389/fnins.2018.00957) · [repo](https://github.com/Eden-Kramer-Lab/StateSpaceMarkedPointProcess) | EM for linear-Gaussian state-space models with mixed (normal, binary, point-process, censored) observations. The M-step re-estimates the state-noise variance from smoothed second moments | Behavioural and clinical data | n/a | 5 |
| E17 | Rezaei, Hadjinicolaou, Cash, Eden, Yousefi (2022) *Neural Comput* 34:1100 (Abs); D4 variant arXiv 2205.10947 (Abs) | [10.1162/neco_a_01491](https://doi.org/10.1162/neco_a_01491) · [arXiv](https://arxiv.org/abs/2205.10947) | Direct discriminative decoder: a network or GP models p(state \| recent activity), combined with a state transition model | Rat hippocampus W-maze, human data (from abstracts) | None reported | 1, 3 (low) |
| E18 | `replay_identification` (Long Tao, Denovellis et al., unpublished) (Code) | [repo](https://github.com/Eden-Kramer-Lab/replay_identification) | Semi-latent replay detector combining spikes or marks with LFP ripple power (GMM on log power) and speed likelihoods; EM refits the encoding model with weights of 1 − p(non-local) | Code and notebooks only | n/a | 4 |
| E19 | Lower priority (Abs only): Huang et al. 2009 *Neural Comput* (T-maze decoding with a decision state) [10.1162/neco.2009.10-08-893](https://doi.org/10.1162/neco.2009.10-08-893); Arai & Kass 2017 *PLoS Comput Biol* (LOST, latent oscillation in spike trains) [10.1371/journal.pcbi.1005596](https://doi.org/10.1371/journal.pcbi.1005596); Rezaei et al. 2018 EMBC (point-process filter vs LSTM; LSTM 7× faster) [10.1109/EMBC.2018.8513154](https://doi.org/10.1109/EMBC.2018.8513154); Wodeyar et al. 2021 *eLife* (real-time phase estimation) [10.7554/eLife.68803](https://doi.org/10.7554/eLife.68803) | | | | | 3, 4 |

### Per-work notes

**E1. Deng et al. 2015, clusterless marked-point-process filter.**
- **Paper:** the intensity λ(x, m) is a product of Gaussian kernels over position and the 4-D peak amplitudes, normalized by an occupancy kernel. Tetrodes are combined by multiplying their likelihoods, and the decoder uses a Riemann sum on a 1-D grid.
- **Paper:** bandwidths were fixed by prior knowledge: position at 1.5% of track length, mark at 20 on the diagonal. Results "remain consistent for a wide range of bandwidths."
- **Paper:** on 1 rat with 5 tetrodes, clusterless gave rMSE 14.3 cm vs 26.0 cm sorted, and 99% HPD coverage of 74.25% vs 70.76%. No latency was measured.
- **Extrapolation:** this is `clusterless_kde` in its original form, so there is nothing new to adopt. It is the historical baseline the package already improves on: log-space KDE, the mark GLM, and bandwidth tuning by inner validation.

**E2. Deng et al. 2016, rapid classification of replay content.**
- **Paper:** a discrete "decision state" with four replay categories is added to the clusterless filter, with 1 ms bins and an empirical movement model sped up 33×.
- **Paper:** in MATLAB on one thread, a 340 ms event with 525 spikes on 18 tetrodes took 320 ms in total. The majority of events reach high posterior certainty (80–95%) within 20 ms of onset, and fewer than 2% fail to classify at a 0.8 threshold.
- **Code:** [`replay_classification`](https://github.com/Eden-Kramer-Lab/replay_classification) (Python, GPL-3.0, last commit 2018) is a predecessor of this package, with no new algorithm.
- **Extrapolation:** the useful idea is the **time-to-confident-decision** metric, the time after event onset at which a causal posterior crosses a threshold. This is the latency that matters for closed loop, and it isn't in the roadmap's item 3a.

**E3 and E4. Marked-point-process goodness of fit.**
- **Paper:** each spike is rescaled to a uniform point in time × mark space. Yousefi 2020 adds transforms that give hypercube-uniform samples, which suit uniformity tests in many mark dimensions.
- **Code:** `popTRT` (MATLAB and Python, no licence, last commit 2018). Yousefi's repository is MATLAB, MIT licence, last commit 2020.
- **Package:** already implemented. `src/non_local_detector/model_checking/clusterless.py` cites Yousefi et al. 2020 and implements `interval_rescaling_transform` and the IRCM transform, with KS and χ² tests. `conditional_marks.py` handles mixture CDFs.

**E5. Arai et al. 2018, JOYFULMoGs, sequential Bayesian encoding.**
- **Paper:** each component c has a rate λ0_c, a Gaussian place field (f_c, σ²_c) and a Gaussian mark (μ_c, Σ_c). Priors are Gamma for the rate, inverse-Wishart for the mark covariance, normal for the means, and inverse-Gamma for the field width. Spikes are stochastically assigned to components (Gibbs sampling, about 15,000 iterations). Components are pruned by occupancy, concentration and stationarity heuristics.
- **Paper:** adaptive version. Data arrive in epochs; each epoch's posterior, fitted to conjugate forms, becomes the next epoch's prior after "uncertainty inflation."
- **Paper:** in simulation, sequential updating approaches the ground truth after several epochs and tracks changing fields, while a stale model degrades. On real data an up-to-date model helps on the early-exposure day, less on the late day, and "the sequential-update advantage largely disappears" when 15 tetrodes are combined.
- **Paper:** there is no head-to-head comparison with the KDE and no runtime numbers. The stated limitation is that it is "still not real-time for most datasets." bioRxiv lists no published version.
- **Code:** `sumojam/gibbs1d.py` (`GibbsSampler1D.gibbs`, lines 165–171 and 424–446) implements the inflation as hyperparameter discounting. The Gamma and inverse-Gamma shape parameters decay as `a·exp(−DT/τ)` with the mode kept fixed (half-life `t_hlf_l0 = t_hlf_q2 = 5 min`). The place-centre prior variance grows by `diffusePerMin²·DT`. The mark-mean and covariance hyperparameters are not discounted in that block.
- **Code:** `mkdecoder.decodeMoG` evaluates the mixture on the position grid per spike (Cython `hc_bcast.evalAtFxdMks_new`) and also has a KDE path. Both repositories are Python 2/3 with Cython, have no licence, and were last touched in 2017 (`EnDe`) and 2020 (`sumojam`, "Initialize" plus one commit). They are unmaintained.
- **Package:** the joint model is effectively the MDN on `feat/clusterless-mdn`. That fits λ_e(x, m) = Σ_k λ_ek(x) N(m; μ_ek, Σ_ek) by EM, with responsibilities that depend on position, which is the maximum-likelihood analogue with spline rate maps in place of single Gaussian fields.
- **Package:** the mark GLM's Dirichlet-process VB mixture already gives the Bayesian pruning. What the package does not have is **time-discounted encoding**.

**E6. Rezaei et al. 2021, real-time Gaussian-mixture point-process filter.**
- **Paper:** with MoG intensities and a linear-Gaussian random walk, prediction and update stay Gaussian mixtures in closed form. At spikes the component count multiplies, so a drop step and a sequential pairwise merge step run each time step.
- **Paper:** the merge uses moment matching, scored by a symmetric KL divergence (a Jensen–Shannon-type weighting via α) with a first-order Taylor approximation, and two thresholds. One-sided KL "tends to merge mixture components… lead[ing] to a biased posterior." Dropping first then merging is about 100× faster than merging first, at similar accuracy.
- **Paper:** in 2-D with 9 tetrodes and 1 ms steps, the filter took about 3.0 ms per step in MATLAB; the 11.9 s "exact" baseline was a naive 2-D Riemann sum. Worst-case steps were up to 2.6× the exact step's time, in under 0.17% of steps. Merging handles up to about 10 components within 1 ms.
- **Code:** `gmm_decoder_matlab/ay_gmm_drop_alpha_optimized.m` and `ay_gmm_merge_alpha_optimized.m` (MATLAB, MIT licence, last commit 2018). The Python package `gmm_decoder/core.py` contains only empty stubs.
- **Extrapolation:** the "4,000×" compares against a naive grid. A 1-D grid HMM step on GPU or CPU in this package is a small matrix-vector product, so the speed-up does not carry over to 1-D tracks.
- **Extrapolation:** two ideas do transfer. (i) The Gaussian-mixture posterior, for 2-D or higher state spaces (item 4c). (ii) Symmetric-divergence merge criteria for **encoding-kernel compression** (items 3d and PR #105). The package's `kde_compression.merge_encoding_spikes` uses a fixed Mahalanobis threshold; this is on branch `feat/kde-compression`, read with `git show`.

**E7. Yousefi et al. 2019, Gaussian-mixture particle filter.**
- **Paper:** the place fields are KDEs of **sorted** cells. At spike times 4,000 particles are drawn, then an EM/BIC refit of the mixture follows; this was more accurate than a standard particle filter.
- **Paper:** 682 ms per step on average for 2-D and 1.21 s for 4-D, in MATLAB. It is the first 4-D (position + velocity) point-process decoder from these groups.
- **Code:** MATLAB, no licence, last commit 2019, with data files in the repository.
- **Extrapolation:** too slow and not clusterless. Relevant only as evidence that decoding velocity sharpens position. The package's velocity plan already covers that.

**E8. Weber 2018 thesis, chapter 3.**
- **Paper:** PCA of the concatenated 160-sample tetrode waveform (4 × 40), computed per tetrode, needs 10–20 PCs for 95% of the variance.
- **Paper:** among geometric features, peak amplitude was "consistently and significantly more informative" than width or slope. Clusterless beat ISO-SPLIT-sorted decoding for most feature sets.
- **Paper:** "in lower dimensions, principal components and spike amplitudes performed comparably; but in higher dimensions [up to 10 PCs], principal components produced significantly more accurate decodes." The recommendation is "higher-dimension principal component vectors when modeling unsorted population spiking data."
- **Paper:** evaluation was by 2-fold rMSE and HPD only. The thesis itself names held-out goodness of fit as the missing step.
- **Extrapolation:** independent tetrode evidence for the roadmap's PCA-12 finding (j16 +6.8 ± 3.4 nats/s). Width and slope are not worth testing.

**E9. Ghahari & Eden 2025, information in individual spikes.**
- **Paper:** KDE clusterless model on peak amplitudes (position bandwidth 5 cm, mark bandwidth 50 µV chosen by decoding error), with leave-one-out per-spike information.
- **Paper:** spikes discarded by sorting carried 48% of the total incremental entropy reduction. In isolation, low-amplitude spikes (below about 200 µV) are less informative, but "with prior history" the low- vs high-amplitude distinction "becomes markedly less pronounced."
- **Paper:** information per spike is much smaller once earlier spikes are accounted for, and depends more on the decoder state than on tuning.
- **Paper:** no code or runtime statement in the text read.
- **Extrapolation:** two experiments follow, both run on per-spike scores the harness can already compute.
  - A detection-threshold sweep, which is both a feature question (area 2) and a speed knob (area 3): fewer spikes means less likelihood work.
  - "Incremental" rather than "isolated" per-spike information as the right lens for which spikes are worth computing in real time.

**E10. Chu et al. 2025, RealtimeDecoder.**
- **Paper:** 6 ms bins, 41 position bins of 5 cm. The encoding model grows during an initial run session of about 10 minutes and is then frozen.
- **Paper:** median posterior update of 1.722 ms. The authors advise a bin no shorter than the 75th percentile of the update time. Network latency was about 0.1 ms on localhost.
- **Paper:** the mark-intensity cost grows by about 0.5 ms per 10,000 encoding spikes. Latency rises with mark dimension, but less than proportionally (1 → 16 dimensions is not a 16× increase).
- **Paper:** CPU only, with one hardware thread per electrode group. GPUs are proposed as future work.
- **Code:** `realtime_decoder/encoder_process.py`, `Encoder.get_joint_prob` (lines 184–257):
  - a Gaussian kernel on squared mark distance only (σ from the config; 20 in the paper);
  - an optional axis-aligned box prefilter (`filter_std` × `filter_n_std`) with a minimum-neighbour count;
  - then `np.histogram` of the training positions weighted by the kernel, divided by occupancy and normalized to sum to one over position.
- **Code:** the no-spike term is `exp(−dt · rate/occupancy)` per electrode group (`decoder_process.py:248–251`). Python with MPI. The GitHub API and repository list no licence file. Last commit 2026-06-06, so it is maintained.
- **Extrapolation:** normalizing each spike's joint intensity over position drops a factor λ(m) that is constant in x. That doesn't change the position posterior within one state, but it means the per-spike term isn't the marked log likelihood. So its information can't be scored in nats against a flat refit without restoring the normalization.
- **Extrapolation:** it is the natural **external latency baseline for item 3a**, and its stage-by-stage timing design (network, mark intensity, posterior) is worth copying.

**E11. Coulter et al. 2025** shows that E10's real-time clusterless decoding is good enough for closed-loop feedback on remote representations. There is no new method.

**E12. Zeng & Eden 2026, causal ripple-replay detection.**
- **Paper:** a binary replay state with a 2-state Markov chain (real data: p(0→1) = 0.001, p(1→1) = 0.99). Replay content follows AR(1) dynamics (α = 0.99, σ = 10 cm), fixed by hand rather than estimated.
- **Paper:** 6 oscillator pairs were chosen by AIC, with LFP parameters fit by maximum likelihood. The spike intensity is a KDE fit separately. Inference is a mixture Kalman filter with 50 particles, plus an optional smoother.
- **Paper:** in simulation, 209 of 218 events were detected with 0 false positives. On real data the full model found 550 events vs 434 from LFP alone, mostly within 30 ms of the offline onset. No runtime numbers and no code link in the text read.
- **Package:** has no LFP or ripple-power observation model; `grep -rn lfp src/non_local_detector` finds none. The predecessor `replay_identification` had one (`likelihoods/lfp_likelihood.py`: a 3-component GMM on log ripple power per state).

**E13–E15.** Background: speed-mixture dynamics (the package's origin), adaptive receptive-field filters, and SMC proposals. E14 is the theory behind time-varying encoding models (item 6b). None has code to reuse.

**E16. COMPASS and `StateSpaceMarkedPointProcess`.**
- **Code:** `compass_em_e.m` (MATLAB, no licence in this repository, last commit 2019; COMPASS itself is MIT). The E-step forms the smoothed moments E[x_k x_kᵀ] = S_k + x̂_k x̂_kᵀ and the lag-one cross-moment E[x_k x_{k−1}ᵀ] (lines 595–620, "Biometrika 1988"). The M-step updates the transition matrix A and the state noise W (lines 1285–1288).
- **Extrapolation:** this is the linear-Gaussian version of item 5a. On a grid HMM the analogous sufficient statistic is the expected transition-count matrix.

**E17. Discriminative decoders.**
- **Paper (abstracts):** they model p(state | activity) directly, with neural networks or GPs.
- **Extrapolation:** they don't produce a marked-point-process likelihood, so they can't be scored by `info_vs_flat_per_s`. They also give up the generative structure the package relies on (mark reference, EM over states). Not recommended; see (c).

**E18. `replay_identification`** (Python, MIT, last commit 2022). A predecessor of this package, with an LFP likelihood, a speed likelihood and EM refits of the encoding model.

---

## Part 2. Liam Paninski

Most of Paninski's output is not about real-time hippocampal decoding. The relevant parts fall into three groups:

- **Clusterless decoding on Neuropixels:** P1.
- **Dense-probe spike processing:** localization, drift registration, denoising, temporal PCA and noise models, and DARTsort's truncated mixture.
- **Older state-space and GLM inference methods.**

Repositories were cloned read-only to `/tmp/lit-review/` (`density_decoding`, `dredge`, `dartsort`, `CEED`, `ibl-neuropixel`, `eks`) and not executed.

### (a) Table of works

| # | Citation | Link | Core idea | Tested on | Real-time evidence | Areas |
|---|---|---|---|---|---|---|
| P1 | Zhang, He, Boussard, Windolf, …, Hurwitz, Paninski (2023) "Bypassing spike sorting: density-based decoding using spike localization from dense multielectrode probes", NeurIPS 36 (FT) + code | [10.1101/2023.09.21.558869](https://doi.org/10.1101/2023.09.21.558869) (PMC10542538) · [repo](https://github.com/yzhang511/density_decoding) | Gaussian mixture over spike features (x, z, max peak-to-peak), with behaviour-dependent component rates and therefore mixing weights; fit by CAVI or ADVI | IBL NP1 (20 sessions), NP2.4, monkey NP1; the tetrode comparison is **simulated** only | Offline; the decode step runs at "0.3× real time" with 50 ms bins | 1, 2, 6 |
| P2 | Boussard, Varol, Lee, Dethe, Paninski (2021) NeurIPS 34, 3-D spike localization (Abs page) | [10.1101/2021.11.05.467503](https://doi.org/10.1101/2021.11.05.467503) · [proceedings](https://proceedings.neurips.cc/paper/2021/hash/b950ea26ca12daae142bd74dba4427c8-Abstract.html) | Point-source localization (x, y, z, α) with a network denoiser; registration on the 3-D point cloud | Neuropixels | None | 2, 6 |
| P3 | Windolf, Yu, Paulk, …, Paninski, Varol (2025) *Nat Methods*, DREDge (Abs + code) | [10.1038/s41592-025-02614-5](https://doi.org/10.1038/s41592-025-02614-5) · [repo](https://github.com/evarol/dredge) | Decentralized registration: pairwise displacement cross-correlations, then a robust least-squares solve, from AP-band spikes or LFP | Human, monkey, acute and chronic mouse (including NP Ultra) | See P4 | 6b, 3 |
| P4 | Windolf, Paulk, Kfir, …, Paninski, Varol (2023) ICASSP, online multiband drift (FT) | [10.1109/icassp49357.2023.10095487](https://doi.org/10.1109/icassp49357.2023.10095487) (PMC10308877) | Chunked online DREDge, each chunk conditioned on the previous one, so cost is linear in recording length | 2 human Neuropixels recordings, 1 mouse | GPU LFP version "much faster than real time" on recorded data; no live test | 6b, 3 |
| P5 | Boussard, Windolf, Hurwitz, Lee, Yu, Winter, Paninski (2023) bioRxiv, DARTsort (Abs + code) | [10.1101/2023.08.11.553023](https://doi.org/10.1101/2023.08.11.553023) · [repo](https://github.com/cwindolf/dartsort) | Truncated-EM mixture: each unit is a mean plus a low-rank basis plus structured noise, on channel neighbourhoods with missing channels marginalized; only a few candidate units evaluated per spike | Simulation (against Kilosort 2.5); Neuropixels | None | 1, 2, 3, 6 |
| P6 | Vishnubhotla, Loh, Paninski, Srivastava et al. (2023) NeurIPS 36, CEED (FT) + code | PMC12373126 · [repo](https://github.com/ankitvishnu23/CEED) | Contrastive 5-D waveform embeddings, trained with noise, collision and channel-dropout augmentations | 2 IBL NP1 recordings; monkey U-probe | None in the main text | 2 |
| P7 | Hurwitz, Xu, Srivastava, Buccino, Hennig (2019) NeurIPS 32 (Abs). **Paninski is not an author**; listed because DARTsort's amortized localizer descends from it | [arXiv:1905.12375](https://arxiv.org/abs/1905.12375) | Amortized variational inference for spike localization: one network pass per spike | Simulated and real high-density arrays | No latency numbers seen | 3, 6 |
| P8 | Calabrese & Paninski (2011) *J Neurosci Methods* 196:159 (Abs) | [10.1016/j.jneumeth.2010.12.002](https://doi.org/10.1016/j.jneumeth.2010.12.002) | Kalman-filter mixture: component waveform means follow a random walk; fit by EM; recursive, so usable online | Simulated and real | "Allows on-line implementation" (claim only) | 6b |
| P9 | Smith & Paninski (2013) *Network* 24:75 (Abs) | [10.3109/0954898X.2013.789568](https://doi.org/10.3109/0954898X.2013.789568) | HMM forward-backward measures the information lost to spike jitter and wrong neuron identities. Averaging over assignments beats using the single best assignment | Simulation | n/a | 1 (1c) |
| P10 | Paninski et al. (2010) *J Comput Neurosci* 29:107 (FT) | [10.1007/s10827-009-0179-x](https://doi.org/10.1007/s10827-009-0179-x) | Banded-Hessian Newton MAP paths in O(T); Laplace marginal likelihood for parameters, presented as an alternative to EM, which "zig-zags" | Simulated and real | "About 1 s of compute per 1 s of data at 1 kHz on a laptop" | 4, 5 |
| P11 | Pnevmatikakis, Rahnama Rad, Huggins, Paninski, low-rank fast Kalman (lab page, under review) | [abstract page](https://sites.stat.columbia.edu/liam/research/abstracts/fast-Kalman-abs.html) | Low-rank perturbation gives an O(k²d) Kalman step | Spatiotemporal neural data | n/a | 4 |
| P12 | Rahnama Rad & Paninski (2010) *Network* 21:142 (Abs) | [10.3109/0954898X.2010.532288](https://doi.org/10.3109/0954898X.2010.532288) | Gaussian-process 2-D rate maps, with smoothness chosen by marginal likelihood | Including place fields | n/a | 5 |
| P13 | Ramirez & Paninski, expected log-likelihood (Abs) | [arXiv:1305.5712](https://arxiv.org/abs/1305.5712) | Replace the GLM exposure sum with an expectation over the covariate distribution | Primate retina | Faster fitting (claim) | 5 |
| P14 | Lawhern, Wu, Hatsopoulos, Paninski (2010) *J Neurosci Methods* 189:267 (Abs) | [10.1016/j.jneumeth.2010.03.024](https://doi.org/10.1016/j.jneumeth.2010.03.024) | Encoding GLM with a latent multi-dimensional hidden state, fit by EM; decoding error down by up to 29% | Monkey M1 Utah array | "Retaining real-time computational efficiency" (claim) | 4 |
| P15 | Zhang, Lyu, Hurwitz, …, Paninski (2025) *Neuron* (Abs) | [10.1016/j.neuron.2025.10.026](https://doi.org/10.1016/j.neuron.2025.10.026) | Multi-session reduced-rank regression and state-space decoding | 433 IBL sessions | n/a | 4, 5 (low) |
| P16 | Ye, Shelton, Shaker, Boussard, … (2025) *Neuron* (Abs) | [10.1016/j.neuron.2025.08.030](https://doi.org/10.1016/j.neuron.2025.08.030) | NP Ultra spatial footprints separate axonal from somatic sources and help identify cell types | Mouse visual cortex | n/a | 2, 6 (low) |
| R1 | `paninski-lab/eks`, Ensemble Kalman Smoother (Lightning Pose; Biderman et al. 2024 *Nat Methods*, [10.1038/s41592-024-02319-1](https://doi.org/10.1038/s41592-024-02319-1)) (Code) | [repo](https://github.com/paninski-lab/eks) | JAX/dynamax smoother; process-noise scale fit by gradient descent on the marginal log likelihood, under `jit(vmap)` | Pose tracking | n/a | 5 (method pattern) |
| R2 | `int-brain-lab/ibl-neuropixel`, `ibldsp.voltage.destripe` (Code; an IBL repository, Paninski's authorship not checked) | [repo](https://github.com/int-brain-lab/ibl-neuropixel) | Neuropixels preprocessing: filtering, ADC-shift correction, bad-channel interpolation, spatial k-filter or common-average reference. P1 uses it | Neuropixels | Offline, and non-causal (see notes) | 6d, 3 |

### Per-work notes

**P1, density-based decoding. The most relevant Paninski paper.**
- **Paper:** the number of components C and the initial means and covariances come from isosplit on the features. Bins are 50 ms within 1.5 s trials, with a GP prior for continuous behaviours.
- **Paper:** the final readout is not the posterior. Per-spike component probabilities are summed into a trials × components × bins weight matrix, then ridge or logistic regression maps it to behaviour. Decoding from the posterior directly "was not robust."
- **Paper:** it beats thresholded multi-unit and Kilosort 2.5 decoders, and on monkey arm force gives R² ≈ 0.6 vs 0.3 for sorted units.
- **Paper:** Table 1 compares it with this package's predecessor (Denovellis et al. 2021 via `replay_trajectory_classification`): simulated 5 tetrodes, correlation 0.97 vs 0.91; NP1 wheel speed, 0.63 vs 0.50. But the supplement sets the KDE mark bandwidth to "1.0" (untuned), so **treat Table 1 as weak evidence.**
- **Code:** PyTorch, MIT, last commit 2025-02-19. `models/advi.py` (lines 87–88; checked) keeps the sklearn GMM means and covariances fixed (`requires_grad=False`) and learns only the low-rank behaviour-dependent rates.
- **Code:** `models/cavi.py` (binary behaviour only) also re-estimates the means and covariances, and its `_decode_m_step` re-estimates them **on the test spikes**.
- **Extrapolation:** ADVI is structurally the package's `clusterless_mark_glm` (a fixed mark mixture plus per-component rate maps), and the package's version is stronger for position (track-graph splines, a state-space posterior, the β·q(m) background). CAVI corresponds to the MDN's EM. Nothing to port.

**P2 and P7, localization.** In DARTsort, `localize/localize_torch.py:localize_amplitude_vectors` runs batched Levenberg–Marquardt on `point_source_mse`, and `transform/amortized_localization.py:AmortizedLocalization` is an MLP returning (x, y, z_abs). The roadmap already records that point-source localization features lost 25–35% on RS58.

**P3 and P4, DREDge.**
- **Code:** Python, MIT, last commit 2026-10-05, maintained; the README recommends SpikeInterface `correct_motion(preset="dredge")`.
- **Code:** `dredgelib.py:online_displacement` processes chunks against the previous chunk; `thomas_solve` is a block-tridiagonal solve; `dredge_lfp.py:register_online_lfp` defaults to 10 s LFP chunks at 250 Hz.
- **Extrapolation:** use it offline through SpikeInterface. A JAX port isn't needed.

**P5, DARTsort.**
- **Code:** PyTorch. `pyproject.toml` says MIT but there is no root LICENSE file. Last commit 2026-10-11, maintained, and the project says it is not yet recommended for production sorting.
- **Code:** `clustering/mixture.py` (about 5,000 lines) is a truncated-EM mixture with "observed part of spike_n | l_n ~ N(mu[l_n], W[l_n]@W[l_n].T + noise)."
  - Each spike is observed only on its channel neighbourhood, and the missing channels are marginalized. `NeighborhoodCovariance` precomputes inverse Cholesky factors and log-determinants for each neighbourhood.
  - Only a few candidate units are evaluated per spike (`n_candidates`, search and explore units, `NeighborhoodLUT`). There is an explicit noise unit, and split/merge decisions use cross-validation.
- **Code:** `util/noise_util.py:EmbeddedNoise` gives structured noise covariances in temporal-PCA space (scalar, diagonal, or a Kronecker rank × channel form).
- **Extrapolation:** the low-rank-plus-noise density and the neighbourhood marginalization are plain linear algebra and port to JAX easily. The split/merge machinery would not port easily, and isn't needed.

**P6, CEED.** PyTorch, MIT, last commit 2024-11, dormant. Augmentations are in `data_aug/wf_data_augs.py`. It reports re-sorting agreement (ARI) of 0.78–0.89 vs 0.39–0.49 for denoised PCA. That measures clustering agreement, not decoding information, and the transformer variant needs multiple GPUs.

**P8, Kalman-filter mixture.** Abstract only, no code found. It is exactly the "time-varying mark model" option in item 6b.

**P9.** Abstract only. A theoretical argument for clusterless over hard sorting, and a way to measure the information sorting loses (item 1c).

**P10 and P11.** Both assume a continuous state with a single-peaked posterior. P10 argues for marginal-likelihood gradients over EM.

**R1, eks.**
- **Code:** JAX plus dynamax, MIT, last commit 2026-07-22. In `core.py`, `optimize_smooth_param` and `_vmap_optimize_singletons` fit each keypoint's process-noise scale by gradient descent on `-post.marginal_loglik`, compiled once with `jit(vmap)`.
- **Extrapolation:** it is linear-Gaussian, not a grid HMM, so the transferable thing is the pattern, not the code.

**R2.** `ibldsp/voltage.py:destripe` uses `scipy.signal.sosfiltfilt`, which is zero-phase and therefore **non-causal**. It's fine for offline extraction in item 6d but not for a real-time path. Its docstring calls it "super slow."

**The paninski-lab GitHub organization.** Besides `eks`, it has `yass` (Apache-2.0, last push 2022-09, retina-oriented, dormant) and pose-estimation tools that aren't relevant here.

---

## (b) Recommendations by roadmap area (all four authors)

All items here are **extrapolation** from the cited evidence. Each idea gives a value and cost rating, says whether the package already does it, and names the test.

"**Protocol**" means the shared evaluation from `PLAN.md`:

- `info_vs_flat_per_s` against the matched position-free refit, plus `bits_per_spike`;
- 5 blocked folds, paired by fold: mean ± SD of the difference, folds better, and the baseline's fold-to-fold SD;
- tuning chosen by blocked inner validation (item 0b);
- the baseline is `markglm:32` on amplitude marks.

Where an idea changes something other than the spike likelihood, the score is named explicitly.

### Area 1. Better likelihood estimators

1. **Low-rank-plus-measured-noise component covariances in the mark GLM** (from P5; also consistent with the roadmap's noise-floor, low-rank-merge and PCA-12 results). *Value: medium. Cost: medium.*
   - **Package:** partly. The mark GLM has full covariances plus `mark_reg_covar`, and the noise floor exists on `feat/mark-glm-biophysical`. Σ_k = Σ_noise + W_k W_kᵀ is not implemented.
   - **Test:** RS58 and j16 on PCA-12, plus a higher-dimensional per-channel PCA set (D ≥ 50) on RS58. Compare full Σ_k with Σ_noise + rank r, r ∈ {2, 4, 8}, r chosen by inner validation. Score by the protocol and also report fit time. A gain should grow with mark dimension; if it doesn't, the idea is dead.
2. **Goodness-of-fit diagnostics to explain where richer models win** (E3, E4). *Value: medium. Cost: low.*
   - **Package:** yes. `model_checking/clusterless.py` implements the Yousefi 2020 transforms.
   - **Test:** on each RS58 and j16 test fold, compute the KS statistic of the rescaled-time column (rate model) and of each conditional-mark column, for the mark GLM, the MDN and the KDE. Check whether the MDN's RS58 gain shows up as better mark-conditional fit or better rate fit, paired by fold against the information difference.
   - This is descriptive, not a promotion test. It helps item 1a decide what kind of model to build next.
3. **Per-spike incremental information for item 1c** (E9, P9). *Value: medium. Cost: low.*
   - **Package:** no. The per-spike log-likelihood terms exist inside the harness, but this decomposition hasn't been run.
   - **Test:** on 1c recordings with curated units, split each clusterless model's held-out information between spikes assigned to curated units and the rest, scored incrementally (with the causal filter prior), not in isolation. Ghahari & Eden found that spikes discarded by sorting carry 48% of the incremental entropy reduction on tetrodes, which predicts a sizeable share on polymer probes too.
4. **Joint mark-and-position mixture fitting** (E5 Arai's Gibbs model; P1's CAVI). *Already covered* by the MDN on `feat/clusterless-mdn`, which uses position-dependent responsibilities and spline rate maps. No new work; item 1a decides its fate.

### Area 2. Waveform features and biophysical priors

1. **Spike-amplitude threshold sweep** (E9). *Value: medium-high. Cost: low.*
   - **Package:** no. Spike sets are fixed by the waveform-feature entries.
   - **Test:** raise the detection threshold after the fact, keeping spikes with max amplitude ≥ t0 + {0, 10, 20, 40, 60} µV on every channel group, on j16, RS58, Jacob and L14. Refit and score each set by the protocol (nats/s from the retained spikes, against that set's own matched flat refit). Report spikes per 6 ms bin alongside, as this doubles as a speed knob for item 3a.
   - If information still rises as the threshold falls to t0, lowering t0 in item 0a's re-extraction is worth testing. That is a spyglass recommendation, not a package change.
2. **PCA of the concatenated multichannel waveform** (E8 on tetrodes; already item 2a). *Confirmatory.* Weber found that PCs beat amplitudes once more than about 4 dimensions are used, up to 10 PCs, on 24 tetrodes. This independently supports the roadmap's j16 PCA-12 result (+6.8 ± 3.4) and supports including PCA-24 on tetrodes in item 2a. No new work.
3. **Localization plus max peak-to-peak (x, z, ptp) on Neuropixels** (P1, P2). *Value: low-medium. Cost: low once item 6d exists.*
   - **Test:** in item 6d, compare the mark GLM on (x, z, ptp) against amplitudes and PCA-12, by the protocol.
   - Do it only on Neuropixels: it lost 25–35% on RS58 polymer probes. That RS58 test may not have included ptp; see (d).

### Area 3. Real-time speed/accuracy trade-offs

1. **Use RealtimeDecoder (E10) as the external latency baseline, and copy its stage-by-stage timing.** *Value: high. Cost: low.*
   - E10 measured on a 32-core CPU with 6 ms bins and 41 position bins: median posterior update 1.722 ms; mark-intensity cost +0.5 ms per 10,000 encoding spikes; one thread per electrode group. It is the only measured, in-vivo-validated real-time clusterless system among these authors.
   - **Package:** no comparable measurement yet (item 3a is planned).
   - **Test:** in item 3a, report network, likelihood and posterior stages separately (median and 99th percentile) on j16 at E10's settings (6 ms bins, 41 bins of 5 cm, CPU). Then repeat at the package's defaults and on GPU.
   - Running E10's code itself needs its own environment and the user's go-ahead; this review did not execute it. Its estimator can instead be rebuilt in the harness (mark-only Gaussian kernel, position histogram, occupancy normalization) and scored by the protocol, restoring the per-spike λ(m) factor that E10's normalization over position drops. That gives a like-for-like information-vs-latency point.
2. **Time-to-confident-decision metric** (E2). *Value: medium. Cost: low.*
   - **Package:** no. The package has causal filtered posteriors over discrete states, but no metric for this.
   - **Test:** for SWR-aligned events, measure the time from onset until the causal (filter-only) probability of the non-local state, or of a target region, first exceeds 0.8. Also report how often that call agrees with the offline smoother.
   - Add it as a descriptive column in item 3a's table for each model and bin size. E2 reports most events classified within 20 ms on tetrodes, which is a reference point.
3. **Symmetric-divergence merge criterion for KDE kernel compression** (E6). *Value: medium. Cost: low.*
   - E6 shows, for **posterior** mixture reduction, that one-sided KL over-merges and biases, while a symmetric criterion doesn't. Carrying this over to **encoding-kernel** dictionaries is my extrapolation.
   - **Package:** no. `merge_encoding_spikes` on `feat/kde-compression` uses a fixed Mahalanobis threshold.
   - **Test:** in the offline compression harness (`/cumulus/edeno/lowocc/compress/fullcov/lowrank_lib.py`), compare the Mahalanobis-threshold merge with a merge that picks pairs by a symmetric KL (or Runnalls-type) cost, at equal kernel budgets (23% and 10%). Score by held-out mark density, then by the protocol through the detector on RS58 and j16.
4. **Top-k component evaluation per spike** (P5). *Value: depends on K. Cost: low.*
   - **Package:** no. `clusterless_mark_glm.py:786-859` evaluates all K components with a dense matrix product.
   - It matters only for whole-shank or Neuropixels mixtures with K in the hundreds.
   - **Test:** in item 3a's synthetic Neuropixels-scale sweep, report latency and information lost as a function of k. **Hard:** identical information at k = K.
5. **CPU thread scaling over electrode groups** (E10 uses one thread per group). *Value: medium. Cost: low.* In item 3a, measure the package's CPU per-step latency as the number of threads (`XLA_FLAGS` intra-op threads) and the number of groups vary, to confirm XLA actually parallelizes across electrodes.
6. **Real-time preprocessing must be causal** (R2). This is a constraint, not an experiment. A Neuropixels real-time path in items 3a and 6d can't reuse `destripe` (it uses `sosfiltfilt`).

### Area 4. Scaling and state-space models

1. **LFP rhythm or ripple-power observations in the discrete-state model** (E12; E18's `lfp_likelihood`). *Value: medium, mainly for real-time detection. Cost: medium.*
   - **Package:** no LFP observation model.
   - **Test:** `info_vs_flat_per_s` scores only the spike likelihood, so it doesn't apply here. Instead, on j16 and RS58 with 5 blocked folds, compare an opt-in ripple-band log-power likelihood per discrete state (as in E18) against spikes only, on:
     - (i) held-out marginal log likelihood of spikes + LFP under each model, paired by fold;
     - (ii) causal detection latency and agreement against the offline smoother's non-local events.
   - E12 reports 550 vs 434 events with spikes + LFP vs LFP alone, mostly within 30 ms, but there is no ground truth on real data.
2. **Gaussian-mixture posterior filters for 2-D or 4-D state spaces** (E6, E7). *Value: low-medium. Cost: high.* Consider them only if item 4b shows dense 2-D transitions dominate and item 4c's structured operator fails.
   - They need Gaussian-mixture spatial intensities; the mark GLM's spline maps have no closed-form product.
   - They ignore track topology.
   - They have per-step costs that vary with the number of components: up to 2.6× the mean step time in E6's worst steps, which hurts a 99th-percentile real-time budget.
   - **Test** (if reached): a 2-D open-field recording, mixture filter against the grid filter. Score the posterior's held-out marginal log likelihood and the 99th-percentile step latency.
3. **A latent hidden state in the encoding GLM** (P14). Speculative. It might absorb the non-positional structure the matched null removes (9–60 nats/s on RS58). It belongs under the velocity plan or item 4d, not here.

### Area 5. Estimating assumed parameters

1. **Estimate the random-walk variance (item 5a) without storing two-slice marginals.** *Value: high. Cost: low-medium.* Two routes, which should agree:
   - **(a) EM with an expected transition-count matrix.** Accumulate N_ij = Σ_t f_t(i) T_ij s_{t+1}(j) / p_{t+1}(j) per continuous state inside the backward scan, which needs O(n_bins²) memory and nothing over time. Then maximize Σ N_ij log T_σ(i,j) over σ by 1-D search, since track-graph boundaries make T_σ a truncated, renormalized kernel with no closed form. This is the grid analogue of COMPASS's linear-Gaussian M-step (E16, `compass_em_e.m:595-620`).
   - **(b) Autodiff of the marginal log likelihood** with respect to log σ, with `jax.checkpoint` on the filter scan (the R1 pattern; P10 argues against EM zig-zag). This needs the track-graph transition construction to be differentiable in σ.
   - **Package:** no. `movement_var` is fixed. `core.py:_smoother_impl` (lines 587–645) returns smoothed marginals only and accumulates no transition counts.
   - **Test:** item 5a's simulation recovery (within 10%, **target**; EM log likelihood non-decreasing, **hard**), and (a) = (b) on small grids. On real data, report per-fold held-out marginal log likelihood of the spikes under the full HMM (5 blocked folds, paired) and the change in decoded replay speed.
   - None of these authors estimated replay dynamics: E12 fixed σ = 10 cm and α = 0.99 by hand, and E6 and E7 set the variance from behaviour. So there is no literature value to check against; that is exactly the gap item 5a fills.
2. **Marginal-likelihood (evidence) selection of the mark GLM's log-rate prior and knot spacing** (P12, P10). *Value: low-medium. Cost: low.* Run it as one extra arm of item 5b, against inner validation, scored by the protocol. Memory `per-neuron-reml-not-adopted` says the sorted-MRF version gained nothing on real data, so expect little.
3. **Bandwidths:** nothing to adopt. E1 fixed them by prior knowledge and E9 by decoding error; item 0b's held-out selection is already better.

### Area 6. High-density probes

1. **A whole-shank mixture over channel neighbourhoods** (P5). *Value: high. Cost: high.* A principled version of item 6a(iv).
   - Fit one mixture per shank on registered geometry. Each spike is observed on its peak-channel neighbourhood, using low-rank-plus-noise Gaussians on temporal-PCA features, marginalizing the unobserved channels, with a rate map per component as in the mark GLM. Each spike counts once, and there are no block boundaries.
   - **Package:** no.
   - **Test:** item 6a on RS58, then the 256-channel probe, against whole-group amplitudes, 8- and 16-contact blocks and item 6a(iii), by the protocol. Validate on item 6c's simulation first; **hard:** the true intensity scores highest.
2. **Time-discounted encoding, if item 6b finds material drift** (E5, E14). *Value: medium if drift is material, otherwise low. Cost: low.*
   - The package's fit functions already take per-time `weights` (`clusterless_mark_glm.py`, "Per-sample weights… in both the mark mixture and the rate maps"). So Arai's forgetting can be approximated without new likelihood code, using weights = exp(−(t_ref − t)/τ).
   - **Test:** this needs forward-chaining blocks (train on blocks before the test block), which **deviates from the 5 symmetric blocked folds**; state that in the report. Compare τ = ∞ (static) against τ chosen by inner validation on the training blocks, paired by test block.
   - Arai's real-data result predicts a small gain once many tetrodes are combined, so gate this on item 6b's drift magnitude.
3. **DREDge registration before feature extraction** (P3, P4). *Cost: low.* Gate on item 6b. Use SpikeInterface `correct_motion(preset="dredge")`, and compare with and without, by the protocol.
4. **Random-walk mark means (Kalman-filter mixture)** (P8). *Cost: medium.* Only if drift persists after registration. Each component mean μ_k(t) follows a random walk, smoothed inside the mark-mixture fit. Arai's code (E5) discounts only the rate and place-field hyperparameters, not the mark means, so P8 is the method that covers mark drift.
   - **Test:** item 6c's drift simulation first, then real data by the protocol.
5. **None of the Eden, Arai or Yousefi work tested dense probes**: every clusterless dataset in Part 1 is tetrodes. For area 6, the Paninski group is the only source here.

## (c) Not worth pursuing

| Idea | Source | Reason |
|---|---|---|
| Gibbs-sampled Bayesian mixture encoding (JOYFULMoGs) | E5 | About 15,000 sweeps; "still not real-time for most datasets"; no comparison with the KDE; unpublished. The MDN covers the joint model by EM at far lower cost |
| Gaussian-mixture or particle posterior filters on 1-D tracks | E6, E7 | The speed-ups are against naive Riemann-sum grids. A 1-D grid HMM step here is a small matrix-vector product. E7 is 682 ms per step and sorted only. Mixtures also lose track topology and give variable step times |
| Width, slope or other geometric waveform features | E8 | Amplitude beat width and slope on 24 tetrodes, and PCA beat amplitude at higher dimension |
| Discriminative decoders (DDD, D4, LSTM) | E17, E19 | No marked-point-process likelihood, so they can't be scored by `info_vs_flat_per_s`. They give up the mark reference and EM over discrete states. No clusterless evidence. The LSTM speed advantage (7×) was against MATLAB filters |
| P1's ridge or logistic readout, and its re-estimation of the mixture on test spikes | P1 | Not a likelihood, per-trial and non-causal; test-data refitting breaks the held-out protocol |
| Laplace or banded-Hessian MAP decoding, low-rank Kalman, MCMC decoding | P10, P11, Ahmadian et al. 2010 (from the Paninski review, PubMed 20964539, not reopened here) | Non-local posteriors have several peaks; the grid HMM is exact and cheap in 1-D and 2-D |
| Porting MATLAB code (GMM_PointProcess, Multi-Dimensional-Decoder, COMPASS) | E6, E7, E16 | The algorithms are short and clear from the papers. Most repositories have no licence. Reimplement in JAX only if an idea above is adopted |
| E10's box prefilter on mark neighbours, ported to JAX | E10 | Its data-dependent neighbour counts would cause recompiles or padding (item 3b). Kernel compression (items 3d and PR #105) and top-k components reach the same goal with static shapes |
| YASS; CEED's transformer | P5, P6 | YASS is dormant and superseded by DARTsort. CEED needs several GPUs and has no decoding validation; learned encoders are out of scope until item 2a finishes |
| Expected log-likelihood GLM fitting | P13 | Already equivalent: occupancy-weighted binned exposure |

## (d) Open questions

1. **Real-time hardware choice.** E10 reached 6 ms bins on a 32-core CPU without a GPU. Should item 3a treat "matches E10 on j16 at E10's settings" as the minimum bar for the CPU path?
2. **Running E10's code.** Is it acceptable to install and run `realtime_decoder` in a separate environment for a like-for-like latency comparison, given it has no licence file? Or should its estimator only be rebuilt in the harness?
3. **Detection thresholds.** What detection threshold do the panel's waveform-feature entries use? The threshold sweep in Area 2, item 1 can only raise it. Lowering it needs item 0a re-extraction.
4. **RS58 localization test.** Did the test that lost 25–35% include max peak-to-peak amplitude, as P1's (x, z, ptp) set does? If not, P1's exact set hasn't been tried on polymer probes.
5. **DARTsort neighbourhoods.** Does DARTsort's neighbourhood machinery fit Frank-lab polymer geometries (RS58's sort groups of 26–32 channels), or only Neuropixels-style regular layouts? Not checked.
6. **Differentiable transitions.** Can the track-graph random-walk transition be made differentiable in σ (Area 5, item 1b)? If not, route (a), EM with expected transition counts, is the path.
7. **Forward-chaining folds.** Should time-discounted encoding (Area 6, item 2) be allowed forward-chaining folds, which differ from the shared protocol? Or is it restricted to simulation (item 6c) unless item 6b shows drift?

## Could not verify

These are left out of the recommendations, or used only as stated.

- **Arai et al. 2018 (E5):** no journal version found; bioRxiv reports none (`published_doi: NA`). The last ~20% of its text (the references area) was not read.
- **Weber thesis (E8):** chapter 3's exact decoding numbers are in figures. Only the text conclusions are used.
- **Deng et al. 2016 (E2):** the per-decision real-time latency was not measured in the paper. Only the whole-event offline time (320 ms for 340 ms) is reported.
- **Zeng & Eden (E12):** no code link in the text read, and no compute times.
- **RealtimeDecoder (E10):** the text and Table 1 disagree on spike-incorporation and event-detection latency (36 vs 12 ms, and 36 vs 30 ms). No licence file in the repository.
- **E11, E13–E17 and E19:** PubMed or arXiv abstracts only.
- **A GitHub account for Kensuke Arai beyond `AraiKensuke`:** none found. **Ali Yousefi's accounts:** `YousefiLab` and `aliyousefi` exist; both were listed, and only state-space toolboxes relevant to area 5 were found.
- **From the Paninski review:**
  - P11 has no journal DOI (lab page "under review").
  - P1's Fig. 5 timings are only in a figure.
  - P3's full text was blocked (paywall, then bioRxiv HTTP 429).
  - P2, P8, P9, P12 and P14 were read as abstracts only.
  - No code was found for P8 or P15.
  - CEED's runtime is in an unread supplement.
  - Paninski's authorship of `ibl-neuropixel` was not checked.
