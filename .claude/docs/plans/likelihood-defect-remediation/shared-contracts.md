# Shared Contracts

These contracts must be settled before changes in the phases that depend on
them. Alternatives are recorded so a later reader does not reopen a closed
decision by accident. [Phase 0](phase-0-core-hmm.md) fixes core HMM conditioning
independently of these likelihood and time/exposure decisions.

**State: C2 and C3a are settled; C1 and C3b are unresolved.** C1's former policy
was withdrawn after it was shown to be non-monotonic. C3b's draft exposure helper
undercounts the full-cell reference and needs redesign. Implementation that
selects either unresolved policy is blocked; work preserving current semantics
or depending only on the settled contracts can be prototyped independently. See the execution order in
[PLAN.md](PLAN.md#execution-order-and-baselines).

- [C1 — Degeneracy policy for log intensities](#c1--degeneracy-policy-for-log-intensities)
- [C2 — Exposure ownership and spike weights](#c2--exposure-ownership-and-spike-weights)
- [C3 — Time vocabulary](#c3--time-vocabulary)

---

## C1 — Degeneracy policy for log intensities

> **PARTIALLY RESOLVED.** The "floor only `-inf`" rule recorded here was found
> to be non-monotonic and is withdrawn. **Settled (2026-09-14):** the GMM
> numerator-ordering defect and ground-process underflow/overflow are fixed as
> pure arithmetic corrections in phase 2 (raw log ratios, log-space ground
> process, no new floors). The zero-rate fallback, mean-rate floor, and
> summed-intensity clip are preserved; the fit-time `occupancy > 0` guard is
> removed because it turned float32 underflow into an `EPS` substitute
> (precision loss, not a support policy). The resulting inconsistency is
> recorded in the phase document. **Deferred:**
> a package-wide degeneracy policy. The recommended direction is a background
> firing model (explicit background rate and mark distribution added to every
> electrode's intensity and expected-count terms) rather than any of options
> 1–3 below; it is a statistical-model change and needs its own proposal,
> including what "unsupported" (zero occupancy, `0/0`) means. Do not remove
> the zero-rate fallback until that replacement exists and is validated.

### The problem (settled)

Flooring the **numerator** before forming `log_rate + log_marginal −
log_occupancy` destroys tail information. For `log_joint = −60`,
`log_occupancy = −8` the true intensity is `−52`; clamping `log_joint` at
`LOG_EPS = log(1e-15) ≈ −34.54` first yields `−26.54` when `log_rate = 0`, an error
of 25.5 log units (~1e11 in intensity). The audited clusterless GMM intensity
paths do this; the ordering defect is settled.

### Why "floor only `-inf`" fails

Flooring exact zero but not tiny finite values **inverts the likelihood ordering
at the boundary** — verified:

| observation | true log-intensity | policy returns |
|---|---|---|
| impossible (zero mass) | `-inf` | **−34.54** |
| possible, 1e-44 | −101.31 | −101.31 |
| possible, 1e-20 | −46.05 | −46.05 |
| possible, 1e-10 | −23.03 | −23.03 |

An **impossible** observation scores −34.54 while a merely very unlikely one
scores −101.31. Impossibility is rewarded over near-impossibility, which
contradicts the Poisson rationale the policy was justified by. Any rule that
floors `-inf` but not values below the floor has this inversion; monotonicity
requires flooring **both** or **neither**.

### Scope is wider than first recorded

"Package-wide" reaches more than the two `clusterless_kde_log` caller clamps:
`clusterless_kde.py:98` and `clusterless_diffusion.py:667` both floor finite
values via `safe_log` in probability space. Whichever policy is chosen must name
every site explicitly.

### Options

1. **Floor neither — keep true `-inf`.** Monotone and mathematically correct: an
   impossible observation *should* have zero likelihood. Requires the all-`-inf`
   row case to be handled downstream; `core.py`'s `_accumulate_chunk_degeneracy`
   already tallies degenerate timesteps, so the machinery exists.
2. **Floor both — clamp the finished intensity at `LOG_EPS`.** Monotone and
   bounded; caps how much evidence one spike can carry, losing discrimination
   below 1e-15. Closest to today's behaviour, and fixes only the ordering bug.
3. **Narrow the scope** — fix only the numerator-before-ratio ordering in the GMM
   and leave every existing floor as-is. Smallest change; leaves the two
   clusterless paths numerically divergent.

### Downstream requirements, whichever is chosen

- **Core conditioning has its own correctness requirement.** Phase 0 reproduces
  probability-mass loss from `_normalize` and avoidable underflow in
  `_condition_on`. C1's floor inventory does not cover either defect. Fix them
  independently and validate the selected C1 policy against the corrected core.
  Positive prior mass on at least one finite log-likelihood state must yield a
  normalized posterior and finite one-step evidence for the phase-0 fixtures.
  True zero support retains the existing prior fallback with `-inf` evidence;
  NaNs must remain visible. No likelihood-flooring choice is made by phase 0.
- **An aggregate ground-process floor applies after aggregation.** Verified:
  8 degenerate electrodes each floored to `LOG_EPS` then combined by `logsumexp`
  give `8 × EPS`, not `EPS`. If the selected policy floors the aggregate once,
  keep per-electrode degeneracy as `-inf` through that combination. This does
  not choose the policy for a finished per-spike intensity.
- **Zero-occupancy *and* zero-marginal is a distinct case.** `log_rate + (-inf) −
  (-inf)` is `NaN`, so an `& ~isnan(...)` guard declines to floor it — verified.
  Detect zero-mass operands *before* the subtraction so a genuine `NaN` from a
  broken computation still reaches `core.py`'s diagnostics.

### Required before choosing: a floor inventory

The scope question cannot be answered without knowing every site. The partial
list recorded earlier was incomplete. Build a **backend × {local, non-local} ×
{spike term, ground process}** matrix covering at least:

| Site | What it floors |
|---|---|
| `clusterless_gmm.py:722`, `:754`, `:876` | joint log density, before the ratio (the defect) |
| `clusterless_kde.py:98` | finished intensity, via `safe_log` in probability space |
| `clusterless_kde.py:158` | assembled block result, `jnp.clip(..., min=LOG_EPS)` |
| `clusterless_kde_log.py:90-92` | occupancy inside the shared helper, via `safe_log` |
| `clusterless_kde_log.py:1328` | assembled non-local result |
| `clusterless_kde_log.py:1905-1912` | local per-spike contribution |
| `clusterless_diffusion.py:667`, `:787` | intensity via `safe_log` |
| sorted backends | interior place fields EPS-floored at fit (`sorted_spikes_kde.py:225-234`, `sorted_spikes_glm.py:351-354`) |

The sorted row matters: including sorted likelihoods would require Phase 2 to
reconcile the existing EPS zero-exposure fallback and field floors explicitly.
Phase 1 is already complete under the existing policy; this future decision
does not retroactively block or reopen it. Phase 7c must use the policy selected
for its baseline and cannot assume every log field is finite.

Three sub-decisions must be made explicitly alongside the main one:

1. **Does the scope include sorted likelihoods?** Any resulting changes to the
   completed Phase 1 fallback belong to Phase 2, with their own validation.
2. **What does zero-numerator-over-zero-occupancy mean?** `0/0` is *unsupported*,
   not "true zero" — the distinction changes whether it floors or propagates.
3. **Is an all-degenerate ground-process aggregate zero, or floored once?**

Apply these requirements at the sites covered by the selected scope; identify
any existing out-of-scope behavior explicitly rather than claiming package-wide
parity.
Phase 2 must distinguish raw log ratios, per-electrode ground-process terms,
their aggregate, and finished likelihoods when specifying where a floor applies.

---

## C2 — Exposure ownership and spike weights

Two mechanisms currently assign a spike to an encoding model, and they disagree.

1. **Hard windows** — `_get_group_spikes` / `_get_group_spike_data`
   (`models/base.py:3804-3826`, `:2838-2860`) select spikes by interval.
2. **Interpolated weights** — `common.interpolate_weights_at_spike_times`
   (`common.py:161-182`) gives each spike `np.interp(t, position_time, weights)`,
   which is fractional near a mask transition.

**Decision: interpolated weights are canonical and sufficient. Delete the hard
windows.**

An earlier draft required windows that both (a) retain every spike with non-zero
interpolated weight and (b) never overlap. Those are mutually unsatisfiable —
verified: for mask `[1,1,0,1,1]` at times `0…4`, a spike at `t=1.75` has weight
`0.25` in that group and `0.75` in its complement, so any window keeping it for
one group must overlap the other's.

The requirement was solving a problem that does not exist. **Interpolated group
weights are a partition of unity**, verified exactly for 2, 3, and 5 groups:

```
2 groups: per-spike weight sum  min=1.000000000000  max=1.000000000000
3 groups: per-spike weight sum  min=1.000000000000  max=1.000000000000
5 groups: per-spike weight sum  min=1.000000000000  max=1.000000000000
```

This is exact by construction, not numerically lucky: `np.interp` is linear and
the per-group one-hot masks sum to the all-ones vector, so
`Σ_g interp(mask_g) ≡ interp(Σ_g mask_g) ≡ 1`. A boundary spike is *apportioned*
between groups, never duplicated. Overlap is therefore harmless, and the
non-overlap rule can go.

**The rule:** select a group's spikes by `interpolated_weight > 0`. No hard
windows, no `time_delta` arithmetic, no run-boundary special cases.

This rule applies within the supported recording domain; it does not choose the
acquisition endpoints or tracking-gap exposure policy deferred to C3b. Phase 5
must preserve event/feature alignment and apply each event weight once. For the
GLM, fractional event ownership requires weighted count sufficient statistics;
weighting ordinary sample counts a second time is not equivalent. Phase 6a's
encoding-cell migration must preserve that Phase 5 ownership contract.

### How much does this change results?

Measured on a 200 s / 100 Hz fixture with one place cell (289 spikes), comparing
interpolated ownership against stepwise cell assignment:

| encoding-group block | transitions | spikes with differing weight | \|Δ mean_rate\| | max \|Δ place field\| / peak |
|---|---|---|---|---|
| 100 s | 1 | 0 / 289 | 0.00% | 0.00% |
| 20 s | 9 | 0 / 289 | 0.00% | 0.00% |
| 5 s | 39 | 0 / 289 | 0.00% | 0.00% |
| 1 s | 199 | 3 / 289 | 0.40% | 0.96% |
| 0.2 s | 999 | 14 / 289 | 0.33% | 1.86% |
| 0.05 s | 3999 | 48 / 289 | 0.29% | 5.75% |
| 0.02 s | 9999 | 164 / 289 | 2.17% | 6.50% |

The longer blocks in this fixture produced identical results because no spikes
fell near their transitions. Block structure alone does not guarantee equality:
any event near a transition can receive a fractional weight. Shorter blocks made
those events more frequent here. Continuous EM weights are also covered by the
chosen interpolation contract.

(A first attempt at this measurement drew spike times *on* the position grid and
found zero difference everywhere. Spikes land at continuous times within a sample
interval; the fixture must jitter them, or the comparison is vacuous.)

**Invariant (scoped to the Phase 1 fixtures).** For the tested KDE exposure
calculation, fitting full arrays with `weights=m.astype(float)` agrees with a
subset fit when spikes are away from mask transitions and encoding coordinates
and exposure conventions are otherwise identical. This is not a universal
equivalence for refitted GLM bases or for a subset time grid that bridges gaps or
redefines sample-cell exposure. Near a mask transition the event weights differ
by construction. Phase 5 tests canonical weighted-event ownership directly;
Phase 6 compares physical exposure on the original acquisition timeline.

An earlier draft asserted this invariant unconditionally on the strength of a test
whose mask boundary happened to have no nearby spikes. Tests asserting it must
either place spikes away from transitions (exact form) or allow a residual —
existing MRF tests already do the latter.

---

## C3 — Time vocabulary

**Split. The decode half is settled; the encoding-exposure half is not.**

### C3a — Decode vocabulary (settled)

Name the three quantities separately and never reuse one array for two roles.

| Name | Shape | Meaning |
|---|---|---|
| `time_edges` | `(n_bins + 1,)` | Decode bin boundaries used for event assignment and deriving centers/durations; not observation coordinates. |
| `time_centers` | `(n_bins,)` | `0.5 * (edges[:-1] + edges[1:])`. Drives position interpolation, local-position kernels, non-local penalties, HMM row coordinates, and the xarray `time` coordinate. |
| `bin_durations` | `(n_bins,)` | `np.diff(edges)`. Scales Poisson intensities. |
| `n_bins` | scalar | `len(time_edges) - 1`. The number of likelihood rows and HMM observations. |

Bin `i` covers `[edges[i], edges[i+1])`, except bin `n_bins - 1` which is
right-closed so a spike at `edges[-1]` is counted. This is the target contract
implemented by Phase 6a. [Phase 3](phase-3-chunk-boundary.md) has implemented
global event binning **under the current unchunked convention**: a chunked
likelihood request now bins every spike against the full decoding timeline and
returns exactly the rows of the full-time result, but the convention itself is
untouched — a spike at `time[-1]` still lands in row `n-2` and the final row
still never owns a spike. Phase 6c's detector guard ships with 6a. Full nonuniform
likelihood support, including duration scaling, is established by 6b/6d.

### C3b — Encoding exposure (UNRESOLVED)

> **Do not implement against this section.** The helper drafted here undercounts;
> the replacement needs a decision this document cannot make.

`position_time` is an array of sample **centers**, not decode edges, so encoding
exposure needs its own per-sample cell widths — `weight_sum × median(diff(...))`
is wrong for jittered or gapped timestamps.

The drafted helper clamped the outer edges to the first and last sample centers.
Compared with an acquisition containing one complete 1 s cell per sample, it
undercounts — verified under that reference convention:

| N uniform centers, dt=1 | clamped helper exposure | full-cell reference exposure |
|---|---|---|
| 2 | 1.00 | 2.00 |
| 5 | 4.00 | 5.00 |
| 100 | 99.00 | 100.00 |
| 1 | 0.00 | 1.00 |

Relative to that reference, Hz rates would be inflated by `N/(N−1)` for N > 1,
and a single sample gets zero exposure. A single center alone cannot determine
the 1 s width assumed in the table. Clamping can describe a different acquisition
domain, but it cannot silently replace full-cell exposure while claiming equal
encoding/decode intervals preserve the old arithmetic. Phase 6b's preservation
check requires matched physical exposure under the chosen endpoint convention.

Two decisions are required before this can be written:

1. **Endpoint policy** — extrapolate half-cells at the ends (`t[0] − dt₀/2`,
   `t[-1] + dt_{N-1}/2`), require explicit acquisition bounds from the caller, or
   reject input too short to define a cell.
2. **Gap policy** — is a long jump in `position_time` exposure (the animal was
   tracked, sampling was sparse) or missing data (tracking dropped)? The current
   code has no policy, and `base.py:2942-2949` already drops NaN position rows
   *before* this point, so interpolation silently bridges dropped-tracking gaps.

Whatever is chosen, encoding exposure is `sum(weights × sample_cell_width)`.

**Do not pass `position_time` to a decode-edge binning helper.** The sorted GLM
currently does exactly that (`sorted_spikes_glm.py:299`, `:339`), which is why
rewriting the shared helper in place would break GLM fitting: N samples would
yield N-1 counts against an N-row design matrix. Phase 6a gives encoding its own
helper.

**Uniform-bin restriction.** `core.py` applies one transition matrix per row
regardless of that row's duration (`_filter_internal` call at `core.py:643`), so
nonuniform bins produce a time-miscalibrated posterior even when the likelihood
is correct. The target detector contract therefore **requires uniform edges**
and raises otherwise. Phase 6c implements that guard with 6a; it is not a claim
that the current code already enforces it. After 6b/6d, nonuniform edges are
supported by the direct `predict_*_log_likelihood` API, where no transition
model is involved. See
[overview.md](overview.md#deferred-with-triggers) for the duration-calibrated
follow-up.
