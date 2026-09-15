# Phase 3 — Backends bin globally, allocate only requested rows

> **REVISED REQUIREMENTS — NEEDS PROTOTYPING.** The boundary-spike defect is
> reproduced below. The former unexecuted implementation snippets and >4× total
> decoding-memory target are withdrawn. This phase owns correct event ownership
> and likelihood-specific allocation bounds. Full-session posterior storage and
> checkpointed smoothing belong to [Phase 7a](phase-7-performance.md).

## Problem

With `n_chunks > 1`, likelihood caching is disabled in `models/base.py`.
The core calls the likelihood function with the full spike arrays and a sliced
time array. Each predictor clips spikes to its own `[time[0], time[-1]]`, so a
spike between adjacent chunks is outside both.

Reproduced under the current timestamp convention:

```text
time = [0..5], spikes = [0.5,1.5,2.5,3.5,4.5], chunks = [0:3], [3:6]
full    : [1 1 1 1 1 0]   total 5
chunked : [1 1 0 1 1 0]   total 4     -> 1 spike lost per boundary
```

The existing `tests/integration/test_core_kde_integration.py` test precomputes
likelihoods and therefore avoids this defect. Regression coverage must exercise
the public path with likelihood caching disabled.

## Scope and dependencies

`n_chunks` is intended to bound likelihood memory, including density evaluation
and reductions. Computing a full-time likelihood and then slicing it does not
meet that requirement. Restricting the final reduction alone also leaves the
clusterless density evaluation over all decoding spikes in memory.

The [representative workload](overview.md#representative-workload) has 1.8 million
time rows and approximately 16,202–64,802 combined hidden bins. Phase 3 must
produce correct likelihood chunks for that future pipeline, but its completion
does not establish bounded memory for the complete HMM or returned posteriors.

Preserve current unchunked event ownership in this phase. Use the settled decode
vocabulary in [C3](shared-contracts.md#c3--time-vocabulary) when naming concepts;
do not independently change the endpoint convention, intensity units, encoding
exposure, C1 floors, or Phase 0 conditioning. Phase 6a migrates the time API, and
6b/6d calibrate intensities and model units. Revalidate chunk ownership against
those contracts before Phase 7 integration.

## Falsification

Before implementation, reproduce the defect through public `predict` with
spikes strictly between adjacent chunk timestamps and caching disabled. Verify
that the unchunked path includes their contributions and the current chunked
path loses them. A helper-only test or cached-likelihood test cannot establish
that the integration defect is fixed.

For memory, profile a single requested likelihood chunk while increasing the
recording length and holding chunk length and fitted model fixed. Identify
full-time spatial arrays or density evaluations over decoding spikes outside
the requested range. Record the failing allocation behavior before rewriting
each affected backend.

## Tasks

### 1. Prototype global event ownership with bounded evaluation

- Prototype an interface that identifies a row range in the full decoding
  timeline. `row_slice` is a candidate API, not a settled signature.
- Assign each decoding spike to its global observation bin consistently, then
  select the spikes belonging to the requested rows **before** expensive density
  evaluation. Slice waveform features using exactly the same selection.
- For clusterless reductions, translate selected global bin IDs into local
  chunk indices and allocate only the chunk's output rows. A full-time
  `segment_sum` followed by slicing is insufficient.
- For sorted backends, form the counts and likelihood arrays needed for the
  requested rows. Avoid repeatedly binning the full recording for every chunk;
  prototype reusable event indices or indexed range lookup.
- Inventory the fixed encoding-model allocations separately from decoding
  workspace. Large encoding-spike × position kernels are a Phase 7c profiling
  target; limiting time rows does not by itself remove them.

No implementation snippet becomes prescriptive until this behavior has been
run against the real backends and compared with the unchunked reference.

### 2. Cover the entire model assembly path

Thread the row range through both sorted and clusterless detector assembly,
including all registered backends and the no-spike model, which bypasses both
registries. Allocate the assembled likelihood as chunk rows × hidden bins.
Align missing-data masks, local-position likelihoods, non-local penalties,
position interpolation, and time-varying transition inputs with those same
global rows. Define ownership of slicing at each layer so arrays are neither
left unsliced nor sliced twice.

### 3. Integrate both core transition paths

Exercise stationary and covariate-dependent transitions independently; both
detector families can use either core path. Preserve the generic likelihood
callback contract: bind the new backend-specific information at the model layer
or provide an explicit adapter, rather than injecting an unsupported keyword
into arbitrary user callables. A legacy callback must not be presented as a
bounded/global-binning implementation without evidence that it satisfies those
contracts. Choose and test the compatibility behavior in the prototype.

### 4. Preserve requested likelihood outputs

When the caller explicitly requests likelihoods, return the correct rows in
global order for cached and uncached prediction. During Phase 3 an explicitly
requested full likelihood result may still consume `T × N` memory; record that
exception. Unrequested likelihood chunks should be released after use. The
incremental output mechanism and full-session memory guarantee are Phase 7a
work, not prerequisites for shipping the boundary-spike correction.

### 5. Release note

Describe which chunk-boundary spikes were lost, the corrected event ownership,
and the behavior of requested likelihood outputs. Report measured likelihood
memory changes separately from total HMM/output memory.

## Validation

Use existing applicable tolerances; numerical contracts and golden data are not
changed to accommodate chunking. Test proposed APIs only after their signatures
have been selected by prototyping.

| Coverage | Required observation |
|---|---|
| Public prediction, cached vs. uncached | Corrected chunked outputs and evidence agree with the unchunked reference for sorted/clusterless detectors and stationary/covariate transitions. |
| Event ownership | Every in-range spike contributes once across chunk boundaries; cover exact boundaries, recording endpoints under the active contract, chunks with no spikes, singleton row ranges, and a ragged final chunk. Validate empty row requests according to the chosen interface. |
| Backend row ranges | Each supported backend's requested rows equal the corresponding full-reference rows; clusterless waveform selection stays aligned with selected spikes. |
| Assembly branches | No-spike states, missing rows, local-position kernels, and non-local penalties use the same global row indices. |
| Generic callbacks | Existing callback arguments remain valid; adapters and any unsupported capability handling are explicit and tested. |
| Requested outputs | Likelihood rows are ordered and complete when requested; unrequested chunk arrays are not accumulated. |

### Memory evidence

Measure likelihood production separately from the HMM and result assembly.
With fitted inputs and chunk length fixed, increasing total duration must not
introduce a full-time spatial likelihood array or density evaluation for all
decoding spikes on each call. Vary chunk length to show which workspaces scale
with requested rows and selected spikes. Declare recording-sized input/index
metadata and resident encoding allocations separately.

Use allocation/shape checks plus measured host/device peaks on representative
backends. Compiler temporary-memory estimates and live-array snapshots are
supporting evidence; neither alone measures the complete transient process
peak. Report the method and its limitations.

The old **>4× reduction in total predict memory** is not an acceptance criterion:
full posterior retention makes it inappropriate for this phase. End-to-end
bounded-memory acceptance is specified in Phase 7a.

Run core/integration tests and the affected backend tests, including golden
regressions. Cached-path goldens are expected to remain unchanged; investigate
unexpected numerical changes under the existing numerical-validation process.

## Review

Have the implementation reviewed for event ownership before density evaluation,
every model assembly return path, generic callback compatibility, and both core
transition paths. Include the parity and likelihood-specific memory evidence.
