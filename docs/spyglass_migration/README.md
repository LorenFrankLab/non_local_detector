# Spyglass companion migration

The [current adapter patch](spyglass-current-explicit-time-grid.patch) targets
Spyglass `cbc30e2088e2b5fa07a6e013f99ace7141bf6d9a` plus the captured uncommitted
observed-time edits. [Baseline hashes](spyglass-current-baseline.json) identify
its exact input files. It preserves unrelated sorting, workflow, and dependency
changes; no live downstream file was modified during this review.

The patch migrates sorted and clusterless prediction/EM, original tracking
support, uniform grids, gap masks, timestamped transition covariates, interval
labels, shared-boundary spike/mark ownership, and ahead/behind consumers. It
includes actual-method and make-to-fit tests plus a downstream migration guide.
It targets the planned `non-local-detector>=0.7,<0.8` release and retains the
current JAX cap. Version 0.7 has not been tagged or published by this work.
The live adapter's existing `non-local-detector==0.6.9` pin protects it until
coordinated rollout.

## Scientific and boundary checks

- Original tracking samples and caller training/group weights remain separate
  from physical neural-recording availability. A connected observed acquisition
  window `[0.11, 0.3]` now gives 0.19 seconds of exposure on 30 Hz tracking;
  one event fits `1 / 0.19 = 5.2631579` Hz instead of 6 Hz. The original
  interpolation basis is clipped with `encoding_time_range`.
- Fitting across disconnected physical observation windows is rejected when
  the native acquisition-range API cannot express that support. Select an
  encoding range within one connected window or fit separate models. Decode
  observation masks may still cover multiple windows. Tracking continuity is
  not redefined to approximate recording availability.
- Timestamped caller masks retain their timeline. Physical observation bounds
  are applied directly to decode bins, avoiding camera-grid censoring and
  shape mismatches. Short decode intervals between camera samples remain valid
  when surrounding measured tracking supports their bins.
- Shared events and marks have one owner. Half-open physical observation stops
  filter events explicitly rather than nudging the decode clock. Shifted-origin
  EM preserves all 100 complete bins when the final edge rounds one ulp past
  a requested stop.
- Interval membership uses O(number of bins) temporary memory. With 18,000
  bins, 10/50/100 windows use about 0.884 MB each; the former 50-window broadcast
  used 23.6 MB. Perfect 500-bin trajectories from 31 camera samples yield 500
  zero ahead/behind distances, with unsupported rows left missing.

The current companion passes **51 hermetic tests**, independently repeated,
on Python 3.11.15, DataJoint 0.14.9, PyNWB 3.1.3, and JAX 0.6.2. Actual `make`
source executes with read-only fetch fixtures and stops before persistence;
its exact prepared inputs then fit real detectors. Four synthetic NWB
write/read round trips feed all four sorted/clusterless prediction/EM paths.
No production table import, database connection, or table write is required.

## Applying and qualifying the current patch

Verify all recorded baseline hashes before applying. During qualification,
all six production/dependency files still matched, but the live
`tests/decoding/test_observed_time.py` independently advanced. Rebase that test
hunk onto its newer fixtures before applying the complete patch; preserve
all existing local edits. The full artifact passes `git apply --check` on its
captured isolated baseline. A plain `cbc30e208` checkout lacks the captured
uncommitted test/adapter edits and is not that exact baseline.

```sh
git apply --check /path/to/spyglass-current-explicit-time-grid.patch
```

Run the included tests in the patched checkout with its supported environment
and this detector source on `PYTHONPATH`:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/tmp/spyglass-migration-numba \
MPLCONFIGDIR=/tmp/spyglass-migration-mpl \
PYTHONPATH=/path/to/non_local_detector/src \
uv run --no-sync python -m pytest --noconftest -o addopts= -o filterwarnings= \
    tests/decoding/test_time_grid_migration.py -q
```

Before release, qualify the rebased adapter with normal DataJoint database
fixtures, selected backend model/result persistence, supported Python
3.10–3.12 and conda profiles, and representative recomputed scientific outputs.
The hermetic environment uses SciPy 1.17.1, satisfying the current declarations.
GLM persistence remains the separately deferred Patsy serialization defect.
Publish matching detector/adapter releases together and repopulate derived
results; hermetic checks do not establish production database compatibility.

## Archived baseline

The [archived adapter patch](spyglass-explicit-time-grid.patch) targets exact
Spyglass commit `8d3cb4408ed5b86f28412c69434e8fa2af3b06aa`. It passes **27 tests**,
including synthetic NWB round trips and between-camera decode intervals.
It predates the newer observed-time feature. Apply-check it only in a clean
checkout of that commit:

```sh
git switch --detach 8d3cb4408ed5b86f28412c69434e8fa2af3b06aa
git apply --check /path/to/spyglass-explicit-time-grid.patch
```

The alternative [legacy cap](spyglass-legacy-compatibility-cap.patch) protects
that archived old adapter below 0.7. Apply it instead of the archived migration,
never together. The current adapter already has an exact 0.6.9 pin and does not
need this cap. SciPy 1.17.1 does not qualify the archived conda profile's
`scipy<1.13` constraint; its conda qualification remains separate from the
current profile.
