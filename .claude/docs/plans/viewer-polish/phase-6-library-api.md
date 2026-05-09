<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 6 — Library-native API + sidecar docs

**Goal:** close the "feels like an internal artifact viewer"
complaint. Most non_local_detector users know "I have a fitted
detector and `predict()` results"; they should never need to learn
`RunBundle` to use the viewer. Commit cluster on
`interactive-decoder-viewer`.

**Estimated effort:** ~4 hours.

**Skill:** `scientific-tdd` for the new launch-API overload.

**Dependencies:**

- **Tasks 6.1–6.4 require Phase 1 only.** They touch the public
  `launch_qt` API, docstrings, the notebook quickstart, and the
  `bundle-from-detector` devtool — all surfaces that don't
  conflict with the rest of the polish work. These tasks can land
  at any review checkpoint after Phase 1.
- **Task 6.5 requires Phase 2.** It edits `_build_controls_bar`,
  which Phase 2 (tasks 2.3 tooltips and 2.4 disabled-overlay
  controls) edits heavily. Doing 6.5 before Phase 2 would force
  rework or merge pain. Run 6.5 after Phase 2 has landed.

Per the single-branch rule, all of Phase 6 still goes on
`interactive-decoder-viewer` sequentially with whatever phase
came before it.

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules apply to every task
> here.

---

## 6.1 — `launch_qt(...)` overload accepting natural inputs

- [ ] Widen
  [viewer/qt.py:launch_qt](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py)
  to accept either:
  - The current `bundle` form (back-compat — pure
    `RunBundle | dict[str, RunBundle]` first positional plus
    existing kwargs), or
  - `detector=..., results=..., spike_times=..., position=...,
    position_time=..., speed=None, name="default"` (builds the
    bundle internally via `RunBundle.from_predict(...)`).
- [ ] Dispatch logic — **fail loud on mixed forms, do not silently
  ignore.** Public-API silent-drop is a footgun: a user who passes
  both a bundle and `detector=...` (e.g. while migrating their
  code) would get the bundle without warning that the detector
  kwarg was thrown away.
  - If `bundle` is provided AND any of the per-component kwargs
    (`detector` / `results` / `spike_times` / `position` /
    `position_time` / `speed`) are non-default → raise
    `ValueError` naming both the bundle and the conflicting
    kwarg(s), and pointing at the two valid call shapes.
  - If `bundle` is provided alone → use it (back-compat).
  - If only per-component kwargs are provided → require all of
    `detector`, `results`, `spike_times`, `position`,
    `position_time` (raise `ValueError` listing which are
    missing); construct the bundle internally.
- [ ] Tests:
  - Notebook-style invocation (per-component kwargs) passes
  - Existing bundle-form tests still pass
  - Mixed form (`bundle=...` + `detector=...`) raises `ValueError`
    with a message that names both the bundle and the conflicting
    kwarg
  - Per-component form missing one required arg raises
    `ValueError` listing the missing arg(s)

---

## 6.2 — Discoverable docstrings

- [ ] `launch_qt` docstring: lead with the per-component example
  (3 lines: fit → predict → launch_qt(detector=..., results=...,
  ...)). Move the bundle form to "Advanced — multi-run".
- [ ] `_DetectorBase.predict` docstring gets a one-line
  cross-reference: "see
  `non_local_detector.visualization.interactive.launch_qt`
  for an interactive view of these results".

---

## 6.3 — Notebook example

- [ ] Add a short `notebooks/interactive_viewer_quickstart.ipynb`
  (or extend an existing notebook with a section): fit a small
  detector, call predict, call `launch_qt(...)` with the four
  natural inputs.
- [ ] Notebook **must not** call `RunBundle` directly — that's the
  whole point.
- [ ] Run the notebook end-to-end; confirm cell outputs are
  reasonable.

---

## 6.4 — `bundle-from-detector` devtool subcommand

- [ ] New subcommand in
  [devtools/__main__.py](../../../../src/non_local_detector/visualization/interactive/devtools/__main__.py):
  `bundle-from-detector --detector model.pkl --results results.nc
  --spikes spikes.npz --position position.parquet --out bundles/foo/`.
- [ ] Devtool emits a `--run-from-dir`-compatible bundle directory
  the user can then point `--run-from-dir` at, optionally followed
  by `build-viewer-cache` for the zarr acceleration.
- [ ] Test: round-trip — generate a bundle from in-memory pieces
  via the devtool, load via `--run-from-dir`, confirm parity.

---

## 6.5 — Controls-bar gestalt grouping

**Issue:** controls bar has 10–15 widgets in one flat row with no
visual grouping. Heer would point to Gestalt grouping.

- [ ] Insert thin vertical `QFrame` separators between three logical
  clusters in
  [_build_controls_bar](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L709-L818):
  - Navigation: center slider + window slider + window label
  - Playback: slice overlay combo + play button + speed combo
  - Swap: model combo + overlay combo + visibility checkboxes
- [ ] Tighten `addSpacing` within clusters; widen between.
- [ ] **[manual]** visual smoke — separator placement and spacing
  (no automated test feasible; surface to user).

---

## Phase 6 done when

- Notebook quickstart runs end-to-end
- Existing bundle-form CLI/launch tests still pass
- Devtool round-trip test passes
- Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification)
- `[manual]` item from 6.5 (visual gestalt smoke) surfaced to user

**If this is the final remaining phase** (i.e. Phases 1–5 are
also done), run the [README.md](README.md) §"Pre-merge checklist
(final)" and **stop and wait for user review** before merging the
`interactive-decoder-viewer` branch to `main`. Otherwise, stop
and wait for user review before proceeding to whichever phase
comes next in the user's chosen order.
