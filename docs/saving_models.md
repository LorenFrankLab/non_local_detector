# Saving and loading models

`save_model` writes a fitted detector to one file, and `load_model` reads it
back in this and every later version of non_local_detector. The file holds
what the fit learned (place fields, encoding samples, fitted transitions) and
the configuration, never Python objects, so loading it runs no code from the
file. Decoding results are saved separately, with `save_results` (see
[Result layout](result_layout.md)).

<!-- docs-test: run -->
```python
import numpy as np

from non_local_detector import SortedSpikesDecoder, time_edges_from_centers
from non_local_detector.model_format import read_model_summary
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

_, position, spike_times, time, _, sampling_frequency, *_ = make_simulated_data(
    n_neurons=5
)
time, position = time[:3000], position[:3000]
spike_times = [spikes[spikes <= time[-1]] for spikes in spike_times]
decoder = SortedSpikesDecoder(sampling_frequency=sampling_frequency).fit(
    time, position, spike_times
)

decoder.save_model("decoder.nld")
loaded = SortedSpikesDecoder.load_model("decoder.nld")

time_edges = time_edges_from_centers(time)
original = decoder.predict(spike_times, time_edges=time_edges)
reloaded = loaded.predict(spike_times, time_edges=time_edges)
assert np.array_equal(original.acausal_posterior, reloaded.acausal_posterior)

summary = read_model_summary("decoder.nld")
place_fields = summary["encoding_models"][0]["place_fields"]  # (5, n_bins), Hz
print(summary["detector_type"], place_fields.shape)
```

`.nld` is the conventional suffix. `load_model` recognizes the format from
the file's first bytes, not its name, and still reads the pickles that
earlier releases wrote (below). It returns an instance of the class it is
called on, or of a subclass: `SortedSpikesDetector.load_model` (from
`non_local_detector.models.base`) loads any sorted-spikes detector, and a
file holding another class raises `ValidationError`.

## What a file holds

A saved model is a zip archive of:

| Member | Contents |
| --- | --- |
| `manifest.json` | `format` (`"non_local_detector.model"`), `format_version`, the `package_version` that wrote it, the `detector_type`, and every array's dtype, shape, dimension names and sha256 |
| `config.json` | The detector's `get_params()`: environments, transitions, initial conditions, observation models, algorithm parameters |
| `state.json` | The fitted state, referring to arrays by member name |
| `arrays/**.npy` | The arrays, loaded with `allow_pickle=False`; members over 4 KB are deflate-compressed |

Objects such as transitions, environments and fitted mixtures appear in the
JSON as `{"$type": "nld.RandomWalk", "state": {...}}`. Loading builds them
only through a registry of known type names, and never imports a module named
by the file. Each array is checked against its sha256 when read, so a
corrupt or edited file raises `DataError`, as does a file holding a member the
manifest does not list, one listed twice, or one declaring more bytes than its
array can hold; such members are refused before they are decompressed.

Only what cannot be rebuilt is stored. Loading rebuilds:

- each environment's grid: an environment with a track graph from its
  configuration, an open field from its stored bin edges and interior mask
  (`Environment.fit_place_grid_from_edges`), with their graph distances;
- the continuous transition, dense matrix or structured operator, from the
  transitions and the environments;
- the patsy design of a covariate-dependent discrete transition, from the
  stored covariates, and the GLM spline basis from its stored inputs.

A transition block built from the fit data, such as `EmpiricalMovement`, is
stored instead, since the training positions are not.

A covariate transition's covariates are stored only as numeric or string
arrays (a categorical as its codes and categories).

A JAX array loads as a JAX array of its stored dtype. One that is 64-bit,
saved with `JAX_ENABLE_X64` set, loads as a NumPy array when x64 is off,
since JAX would round it to 32 bits.

Rebuilding repeats the fit's arithmetic, so on the same machine a reloaded
model decodes exactly as the original. A newer NumPy, patsy or
non_local_detector could rebuild these structures differently, so the file
also stores small fingerprints of them (weighted sums and sums of squares).
Before a loaded model first decodes, the rebuilt structures are compared with
the fingerprints at a relative tolerance of 1e-6 (float32 rounding), which
accepts last-bit differences between platforms and rounding-level changes to
the code that rebuilds them. The GLM spline basis is checked the same way. If they differ, decoding raises
`ValidationError` asking for a refit; loading and inspecting the model still
work. The bin edges, centers and interior mask are stored as well: a grid rebuilt
within tolerance takes those exact values, so the encoding models, which record
an exact hash of their grid, decode on any platform.

## Transition formulas from a file

patsy evaluates a covariate transition's formula as Python, and a saved file
may be someone else's, so `load_model` checks a stored formula against a
grammar before patsy sees it. The grammar allows the names of the stored
covariates, numbers, arithmetic (`**` only with a small numeric exponent),
lists of numbers, `np.pi`, the package's spline and transform functions
(`quantile_bs`, `cyclic_bs`, `bs`, `cr`, `cc`, `te`, `center`, `standardize`,
`I`), elementwise numpy functions (`np.log`, `np.log1p`, `np.log2`,
`np.log10`, `np.exp`, `np.expm1`, `np.sqrt`, `np.square`, `np.abs`, `np.sign`,
`np.floor`, `np.ceil`, `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`,
`np.arctan`, `np.arctan2`, `np.minimum`, `np.maximum`) and `C()` with
`levels=` and a contrast of patsy's `Treatment`, `Sum`, `Helmert`, `Poly` or
`Diff` called with literal arguments, such as `C(arm, Treatment('b'))`.
Integers computed from literals must stay within int64, a formula may have
at most 2,000 characters and nest at most 32 deep, a spline's `df` and
`n_knots` are integers from 1 to 1,000 and its `degree` from 0 to 10,
`knots=` and `levels=` are lists of at most 1,000 entries, and a formula makes
at most 32 calls.

By default, loading never runs code from a `.nld` file. A crafted file can
still make loading or decoding slow or memory-hungry, for example with a
formula whose spline terms interact into a very large design, which these
limits do not prevent, or with an array of ones, which a file stores as its
dtype and shape alone (`$fill`), so the archive's checks on each member's
size do not bound it. Load files from untrusted sources with care.

A file whose formula is outside the grammar loads for inspection only:
`load_model` warns, naming the formula, and does not call patsy, so the design
(and the training-time `discrete_state_transitions_`) is not rebuilt. The
transition's `formula` is an inert `UntrustedFormula`, whose `.text` is the
stored formula; patsy cannot evaluate it. Copies, pickles and
`sklearn.clone` of the model carry it, so decoding, fitting,
`estimate_parameters`, `discrete_transition_intervals`,
`plot_discrete_state_transition` and `save_model` on the model or any copy of
it raise `ValidationError`, for a model saved fitted or unfitted, until you
give the transition a formula of your own (a plain string). A formula within
the grammar is a `LoadedFormula`, a string checked again, with the covariate
names, whenever a design is built from it, for example when an unfitted saved
model is fitted. If you trust the file, for example because you saved it
yourself, load it with `load_model(path, trust_formulas=True)`: the formula is
then a plain string and any formula is rebuilt. `read_model_summary` never
evaluates a formula, so it reads any file.

Formulas you write in your own session are your own code and are not checked:
any patsy formula fits and decodes. Pickles from earlier releases are not
checked either, and `trust_formulas` does not apply to them: unpickling can
run arbitrary code anyway, so only load pickles you trust. `convert_model` writes the pickle's
formula unchanged, so a converted model with a formula outside the grammar
needs `trust_formulas=True` when loaded.

## Inspecting a model

`read_model_summary(path)` returns the configuration, each environment's grid
and each encoding model's per-bin summaries (`place_fields`, `mean_rates`,
occupancy, ...) as plain dicts and arrays, without building the detector and
without reading encoding spike samples. It is the fast way to look at a large
clusterless model.

## Reading a file without non_local_detector

The format needs only `zipfile`, `json` and NumPy. This reads the grid and the
place fields of a sorted-spikes model:

<!-- no-nld-reader -->
```python
import json
import zipfile

import numpy as np

with zipfile.ZipFile("decoder.nld") as archive:
    state = json.loads(archive.read("state.json"))

    def array(node):
        return np.load(archive.open(node["$array"]), allow_pickle=False)

    environment = state["environments"][0]
    centers = array(environment["place_bin_centers"])  # (n_bins, n_dims)
    interior = array(environment["is_track_interior"])  # bins in each dimension
    model = state["encoding_models"][0]["model"]
    place_fields = array(model["place_fields"])  # (n_neurons, n_bins), Hz
```
<!-- end-no-nld-reader -->

`state["encoding_models"]` lists each encoding model with its
`environment_name` and `encoding_group`; `config.json` and the manifest
describe the rest.

## Models saved by earlier releases

Earlier releases pickled the detector. `load_model` still loads those files
for inspection. Only load pickles you trust: unpickling can execute arbitrary
code. A pickled model fitted before rates were in Hz (the 0.6 releases)
refuses to decode and asks for a refit.

`convert_model(old_path, new_path)` (in `non_local_detector.model_format`)
rewrites a pickle that can decode in the new format. The fingerprints are
computed at conversion, from the structures the pickle holds. It refuses
models that cannot decode, those fitted before rates were in Hz and those
using the removed `clusterless_kde_log` algorithm, which stay pickles.

## Custom transitions and detector subclasses

A model whose parameters hold a class of your own, such as a custom
continuous transition, or that is an instance of a detector subclass, can be
saved once the class is registered:

```python
from dataclasses import dataclass

from non_local_detector.model_format import register_type


@dataclass
class SpeedTransition:
    environment_name: str = ""
    speed_scale: float = 1.0

    def make_state_transition(self, environments, context=None): ...


register_type(SpeedTransition, "mylab.SpeedTransition")
```

A dataclass is stored as its `__init__` fields; another class passes
`to_state` and `from_state` functions. The name is stored in the file, so keep
it when the class moves. Register the class in every session that saves or
loads such a model, for example when your package is imported: `save_model`
refuses an unregistered class with a message naming it, and `load_model`
refuses a file naming an unregistered type and names the package to import. A
transition whose block is built from the fit data passes
`block_needs_fit_data=True`, so its block is stored. See `register_type` for
the details.

## Versions

`format_version` (currently 1) changes when stored names or meanings change.
A newer release reads every older version through migrations of the stored
data, run before anything is built. A file written in a newer format version
than the installed release reads is refused, naming the version to upgrade
to.

A field added to the format has a default, so files written before it still
load. A release loading a file written by a newer one ignores the names it
does not read, with a `CompatibilityWarning` listing them. A fitted file
missing a name its kind of model always has raises `DataError`, as does an
encoding model missing a key its likelihood algorithm stores, or naming an
algorithm the installed release does not register.
