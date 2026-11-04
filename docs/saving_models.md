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

_, position, spike_times, time, *_ = make_simulated_data(n_neurons=5)
time, position = time[:3000], position[:3000]
spike_times = [spikes[spikes <= time[-1]] for spikes in spike_times]
decoder = SortedSpikesDecoder().fit(time, position, spike_times)

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
earlier releases wrote (below).

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

patsy evaluates a covariate transition's formula as Python, so a stored
formula is checked against the formula grammar before patsy sees it, and the
covariates are stored only as numeric or string arrays (a categorical as its
codes and categories). A file whose formula is outside
the grammar loads for inspection only, as such a pickle does: `load_model`
warns and does not rebuild the design (or the training-time
`discrete_state_transitions_`), and decoding or saving it again is refused.

Rebuilding repeats the fit's arithmetic, so on the same machine a reloaded
model decodes exactly as the original. A newer NumPy, patsy or
non_local_detector could rebuild these structures differently, so the file
also stores small fingerprints of them (weighted sums and sums of squares).
Before a loaded model first decodes, the rebuilt structures are compared with
the fingerprints at a relative tolerance of 1e-10, which accepts last-bit
differences between platforms. If they differ, decoding raises
`ValidationError` asking for a refit; loading and inspecting the model still
work.

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
