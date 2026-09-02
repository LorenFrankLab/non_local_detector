# Clusterless mark-model options for high-channel-count probes

## Motivation

The clusterless KDE algorithms treat each spike's waveform features as a mark and
estimate a mark kernel between encoding and decoding spikes. This works well for
tetrodes, where a mark is often only four amplitudes, but it becomes statistically
and computationally difficult when a shank has tens or hundreds of contacts.

`clusterless_diffusion` changes how animal position is smoothed, but deliberately
retains the same Gaussian mark kernel as `clusterless_kde`. It therefore respects
environmental geometry and removes the pairwise position kernel, but it does not
solve the curse of dimensionality in waveform-feature space.

High-density probes provide useful structure: because of contact spacing and the
biophysics of extracellular propagation, correlations are mostly local to a few
nearby contacts. A scalable mark model should encode that locality explicitly.

Two geometries must remain distinct:

- **Environmental geometry** describes the animal's state space and is handled by
  graph diffusion.
- **Probe geometry** describes physical contact locations and should be handled by
  the mark representation or mark-density model.

## Product requirements

The preferred solution must be both easy for working neuroscientists to configure
and efficient enough for long, high-channel-count recordings.

The normal user should provide only data already associated with the recording:

- spike times;
- per-spike amplitudes or waveforms;
- contact coordinates and, for multi-shank probes, shank identifiers.

Probe graphs, neighborhoods, feature scaling, compression dimensions, covariance
regularization, and candidate indexes should be inferred automatically. Advanced
overrides can be available, but users should not have to choose a GMRF precision,
tissue conductivity, spatial-decay exponent, or PCA rank to obtain a good default.

The implementation should target two independent efficiency gains:

1. **Fixed local mark dimension.** Runtime and statistical difficulty should depend
   on a small physical neighborhood, not total probe contact count.
2. **Fewer mark comparisons.** A decode spike should not be compared with every
   encoding spike on a shank when their probe locations are too far apart to share
   a source.

Graph diffusion already removes the pairwise animal-position kernel, but its
low-rank diffusion work remains. Improving the mark model reduces the mark-kernel
part of prediction; it does not eliminate the diffusion matrix operations.

## Proposed user-facing modes

The probe graph should normally be an internal object constructed from contact
coordinates. A minimal interface could be

```python
ClusterlessDecoder(
    clusterless_algorithm="clusterless_diffusion",
    clusterless_algorithm_params={
        "mark_model": "local_amplitude",
        "contact_positions": contact_positions,
        "shank_ids": shank_ids,
    },
)
```

Suggested modes are:

- `local_amplitude`: recommended high-channel-count default; automatically build
  geometry-centered amplitude marks and a spatial candidate index.
- `local_waveform`: extract the same local geometry patch, then apply automatic
  temporal compression fitted on encoding data only.
- `precomputed`: preserve the current expert workflow for source locations,
  template coefficients, or custom features supplied upstream.

The model should provide simple diagnostics: example footprints plotted on contact
geometry, fraction of signal energy captured by the neighborhood, missing-contact
and boundary rates, candidate counts, and retained variance for compressed
waveforms. Warn when the local neighborhood consistently misses substantial signal.

## Option 1: geometry-centered local amplitudes

Assign each spike a peak contact or estimated source coordinate, and retain only
the amplitude footprint within a physical radius of that location. Select contacts
using their physical coordinates, not `peak_contact +/- k` array indices; index
neighborhoods are only a reasonable shortcut for a single regularly spaced column.

An amplitude mark could be

```text
mark = (
    shank_id,
    absolute_source_coordinate,
    log_peak_amplitude,
    normalized_local_amplitude_footprint,
)
```

The footprint can be interpolated onto a canonical peak-relative physical stencil
to obtain a fixed-length vector. Preserve a mask for disabled contacts and for
stencils truncated at probe boundaries. Keeping absolute source position prevents
similar local shapes from distant probe regions from being conflated; keeping peak
amplitude separately prevents footprint normalization from discarding useful scale.

A factored kernel could be

```text
K_mark = K_probe_location * K_amplitude * K_local_footprint
```

Use compact support for `K_probe_location` and index encoding spikes by probe
neighborhood. This reduces both feature dimension and the number of encoding
spikes compared for each decoding spike.

**Advantages:** simple, interpretable, compatible with KDE, and directly motivated
by probe physics.

**Risks:** hard peak assignments can be unstable under noise or drift; canonical
stencils require careful handling of staggered layouts, multiple shanks, and
missing contacts.

## Option 2: local waveform patches with temporal compression

For whole waveforms, a spatial crop alone may still contain hundreds of samples.
After selecting a geometry-local patch, compress each contact's temporal waveform
with a small shared basis, such as PCA, wavelets, or template coefficients.

For `K` nearby contacts and `P` temporal coefficients per contact, the continuous
mark has approximately `K * P` components instead of
`n_contacts * n_waveform_samples`. Sharing the temporal basis across contact
positions makes the transform approximately translation-equivariant along a
repeating probe layout.

**Advantages:** retains waveform shape while remaining a mostly drop-in replacement
for amplitude features.

**Risks:** the basis must be fitted only on encoding data; a fixed basis can become
stale under drift or recording-condition changes.

## Option 3: continuous source localization

Fit the local amplitude footprint to a compact physical source model. The mark may
contain

```text
(shank, axial_position, lateral_position, amplitude, spatial_spread,
 waveform_shape_coefficients)
```

This replaces dozens of raw amplitudes with a handful of interpretable latent
parameters. Continuous location may be more drift-tolerant than absolute peak
contact identity. A hybrid model can retain a few residual footprint coefficients
to capture deviations from the source model.

**Advantages:** strongest dimensionality reduction, geometry-aware, and naturally
supports irregular layouts.

**Risks:** localization uncertainty and biophysical model mismatch become part of
the mark noise; fitting the source model adds preprocessing cost.

## Option 4: banded Gaussian or Gaussian Markov random field

Keep a larger contact vector but constrain its covariance or precision matrix using
probe adjacency. For correlation range `k`, a banded model has roughly `O(C * k)`
parameters rather than `O(C^2)` for `C` contacts. A mixture of banded Gaussian
components can represent multiple local waveform populations without assigning
permanent sorted-unit labels.

**Advantages:** produces a normalized density, uses all contacts, and incorporates
local conditional dependence explicitly.

**Risks:** a single Gaussian is unlikely to capture multimodal clusterless marks;
mixtures require component selection and drift handling. Merely using a banded
distance inside a full-dimensional KDE improves the metric but does not remove the
KDE sample-complexity or pairwise-comparison problem.

This is mathematically related to `sorted_spikes_mrf`, but it operates on a
different graph and variable. `sorted_spikes_mrf` uses the environmental graph
Laplacian to regularize Poisson place fields over animal-position bins. A probe
GMRF would use a contact graph to model amplitudes or waveform residuals across
contacts, usually with a Gaussian or mixture observation model. The existing
Poisson IRLS implementation is therefore not directly reusable, although the
graph-Laplacian and reduced-rank ideas are.

A particularly attractive hybrid is a simple physical source model plus a local
probe-graph residual:

```text
observed footprint = predicted source footprint + local graph residual
residual ~ Normal(0, inverse(alpha * I + beta * L_probe))
```

The source model captures location, scale, and expected attenuation; the GMRF
absorbs locally correlated deviations and noise. The regularization parameters
should be estimated automatically. This is a supporting representation or
component model, not by itself the main efficiency improvement.

## Option 5: contact-local autoregressive density

Factor the full mark density into normalized local conditionals, for example

```text
p(mark) = p(first local block)
          * product_c p(mark_c | nearby preceding contacts)
```

Each conditional can be Gaussian, a small mixture, or a compact learned model.
Unlike multiplying likelihoods from independently fitted overlapping windows, an
autoregressive factorization defines a proper joint density and does not count the
same evidence repeatedly.

**Advantages:** flexible, normalized, and scales approximately linearly with probe
length when the dependency range is fixed.

**Risks:** more invasive than replacing `kde_distance`; the ordering and conditional
model must respect multi-column and multi-shank probe geometry.

## Option 6: local template mixture

Represent each spike with a latent local waveform template plus low-dimensional
residuals. Each template has a source location, local amplitude footprint, waveform
shape, and possibly a drift trajectory. Graph diffusion can maintain a spatial
animal-position field for each soft template component.

This is analogous to soft spike sorting: templates organize the mark density, but
spikes need not receive permanent unit labels.

**Advantages:** naturally represents multimodal marks and can be substantially more
sample-efficient than KDE. At prediction, cost depends on the number of nearby
components rather than the number of local encoding spikes, making this the most
promising scalable second-stage model for long recordings.

**Risks:** template number, initialization, component collapse, and drift make the
fit more complicated.

## Option 7: geometry-aware learned encoder

Use a convolutional or graph-based encoder over `(contact coordinate, waveform)`
pairs. Spatial receptive fields should span only the expected correlation range,
while temporal filters encode waveform shape. The encoder can produce a compact
mark embedding or the parameters of a normalized density model.

**Advantages:** handles raw waveforms, irregular probe geometry, and nonlinear
features.

**Risks:** an arbitrary embedding does not itself define a calibrated mark density.
Training only for reconstruction may preserve irrelevant noise, while training
against decoded position can leak behavioral structure into the observation model.
A probabilistic objective and encoding-only fitting are important.

## Option 8: multiscale marks

Combine a fine local footprint with a few coarse summaries of the larger shank,
such as amplitude center of mass, total energy, axial spread, or regional energy.
This preserves weak nonlocal context without exposing the density estimator to the
entire raw contact vector.

**Advantages:** a practical compromise when distant contacts contain small but useful
information.

**Risks:** introduces additional feature scaling and bandwidth choices; coarse
summaries must not duplicate the same evidence already represented locally.

## Important modeling constraint

Do not treat overlapping contact neighborhoods as independent electrodes and add
all their log likelihoods. That counts one physical spike multiple times and breaks
the marked-point-process interpretation. Each spike should enter the observation
model once, either through one geometry-centered mark or through a properly
normalized joint factorization.

Deterministic compression is acceptable: after defining the transformed feature as
the observed mark, the model estimates the density of that transformed mark rather
than claiming to model the discarded raw waveform coordinates.

## Efficient execution architecture

### Geometry-local candidate indexing

Assign every spike one peak contact or inexpensive source estimate, then bucket it
by shank and physical probe tile. Sort and index encoding spikes once at fit. A
decode spike in tile `b` is compared only with encoding spikes in `b` and adjacent
tiles permitted by a compact-support probe-location kernel.

```text
shank
└── physical probe tile
    ├── encoding source positions
    ├── local marks
    ├── animal-position bins
    └── encoding weights
```

The current pairwise mark-KDE cost is approximately

```text
O(n_decode * n_encoding * mark_dimension).
```

With local indexing it becomes

```text
O(sum_b n_decode[b] * n_encoding[neighbors(b)] * local_mark_dimension).
```

The kernel should have true compact support or a controlled truncation policy so
candidate omission is part of the defined estimator rather than an undocumented
approximation.

For JAX/GPU execution, prefer bucketed dense matrix operations over a generic
irregular sparse matrix until benchmarks show otherwise. Group decode spikes by
probe tile, evaluate one dense local kernel per populated group, then scatter back
to decode-spike order. Bucket similarly sized groups to limit padding and
recompilation.

### Local mixture compression

Candidate indexing still scales with the number of local encoding spikes. For long
recordings, fit a small set of local mark components and associate each component
with a graph-diffused animal-position field. Prediction then evaluates

```text
lambda(animal_position, mark)
    = sum_local_components
        p(mark | component) * component_position_intensity(animal_position)
```

The mark cost becomes approximately

```text
O(n_decode * n_local_components * local_mark_dimension),
```

where `n_local_components` can be much smaller than the local encoding-spike count.
Component count, covariance regularization, and fallback behavior should be chosen
automatically so this does not introduce a new expert-tuning burden.

## Recommended implementation sequence

Start with the least invasive, usable, and efficient geometry-aware model:

1. Assign one peak contact or continuous source estimate per spike.
2. Select contacts within a fixed physical radius on the same shank.
3. Interpolate amplitudes onto a fixed peak-relative stencil.
4. Store absolute source location and log peak amplitude separately.
5. Build a shank/tile index and use a compact-support probe-location kernel to
   restrict comparisons to nearby encoding spikes.
6. Implement the hot path as bucketed local dense kernels and measure it against
   the current full-shank pairwise KDE.
7. Add shared temporal PCA only when raw waveform shape is needed.
8. Add automatically selected local mixture compression when the local encoding
   set is large enough for pairwise comparisons to dominate.
9. Evaluate a probe-GMRF covariance or source-model-plus-GMRF residual within local
   mixture components; keep its parameters automatic and its use optional.

This can reuse the existing graph-diffusion position model while replacing only the
mark representation and mark-kernel evaluation. It also provides a straightforward
baseline against which the more structured density models can be measured. The
probe graph supplies geometry, locality, and weak biophysical regularization, but
it should remain mostly invisible in the standard user workflow.

Evaluate options using held-out mark log likelihood, decoding accuracy and
calibration, sample efficiency versus contact count, sensitivity to drift and
missing contacts, and fit/predict runtime. Neighborhood radius, geometry handling,
feature normalization, temporal compression, candidate count, and component count
should be explicit ablations. Benchmark the mark-kernel and diffusion portions
separately so improvements to one are not hidden by the other.
