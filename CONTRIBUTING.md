# Contributing

Contributions of all kinds are welcome: bug reports, documentation fixes, tests, and code. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Issues

Search [existing issues](https://github.com/LorenFrankLab/non_local_detector/issues) first, then open a new one using the bug report or feature request form. For a bug, include the package, Python and JAX versions, whether you ran on CPU or GPU, and a minimal example, ideally on simulated data (`non_local_detector.simulate`). Please do not attach private recordings.

## Development setup

Fork and clone the repository, then install it with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/<your-username>/non_local_detector.git
cd non_local_detector
uv sync
uv run pre-commit install
```

`uv sync` installs the package in editable mode with the development dependencies pinned in `uv.lock`, so your tools match CI. Run everything through `uv run`. A conda environment is an alternative: create it from `environment.yml` (or `environment_gpu.yml` for a CUDA GPU) and run `pip install -e .`, but the locked versions are what CI checks.

## Running the tests

```bash
uv run pytest tests/test_version_import.py          # one file
uv run pytest tests/core -m "not slow"              # one directory, skipping slow tests
uv run pytest -k "random_walk"                      # by name
uv run pytest -m unit                               # by marker
```

The full suite takes a long time; during development run the tests for the code you touched, and CI runs the rest on every pull request. Markers (declared in `pyproject.toml`, and `--strict-markers` is on):

| Marker | Meaning |
| --- | --- |
| `unit` | Unit tests |
| `integration` | End-to-end and multi-component tests |
| `property` | Property-based tests (Hypothesis) of mathematical invariants |
| `slow` | Slow tests; deselect with `-m "not slow"` |

Every warning is an error in the test suite, unless a test expects it (`pytest.warns`) or marks it incidental with a reason (`pytest.mark.filterwarnings`). Tests should be deterministic (fixed seeds) and use the shared fixtures in `tests/conftest.py`. New tests should say what they verify and fail when the behavior breaks.

## Code style

- [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and strict [mypy](https://mypy.readthedocs.io/) on `src/non_local_detector/`. Add type annotations to every function.
- Docstrings are [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style and give array shapes and units, e.g. `(n_time, n_position_dims)` and seconds or Hz.
- Vectorize with NumPy or JAX rather than looping over arrays in Python.

<!-- --8<-- [start:array-annotations] -->
### Array annotations

Follow JAX's guidance (`jax.typing`, JEP 12049): lenient inputs, concrete outputs.

- NumPy-side code: inputs are `numpy.typing.ArrayLike` when the function converts them (`np.asarray`), outputs are `NDArray[<dtype>]` or `np.ndarray`. If a function needs an ndarray (`.shape`, `.ndim`, in-place use), annotate `np.ndarray`.
- JAX-side code: inputs are `jax.typing.ArrayLike`, outputs are `jax.Array`. Never write `jnp.ndarray`, which is only an alias of `jax.Array`. Do not accept lists or tuples in place of arrays in jitted code.
  If the body calls array methods (`.dtype`, `.sum`, indexing), which the scalar members of `jax.typing.ArrayLike` lack, annotate inputs `jax.Array | np.ndarray` instead (as `core.py` does with its `ArrayLike` alias).
- Dtypes are `DTypeLike` (`numpy.typing` or `jax.typing`, matching the side).
- Keep a list type where the API takes a list of per-neuron or per-electrode arrays. For an input the function converts, write `Sequence[ArrayLike]`: `list` is invariant, so a `list[ArrayLike]` parameter rejects a caller's `list[np.ndarray]`.
- Shapes go in the NumPy-style docstring, e.g. `(n_time, n_position_bins)`, not in the annotation.
<!-- --8<-- [end:array-annotations] -->

Run all of the checks, the same ones CI runs, with:

```bash
uv run pre-commit run --all-files
```

or individually: `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy`.

## Numerical changes

This package produces scientific results, so a change that alters numbers needs more care than one that does not.

- A refactor must leave outputs unchanged (to floating-point noise, about 1e-14). Compare outputs before and after.
- A deliberate algorithm change should stay within about 1e-10 of the old results, or come with a justification for the larger difference.
- After touching the HMM algorithms, likelihoods, transitions or initial conditions, check the invariants: posteriors sum to one, transition matrices are row-stochastic, log-probabilities are finite, covariances are positive semi-definite, and likelihoods are non-negative. The property tests under `tests/properties/` check these.
- The golden regression tests (`tests/test_golden_regression.py`, pickled data in `tests/golden_data/`) pin known-good outputs. Do not regenerate their data to make a failing test pass. If a difference is intended, explain in the pull request what changed, by how much, why it is correct, and which invariants still hold, and a maintainer will approve updating the data.
- Do not loosen tolerances or convergence criteria without the same explanation.
- After changing JAX code, check for unexpected recompilation, NaN/Inf in outputs, and unchanged shapes. If you changed GPU-specific paths, run the relevant tests on a CUDA machine.

## Adding a likelihood algorithm

A likelihood algorithm is a fit and a predict function registered in `_SORTED_SPIKES_ALGORITHMS` or `_CLUSTERLESS_ALGORITHMS` of `non_local_detector.likelihoods`; a detector can only select a registered name.

- The fit returns its encoding model as a plain dict. Describe it with a `TypedDict` derived from `SortedSpikesEncodingModel` or `ClusterlessEncodingModel` (`non_local_detector.likelihoods.encoding_model`) and annotate the fit with it, so mypy checks the returned keys and values. Set `algorithm` to the registered name.
- The detector calls `predict(position_time, position, *spike_data, **encoding_model, ...)`. After the data arguments every fit and predict parameter is keyword-only, and predict declares every key of the encoding model (no `**kwargs`), including fit diagnostics it does not use.
- Annotate the data arguments as the protocols `SortedSpikesFit` / `ClusterlessFit` and `SortedSpikesPredict` / `ClusterlessPredict` of `non_local_detector.likelihoods` type them (the records' `fit` and `predict` fields have these types, so mypy checks the detector's calls): `position_time`, `position` and `weights` are `ArrayLike`, and `spike_times` and `spike_waveform_features` `Sequence[ArrayLike]`. The detector passes NumPy arrays; convert at entry with `as_array` and `as_arrays` from `likelihoods.common`, which keep a JAX array as is (a sharded input must not be gathered to the host) and convert any other array-like with `np.asarray`. Predict must accept `position_time=None` and `position=None` when `is_local` is False, because a detector whose states read no position (no Local state, no non-local position penalty) passes None for both; read them on the local path through `require_position`, which raises `ValidationError` for None.
- Register a `LikelihoodAlgorithm(fit, predict, EncodingModelType)`, or for clusterless data a `ClusterlessLikelihoodAlgorithm` with its `mark_reference_scale` (a function of the fit parameters whose defaults equal the fit's, which the contract tests check) and, if model checking supports it, the name of its intensity class in `model_checking._clusterless_models` (`electrode_intensity`).
- Name in `execution_settings` the fit parameters that only control how predict runs, such as block sizes, memory budgets and `disable_progress_bar`. The fit records them in the encoding model, and a detector passes predict the value its current algorithm params set, so they can change without refitting.
- `tests/likelihoods/test_algorithm_contract.py` checks every registered algorithm against these rules. Add the new name to its `_CHANGED_FIT_PARAMETER` (a fit parameter that is not an execution setting, which prediction must ignore until the next fit; the test raises `KeyError` without it) and, if it has execution settings beyond `disable_progress_bar`, to `_CHANGED_SETTINGS` and `_INVALID_SETTINGS`.
- The detector-level tests fit every registered algorithm through `tests/models/_helpers.py`'s `_decoder` on small fixtures; add parameters there if the defaults do not fit them (as the sorted GLM's knot spacing and the clusterless GMM's component counts do).
- List each new class that its encoding model holds (a fitted density model, a basis) in `model_format/_builtins.py` with its fields, or `save_model` refuses the fitted detector (see "Saved-model format").
- Add the encoding model's keys to `_STORED_KEYS` in `tests/model_format/test_encoding_model_format.py`: they are stored in saved models, and `load_model` refuses an encoding model whose keys differ from its type's. Changing a registered algorithm's keys needs a format version (see "Saved-model format").
- Add the algorithm to the user-facing lists, the likelihood models in `README.md` ("Architecture") and the likelihood table in `docs/getting-started/core-concepts.md`; the contract tests check that both name every registered algorithm.

## Saved-model format

`save_model` writes the format of `non_local_detector.model_format`, and every later release must read the files earlier ones wrote. The stored names are the format: the file names of the detector's fitted state (`_STORED` in `model_format/_detector.py`), the fields each saved class stores (`model_format/_builtins.py`) and the fixed `nld.` type names listed there.

- Renaming or removing a stored name, or changing what a stored value means, needs a `FORMAT_VERSION` bump and an entry in `MIGRATIONS` that rewrites the stored documents and arrays of the previous version.
- Reading is strict: a name or field the reader does not know raises `DataError` (the schema is in `model_format/_detector.py`: `_MANIFEST_NAMES`, `_FITTED_STATE`, `_STORED`, `_OPTIONAL_STORED`, `_ENTRY_FIELDS`; each type's fields in `model_format/_builtins.py`). So adding any stored name or field also bumps `FORMAT_VERSION`, so that earlier releases ask for an upgrade instead. The reader supplies the new field's value for older files: give it in the type's `"defaults"` option in `model_format/_builtins.py`, or add it in the `MIGRATIONS` entry. A field missing without a default raises `DataError`.
- Loading rebuilds environment grids, continuous transitions and covariate designs and requires them to match the saved fingerprints to a relative 1e-6. A change to that rebuilding code larger than 1e-6 relative locks out every saved model: it needs a format migration, or a changelog note that such models must be refit.
- A new class that a detector's parameters or fitted state can hold must be listed in `model_format/_builtins.py` with its fields, or `save_model` refuses it. `tests/model_format` checks the listed fields against each dataclass.
- Version 1 freezes at the first release that ships the format. Until then its schema may change without a migration. When it freezes, add one small golden file per detector family and format version to `tests/compat_data/`, with a test that loads and decodes each, as for the legacy pickles there.

## Changelog

Add an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for every user-visible change (new features, behavior changes, fixes, deprecations). Say what changed for users, and mark changes that alter results.

## Pull requests

1. Create a branch from `main`.
2. Make small, self-contained commits, one logical change each.
3. Add or update tests and docstrings, and update documentation and the changelog.
   Preview the documentation site with `uv sync --group docs && uv run mkdocs serve`.
4. Run `uv run pre-commit run --all-files` and the relevant tests.
5. Open a pull request and fill in the template. Link the issue it fixes. CI runs the lint, type, test (several Python versions and dependency ranges) and packaging jobs, and all must pass before merging.
