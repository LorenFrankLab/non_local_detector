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
- Keep `list[...]` where the API takes a list of per-neuron or per-electrode arrays.
- Shapes go in the NumPy-style docstring, e.g. `(n_time, n_position_bins)`, not in the annotation.
<!-- --8<-- [end:array-annotations] -->

Run all of the checks, the same ones CI runs, with:

```bash
uv run pre-commit run --all-files
```

or individually: `uv run ruff check src/`, `uv run ruff format src/`, `uv run mypy`.

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
- Register a `LikelihoodAlgorithm(fit, predict, EncodingModelType)`, or for clusterless data a `ClusterlessLikelihoodAlgorithm` with its `mark_reference_scale` and, if model checking supports it, its `electrode_intensity`.
- `tests/likelihoods/test_algorithm_contract.py` checks every registered algorithm against these rules.

## Changelog

Add an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for every user-visible change (new features, behavior changes, fixes, deprecations). Say what changed for users, and mark changes that alter results.

## Pull requests

1. Create a branch from `main`.
2. Make small, self-contained commits, one logical change each.
3. Add or update tests and docstrings, and update documentation and the changelog.
   Preview the documentation site with `uv sync --group docs && uv run mkdocs serve`.
4. Run `uv run pre-commit run --all-files` and the relevant tests.
5. Open a pull request and fill in the template. Link the issue it fixes. CI runs the lint, type, test (several Python versions and dependency ranges) and packaging jobs, and all must pass before merging.
