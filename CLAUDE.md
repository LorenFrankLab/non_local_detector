# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`non_local_detector` is a Python package for decoding non-local neural activity from electrophysiological data. It uses Bayesian inference with Hidden Markov Models (HMMs) and various likelihood models to detect spatial replay events and decode position from neural spike data.

## Development Commands

### Installation

- **Development setup**: `mamba create env -f environment.yml` (CPU) or `mamba create env -f environment_gpu.yml` (GPU)
- **From pip**: `pip install non_local_detector` (CPU) or `pip install non_local_detector[gpu]` (GPU with CUDA)

### Testing

- **Run tests**: `pytest` (uses pytest framework defined in pyproject.toml)
- **Run single test**: `pytest src/non_local_detector/tests/test_version_import.py`
- **Test with coverage**: `pytest --cov`

### Code Quality

- **Lint and fix**: `ruff check --fix src/`
- **Format code**: `ruff format src/` or `black src/`
- **Type checking**: `mypy src/non_local_detector/`
- **Check all quality tools**: `ruff check src/ && ruff format --check src/ && black --check src/ && mypy src/non_local_detector/`

### Building

- **Build package**: Uses hatchling build system (defined in pyproject.toml)
- **Version management**: Automatic versioning via hatch-vcs from git tags

## Architecture Overview

### Core Components

1. **Likelihood Models** (`src/non_local_detector/likelihoods/`):
   - `sorted_spikes_glm`: GLM-based likelihood for sorted spike data
   - `sorted_spikes_kde`: KDE-based likelihood for sorted spike data
   - `clusterless_kde`: KDE-based likelihood for clusterless (continuous) spike data
   - `clusterless_gmm`: GMM-based likelihood for clusterless spike data
   - Algorithms are registered in `_SORTED_SPIKES_ALGORITHMS` and `_CLUSTERLESS_ALGORITHMS` dictionaries

2. **State Transition Models**:
   - **Continuous**: `RandomWalk`, `EmpiricalMovement`, `Identity`, `Uniform` transitions
   - **Discrete**: Stationary/non-stationary diagonal and custom transition matrices

3. **Decoder Models** (`src/non_local_detector/models/`):
   - `NonLocalClusterlessDetector` / `NonLocalSortedSpikesDetector`: Main replay detection models
   - `ContFragClusterlessClassifier` / `ContFragSortedSpikesClassifier`: Fragmented decoding
   - `ClusterlessDecoder` / `SortedSpikesDecoder`: Basic position decoding
   - `MultiEnvironmentClusterlessClassifier`: Multi-environment decoding

4. **Core Algorithms** (`src/non_local_detector/core.py`):
   - Forward-backward algorithm implementation adapted from dynamax
   - Chunked filtering for large datasets
   - Viterbi algorithm for most likely sequences
   - JAX-based implementations for GPU acceleration

### Key Dependencies

- **JAX**: Primary computational backend for GPU acceleration
- **NumPy/SciPy**: Numerical computing
- **pandas/xarray**: Data handling
- **scikit-learn**: Machine learning utilities
- **track_linearization**: Spatial trajectory processing
- **matplotlib/seaborn**: Visualization

### Data Flow

1. **Encoding Models**: Fit likelihood models to training data (position + spikes)
2. **State Transitions**: Define movement models and discrete state transitions
3. **Decoding**: Apply HMM filtering/smoothing to decode position from test data
4. **Analysis**: Extract replay events, compute metrics, visualize results

### Testing Structure

- Minimal test suite in `src/non_local_detector/tests/`
- Extensive notebook-based testing and validation in `notebooks/`
- Test notebooks cover different likelihood models, simulation, and real data analysis

### Key Files to Understand

- `src/non_local_detector/core.py`: Core HMM algorithms
- `src/non_local_detector/models/base.py`: Base classes for all models
- `src/non_local_detector/likelihoods/__init__.py`: Likelihood algorithm registry
- `src/non_local_detector/__init__.py`: Public API exports

## Development Notes

- The codebase uses JAX for numerical computations, enabling GPU acceleration
- Models follow scikit-learn estimator patterns (fit/predict interface)
- Extensive use of xarray for labeled multidimensional data
- The package handles both sorted spike data (discrete units) and clusterless data (continuous features)
- Environment configuration supports both CPU and GPU installations via different dependency sets
