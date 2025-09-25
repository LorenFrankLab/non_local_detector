# Non-Local Detector

[![Tests](https://github.com/LorenFrankLab/non_local_detector/actions/workflows/test_package_build.yml/badge.svg)](https://github.com/LorenFrankLab/non_local_detector/actions/workflows/test_package_build.yml)
[![PyPI version](https://badge.fury.io/py/non-local-detector.svg)](https://badge.fury.io/py/non-local-detector)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for decoding non-local neural activity from electrophysiological data using Bayesian inference with Hidden Markov Models (HMMs). This package specializes in detecting spatial replay events and decoding position from neural spike data.

## 🧠 Overview

The `non_local_detector` package provides state-of-the-art algorithms for:

- **Replay Detection**: Identify when the brain is replaying past experiences during rest periods
- **Position Decoding**: Decode spatial position from neural activity in real-time
- **Multi-Environment Analysis**: Handle complex experimental paradigms with multiple spatial contexts
- **GPU Acceleration**: Leverage JAX for high-performance computing on GPUs

### Key Features

- 🔬 **Multiple Likelihood Models**: GLM, KDE, and GMM-based approaches for both sorted and clusterless spike data
- ⚡ **JAX-Powered**: GPU-accelerated computations for large-scale analyses
- 🎯 **Scikit-learn Compatible**: Familiar fit/predict interface for easy integration
- 📊 **Rich Visualizations**: Built-in plotting functions for results analysis
- 🧪 **Extensive Testing**: Comprehensive test suite and validation notebooks

## 🚀 Installation

### Quick Install (CPU)

```bash
pip install non_local_detector
```

### GPU Installation (Recommended for Large Datasets)

For CUDA-enabled GPU acceleration:

```bash
pip install non_local_detector[gpu]
```

Or using conda/mamba:

```bash
mamba install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
mamba install non_local_detector -c edeno
```

### Development Installation

For development and contributing:

```bash
git clone https://github.com/LorenFrankLab/non_local_detector.git
cd non_local_detector
mamba env create -f environment.yml  # or environment_gpu.yml for GPU
conda activate non_local_detector
pip install -e .[dev]
```

## 📖 Quick Start

### Basic Replay Detection

```python
import numpy as np
from non_local_detector import NonLocalClusterlessDetector

# Initialize detector
detector = NonLocalClusterlessDetector(
    environments=[environment],  # Your spatial environment
    observation_model="clusterless_kde",
    transition_type="random_walk"
)

# Fit the model
detector.fit(
    position=training_position,
    spike_times=training_spike_times,  # List of arrays, one per electrode
    spike_waveform_features=training_waveform_features,  # List of arrays, one per electrode
    time=training_time
)

# Detect replay events
results = detector.predict(
    spike_times=test_spike_times,
    spike_waveform_features=test_waveform_features,
    time=test_time
)

# Analyze results
state_probability = results.acausal_state_probabilities
decoded_position = results.acausal_posterior.unstack("state_bins").sum("position")
```

### Working with Sorted Spikes

```python
from non_local_detector import NonLocalSortedSpikesDetector

# For traditional spike-sorted data
detector = NonLocalSortedSpikesDetector(
    environments=[environment],
    observation_model="sorted_spikes_kde"
)

detector.fit(
    position=position,
    spike_times=spike_times,  # List of arrays, one per neuron
    time=time
)
```

## 🏗️ Architecture

### Core Components

1. **Likelihood Models** (`non_local_detector.likelihoods`)
   - `sorted_spikes_glm`: GLM-based likelihood for sorted spike data
   - `sorted_spikes_kde`: KDE-based likelihood for sorted spike data
   - `clusterless_kde`: KDE-based likelihood for clusterless data
   - `clusterless_gmm`: GMM-based likelihood for clusterless data

2. **State Transition Models**
   - **Continuous**: `RandomWalk`, `EmpiricalMovement`, `Identity`, `Uniform`
   - **Discrete**: Stationary/non-stationary diagonal and custom transitions

3. **Decoder Models** (`non_local_detector.models`)
   - `NonLocalDetector`: Main replay detection models
   - `ContFragClassifier`: Fragmented decoding for interrupted sequences
   - `ClusterlessDecoder`/`SortedSpikesDecoder`: Basic position decoding
   - `MultiEnvironmentClassifier`: Multi-environment decoding

## 🛠️ Development

### Quality Standards

This project maintains high code quality standards:

- **Linting**: `ruff check src/` (zero issues required)
- **Formatting**: `ruff format src/` and `black src/`
- **Type Checking**: `mypy src/non_local_detector/` (gradual adoption)
- **Testing**: `pytest` with coverage reporting

### Development Workflow

1. **Setup Development Environment**

   ```bash
   mamba env create -f environment.yml
   conda activate non_local_detector
   pip install -e .[dev]
   ```

2. **Run Quality Checks**

   ```bash
   # Full quality check (required before committing)
   ruff check src/ && ruff format --check src/ && black --check src/ && pytest

   # Auto-fix linting issues
   ruff check --fix src/
   ruff format src/
   ```

3. **Testing**

   ```bash
   # Run all tests
   pytest

   # With coverage
   pytest --cov=non_local_detector --cov-report=html

   # Single test file
   pytest src/non_local_detector/tests/test_version_import.py
   ```

4. **Building**

   ```bash
   # Build package
   python -m build

   # Check distribution
   twine check dist/*
   ```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Quality Gate**: All PRs must pass linting, formatting, and type checking
- **Multi-Version Testing**: Tests run on Python 3.10, 3.11, and 3.12
- **Package Building**: Automatic wheel and source distribution building
- **Coverage Reporting**: Integration with Codecov

## 📚 Documentation

- **API Reference**: [Coming Soon]
- **Tutorials**: See `notebooks/` directory for example analyses
- **Contributing Guide**: See development section above
- **Issue Tracker**: [GitHub Issues](https://github.com/LorenFrankLab/non_local_detector/issues)

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run quality checks locally (see Development Workflow)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Quality Requirements

All contributions must:

- Pass all quality checks (`ruff`, `black`, `mypy`)
- Include tests for new functionality
- Follow the existing code style and patterns
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Author**: Eric Denovellis
- **Institution**: Loren Frank Lab, UCSF
- **Funding**: [Add funding sources if applicable]

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/LorenFrankLab/non_local_detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LorenFrankLab/non_local_detector/discussions)

## 🔗 Related Projects

- [track_linearization](https://github.com/LorenFrankLab/track_linearization): Spatial trajectory processing
- [dynamax](https://github.com/probml/dynamax): JAX-based probabilistic state space models
