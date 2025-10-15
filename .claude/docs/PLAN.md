# Implementation Plan for Code Review Recommendations

This plan addresses the recommendations from REVIEW.md, prioritized by impact and implementation effort.

## Phase 1: Quick Wins (1-2 weeks)

### 1.1 Custom Exception Classes

**Priority: High | Effort: Low**

Create a dedicated exceptions module to improve error handling and user experience.

```python
# src/non_local_detector/exceptions.py
class NonLocalDetectorError(Exception):
    """Base exception for non_local_detector package."""
    pass

class ValidationError(NonLocalDetectorError):
    """Raised when input validation fails."""
    pass

class FittingError(NonLocalDetectorError):
    """Raised when model fitting fails."""
    pass

class ConfigurationError(NonLocalDetectorError):
    """Raised when configuration is invalid."""
    pass
```

**Implementation:**

- Create `src/non_local_detector/exceptions.py`
- Replace generic `ValueError`/`RuntimeError` with specific exceptions
- Add helpful error messages with suggested fixes
- Focus on high-visibility areas: model fitting, parameter validation

### 1.2 Constants Configuration Module

**Priority: High | Effort: Low**

Extract hardcoded constants to a centralized configuration.

```python
# src/non_local_detector/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class NumericalConfig:
    """Numerical computation constants."""
    EPS: float = 1e-15
    LOG_EPS: float = -34.54  # log(1e-15)
    CONVERGENCE_TOLERANCE: float = 1e-4
    MAX_ITERATIONS: int = 1000

@dataclass(frozen=True)
class DefaultAlgorithmParams:
    """Default parameters for algorithms."""
    CLUSTERLESS_WAVEFORM_STD: float = 24.0
    CLUSTERLESS_POSITION_STD: float = 6.0
    SORTED_SPIKES_POSITION_STD: float = 6.0
    DEFAULT_BLOCK_SIZE: int = 10_000
    DEFAULT_SAMPLING_FREQUENCY: float = 500.0
    DEFAULT_NO_SPIKE_RATE: float = 1e-10

# Global instances
NUMERICAL = NumericalConfig()
DEFAULTS = DefaultAlgorithmParams()
```

**Implementation:**

- Create configuration module
- Update imports in `core.py`, `common.py`, and `base.py`
- Make constants easily configurable for advanced users

### 1.3 Enhanced Documentation Examples

**Priority: Medium | Effort: Low**

Add practical usage examples to complex classes.

**Target classes:**

- `KDEModel` (already good, enhance further)
- `NonLocalClusterlessDetector`
- `Environment`
- Core filtering functions

## Phase 2: Structural Improvements (2-4 weeks)

### 2.1 Base Class Refactoring

**Priority: High | Effort: Medium**

Address the 2,609-line `base.py` file by extracting reusable components.

**Step 1: Parameter Validation Mixins**

```python
# src/non_local_detector/mixins/validation.py
class ParameterValidationMixin:
    def _validate_array_shape(self, arr, expected_shape, name):
        """Centralized array shape validation."""
        pass

    def _validate_positive_scalar(self, value, name):
        """Centralized positive scalar validation."""
        pass

    def _validate_probability_distribution(self, dist, name):
        """Centralized probability distribution validation."""
        pass
```

**Step 2: Fitting Strategy Pattern**

```python
# src/non_local_detector/strategies/fitting.py
class FittingStrategy:
    """Abstract base class for fitting strategies."""
    def fit_encoding_models(self, ...): pass
    def fit_transitions(self, ...): pass
    def validate_inputs(self, ...): pass

class ClusterlessFittingStrategy(FittingStrategy):
    """Fitting strategy for clusterless data."""
    pass

class SortedSpikesFittingStrategy(FittingStrategy):
    """Fitting strategy for sorted spikes data."""
    pass
```

### 2.2 Environment Class Decomposition

**Priority: Medium | Effort: Medium**

Break down the 1,301-line Environment class using composition.

```python
# src/non_local_detector/environment/
#   __init__.py
#   core.py          # Core Environment class
#   track.py         # Track-related functionality
#   geometry.py      # Geometric computations
#   validation.py    # Input validation
#   builders.py      # Environment builders/factories
```

**Benefits:**

- Easier testing of individual components
- Clearer separation of concerns
- More maintainable codebase

### 2.3 Abstract Base Classes

**Priority: Medium | Effort: Medium**

Create abstract base classes to reduce duplication between Detector/Classifier classes.

```python
# src/non_local_detector/abc/
class AbstractStatefulModel(ABC):
    """Common interface for stateful models."""

    @abstractmethod
    def fit(self, ...): pass

    @abstractmethod
    def predict_state_probabilities(self, ...): pass

    def _validate_fit_inputs(self, ...):
        """Common input validation."""
        pass

class AbstractDetector(AbstractStatefulModel):
    """Base for detection models."""
    pass

class AbstractClassifier(AbstractStatefulModel):
    """Base for classification models."""
    pass
```

## Phase 3: Advanced Improvements (4-8 weeks)

### 3.1 Property-Based Testing

**Priority: Medium | Effort: Medium**

Add hypothesis-based tests for numerical algorithms.

```python
# tests/property_based/test_core_properties.py
from hypothesis import given, strategies as st

@given(
    initial_dist=st.arrays(dtype=float, shape=st.integers(2, 20)),
    transition_matrix=st.arrays(...),
    log_likelihoods=st.arrays(...)
)
def test_filter_output_properties(initial_dist, transition_matrix, log_likelihoods):
    """Test mathematical properties of filtering algorithm."""
    # Normalize inputs
    initial_dist = initial_dist / initial_dist.sum()

    # Run filtering
    result = filter(initial_dist, transition_matrix, log_likelihoods)

    # Check properties
    assert all_probabilities_sum_to_one(result)
    assert all_values_non_negative(result)
    assert marginal_likelihood_is_finite(result)
```

### 3.2 Plugin Architecture Enhancement

**Priority: Low | Effort: High**

Formalize the plugin system for custom likelihood models.

```python
# src/non_local_detector/plugins/
class LikelihoodPlugin:
    """Base class for likelihood plugins."""

    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    def fit(self, ...): pass

    @abstractmethod
    def predict_log_likelihood(self, ...): pass

    def validate_requirements(self): pass

# Registration system
class PluginRegistry:
    def register_likelihood(self, plugin: LikelihoodPlugin): pass
    def get_likelihood(self, name: str): pass
```

### 3.3 Performance Benchmarking Suite

**Priority: Low | Effort: High**

Create comprehensive benchmarks to track performance improvements.

```python
# benchmarks/
#   benchmark_core.py      # Core algorithm benchmarks
#   benchmark_likelihoods.py  # Likelihood computation benchmarks
#   benchmark_memory.py    # Memory usage benchmarks
#   benchmark_scaling.py   # Scaling behavior benchmarks
```

## Phase 4: Long-term Enhancements (Future)

### 4.1 Async I/O Integration

**Priority: Low | Effort: High**

For large dataset loading and processing.

### 4.2 Configuration Management System

**Priority: Low | Effort: Medium**

YAML/TOML-based configuration for complex experiments.

### 4.3 Distributed Computing Support

**Priority: Low | Effort: Very High**

JAX pmap/sharding for multi-GPU processing.

## Implementation Strategy

### Development Process

1. **Create feature branches** for each phase
2. **Maintain backward compatibility** during transitions
3. **Add deprecation warnings** before removing old APIs
4. **Comprehensive testing** for each change
5. **Documentation updates** alongside code changes

### Risk Mitigation

- **Incremental rollout**: Implement changes in small, testable chunks
- **Feature flags**: Use configuration to toggle new vs old implementations
- **Extensive testing**: Both unit tests and integration tests for each change
- **Performance monitoring**: Ensure changes don't regress performance

### Success Metrics

- Reduced complexity in large classes (target: <500 lines per class)
- Improved error messages (user feedback)
- Better test coverage (target: >90%)
- Maintained or improved performance
- Cleaner public API surface

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| 1 | 1-2 weeks | Exceptions, constants, documentation |
| 2 | 2-4 weeks | Refactored base classes, decomposed Environment |
| 3 | 4-8 weeks | Property testing, plugin system |
| 4 | Future | Async I/O, distributed computing |

This plan maintains the package's excellent foundation while systematically addressing the identified improvement areas.
