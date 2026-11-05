# Warnings

Every warning the package raises itself is an instance of
`NonLocalDetectorWarning`, so you can silence or escalate the
package's warnings without matching message text. Warnings from other libraries
(JAX, scikit-learn, ...) are not instances of it.

| Category | Meaning | Examples |
| --- | --- | --- |
| `DataQualityWarning` | The data are sparse, missing or off the track. | Zero training exposure, an electrode with no encoding spikes, position that moves farther between samples than `position_std`, positions snapped to the nearest bin, decode bins outside the tracked period, spike times far outside the fit or decode time range (ms instead of s, or another clock), or at fit almost none inside the position span with spikes reaching far past it (ms instead of s on a clock starting near 0). |
| `NumericalWarning` | A fit did not converge or hit a degenerate case. | EM not converged or its objective decreasing, a singular Poisson regression, discarded Gaussian mixture restarts, a NaN kernel density. |
| `ParameterWarning` | A setting was adjusted or is likely to give a poor or costly fit. | `position_std` below the bin spacing, a basis rank or mixture size reduced to fit the data, dense EM that may not fit in memory. |
| `CompatibilityWarning` | A deprecated argument or legacy saved form was used. Also a `DeprecationWarning`. | Unset fitted attributes passed to `Environment()`. |

All of them are importable from `non_local_detector` and
`non_local_detector.exceptions`.

## Filtering

```python
import warnings

from non_local_detector import (
    DataQualityWarning,
    NonLocalDetectorWarning,
    NumericalWarning,
)

# Hide the package's non-convergence warnings.
warnings.filterwarnings("ignore", category=NumericalWarning)

# Turn data-quality problems into errors.
warnings.filterwarnings("error", category=DataQualityWarning)

# Hide everything the package warns about.
warnings.filterwarnings("ignore", category=NonLocalDetectorWarning)
```

In pytest, use `pytest.warns(NumericalWarning)` or
`pytest.mark.filterwarnings("error::non_local_detector.NumericalWarning")`.
Existing filters on `UserWarning` (or `DeprecationWarning` for
`CompatibilityWarning`) keep working because each class subclasses the category
it replaced. The base class `NonLocalDetectorWarning` is a plain `Warning`, as
in astropy: `DataQualityWarning`, `NumericalWarning` and `ParameterWarning` are
also `UserWarning`s, and `CompatibilityWarning` is a `DeprecationWarning` only,
so a filter on `UserWarning` no longer catches deprecations. Default visibility
is unchanged.
