# Warnings

Every warning the package raises itself is an instance of
`NonLocalDetectorWarning`, so you can silence or escalate the
package's warnings without matching message text. Warnings from other libraries
(JAX, scikit-learn, ...) are not instances of it.

| Category | Meaning | Examples |
| --- | --- | --- |
| `DataQualityWarning` | The data are sparse, missing or off the track. | Zero training exposure, an electrode with no encoding spikes, position that moves farther between samples than `position_std`, positions snapped to the nearest bin, more than 1% of positions over a bin outside the fitted grid (larger units than the grid, a `position_range` that does not match the data, or decoding where the fitting positions never went), decode bins outside the tracked period, decode bins reaching a gap in `position_time`, every decode bin missing because of position or the intervals, `encoding_time_range` over one sample interval past the position samples, spike times far outside the fit or decode time range (ms instead of s, or another clock), spike times whose dtype cannot resolve a decode bin (float32 on a Unix-seconds clock), at fit no spike inside the position span, almost none inside it with spikes reaching far past it (spikes in ms instead of s on a clock starting near 0), or spikes spanning under 1% of it (`position_time` in ms); a saved model holding names a newer version wrote, which loading ignores; in sorted-spike time rescaling, spikes at the first sample of a run of sample times (left out: they lie before the run), or more than 5% of the bins with a spike holding more than one (the count per bin is not tested). |
| `NumericalWarning` | A fit did not converge or hit a degenerate case. | EM not converged or its objective decreasing, a singular Poisson regression, discarded Gaussian mixture restarts, a NaN kernel density. |
| `ParameterWarning` | A setting was adjusted or is likely to give a poor or costly fit. | `position_std` below the bin spacing, a basis rank or mixture size reduced to fit the data, dense EM that may not fit in memory, decoding a detector whose state-defining parameters changed after fitting, a saved model's transition formula outside the formula grammar, which loading leaves unevaluated. |
| `CompatibilityWarning` | A deprecated argument or legacy saved form was used. Also a `DeprecationWarning`. | Unset fitted attributes passed to `Environment()`. |

All of them are importable from `non_local_detector` and
`non_local_detector.exceptions`. A warning about a method's data fires once per
public call, however many chunks or EM iterations the call runs. The
[data contract](data_contract.md) says what the package expects of your data.

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
