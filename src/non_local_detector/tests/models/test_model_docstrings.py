"""Model class docstrings: complete, current, and alike within each pair.

Each model has a sorted-spikes and a clusterless class with the same states
and defaults. Their docstrings are written out in each class, so these tests
keep them from drifting: every constructor parameter is documented in order
with its current default, and the two docstrings of a pair differ only in
the data-type wording listed in ``_CLUSTERLESS_TO_SORTED``.
"""

import inspect

import pytest

from non_local_detector import models

pytestmark = pytest.mark.unit

MODEL_CLASSES = [
    models.SortedSpikesDecoder,
    models.ClusterlessDecoder,
    models.ContFragSortedSpikesClassifier,
    models.ContFragClusterlessClassifier,
    models.MultiEnvironmentSortedSpikesClassifier,
    models.MultiEnvironmentClusterlessClassifier,
    models.NoSpikeContFragSortedSpikesClassifier,
    models.NoSpikeContFragClusterlessClassifier,
    models.NonLocalSortedSpikesDetector,
    models.NonLocalClusterlessDetector,
]


def _parameter_entries(cls):
    """Entries of the class docstring's Parameters section, by name.

    Returns
    -------
    entries : dict[str, str]
        Each documented parameter's name and its entry (type line and
        description), in order.
    """
    doc = inspect.getdoc(cls) or ""
    assert "Parameters\n----------\n" in doc, f"{cls.__name__} has no Parameters"
    section = doc.split("Parameters\n----------\n", 1)[1].split("\nExamples\n", 1)[0]
    entries: dict[str, str] = {}
    name = None
    for line in section.splitlines():
        if line and not line[0].isspace():
            name = line.split(" :")[0]
            entries[name] = line
        elif name is not None:
            entries[name] += "\n" + line
    return entries


def _documented_parameters(cls):
    """Names of the entries in the class docstring's Parameters section."""
    return list(_parameter_entries(cls))


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_parameters_section_matches_signature(cls):
    """The Parameters section lists the constructor's parameters in order."""
    signature = list(inspect.signature(cls.__init__).parameters)
    assert _documented_parameters(cls) == signature[1:]


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_docstring_names_its_own_class_and_data(cls):
    """The examples construct this class and the text names its data type."""
    doc = inspect.getdoc(cls) or ""
    data = "clusterless" if "Clusterless" in cls.__name__ else "sorted spike"
    assert f"{cls.__name__}()" in doc
    assert data in doc.splitlines()[0]


def _documented_default(default):
    """How a scalar default is written in a docstring entry."""
    return f'"{default}"' if isinstance(default, str) else repr(default)


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_documented_scalar_defaults_match_signature(cls):
    """Each parameter with a scalar default states that default, so changing
    a default without its documentation fails here."""
    entries = _parameter_entries(cls)
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        default = parameter.default
        if isinstance(default, (bool, int, float, str)):
            assert _documented_default(default) in entries[name], (
                f"{cls.__name__}.{name}: default {default!r} not documented"
            )


MODEL_PAIRS = list(zip(MODEL_CLASSES[::2], MODEL_CLASSES[1::2], strict=True))

# The only wording allowed to differ between a clusterless docstring and its
# sorted-spikes twin, as (clusterless text, sorted-spikes text), applied in
# order.
_CLUSTERLESS_TO_SORTED = [
    (
        """clusterless_algorithm : str, optional
    Clusterless likelihood algorithm, a key of ``_CLUSTERLESS_ALGORITHMS``,
    by default "clusterless_kde".
clusterless_algorithm_params : dict or None, optional
    Parameters for the clusterless algorithm. If ``None`` (the default),
    the module-level ``_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS`` values are
    copied into a fresh dict at fit time. User-supplied dicts are stored
    as-is for sklearn ``clone()`` compatibility and copied on use.""",
        """sorted_spikes_algorithm : str, optional
    Sorted-spikes likelihood algorithm, a key of
    ``_SORTED_SPIKES_ALGORITHMS``, by default "sorted_spikes_kde".
sorted_spikes_algorithm_params : dict or None, optional
    Parameters for the sorted-spikes algorithm. If ``None`` (the default),
    the module-level ``_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS`` values are
    copied into a fresh dict at fit time. User-supplied dicts are stored
    as-is for sklearn ``clone()`` compatibility and copied on use.""",
    ),
    (
        """no_spike_rate : float, optional
    Firing rate in Hz of each electrode in a No-Spike state, by default
    1e-10. The marks of its spikes are scored with the training-fitted
    reference density ``mark_reference_``.""",
        """no_spike_rate : float, optional
    Firing rate in Hz of each neuron in a No-Spike state, by default
    1e-10.""",
    ),
    (
        "position_time, position, spike_times, spike_waveform_features",
        "position_time, position, spike_times",
    ),
    ("spike_times, spike_waveform_features", "spike_times"),
    ("clusterless spike data", "sorted spike data"),
]


@pytest.mark.parametrize(
    "sorted_cls, clusterless_cls",
    MODEL_PAIRS,
    ids=[sorted_cls.__name__ for sorted_cls, _ in MODEL_PAIRS],
)
def test_pair_docstrings_differ_only_in_data_type(sorted_cls, clusterless_cls):
    """Rewriting the clusterless docstring's data-type wording gives the
    sorted-spikes docstring exactly."""
    clusterless_doc = inspect.getdoc(clusterless_cls) or ""
    for clusterless_text, sorted_text in _CLUSTERLESS_TO_SORTED:
        assert clusterless_text in clusterless_doc, (
            f"{clusterless_cls.__name__} lacks its data-type text: "
            f"{clusterless_text.splitlines()[0]!r}"
        )
        clusterless_doc = clusterless_doc.replace(clusterless_text, sorted_text)
    clusterless_doc = clusterless_doc.replace(
        clusterless_cls.__name__, sorted_cls.__name__
    )
    assert clusterless_doc == inspect.getdoc(sorted_cls)
