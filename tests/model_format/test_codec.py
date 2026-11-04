"""The JSON tags of a saved model resolve only to the codec's own handlers."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.model_format._codec import ArrayWriter, Decoder, Encoder
from non_local_detector.model_format._container import StoredModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "tag", ["$_init__", "$_class__", "$_as", "$_type", "$read_array"]
)
def test_a_tag_that_is_not_a_handler_is_refused(tag):
    decoder = Decoder(lambda member: None, plain=True)

    with pytest.raises(ValidationError, match="Unknown tag"):
        decoder.decode({tag: {"x": 1}})
    assert decoder.plain is True


@pytest.mark.parametrize(
    "value",
    [
        np.array(6.0),
        np.array(3, dtype=np.int64),
        np.array(True),
        np.array(2.5, dtype=np.float32),
        np.float64(6.0),
        np.float32(2.5),
        np.int64(3),
        np.bool_(True),
        jnp.asarray(np.float32(1.5)),
    ],
    ids=lambda value: f"{type(value).__name__}-{np.asarray(value).dtype}",
)
def test_scalars_and_zero_dimensional_arrays_keep_type_dtype_and_shape(value):
    writer = ArrayWriter()
    node = Encoder(writer).encode({"value": value}, "test")
    stored = StoredModel(
        {"arrays": writer.index}, None, {}, _read_member=writer.members.__getitem__
    )

    decoded = Decoder(stored.read_array).decode(node)["value"]

    assert type(decoded) is type(value)
    assert np.asarray(decoded).dtype == np.asarray(value).dtype
    assert np.shape(decoded) == np.shape(value)
    assert np.asarray(decoded) == np.asarray(value)


def test_an_array_unlike_its_manifest_entry_is_refused():
    writer = ArrayWriter()
    node = Encoder(writer).encode(np.arange(4.0), "test")
    (entry,) = writer.index.values()
    entry["shape"] = [2, 2]
    stored = StoredModel(
        {"arrays": writer.index}, None, {}, _read_member=writer.members.__getitem__
    )

    with pytest.raises(DataError, match="manifest"):
        Decoder(stored.read_array).decode(node)
