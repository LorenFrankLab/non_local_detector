"""The JSON tags of a saved model resolve only to the codec's own handlers."""

import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.model_format._codec import Decoder

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "tag", ["$_init__", "$_class__", "$_as", "$_type", "$read_array"]
)
def test_a_tag_that_is_not_a_handler_is_refused(tag):
    decoder = Decoder(lambda member: None, plain=True)

    with pytest.raises(ValidationError, match="Unknown tag"):
        decoder.decode({tag: {"x": 1}})
    assert decoder.plain is True
