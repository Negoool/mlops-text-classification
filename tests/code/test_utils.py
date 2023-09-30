import pytest

from text2speech.utils import convert_to_words


@pytest.mark.parametrize(
    "text, text_wo_digits",
    [("100 times more", "one hundred times more"), ("18 Feb 2023", "eighteen Feb two thousand and twenty-three")],
)
def test_convert_to_words(text, text_wo_digits):
    assert convert_to_words(text) == text_wo_digits
