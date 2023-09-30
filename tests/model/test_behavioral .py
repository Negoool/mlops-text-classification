import pytest

from text2speech.infer import Infer


@pytest.fixture
def load_model(scopre="module"):
    inferencer = Infer()
    return inferencer


def test_length_speech(load_model):
    text = "today is Monday of the first january month"
    speech = load_model.execute(text)
    speech_length = round(speech.shape[0] / 16000, 2)
    assert 2 < speech_length and speech_length < 8
