import pytest
import torch

from text2speech.infer import Infer


@pytest.fixture
def load_model(scopre="module"):
    inferencer = Infer()
    return inferencer


def test_tokenize_text(load_model):
    text = "one hundred days is left till fall"
    tokenized_text = load_model.tokenizer_text(text)
    assert "input_ids" in tokenized_text
    assert tokenized_text["input_ids"].ndim == 2
    assert tokenized_text["input_ids"].shape[0] == 1


def test_execute(load_model):
    text = "Today is a very warm day"
    speech = load_model.execute(text)
    assert isinstance(speech, torch.Tensor)
    assert speech.ndim == 1
