import time
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from IPython.display import Audio
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from text2speech.logger_configure import get_logger
from text2speech.utils import convert_to_words

logger = get_logger(__name__)


class Infer:
    def __init__(
        self,
        model_name: str = "microsoft/speecht5_tts",
        vocoder: str = "microsoft/speecht5_hifigan",
        embedding: str = "Matthijs/cmu-arctic-xvectors",
    ) -> None:
        """Responsible for mahe speech on a given text based on a given speaker id
        Args:
            model_name: The name if the model, the same is used for processing
            vocoder: The vocoder to chage the spectrum to waveform
            voice_embedding: An embedding of different voices used"""
        # Load models
        logger.info("Loading the model and preprocessor")
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

        # Load vovector
        logger.info("Loading  vovector")
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder)

        # Load embeddings
        logger.info("Loading embedding")
        self.embedding = load_dataset(embedding, split="validation")

        self.metadata = {
            "model_name": model_name,
            "preporcessor": model_name,
            "vocoder": vocoder,
            "voice_embeddings": embedding,
        }

    def preprocess_text(self, text):
        return convert_to_words(text)

    def tokenizer_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        return inputs

    def execute(self, text: str, speaker_id: int = 7000) -> torch.Tensor:
        assert speaker_id < self.embedding.num_rows
        speaker_embedding = torch.tensor(self.embedding[speaker_id]["xvector"]).unsqueeze(0)

        preprocessed_text = self.preprocess_text(text)
        inputs = self.tokenizer_text(preprocessed_text)

        speech = self.model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=self.vocoder)
        return speech

    def listen_to_speach(self, waveform):
        return Audio(waveform, rate=16000)

    def save_audio(self, waveform, save_loc):
        sf.write(save_loc, waveform.numpy(), samplerate=16000)
        return save_loc


if __name__ == "__main__":
    infer = Infer()
    infer.execute(
        "Users number 1000 (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations."
    )
