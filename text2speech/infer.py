import time
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from IPython.display import Audio
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from text2speech.logger_configure import get_logger

logger = get_logger(__name__)


class Infer:
    def __init__(self, model_name: str = "microsoft/speecht5_tts", saved_loc: str = "./") -> None:
        """Responsible for making inference on a given text adn speaker id"""
        # Load models
        logger.info("Loading the model and preprocessor")
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

        # Load vovector
        logger.info("Loading  vovector")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load embeddings
        logger.info("Loading embedding")
        self.embedding = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

        self.saved_loc = Path(saved_loc)

    def tts(self, text: str, speaker_id: int) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt")
        speaker_embedding = torch.tensor(self.embedding[speaker_id]["xvector"]).unsqueeze(0)
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=self.vocoder)
        return speech

    def listen_to_speach(self, waveform):
        return Audio(waveform, rate=16000)

    def save_audio(self, waveform):
        sf.write(self.saved_loc / "speech.wav", waveform.numpy(), samplerate=16000)

    def run(self, text: str, speacker_id: int = 7000) -> None:
        st = time.time()
        speech = self.tts(text, speacker_id)
        infer_time = time.time() - st
        logger.info({"infer_time": infer_time, "text_length": len(text), "speech_length": round(speech.shape[0] / 16000, 2)})
        self.save_audio(speech)


if __name__ == "__main__":
    infer = Infer()
    infer.run(
        "Users number 1000 (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations."
    )
