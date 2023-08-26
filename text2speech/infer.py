import logging
import time

import torch
from datasets import load_dataset
from IPython.display import Audio
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from text2speech.logger_configure import get_logger


class Infer:
    def __init__(self, model_name: str = "microsoft/speecht5_tts") -> None:
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

    def tts(self, text: str, speaker_id: int) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt")
        speaker_embedding = torch.tensor(self.embedding[speaker_id]["xvector"]).unsqueeze(0)
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=self.vocoder)
        return speech

    def listen_to_speach(self, waveform):
        return Audio(waveform, rate=16000)

    def run(self, text: str, speacker_id: int = 16) -> None:
        st = time.time()
        speech = self.tts(text, speacker_id)
        infer_time = time.time() - st
        logger.debug({"infer_time": infer_time, "text_length": len(text), "speech_length": round(speech.shape[0] / 16000, 2)})
        # self.listen_to_speach(speech)
