import torch
from datasets import load_dataset
from IPython.display import Audio
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)


class Infer:
    def __init__(self, model_name: str = "microsoft/speecht5_tts") -> None:
        """Responsible for making inference on a given text adn speaker id"""
        # Load models
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

        # Load vovector
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load embeddings
        self.embedding = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def tts(self, text: str, speaker_id: int) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt")
        speaker_embedding = torch.tensor(self.embedding[speaker_id]["xvector"]).unsqueeze(0)
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=self.vocoder)
        return speech

    def listen_to_speach(self, waveform):
        return Audio(waveform, rate=16000)

    def run(self, text: str, speacker_id: int = 16) -> None:
        speech = self.tts(text, speacker_id)
        self.listen_to_speach(speech)
