from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from IPython.display import Audio


class Infer:
    def __init__(self, model_name="microsoft/speecht5_tts"):
        # Load models
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

        # Load vovector
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load embeddings
        self.embedding = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )

    def tts(self, text, speaker_id):
        inputs = self.processor(text=text, return_tensors="pt")
        speaker_embedding = torch.tensor(
            self.embedding[speaker_id]["xvector"]
        ).unsqueeze(0)
        speech = self.model.generate_speech(
            inputs["input_ids"], speaker_embedding, vocoder=self.vocoder
        )
        return speech

    def listen_to_speach(self, waveform):
        return Audio(waveform, rate=16000)

    def run(self, text, speacker_id=16):
        speech = self.tts(text, speacker_id)
        self.listen_to_speach(speech)
