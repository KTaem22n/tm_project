# utils/speaker_encoder.py
import torch
from speechbrain.pretrained import EncoderClassifier


class SpeakerEncoder:
    def __init__(self, model_path="models/msa/ecapa_tdnn"):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_path
        )

    def encode(self, audio):
        """
        audio: numpy array
        returns: speaker vector (192-dim)
        """
        with torch.no_grad():
            embedding = self.model.encode_batch(
                torch.tensor(audio).unsqueeze(0)
            )
        return embedding.squeeze().cpu().numpy()