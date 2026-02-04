# utils/synthesizer.py
from models.hifigan import Generator


class Synthesizer:
    def __init__(self, hifigan_path="models/msa/hifigan"):
        self.vocoder = Generator(...)  # HiFi-GAN 로드

    def synthesize(self, content_features, f0, speaker_vector):
        """
        음성 합성
        """
        # TODO: 구현
        pass