# utils/feature_extractor.py
from transformers import HubertModel
import librosa


class FeatureExtractor:
    def __init__(self, hubert_path="models/msa/hubert"):
        self.hubert = HubertModel.from_pretrained(hubert_path)

    def extract_content(self, audio, sr=16000):
        """HuBERT 컨텐츠 특징 추출"""
        # TODO: 구현
        pass

    def extract_f0(self, audio, sr=16000):
        """F0 추출 (YAAPT)"""
        f0, _ = librosa.piptrack(y=audio, sr=sr)
        return f0