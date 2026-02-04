# utils/conversation_anonymizer.py
import numpy as np
from utils.vector_utils import VectorProcessor


class ConversationAnonymizer:
    def __init__(self, external_pool_path, method='aggregated_similarity'):
        self.pool = np.load(external_pool_path)
        self.method = method
        self.vector_proc = VectorProcessor()

    def anonymize(self, speaker_vectors):
        """
        speaker_vectors: dict {spk_id: vector}
        returns: dict {spk_id: anonymized_vector}
        """
        if self.method == 'aggregated_similarity':
            return self._apply_AS(speaker_vectors)
        elif self.method == 'differential_similarity':
            return self._apply_DS(speaker_vectors)

    def _apply_AS(self, speaker_vectors):
        # Algorithm 1 from MSA paper
        # TODO: 구현
        pass

    def _apply_DS(self, speaker_vectors):
        # Algorithm 1 from MSA paper (DS variant)
        # TODO: 구현
        pass