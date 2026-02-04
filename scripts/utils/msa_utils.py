#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Scripts for MSA-EENDEDA Integration
--------------------------------------------
Collection of helper functions for audio processing, RTTM handling,
and vector operations.
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# RTTM Utilities
# ============================================================================

class RTTMParser:
    """Parse and manipulate RTTM files."""

    @staticmethod
    def parse_file(rttm_path: Path) -> List[Dict]:
        """
        Parse RTTM file.

        Returns:
            List of segment dicts with keys:
                - file_id, channel, start, duration, speaker_id
        """
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    segments.append({
                        'file_id': parts[1],
                        'channel': parts[2],
                        'start': float(parts[3]),
                        'duration': float(parts[4]),
                        'speaker_id': parts[7]
                    })
        return segments

    @staticmethod
    def write_file(segments: List[Dict], output_path: Path):
        """Write segments to RTTM file."""
        with open(output_path, 'w') as f:
            for seg in segments:
                f.write(
                    f"SPEAKER {seg['file_id']} {seg.get('channel', 1)} "
                    f"{seg['start']:.3f} {seg['duration']:.3f} "
                    f"<NA> <NA> {seg['speaker_id']} <NA> <NA>\n"
                )

    @staticmethod
    def get_speaker_segments(segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group segments by speaker ID."""
        speaker_segs = {}
        for seg in segments:
            spk = seg['speaker_id']
            if spk not in speaker_segs:
                speaker_segs[spk] = []
            speaker_segs[spk].append(seg)
        return speaker_segs

    @staticmethod
    def merge_adjacent_segments(
            segments: List[Dict],
            gap_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Merge adjacent segments from same speaker if gap < threshold.

        Args:
            segments: List of segment dicts
            gap_threshold: Maximum gap in seconds to merge
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segs = sorted(segments, key=lambda x: (x['speaker_id'], x['start']))

        merged = []
        current = sorted_segs[0].copy()

        for seg in sorted_segs[1:]:
            if seg['speaker_id'] != current['speaker_id']:
                merged.append(current)
                current = seg.copy()
                continue

            gap = seg['start'] - (current['start'] + current['duration'])

            if gap <= gap_threshold:
                # Merge
                current['duration'] = (
                        seg['start'] + seg['duration'] - current['start']
                )
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)
        return merged


# ============================================================================
# Audio Utilities
# ============================================================================

class AudioProcessor:
    """Audio loading, segmentation, and reconstruction."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def load_audio(self, path: Path, target_sr: Optional[int] = None) -> np.ndarray:
        """Load audio file and resample if needed."""
        audio, file_sr = sf.read(path)

        if target_sr is None:
            target_sr = self.sr

        if file_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return audio

    def extract_segment(
            self,
            audio: np.ndarray,
            start: float,
            duration: float,
            sr: Optional[int] = None
    ) -> np.ndarray:
        """Extract audio segment given start time and duration."""
        if sr is None:
            sr = self.sr

        start_sample = int(start * sr)
        end_sample = int((start + duration) * sr)

        return audio[start_sample:end_sample]

    def aggregate_segments(
            self,
            audio: np.ndarray,
            segments: List[Dict],
            sr: Optional[int] = None
    ) -> np.ndarray:
        """Concatenate multiple segments into single audio."""
        if sr is None:
            sr = self.sr

        chunks = []
        for seg in sorted(segments, key=lambda x: x['start']):
            chunk = self.extract_segment(audio, seg['start'], seg['duration'], sr)
            chunks.append(chunk)

        if not chunks:
            return np.array([])

        return np.concatenate(chunks)

    def aggregate_by_speaker(
            self,
            audio: np.ndarray,
            segments: List[Dict],
            sr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate audio segments grouped by speaker."""
        speaker_segs = RTTMParser.get_speaker_segments(segments)

        speaker_audio = {}
        for spk_id, segs in speaker_segs.items():
            speaker_audio[spk_id] = self.aggregate_segments(audio, segs, sr)

        return speaker_audio

    def reconstruct_conversation(
            self,
            speaker_audio: Dict[str, np.ndarray],
            segments: List[Dict],
            total_duration: Optional[float] = None,
            sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct conversation from speaker-level audio using RTTM.

        Args:
            speaker_audio: Dict mapping speaker_id -> concatenated audio
            segments: Original RTTM segments
            total_duration: Total duration in seconds (optional)
            sr: Sample rate
        """
        if sr is None:
            sr = self.sr

        # Determine total length
        if total_duration is None:
            max_end = max(seg['start'] + seg['duration'] for seg in segments)
            total_samples = int(max_end * sr)
        else:
            total_samples = int(total_duration * sr)

        # Initialize output
        output = np.zeros(total_samples)

        # Track position in each speaker's concatenated audio
        speaker_positions = {spk: 0 for spk in speaker_audio.keys()}

        # Place segments
        for seg in sorted(segments, key=lambda x: x['start']):
            spk_id = seg['speaker_id']

            if spk_id not in speaker_audio:
                logger.warning(f"Speaker {spk_id} not found in speaker_audio")
                continue

            start_sample = int(seg['start'] * sr)
            duration_samples = int(seg['duration'] * sr)

            # Get segment from speaker's audio
            spk_pos = speaker_positions[spk_id]
            spk_audio = speaker_audio[spk_id]

            if spk_pos >= len(spk_audio):
                logger.warning(
                    f"Position {spk_pos} exceeds audio length {len(spk_audio)} "
                    f"for speaker {spk_id}"
                )
                continue

            segment = spk_audio[spk_pos:spk_pos + duration_samples]

            # Handle segment length mismatch
            actual_len = len(segment)
            if actual_len < duration_samples:
                logger.debug(
                    f"Segment shorter than expected: {actual_len} < {duration_samples}"
                )

            # Place in output (handle overlaps by addition)
            end_sample = min(start_sample + actual_len, total_samples)
            output[start_sample:end_sample] += segment[:end_sample - start_sample]

            speaker_positions[spk_id] += actual_len

        return output

    def save_audio(self, audio: np.ndarray, path: Path, sr: Optional[int] = None):
        """Save audio to file."""
        if sr is None:
            sr = self.sr
        sf.write(path, audio, sr)


# ============================================================================
# Speaker Vector Utilities
# ============================================================================

class VectorProcessor:
    """Handle speaker vector operations for anonymization."""

    @staticmethod
    def compute_similarity_matrix(
            vectors: np.ndarray,
            metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            vectors: Array of shape (N, D)
            metric: 'cosine' or 'euclidean'
        """
        if metric == 'cosine':
            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / (norms + 1e-8)
            # Compute cosine similarity
            return normalized @ normalized.T
        elif metric == 'euclidean':
            # Compute pairwise distances
            diff = vectors[:, None, :] - vectors[None, :, :]
            distances = np.sqrt((diff ** 2).sum(axis=2))
            # Convert to similarity
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def find_farthest_vectors(
            query: np.ndarray,
            pool: np.ndarray,
            k: int,
            metric: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k farthest vectors from pool.

        Args:
            query: Query vector of shape (D,)
            pool: Pool vectors of shape (N, D)
            k: Number of vectors to return
            metric: Distance metric

        Returns:
            indices, distances
        """
        if metric == 'cosine':
            # Compute cosine similarities
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            pool_norm = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
            similarities = pool_norm @ query_norm

            # Farthest = most negative/smallest similarity
            indices = np.argsort(similarities)[:k]
            return indices, similarities[indices]
        else:
            # Euclidean distance
            distances = np.linalg.norm(pool - query, axis=1)
            indices = np.argsort(distances)[-k:]  # Largest distances
            return indices, distances[indices]

    @staticmethod
    def apply_differential_similarity_constraint(
            original_vectors: np.ndarray,
            candidate_pool: np.ndarray,
            num_farthest: int = 200,
            num_candidates: int = 10000
    ) -> np.ndarray:
        """
        Apply differential similarity (DS) anonymization.
        Minimizes |sim(anon_i, anon_j) - sim(orig_i, orig_j)|
        """
        N = len(original_vectors)

        # Compute original similarity matrix
        S_orig = VectorProcessor.compute_similarity_matrix(original_vectors)

        # Get candidate vectors for each speaker
        candidates = []
        for i in range(N):
            indices, _ = VectorProcessor.find_farthest_vectors(
                original_vectors[i],
                candidate_pool,
                num_farthest
            )
            candidates.append(candidate_pool[indices])

        # Greedy search (simplified version of Algorithm 1)
        selected = []
        for i in range(N):
            best_idx = 0
            best_score = float('inf')

            for idx in range(len(candidates[i])):
                # Compute score for this candidate
                score = 0
                for j in range(len(selected)):
                    sim_anon = VectorProcessor._cosine_similarity(
                        candidates[i][idx],
                        selected[j]
                    )
                    sim_orig = S_orig[i, j]
                    score += abs(sim_anon - sim_orig)

                if score < best_score:
                    best_score = score
                    best_idx = idx

            selected.append(candidates[i][best_idx])

        return np.array(selected)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Process an audio file with RTTM

    # Initialize processors
    audio_proc = AudioProcessor(sr=16000)
    rttm_parser = RTTMParser()

    # Load RTTM
    segments = rttm_parser.parse_file(Path("example.rttm"))
    print(f"Found {len(segments)} segments")

    # Load audio
    audio = audio_proc.load_audio(Path("example.wav"))
    print(f"Audio length: {len(audio) / 16000:.2f}s")

    # Aggregate by speaker
    speaker_audio = audio_proc.aggregate_by_speaker(audio, segments)
    print(f"Speakers: {list(speaker_audio.keys())}")

    # Example: Reconstruct
    reconstructed = audio_proc.reconstruct_conversation(
        speaker_audio,
        segments
    )
    print(f"Reconstructed length: {len(reconstructed) / 16000:.2f}s")
