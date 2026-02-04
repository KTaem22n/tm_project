#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MSA with EEND-EDA-VIB Integration
----------------------------------
This script integrates EEND-EDA-VIB diarization with MSA anonymization.

Directory structure expected:
/home/ktaemin/tm_project/
├── eendeda_output/          # Output from infer_4_anon.py
│   ├── rttm/
│   ├── spkvec/
│   └── manifest.json
├── msa_anonymization/       # MSA anonymization models
├── output/                  # Final anonymized output
└── this_script.py

Usage:
    python msa_eendeda_integration.py \
        --eendeda-output ./eendeda_output \
        --audio-dir ./audio_files \
        --output-dir ./output \
        --msa-config ./msa_config.yaml
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EENDEDAtoMSABridge:
    """
    Bridge between EEND-EDA-VIB output and MSA anonymization input.
    Handles RTTM parsing, audio segmentation, and speaker vector preparation.
    """

    def __init__(self, eendeda_output_dir: Path, audio_dir: Path):
        self.eendeda_dir = Path(eendeda_output_dir)
        self.audio_dir = Path(audio_dir)
        self.rttm_dir = self.eendeda_dir / "rttm"
        self.spkvec_dir = self.eendeda_dir / "spkvec"

        # Load manifest
        manifest_path = self.eendeda_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded manifest with {len(self.manifest)} utterances")

    def parse_rttm(self, rttm_path: Path) -> List[Dict]:
        """
        Parse RTTM file to extract speaker segments.

        Returns:
            List of dicts with keys: speaker, start, duration
        """
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == "SPEAKER":
                    segments.append({
                        'speaker': parts[7],  # spk0, spk1, etc.
                        'start': float(parts[3]),
                        'duration': float(parts[4])
                    })
        return segments

    def aggregate_speaker_audio(
            self,
            audio_path: Path,
            segments: List[Dict],
            sr: int = 16000
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate audio segments by speaker from RTTM.

        Args:
            audio_path: Path to original audio file
            segments: List of segment dicts from RTTM
            sr: Sample rate

        Returns:
            Dict mapping speaker_id -> concatenated audio array
        """
        # Load audio
        audio, file_sr = sf.read(audio_path)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        # Group by speaker
        speaker_audio = {}
        for seg in segments:
            spk_id = seg['speaker']
            start_sample = int(seg['start'] * sr)
            end_sample = int((seg['start'] + seg['duration']) * sr)

            segment_audio = audio[start_sample:end_sample]

            if spk_id not in speaker_audio:
                speaker_audio[spk_id] = []
            speaker_audio[spk_id].append(segment_audio)

        # Concatenate segments for each speaker
        for spk_id in speaker_audio:
            speaker_audio[spk_id] = np.concatenate(speaker_audio[spk_id])

        return speaker_audio

    def load_attractor_vectors(self, utt_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load attractor vectors for an utterance.

        Returns:
            z_mu, z_logvar, att_probs as numpy arrays
        """
        z_mu = np.load(self.spkvec_dir / f"{utt_name}.z_mu.npy")
        z_logvar = np.load(self.spkvec_dir / f"{utt_name}.z_logvar.npy")
        att_probs = np.load(self.spkvec_dir / f"{utt_name}.att_probs.npy")

        return z_mu, z_logvar, att_probs

    def prepare_for_anonymization(self, utt_name: str, audio_path: Path) -> Dict:
        """
        Prepare all data needed for MSA anonymization.

        Returns:
            Dict containing:
                - speaker_audio: Dict[speaker_id -> audio_array]
                - attractor_vectors: Dict[speaker_id -> vector]
                - segments: Original RTTM segments
                - num_speakers: Number of speakers
        """
        # Find manifest entry
        entry = next((e for e in self.manifest if e['utt'] == utt_name), None)
        if entry is None:
            raise ValueError(f"Utterance {utt_name} not found in manifest")

        # Parse RTTM
        segments = self.parse_rttm(Path(entry['rttm']))

        # Aggregate audio by speaker
        speaker_audio = self.aggregate_speaker_audio(audio_path, segments)

        # Load attractor vectors
        z_mu, z_logvar, att_probs = self.load_attractor_vectors(utt_name)

        # Map attractors to speakers (assuming spk0->z_mu[0], spk1->z_mu[1], etc.)
        speaker_ids = sorted(set(seg['speaker'] for seg in segments))
        attractor_map = {}
        for i, spk_id in enumerate(speaker_ids):
            if i < len(z_mu):
                attractor_map[spk_id] = {
                    'z_mu': z_mu[i],
                    'z_logvar': z_logvar[i],
                    'att_prob': att_probs[i] if i < len(att_probs) else 0.0
                }

        return {
            'utt_name': utt_name,
            'speaker_audio': speaker_audio,
            'attractor_vectors': attractor_map,
            'segments': segments,
            'num_speakers': len(speaker_ids),
            'speaker_ids': speaker_ids
        }


class MSAAnonymizer:
    """
    MSA anonymization wrapper.
    This integrates with the MSA anonymization components.
    """

    def __init__(self, msa_model_path: Path, device: str = 'cuda'):
        self.device = device
        self.model_path = Path(msa_model_path)

        # Initialize MSA components
        # Note: Actual MSA imports would go here
        # from msa.anonymization import SelectionBasedAnonymizer, etc.
        logger.info(f"Initializing MSA anonymizer from {msa_model_path}")

    def anonymize_conversation(
            self,
            conversation_data: Dict,
            external_pool_path: Optional[Path] = None
    ) -> Dict:
        """
        Anonymize a multi-speaker conversation.

        Args:
            conversation_data: Output from EENDEDAtoMSABridge.prepare_for_anonymization
            external_pool_path: Path to external speaker vector pool

        Returns:
            Dict containing anonymized audio and metadata
        """
        speaker_audio = conversation_data['speaker_audio']
        attractor_vectors = conversation_data['attractor_vectors']
        segments = conversation_data['segments']
        speaker_ids = conversation_data['speaker_ids']

        logger.info(f"Anonymizing {conversation_data['num_speakers']} speakers")

        # Extract original speaker vectors (use z_mu from attractors)
        original_vectors = {}
        for spk_id in speaker_ids:
            if spk_id in attractor_vectors:
                original_vectors[spk_id] = attractor_vectors[spk_id]['z_mu']

        # Apply conversation-level anonymization
        # This would use the actual MSA anonymization logic
        anonymized_vectors = self._apply_conversation_anonymization(
            original_vectors,
            method='differential_similarity'  # or 'aggregated_similarity'
        )

        # Anonymize each speaker's audio
        anonymized_audio = {}
        for spk_id in speaker_ids:
            audio = speaker_audio[spk_id]
            anon_vector = anonymized_vectors[spk_id]

            # This would call actual MSA speech synthesis
            anon_audio = self._anonymize_speaker_audio(audio, anon_vector)
            anonymized_audio[spk_id] = anon_audio

        # Reconstruct conversation
        reconstructed = self._reconstruct_conversation(
            anonymized_audio,
            segments,
            original_length=sum(len(a) for a in speaker_audio.values())
        )

        return {
            'anonymized_audio': reconstructed,
            'anonymized_vectors': anonymized_vectors,
            'original_segments': segments,
            'metadata': {
                'num_speakers': conversation_data['num_speakers'],
                'speaker_ids': speaker_ids
            }
        }

    def _apply_conversation_anonymization(
            self,
            original_vectors: Dict[str, np.ndarray],
            method: str = 'differential_similarity'
    ) -> Dict[str, np.ndarray]:
        """
        Apply conversation-level speaker anonymization.
        Implements either DS or AS from the MSA paper.
        """
        # This is a placeholder - actual implementation would use
        # the selection algorithm from MSA (Algorithm 1 in paper)
        logger.info(f"Applying {method} anonymization")

        # Simplified version - in reality this would:
        # 1. Load external pool
        # 2. Compute similarity matrices
        # 3. Apply greedy search with DS or AS constraints

        anonymized = {}
        for spk_id, vec in original_vectors.items():
            # Placeholder: just add noise (replace with actual MSA logic)
            anonymized[spk_id] = vec + np.random.randn(*vec.shape) * 0.1

        return anonymized

    def _anonymize_speaker_audio(
            self,
            audio: np.ndarray,
            anonymized_vector: np.ndarray
    ) -> np.ndarray:
        """
        Anonymize single speaker's audio using MSA synthesis pipeline.
        """
        # This would use the actual MSA synthesis:
        # 1. Extract F0 with YAAPT
        # 2. Extract content features with HuBERT
        # 3. Synthesize with HiFi-GAN using anonymized vector

        logger.info(f"Anonymizing audio segment of length {len(audio)}")

        # Placeholder - return original (replace with actual synthesis)
        return audio

    def _reconstruct_conversation(
            self,
            anonymized_audio: Dict[str, np.ndarray],
            segments: List[Dict],
            original_length: int,
            sr: int = 16000
    ) -> np.ndarray:
        """
        Reconstruct full conversation from anonymized speaker segments.
        """
        # Create empty audio buffer
        total_samples = max(
            int((seg['start'] + seg['duration']) * sr)
            for seg in segments
        )
        reconstructed = np.zeros(total_samples)

        # Track position in each speaker's audio
        speaker_positions = {spk: 0 for spk in anonymized_audio.keys()}

        # Place segments in timeline
        for seg in sorted(segments, key=lambda x: x['start']):
            spk_id = seg['speaker']
            start_sample = int(seg['start'] * sr)
            duration_samples = int(seg['duration'] * sr)

            # Get segment from anonymized audio
            spk_audio = anonymized_audio[spk_id]
            spk_pos = speaker_positions[spk_id]

            segment = spk_audio[spk_pos:spk_pos + duration_samples]

            # Place in reconstructed audio
            end_sample = start_sample + len(segment)
            reconstructed[start_sample:end_sample] += segment

            speaker_positions[spk_id] += len(segment)

        return reconstructed


def main():
    parser = argparse.ArgumentParser(description="MSA with EEND-EDA-VIB Integration")
    parser.add_argument("--eendeda-output", required=True,
                        help="Directory with EEND-EDA-VIB output (from infer_4_anon.py)")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory containing original audio files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for anonymized audio")
    parser.add_argument("--msa-model-path", required=True,
                        help="Path to MSA anonymization models")
    parser.add_argument("--external-pool", default=None,
                        help="Path to external speaker vector pool")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # Initialize components
    bridge = EENDEDAtoMSABridge(args.eendeda_output, args.audio_dir)
    anonymizer = MSAAnonymizer(args.msa_model_path, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each utterance
    results = []
    for entry in bridge.manifest:
        utt_name = entry['utt']
        logger.info(f"\nProcessing: {utt_name}")

        # Find audio file
        audio_path = Path(args.audio_dir) / f"{utt_name}.wav"
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        # Prepare data
        conv_data = bridge.prepare_for_anonymization(utt_name, audio_path)

        # Anonymize
        result = anonymizer.anonymize_conversation(
            conv_data,
            external_pool_path=args.external_pool
        )

        # Save anonymized audio
        output_path = output_dir / f"{utt_name}_anonymized.wav"
        sf.write(output_path, result['anonymized_audio'], 16000)

        logger.info(f"Saved: {output_path}")
        results.append({
            'utt': utt_name,
            'output': str(output_path),
            'num_speakers': result['metadata']['num_speakers']
        })

    # Save results manifest
    with open(output_dir / "anonymization_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nComplete! Processed {len(results)} utterances")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()