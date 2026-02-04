#!/usr/bin/env python3
"""
완전한 MSA-EENDEDA 파이프라인
"""
import argparse
from pathlib import Path
from utils.speaker_encoder import SpeakerEncoder
from utils.feature_extractor import FeatureExtractor
from utils.conversation_anonymizer import ConversationAnonymizer
from utils.synthesizer import Synthesizer
from utils.msa_utils import AudioProcessor,RTTMParser

import json


def run_pipeline(args):
    # 초기화
    audio_proc = AudioProcessor()
    speaker_enc = SpeakerEncoder()
    feat_ext = FeatureExtractor()
    anonymizer = ConversationAnonymizer(args.external_pool)
    synthesizer = Synthesizer()

    # Manifest 로드
    with open(args.eendeda_output / "manifest.json") as f:
        manifest = json.load(f)

    for entry in manifest:
        print(f"\n{'=' * 50}")
        print(f"처리: {entry['utt']}")
        print(f"{'=' * 50}")

        # 1. RTTM 파싱
        segments = RTTMParser.parse_file(Path(entry['rttm']))

        # 2. 오디오 로드 및 화자별 집계
        audio = audio_proc.load_audio(args.audio_dir / f"{entry['utt']}.wav")
        speaker_audio = audio_proc.aggregate_by_speaker(audio, segments)

        # 3. 화자 벡터 추출
        speaker_vectors = {}
        for spk_id, spk_audio in speaker_audio.items():
            speaker_vectors[spk_id] = speaker_enc.encode(spk_audio)
        print(f"  ✓ 화자 벡터 추출: {len(speaker_vectors)}개")

        # 4. 대화 수준 익명화
        anon_vectors = anonymizer.anonymize(speaker_vectors)
        print(f"  ✓ 익명화 완료")

        # 5. 각 화자 음성 합성
        anon_audio = {}
        for spk_id, spk_audio in speaker_audio.items():
            content = feat_ext.extract_content(spk_audio)
            f0 = feat_ext.extract_f0(spk_audio)

            anon_audio[spk_id] = synthesizer.synthesize(
                content, f0, anon_vectors[spk_id]
            )
        print(f"  ✓ 음성 합성 완료")

        # 6. 대화 재구성
        reconstructed = audio_proc.reconstruct_conversation(
            anon_audio, segments, len(audio) / 16000
        )

        # 7. 저장
        output_path = args.output_dir / f"{entry['utt']}_anonymized.wav"
        audio_proc.save_audio(reconstructed, output_path)
        print(f"  ✓ 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eendeda-output", type=Path,
                        default="output/eendeda")
    parser.add_argument("--audio-dir", type=Path,
                        default="data/audio")
    parser.add_argument("--output-dir", type=Path,
                        default="output/anonymized")
    parser.add_argument("--external-pool", type=Path,
                        default="data/external_pool/librispeech_xvectors.npy")

    args = parser.parse_args()
    run_pipeline(args)