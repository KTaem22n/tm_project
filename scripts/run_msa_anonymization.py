#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_msa_anonymization.py
유연한 MSA 익명화 - 명령줄 인자 지원
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import json

# 기존 유틸리티 임포트
# sys.path를 조정하여 utils 임포트 가능하도록
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.audio_utils import AudioProcessor
    from utils.rttm_utils import RTTMParser
except:
    # utils가 없으면 간단한 버전 사용
    class AudioProcessor:
        def __init__(self, sr=16000):
            self.sr = sr

        def load_audio(self, path):
            audio, file_sr = sf.read(path)
            if file_sr != self.sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio

        def aggregate_by_speaker(self, audio, segments, sr=None):
            if sr is None:
                sr = self.sr
            speaker_audio = {}
            for seg in segments:
                spk_id = seg['speaker_id']
                start = int(seg['start'] * sr)
                end = int((seg['start'] + seg['duration']) * sr)
                segment = audio[start:end]
                if spk_id not in speaker_audio:
                    speaker_audio[spk_id] = []
                speaker_audio[spk_id].append(segment)

            for spk in speaker_audio:
                speaker_audio[spk] = np.concatenate(speaker_audio[spk])
            return speaker_audio

        def reconstruct_conversation(self, speaker_audio, segments, total_duration=None, sr=None):
            if sr is None:
                sr = self.sr
            if total_duration is None:
                max_end = max(seg['start'] + seg['duration'] for seg in segments)
                total_samples = int(max_end * sr)
            else:
                total_samples = int(total_duration * sr)

            output = np.zeros(total_samples)
            speaker_positions = {spk: 0 for spk in speaker_audio.keys()}

            for seg in sorted(segments, key=lambda x: x['start']):
                spk_id = seg['speaker_id']
                if spk_id not in speaker_audio:
                    continue

                start_sample = int(seg['start'] * sr)
                duration_samples = int(seg['duration'] * sr)
                spk_pos = speaker_positions[spk_id]
                spk_audio = speaker_audio[spk_id]

                if spk_pos >= len(spk_audio):
                    continue

                segment = spk_audio[spk_pos:spk_pos + duration_samples]
                actual_len = len(segment)
                end_sample = min(start_sample + actual_len, total_samples)
                output[start_sample:end_sample] += segment[:end_sample - start_sample]
                speaker_positions[spk_id] += actual_len

            return output

        def save_audio(self, audio, path, sr=None):
            if sr is None:
                sr = self.sr
            sf.write(path, audio, sr)


    class RTTMParser:
        @staticmethod
        def parse_file(rttm_path):
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


def main():
    parser = argparse.ArgumentParser(
        description="MSA 익명화 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--eendeda-output',
        required=True,
        type=Path,
        help='EEND-EDA 출력 디렉토리 (RTTM과 attractor 포함)'
    )

    parser.add_argument(
        '--audio-dir',
        required=True,
        type=Path,
        help='원본 오디오 파일 디렉토리'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='익명화된 오디오 출력 디렉토리'
    )

    parser.add_argument(
        '--rttm-dir',
        type=Path,
        default=None,
        help='RTTM 디렉토리 (기본: eendeda-output/rttm)'
    )

    args = parser.parse_args()

    # 기본값 설정
    if args.rttm_dir is None:
        args.rttm_dir = args.eendeda_output / "rttm"

    # 디렉토리 존재 확인
    if not args.eendeda_output.exists():
        print(f"❌ EEND-EDA 출력 디렉토리 없음: {args.eendeda_output}")
        return

    if not args.audio_dir.exists():
        print(f"❌ 오디오 디렉토리 없음: {args.audio_dir}")
        return

    if not args.rttm_dir.exists():
        print(f"❌ RTTM 디렉토리 없음: {args.rttm_dir}")
        return

    # 출력 디렉토리 생성
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MSA 익명화")
    print(f"{'=' * 70}")
    print(f"EEND-EDA 출력: {args.eendeda_output}")
    print(f"원본 오디오: {args.audio_dir}")
    print(f"RTTM: {args.rttm_dir}")
    print(f"출력: {args.output_dir}")
    print(f"{'=' * 70}\n")

    # Manifest 로드
    manifest_path = args.eendeda_output / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"✓ Manifest 로드: {len(manifest)}개 발화\n")
    else:
        # manifest 없으면 RTTM 파일 기반으로 생성
        manifest = []
        for rttm_file in args.rttm_dir.glob("*.rttm"):
            manifest.append({'utt': rttm_file.stem})
        print(f"✓ RTTM 기반 Manifest 생성: {len(manifest)}개 발화\n")

    # 프로세서 초기화
    audio_proc = AudioProcessor(sr=16000)
    rttm_parser = RTTMParser()

    # 결과 리스트
    results = []
    processed = 0
    failed = 0

    # 각 발화 처리
    for entry in manifest:
        utt_name = entry['utt']

        try:
            print(f"처리 중: {utt_name}")

            # RTTM 파싱
            rttm_path = args.rttm_dir / f"{utt_name}.rttm"
            if not rttm_path.exists():
                print(f"  ⚠️  RTTM 파일 없음: {rttm_path}")
                failed += 1
                continue

            segments = rttm_parser.parse_file(rttm_path)
            print(f"  - 세그먼트 수: {len(segments)}")

            # 화자 수 확인
            speaker_ids = sorted(set(seg['speaker_id'] for seg in segments))
            print(f"  - 화자 수: {len(speaker_ids)}")

            # 오디오 파일 찾기
            audio_path = None
            for ext in ['.wav', '.flac']:
                candidate = args.audio_dir / f"{utt_name}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break

            if audio_path is None:
                print(f"  ⚠️  오디오 파일 없음: {utt_name}")
                failed += 1
                continue

            # 오디오 로드
            audio = audio_proc.load_audio(audio_path)

            # 화자별 오디오 집계
            speaker_audio = audio_proc.aggregate_by_speaker(audio, segments)

            # TODO: 여기서 실제 MSA 익명화 수행
            # 현재는 재구성만 수행 (테스트)
            anonymized_audio = speaker_audio.copy()

            # 대화 재구성
            reconstructed = audio_proc.reconstruct_conversation(
                anonymized_audio,
                segments,
                total_duration=len(audio) / 16000
            )

            # 저장
            output_path = args.output_dir / f"{utt_name}_anonymized.wav"
            audio_proc.save_audio(reconstructed, output_path)
            print(f"  ✓ 저장 완료: {output_path}")

            results.append({
                'utt': utt_name,
                'num_segments': len(segments),
                'num_speakers': len(speaker_ids),
                'output': str(output_path),
                'status': 'success'
            })
            processed += 1

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results.append({
                'utt': utt_name,
                'status': 'failed',
                'error': str(e)
            })
            failed += 1

    # 결과 저장
    result_json = args.output_dir / "anonymization_results.json"
    with open(result_json, 'w') as f:
        json.dump({
            'total': len(manifest),
            'processed': processed,
            'failed': failed,
            'results': results
        }, f, indent=2)

    # 요약 출력
    print(f"\n{'=' * 70}")
    print(f"완료!")
    print(f"{'=' * 70}")
    print(f"총 발화: {len(manifest)}")
    print(f"처리 성공: {processed}")
    print(f"처리 실패: {failed}")
    print(f"결과 저장: {result_json}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
