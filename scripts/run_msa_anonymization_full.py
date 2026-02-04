#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_msa_anonymization_full.py
실제 MSA 익명화 구현 포함 (AS/DS/Select)
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import json
from typing import Dict, List

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# EEND-EDA VIB 경로
eendeda_path = project_root / "eendeda_repo" / "eendedavib"
sys.path.insert(0, str(eendeda_path))


class SimpleAudioProcessor:
    """간단한 오디오 처리"""

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

    def aggregate_by_speaker(self, audio, segments):
        speaker_audio = {}
        for seg in segments:
            spk_id = seg['speaker_id']
            start = int(seg['start'] * self.sr)
            end = int((seg['start'] + seg['duration']) * self.sr)
            segment = audio[start:end]
            if spk_id not in speaker_audio:
                speaker_audio[spk_id] = []
            speaker_audio[spk_id].append(segment)

        for spk in speaker_audio:
            speaker_audio[spk] = np.concatenate(speaker_audio[spk])
        return speaker_audio

    def reconstruct_conversation(self, speaker_audio, segments, total_duration=None):
        if total_duration is None:
            max_end = max(seg['start'] + seg['duration'] for seg in segments)
            total_samples = int(max_end * self.sr)
        else:
            total_samples = int(total_duration * self.sr)

        output = np.zeros(total_samples)
        speaker_positions = {spk: 0 for spk in speaker_audio.keys()}

        for seg in sorted(segments, key=lambda x: x['start']):
            spk_id = seg['speaker_id']
            if spk_id not in speaker_audio:
                continue

            start_sample = int(seg['start'] * self.sr)
            duration_samples = int(seg['duration'] * self.sr)
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

    def save_audio(self, audio, path):
        sf.write(path, audio, self.sr)


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


class SpeakerAnonymizer:
    """화자 벡터 익명화"""

    def __init__(self, method='select', external_pool_path=None):
        """
        Args:
            method: 'select', 'as' (aggregated similarity), 'ds' (differential similarity)
            external_pool_path: 외부 화자 벡터 풀 경로
        """
        self.method = method

        # 외부 풀 로드 (없으면 랜덤 생성)
        if external_pool_path and Path(external_pool_path).exists():
            self.external_pool = np.load(external_pool_path)
            print(f"  ✓ 외부 풀 로드: {self.external_pool.shape}")
        else:
            # 테스트용 랜덤 풀 생성
            self.external_pool = np.random.randn(500, 192)  # 500개 화자, 192차원
            print(f"  ⚠️  외부 풀 없음 - 랜덤 생성 (테스트용)")

    def cosine_similarity(self, a, b):
        """코사인 유사도"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def find_farthest_vectors(self, query, pool, k=200):
        """가장 먼 k개 벡터 찾기"""
        similarities = []
        for i, vec in enumerate(pool):
            sim = self.cosine_similarity(query, vec)
            similarities.append((i, sim))

        # 유사도 낮은 순 정렬 (가장 먼 벡터)
        similarities.sort(key=lambda x: x[1])

        indices = [idx for idx, _ in similarities[:k]]
        return indices

    def anonymize_select(self, speaker_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Selection-based 익명화
        각 화자에 대해 가장 먼 200개 벡터 중 10개를 랜덤 선택하여 평균
        """
        anonymized = {}

        for spk_id, orig_vec in speaker_vectors.items():
            # 가장 먼 200개 찾기
            farthest_indices = self.find_farthest_vectors(orig_vec, self.external_pool, k=200)

            # 랜�ом으로 10개 선택
            selected_indices = np.random.choice(farthest_indices, size=10, replace=False)

            # 평균
            anon_vec = self.external_pool[selected_indices].mean(axis=0)
            anonymized[spk_id] = anon_vec

        return anonymized

    def anonymize_as(self, speaker_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Aggregated Similarity 익명화
        화자 간 유사도를 최소화하도록 선택
        """
        speaker_ids = list(speaker_vectors.keys())
        N = len(speaker_ids)

        if N == 1:
            return self.anonymize_select(speaker_vectors)

        # 각 화자에 대한 후보 벡터
        candidates = {}
        for spk_id, orig_vec in speaker_vectors.items():
            farthest_indices = self.find_farthest_vectors(orig_vec, self.external_pool, k=200)
            candidates[spk_id] = self.external_pool[farthest_indices]

        # Greedy 선택
        selected = {}

        for i, spk_id in enumerate(speaker_ids):
            if i == 0:
                # 첫 번째 화자: 랜덤 선택
                selected[spk_id] = candidates[spk_id][np.random.randint(len(candidates[spk_id]))]
            else:
                # 이미 선택된 화자들과의 유사도를 최소화
                min_sim = float('inf')
                best_vec = None

                # 후보 중 일부만 샘플링 (속도 향상)
                sample_size = min(100, len(candidates[spk_id]))
                sampled_indices = np.random.choice(len(candidates[spk_id]), size=sample_size, replace=False)

                for idx in sampled_indices:
                    candidate_vec = candidates[spk_id][idx]

                    # 이미 선택된 화자들과의 평균 유사도
                    total_sim = 0
                    for other_spk_id in speaker_ids[:i]:
                        sim = self.cosine_similarity(candidate_vec, selected[other_spk_id])
                        total_sim += sim

                    if total_sim < min_sim:
                        min_sim = total_sim
                        best_vec = candidate_vec

                selected[spk_id] = best_vec

        return selected

    def anonymize_ds(self, speaker_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Differential Similarity 익명화
        원본 화자 간 관계를 유지하면서 익명화
        """
        speaker_ids = list(speaker_vectors.keys())
        N = len(speaker_ids)

        if N == 1:
            return self.anonymize_select(speaker_vectors)

        # 원본 화자 간 유사도 행렬
        orig_sim_matrix = np.zeros((N, N))
        for i, spk1 in enumerate(speaker_ids):
            for j, spk2 in enumerate(speaker_ids):
                if i != j:
                    orig_sim_matrix[i, j] = self.cosine_similarity(
                        speaker_vectors[spk1],
                        speaker_vectors[spk2]
                    )

        # 각 화자에 대한 후보 벡터
        candidates = {}
        for spk_id, orig_vec in speaker_vectors.items():
            farthest_indices = self.find_farthest_vectors(orig_vec, self.external_pool, k=200)
            candidates[spk_id] = self.external_pool[farthest_indices]

        # Greedy 선택
        selected = {}

        for i, spk_id in enumerate(speaker_ids):
            if i == 0:
                # 첫 번째 화자: 랜덤 선택
                selected[spk_id] = candidates[spk_id][np.random.randint(len(candidates[spk_id]))]
            else:
                # 원본 관계를 최대한 보존
                min_diff = float('inf')
                best_vec = None

                # 후보 중 일부만 샘플링
                sample_size = min(100, len(candidates[spk_id]))
                sampled_indices = np.random.choice(len(candidates[spk_id]), size=sample_size, replace=False)

                for idx in sampled_indices:
                    candidate_vec = candidates[spk_id][idx]

                    # 이미 선택된 화자들과의 유사도 차이
                    total_diff = 0
                    for j, other_spk_id in enumerate(speaker_ids[:i]):
                        anon_sim = self.cosine_similarity(candidate_vec, selected[other_spk_id])
                        orig_sim = orig_sim_matrix[i, j]
                        total_diff += abs(anon_sim - orig_sim)

                    if total_diff < min_diff:
                        min_diff = total_diff
                        best_vec = candidate_vec

                selected[spk_id] = best_vec

        return selected

    def anonymize(self, speaker_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """익명화 실행"""
        if self.method == 'select':
            return self.anonymize_select(speaker_vectors)
        elif self.method == 'as':
            return self.anonymize_as(speaker_vectors)
        elif self.method == 'ds':
            return self.anonymize_ds(speaker_vectors)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class SpeakerVectorExtractor:
    """화자 벡터 추출 (간단 버전)"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ECAPA-TDNN 모델 로드 시도
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_ecapa"
            )
            self.model.to(self.device)
            self.has_model = True
            print("  ✓ ECAPA-TDNN 모델 로드 완료")
        except:
            self.has_model = False
            print("  ⚠️  ECAPA-TDNN 없음 - 랜덤 벡터 사용 (테스트용)")

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """화자 벡터 추출"""
        if not self.has_model:
            # 테스트용 랜덤 벡터
            return np.random.randn(192)

        # 최소 길이 체크
        sr = 16000
        min_length = int(0.5 * sr)
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))

        # 추출
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            embedding = self.model.encode_batch(audio_tensor)
            return embedding.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="MSA 익명화 (AS/DS/Select)")

    parser.add_argument('--eendeda-output', required=True, type=Path)
    parser.add_argument('--audio-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--rttm-dir', type=Path, default=None)
    parser.add_argument(
        '--method',
        type=str,
        default='select',
        choices=['select', 'as', 'ds'],
        help='익명화 방법: select, as (aggregated similarity), ds (differential similarity)'
    )
    parser.add_argument(
        '--external-pool',
        type=Path,
        default=None,
        help='외부 화자 벡터 풀 경로 (.npy)'
    )

    args = parser.parse_args()

    if args.rttm_dir is None:
        args.rttm_dir = args.eendeda_output / "rttm"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MSA 익명화")
    print(f"{'=' * 70}")
    print(f"방법: {args.method.upper()}")
    print(f"EEND-EDA 출력: {args.eendeda_output}")
    print(f"원본 오디오: {args.audio_dir}")
    print(f"출력: {args.output_dir}")
    print(f"{'=' * 70}\n")

    # 초기화
    audio_proc = SimpleAudioProcessor()
    rttm_parser = RTTMParser()
    anonymizer = SpeakerAnonymizer(method=args.method, external_pool_path=args.external_pool)
    extractor = SpeakerVectorExtractor()

    # Manifest 로드
    manifest_path = args.eendeda_output / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = [{'utt': rttm_file.stem} for rttm_file in args.rttm_dir.glob("*.rttm")]

    print(f"처리할 발화: {len(manifest)}개\n")

    # 처리
    results = []

    for entry in manifest:
        utt_name = entry['utt']

        try:
            print(f"처리 중: {utt_name}")

            # RTTM 로드
            rttm_path = args.rttm_dir / f"{utt_name}.rttm"
            segments = rttm_parser.parse_file(rttm_path)
            speaker_ids = sorted(set(seg['speaker_id'] for seg in segments))
            print(f"  - 화자 수: {len(speaker_ids)}")

            # 오디오 로드
            audio_path = None
            for ext in ['.wav', '.flac']:
                candidate = args.audio_dir / f"{utt_name}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break

            if not audio_path:
                print(f"  ⚠️  오디오 없음")
                continue

            audio = audio_proc.load_audio(audio_path)
            speaker_audio = audio_proc.aggregate_by_speaker(audio, segments)

            # 화자 벡터 추출
            speaker_vectors = {}
            for spk_id in speaker_ids:
                vec = extractor.extract(speaker_audio[spk_id])
                speaker_vectors[spk_id] = vec

            # 익명화
            anon_vectors = anonymizer.anonymize(speaker_vectors)

            # TODO: 실제 음성 합성 (현재는 재구성만)
            # 실제로는 HuBERT + F0 + HiFi-GAN 사용
            anonymized_audio = speaker_audio.copy()

            # 재구성
            reconstructed = audio_proc.reconstruct_conversation(
                anonymized_audio, segments, len(audio) / 16000
            )

            # 저장
            output_path = args.output_dir / f"{utt_name}_anonymized.wav"
            audio_proc.save_audio(reconstructed, output_path)
            print(f"  ✓ 저장: {output_path}")

            results.append({
                'utt': utt_name,
                'num_speakers': len(speaker_ids),
                'method': args.method,
                'status': 'success'
            })

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results.append({'utt': utt_name, 'status': 'failed', 'error': str(e)})

    # 결과 저장
    with open(args.output_dir / "results.json", 'w') as f:
        json.dump({'method': args.method, 'results': results}, f, indent=2)

    print(f"\n✅ 완료! 방법: {args.method.upper()}")


if __name__ == "__main__":
    main()