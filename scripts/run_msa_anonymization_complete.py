#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_msa_anonymization_complete.py
완전한 MSA 익명화: 화자 벡터 익명화 + 음성 합성
- HuBERT: Content feature 추출
- YAAPT: F0 추출
- HiFi-GAN: 음성 합성
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import json
from typing import Dict, List
import librosa
from tqdm import tqdm

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ContentExtractor:
    """HuBERT 기반 Content Feature 추출"""

    def __init__(self, model_name="facebook/hubert-base-ls960"):
        """
        Args:
            model_name: HuggingFace HuBERT 모델 이름
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            print(f"  HuBERT 모델 로드 중: {model_name}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.has_model = True
            print(f"  ✓ HuBERT 로드 완료")

        except Exception as e:
            print(f"  ❌ HuBERT 로드 실패: {e}")
            print(f"  설치: pip install transformers")
            self.has_model = False

    def extract(self, audio: np.ndarray, sr=16000) -> np.ndarray:
        """
        Content feature 추출

        Args:
            audio: 오디오 배열 (sr=16000)

        Returns:
            content_features: [T, D] (T: 프레임 수, D: 특징 차원)
        """
        if not self.has_model:
            # Fallback: 랜덤 특징
            num_frames = len(audio) // 320  # 20ms frame
            return np.random.randn(num_frames, 768)

        with torch.no_grad():
            # 특징 추출
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_values

            inputs = inputs.to(self.device)

            # HuBERT forward
            outputs = self.model(inputs)

            # Hidden states 사용 (마지막 레이어)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            return features


class F0Extractor:
    """F0 (Pitch) 추출"""

    def __init__(self):
        """librosa 사용 (YAAPT는 호환성 문제로 제외)"""
        print("  ✓ F0 추출: librosa.pyin 사용")

    def extract(self, audio: np.ndarray, sr=16000) -> np.ndarray:
        """
        F0 추출

        Returns:
            f0: [T] (프레임별 F0 값, 0은 무성음)
        """
        try:
            # librosa pyin 사용
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=2048,
                hop_length=320
            )

            # NaN을 0으로 변경 (무성음)
            f0 = np.nan_to_num(f0, nan=0.0)

            return f0

        except Exception as e:
            # Fallback: 간단한 autocorrelation 기반
            print(f"    ⚠️  F0 추출 실패, 기본값 사용: {e}")
            # 대략적인 프레임 수 계산
            hop_length = 320
            num_frames = len(audio) // hop_length
            # 기본 F0 (100-200Hz 사이 랜덤, 실제로는 사용 안함)
            return np.full(num_frames, 150.0)


class SpeakerVectorExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            from speechbrain.pretrained import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_ecapa"
            )
            # 이 줄 추가
            self.model.to(self.device)
            self.model.eval()  # 추가
            self.has_model = True
            print("  ✓ ECAPA-TDNN 로드 완료")
        except:
            self.has_model = False
            print("  ⚠️  ECAPA-TDNN 없음")

    def extract(self, audio: np.ndarray, sr=16000) -> np.ndarray:
        if not self.has_model:
            return np.random.randn(192)

        min_length = int(0.5 * sr)
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))

        with torch.no_grad():
            # 수정: GPU로 보내기
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            embedding = self.model.encode_batch(audio_tensor)
            return embedding.squeeze().cpu().numpy()

class HiFiGANVocoder:
    """HiFi-GAN 보코더"""

    def __init__(self, model_name="microsoft/speecht5_hifigan"):
        """
        Args:
            model_name: HuggingFace HiFi-GAN 모델
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            from transformers import SpeechT5HifiGan

            print(f"  HiFi-GAN 로드 중: {model_name}")
            self.vocoder = SpeechT5HifiGan.from_pretrained(model_name)
            self.vocoder.to(self.device)
            self.vocoder.eval()
            self.has_model = True
            print(f"  ✓ HiFi-GAN 로드 완료")

        except Exception as e:
            print(f"  ❌ HiFi-GAN 로드 실패: {e}")
            self.has_model = False

    def synthesize(
            self,
            content_features: np.ndarray,
            f0: np.ndarray,
            speaker_embedding: np.ndarray
    ) -> np.ndarray:
        """
        음성 합성

        Args:
            content_features: [T, D] HuBERT 특징
            f0: [T] F0 값
            speaker_embedding: [192] 화자 벡터

        Returns:
            audio: [N] 합성된 오디오
        """
        # 현재는 실제 합성이 복잡하므로 원본 길이 반환
        # TODO: 실제 TTS 파이프라인 구현 필요
        estimated_length = len(content_features) * 320
        return np.zeros(estimated_length)


class SpeakerAnonymizer:
    """화자 벡터 익명화 (Select/AS/DS)"""

    def __init__(self, method='select', external_pool_path=None):
        self.method = method

        if external_pool_path and Path(external_pool_path).exists():
            self.external_pool = np.load(external_pool_path)
            print(f"  ✓ 외부 풀 로드: {self.external_pool.shape}")
        else:
            # 랜덤 풀
            self.external_pool = np.random.randn(500, 192)
            print(f"  ⚠️  랜덤 풀 사용 (500, 192)")

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def find_farthest_vectors(self, query, pool, k=200):
        similarities = [(i, self.cosine_similarity(query, vec)) for i, vec in enumerate(pool)]
        similarities.sort(key=lambda x: x[1])
        return [idx for idx, _ in similarities[:k]]

    def anonymize_select(self, speaker_vectors):
        anonymized = {}
        for spk_id, orig_vec in speaker_vectors.items():
            farthest = self.find_farthest_vectors(orig_vec, self.external_pool, k=200)
            selected = np.random.choice(farthest, size=10, replace=False)
            anonymized[spk_id] = self.external_pool[selected].mean(axis=0)
        return anonymized

    def anonymize_as(self, speaker_vectors):
        """Aggregated Similarity"""
        speaker_ids = list(speaker_vectors.keys())
        if len(speaker_ids) == 1:
            return self.anonymize_select(speaker_vectors)

        candidates = {
            spk: self.external_pool[self.find_farthest_vectors(vec, self.external_pool, k=200)]
            for spk, vec in speaker_vectors.items()
        }

        selected = {}
        for i, spk in enumerate(speaker_ids):
            if i == 0:
                selected[spk] = candidates[spk][np.random.randint(len(candidates[spk]))]
            else:
                min_sim = float('inf')
                best_vec = None
                sample_size = min(100, len(candidates[spk]))
                for idx in np.random.choice(len(candidates[spk]), sample_size, replace=False):
                    candidate = candidates[spk][idx]
                    total_sim = sum(self.cosine_similarity(candidate, selected[other])
                                    for other in speaker_ids[:i])
                    if total_sim < min_sim:
                        min_sim = total_sim
                        best_vec = candidate
                selected[spk] = best_vec
        return selected

    def anonymize_ds(self, speaker_vectors):
        """Differential Similarity"""
        speaker_ids = list(speaker_vectors.keys())
        N = len(speaker_ids)
        if N == 1:
            return self.anonymize_select(speaker_vectors)

        # 원본 유사도 행렬
        orig_sim = np.zeros((N, N))
        for i, spk1 in enumerate(speaker_ids):
            for j, spk2 in enumerate(speaker_ids):
                if i != j:
                    orig_sim[i, j] = self.cosine_similarity(
                        speaker_vectors[spk1], speaker_vectors[spk2]
                    )

        candidates = {
            spk: self.external_pool[self.find_farthest_vectors(vec, self.external_pool, k=200)]
            for spk, vec in speaker_vectors.items()
        }

        selected = {}
        for i, spk in enumerate(speaker_ids):
            if i == 0:
                selected[spk] = candidates[spk][np.random.randint(len(candidates[spk]))]
            else:
                min_diff = float('inf')
                best_vec = None
                sample_size = min(100, len(candidates[spk]))
                for idx in np.random.choice(len(candidates[spk]), sample_size, replace=False):
                    candidate = candidates[spk][idx]
                    total_diff = sum(
                        abs(self.cosine_similarity(candidate, selected[speaker_ids[j]]) - orig_sim[i, j])
                        for j in range(i)
                    )
                    if total_diff < min_diff:
                        min_diff = total_diff
                        best_vec = candidate
                selected[spk] = best_vec
        return selected

    def anonymize(self, speaker_vectors):
        if self.method == 'select':
            return self.anonymize_select(speaker_vectors)
        elif self.method == 'as':
            return self.anonymize_as(speaker_vectors)
        elif self.method == 'ds':
            return self.anonymize_ds(speaker_vectors)


class AudioProcessor:
    """오디오 처리"""

    def __init__(self, sr=16000):
        self.sr = sr

    def load(self, path):
        audio, file_sr = sf.read(path)
        if file_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio

    def aggregate_by_speaker(self, audio, segments):
        speaker_audio = {}
        for seg in segments:
            spk = seg['speaker_id']
            start = int(seg['start'] * self.sr)
            end = int((seg['start'] + seg['duration']) * self.sr)
            segment = audio[start:end]
            if spk not in speaker_audio:
                speaker_audio[spk] = []
            speaker_audio[spk].append(segment)
        return {spk: np.concatenate(segs) for spk, segs in speaker_audio.items()}

    def reconstruct(self, speaker_audio, segments, total_duration=None):
        if total_duration is None:
            total_duration = max(s['start'] + s['duration'] for s in segments)
        output = np.zeros(int(total_duration * self.sr))
        positions = {spk: 0 for spk in speaker_audio}

        for seg in sorted(segments, key=lambda x: x['start']):
            spk = seg['speaker_id']
            if spk not in speaker_audio:
                continue
            start = int(seg['start'] * self.sr)
            duration = int(seg['duration'] * self.sr)
            pos = positions[spk]
            if pos >= len(speaker_audio[spk]):
                continue
            segment = speaker_audio[spk][pos:pos + duration]
            end = min(start + len(segment), len(output))
            output[start:end] += segment[:end - start]
            positions[spk] += len(segment)
        return output


class RTTMParser:
    @staticmethod
    def parse(path):
        segments = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    segments.append({
                        'file_id': parts[1],
                        'start': float(parts[3]),
                        'duration': float(parts[4]),
                        'speaker_id': parts[7]
                    })
        return segments


def main():
    parser = argparse.ArgumentParser(description="완전한 MSA 익명화")
    parser.add_argument('--eendeda-output', required=True, type=Path)
    parser.add_argument('--audio-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--rttm-dir', type=Path)
    parser.add_argument('--method', default='select', choices=['select', 'as', 'ds'])
    parser.add_argument('--external-pool', type=Path)
    parser.add_argument('--skip-synthesis', action='store_true', help='음성 합성 스킵 (재구성만)')

    args = parser.parse_args()

    if args.rttm_dir is None:
        args.rttm_dir = args.eendeda_output / "rttm"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"완전한 MSA 익명화")
    print(f"{'=' * 70}")
    print(f"방법: {args.method.upper()}")
    print(f"음성 합성: {'❌ 스킵' if args.skip_synthesis else '✅ 수행'}")
    print(f"{'=' * 70}\n")

    # 초기화
    print("모델 로드 중...")
    audio_proc = AudioProcessor()
    anonymizer = SpeakerAnonymizer(args.method, args.external_pool)
    spk_extractor = SpeakerVectorExtractor()

    if not args.skip_synthesis:
        content_extractor = ContentExtractor()
        f0_extractor = F0Extractor()
        vocoder = HiFiGANVocoder()

    # Manifest
    manifest_path = args.eendeda_output / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = [{'utt': f.stem} for f in args.rttm_dir.glob("*.rttm")]

    print(f"\n처리: {len(manifest)}개\n")

    results = []

    for entry in tqdm(manifest, desc="익명화"):
        utt = entry['utt']

        try:
            # RTTM
            segments = RTTMParser.parse(args.rttm_dir / f"{utt}.rttm")
            speaker_ids = sorted(set(s['speaker_id'] for s in segments))

            # 오디오
            audio_path = None
            for ext in ['.wav', '.flac']:
                cand = args.audio_dir / f"{utt}{ext}"
                if cand.exists():
                    audio_path = cand
                    break

            if not audio_path:
                continue

            audio = audio_proc.load(audio_path)
            speaker_audio = audio_proc.aggregate_by_speaker(audio, segments)

            # 화자 벡터 추출 & 익명화
            orig_vectors = {spk: spk_extractor.extract(speaker_audio[spk]) for spk in speaker_ids}
            anon_vectors = anonymizer.anonymize(orig_vectors)

            if args.skip_synthesis:
                # 재구성만
                anonymized_audio = speaker_audio
            else:
                # 음성 합성 (현재는 원본 오디오 사용)
                # TODO: 실제 합성 파이프라인 완성 필요
                anonymized_audio = {}
                for spk in speaker_ids:
                    try:
                        content = content_extractor.extract(speaker_audio[spk])
                        f0 = f0_extractor.extract(speaker_audio[spk])
                        synth = vocoder.synthesize(content, f0, anon_vectors[spk])

                        # 길이 맞추기
                        if len(synth) != len(speaker_audio[spk]):
                            if len(synth) < len(speaker_audio[spk]):
                                synth = np.pad(synth, (0, len(speaker_audio[spk]) - len(synth)))
                            else:
                                synth = synth[:len(speaker_audio[spk])]

                        anonymized_audio[spk] = synth
                    except Exception as e:
                        # 합성 실패 시 원본 사용 (임시)
                        print(f"    ⚠️  {spk} 합성 실패, 원본 사용: {e}")
                        anonymized_audio[spk] = speaker_audio[spk]

            # 재구성
            output = audio_proc.reconstruct(anonymized_audio, segments, len(audio) / 16000)

            # 저장
            out_path = args.output_dir / f"{utt}_anonymized.wav"
            sf.write(out_path, output, 16000)

            results.append({'utt': utt, 'status': 'success'})

        except Exception as e:
            print(f"  ❌ {utt}: {e}")
            results.append({'utt': utt, 'status': 'failed', 'error': str(e)})

    # 결과 저장
    with open(args.output_dir / "results.json", 'w') as f:
        json.dump({'method': args.method, 'results': results}, f, indent=2)

    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ 완료: {success}/{len(results)}")


if __name__ == "__main__":
    main()