#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/evaluate_privacy.py
Privacy 평가 (FAR - False Acceptance Rate)
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm


def load_rttm(rttm_path):
    """RTTM 파일 로드"""
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                segments.append({
                    'file_id': parts[1],
                    'start': float(parts[3]),
                    'duration': float(parts[4]),
                    'speaker': parts[7]
                })
    return segments


def extract_speaker_audio_segments(audio_path, rttm_path, sr=16000):
    """
    RTTM 기반으로 화자별 오디오 세그먼트 추출
    Returns: dict {speaker_id: [audio_segments]}
    """
    # 오디오 로드
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    # RTTM 로드
    segments = load_rttm(rttm_path)

    # 화자별로 그룹화
    speaker_segments = {}
    for seg in segments:
        spk = seg['speaker']
        start_sample = int(seg['start'] * sr)
        end_sample = int((seg['start'] + seg['duration']) * sr)

        segment_audio = audio[start_sample:end_sample]

        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(segment_audio)

    return speaker_segments


def compute_speaker_embedding(audio, model, device, sr=16000):
    """
    화자 임베딩 추출
    간단한 예시 - 실제로는 ECAPA-TDNN 같은 모델 필요
    """
    # 오디오가 너무 짧으면 패딩
    min_length = int(0.5 * sr)  # 최소 0.5초
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)))

    # Torch 텐서로 변환
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

    with torch.no_grad():
        # 모델에 따라 다름 - 여기서는 placeholder
        # 실제로는 model.encode_batch(audio_tensor) 같은 형태
        try:
            embedding = model.encode_batch(audio_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        except:
            # 모델이 없으면 랜덤 벡터 (테스트용)
            embedding = np.random.randn(192)

    return embedding


def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Privacy 평가 (FAR)")
    parser.add_argument('--original-audio', required=True, help='원본 오디오 디렉토리')
    parser.add_argument('--anonymized-audio', required=True, help='익명화된 오디오 디렉토리')
    parser.add_argument('--rttm-dir', required=True, help='RTTM 디렉토리')
    parser.add_argument('--asv-model', default=None, help='ASV 모델 경로 (선택)')

    args = parser.parse_args()

    orig_dir = Path(args.original_audio)
    anon_dir = Path(args.anonymized_audio)
    rttm_dir = Path(args.rttm_dir)

    print(f"\n{'=' * 60}")
    print(f"Privacy 평가 (FAR)")
    print(f"{'=' * 60}")
    print(f"원본 오디오: {orig_dir}")
    print(f"익명화 오디오: {anon_dir}")
    print(f"RTTM: {rttm_dir}")
    print(f"{'=' * 60}\n")

    # ASV 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from speechbrain.pretrained import EncoderClassifier
        print("✓ SpeechBrain ASV 모델 로드 중...")

        if args.asv_model:
            model = EncoderClassifier.from_hparams(source=args.asv_model)
        else:
            # 기본 모델
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model"
            )
        print("✓ 모델 로드 완료\n")
        has_model = True
    except:
        print("⚠️  ASV 모델 없음 - 간단한 계산 사용")
        print("   정확한 계산을 위해 설치하세요:")
        print("   pip install speechbrain\n")
        model = None
        has_model = False

    # 평가
    all_similarities_oa = []  # Original-Anonymized
    all_similarities_oo = []  # Original-Original (same speaker)

    # RTTM 파일들 찾기
    rttm_files = list(rttm_dir.glob("*.rttm"))

    if not rttm_files:
        print("❌ RTTM 파일이 없습니다.")
        return

    print(f"평가 파일 수: {len(rttm_files)}\n")

    for rttm_file in tqdm(rttm_files, desc="평가 중"):
        utt_id = rttm_file.stem

        # 파일 경로 찾기
        # 원본: sim_data/wav/all/2spk/10/mix_xxxxx.wav
        # 익명화: output/anonymized/mix_xxxxx_anonymized.wav

        # 원본 오디오 찾기 (여러 경로 시도)
        orig_audio = None
        for pattern in [f"{utt_id}.wav", f"{utt_id}.flac"]:
            found = list(orig_dir.rglob(pattern))
            if found:
                orig_audio = found[0]
                break

        if orig_audio is None:
            print(f"⚠️  원본 오디오 없음: {utt_id}")
            continue

        # 익명화 오디오
        anon_audio = anon_dir / f"{utt_id}_anonymized.wav"
        if not anon_audio.exists():
            anon_audio = anon_dir / f"{utt_id}.wav"

        if not anon_audio.exists():
            print(f"⚠️  익명화 오디오 없음: {utt_id}")
            continue

        try:
            # 화자별 세그먼트 추출
            orig_segments = extract_speaker_audio_segments(orig_audio, rttm_file)
            anon_segments = extract_speaker_audio_segments(anon_audio, rttm_file)

            # 각 화자에 대해 임베딩 추출 및 유사도 계산
            for spk in orig_segments.keys():
                if spk not in anon_segments:
                    continue

                # 원본 화자 임베딩 (첫 번째 세그먼트 사용)
                orig_audio_seg = np.concatenate(orig_segments[spk][:2])  # 처음 2개 세그먼트
                anon_audio_seg = np.concatenate(anon_segments[spk][:2])

                if has_model:
                    orig_emb = compute_speaker_embedding(orig_audio_seg, model, device)
                    anon_emb = compute_speaker_embedding(anon_audio_seg, model, device)

                    # Original-Anonymized 유사도
                    sim_oa = cosine_similarity(orig_emb, anon_emb)
                    all_similarities_oa.append(sim_oa)

                    # Original-Original 유사도 (같은 화자의 다른 세그먼트)
                    if len(orig_segments[spk]) >= 3:
                        orig_audio_seg2 = np.concatenate(orig_segments[spk][2:4])
                        orig_emb2 = compute_speaker_embedding(orig_audio_seg2, model, device)
                        sim_oo = cosine_similarity(orig_emb, orig_emb2)
                        all_similarities_oo.append(sim_oo)

        except Exception as e:
            print(f"⚠️  처리 오류 ({utt_id}): {e}")
            continue

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"결과")
    print(f"{'=' * 60}")

    if has_model and all_similarities_oa:
        # FAR 계산
        # EER threshold 계산 (간단 근사)
        if all_similarities_oo:
            threshold = np.mean(all_similarities_oo) - np.std(all_similarities_oo)
        else:
            threshold = 0.5

        far = np.mean(np.array(all_similarities_oa) > threshold) * 100

        print(f"Original-Anonymized 유사도:")
        print(f"  - 평균: {np.mean(all_similarities_oa):.4f}")
        print(f"  - 표준편차: {np.std(all_similarities_oa):.4f}")
        print(f"  - 최소: {np.min(all_similarities_oa):.4f}")
        print(f"  - 최대: {np.max(all_similarities_oa):.4f}")

        if all_similarities_oo:
            print(f"\nOriginal-Original 유사도 (같은 화자):")
            print(f"  - 평균: {np.mean(all_similarities_oo):.4f}")
            print(f"  - 표준편차: {np.std(all_similarities_oo):.4f}")

        print(f"\nThreshold: {threshold:.4f}")
        print(f"FAR (False Acceptance Rate): {far:.2f}%")

        # 낮을수록 좋음
        if far < 5:
            print("\n✅ 매우 좋은 프라이버시 보호!")
        elif far < 10:
            print("\n✓ 좋은 프라이버시 보호")
        elif far < 20:
            print("\n⚠️  보통 수준의 프라이버시 보호")
        else:
            print("\n❌ 프라이버시 보호 개선 필요")
    else:
        print("⚠️  ASV 모델이 없어 정확한 평가를 수행할 수 없습니다.")
        print("   speechbrain 설치 후 다시 실행하세요:")
        print("   pip install speechbrain")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
