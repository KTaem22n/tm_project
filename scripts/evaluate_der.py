#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/evaluate_der.py
DER (Diarization Error Rate) 평가
"""

import argparse
from pathlib import Path
import sys


def parse_rttm(rttm_path):
    """
    간단한 RTTM 파싱
    Returns: list of (start, end, speaker_id)
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({
                    'start': start,
                    'end': start + duration,
                    'speaker': speaker
                })
    return segments


def compute_simple_der(ref_segments, hyp_segments, collar=0.25):
    """
    간단한 DER 계산

    DER = (False Alarm + Missed Speech + Speaker Error) / Total Reference Time
    """
    if not ref_segments:
        return 0.0

    # 전체 시간 계산
    total_ref_time = sum(seg['end'] - seg['start'] for seg in ref_segments)

    if total_ref_time == 0:
        return 0.0

    # 간단한 근사 계산
    # 실제로는 프레임 단위로 계산해야 하지만, 여기서는 세그먼트 기반 근사

    # Missed Speech: reference에는 있는데 hypothesis에 없는 시간
    missed = 0.0
    for ref_seg in ref_segments:
        ref_start = ref_seg['start']
        ref_end = ref_seg['end']

        # hypothesis와 겹치는 부분 찾기
        overlap = 0.0
        for hyp_seg in hyp_segments:
            hyp_start = max(ref_start, hyp_seg['start'])
            hyp_end = min(ref_end, hyp_seg['end'])
            if hyp_start < hyp_end:
                overlap += (hyp_end - hyp_start)

        segment_duration = ref_end - ref_start
        missed += max(0, segment_duration - overlap)

    # False Alarm: hypothesis에는 있는데 reference에 없는 시간
    false_alarm = 0.0
    for hyp_seg in hyp_segments:
        hyp_start = hyp_seg['start']
        hyp_end = hyp_seg['end']

        overlap = 0.0
        for ref_seg in ref_segments:
            ref_start = max(hyp_start, ref_seg['start'])
            ref_end = min(hyp_end, ref_seg['end'])
            if ref_start < ref_end:
                overlap += (ref_end - ref_start)

        segment_duration = hyp_end - hyp_start
        false_alarm += max(0, segment_duration - overlap)

    # Speaker Error: 화자가 틀린 시간 (간단 근사)
    speaker_error = 0.0
    for ref_seg in ref_segments:
        for hyp_seg in hyp_segments:
            # 시간이 겹치는 부분
            overlap_start = max(ref_seg['start'], hyp_seg['start'])
            overlap_end = min(ref_seg['end'], hyp_seg['end'])

            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                # 화자가 다르면 에러
                if ref_seg['speaker'] != hyp_seg['speaker']:
                    speaker_error += overlap_duration

    # DER 계산
    der = (missed + false_alarm + speaker_error) / total_ref_time

    return der * 100  # 퍼센트로 반환


def main():
    parser = argparse.ArgumentParser(description="DER 평가")
    parser.add_argument('--reference', required=True, help='Ground Truth RTTM 디렉토리')
    parser.add_argument('--hypothesis', required=True, help='예측 RTTM 디렉토리')
    parser.add_argument('--collar', type=float, default=0.25, help='Collar (초)')

    args = parser.parse_args()

    ref_dir = Path(args.reference)
    hyp_dir = Path(args.hypothesis)

    if not ref_dir.exists():
        print(f"❌ Reference 디렉토리 없음: {ref_dir}")
        return

    if not hyp_dir.exists():
        print(f"❌ Hypothesis 디렉토리 없음: {hyp_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"DER 평가")
    print(f"{'=' * 60}")
    print(f"Reference: {ref_dir}")
    print(f"Hypothesis: {hyp_dir}")
    print(f"Collar: {args.collar}초")
    print(f"{'=' * 60}\n")

    # pyannote 사용 가능한지 확인
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
        use_pyannote = True
        print("✓ pyannote.metrics 사용\n")
    except ImportError:
        use_pyannote = False
        print("⚠️  pyannote.metrics 없음 - 간단한 DER 계산 사용")
        print("   정확한 계산을 위해 설치하세요: pip install pyannote.core pyannote.metrics\n")

    # 평가
    total_der = 0.0
    count = 0

    for ref_file in sorted(ref_dir.glob("*.rttm")):
        utt_id = ref_file.stem
        hyp_file = hyp_dir / f"{utt_id}.rttm"

        if not hyp_file.exists():
            print(f"⚠️  예측 파일 없음: {utt_id}")
            continue

        if use_pyannote:
            # pyannote 사용
            # Reference 로드
            ref_annotation = Annotation()
            ref_segments = parse_rttm(ref_file)
            for seg in ref_segments:
                ref_annotation[Segment(seg['start'], seg['end'])] = seg['speaker']

            # Hypothesis 로드
            hyp_annotation = Annotation()
            hyp_segments = parse_rttm(hyp_file)
            for seg in hyp_segments:
                hyp_annotation[Segment(seg['start'], seg['end'])] = seg['speaker']

            # DER 계산
            metric = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
            der = metric(ref_annotation, hyp_annotation)

        else:
            # 간단한 계산
            ref_segments = parse_rttm(ref_file)
            hyp_segments = parse_rttm(hyp_file)
            der = compute_simple_der(ref_segments, hyp_segments, args.collar) / 100

        print(f"{utt_id}: DER = {der * 100:.2f}%")
        total_der += der
        count += 1

    if count > 0:
        avg_der = total_der / count
        print(f"\n{'=' * 60}")
        print(f"평균 DER: {avg_der * 100:.2f}%")
        print(f"평가 파일 수: {count}")
        print(f"{'=' * 60}\n")
    else:
        print("\n⚠️  평가할 파일이 없습니다.\n")


if __name__ == "__main__":
    main()
