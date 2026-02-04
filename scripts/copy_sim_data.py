#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sim_data의 Kaldi 데이터를 프로젝트로 복사
"""

import argparse
import shutil
import json
from pathlib import Path


def copy_kaldi_data(
        sim_data_root: Path,
        project_root: Path,
        n_spk: str,
        ratio: str
):
    """
    특정 설정의 Kaldi 데이터를 프로젝트로 복사

    Args:
        sim_data_root: sim_data 루트 (예: /home/ktaemin/sim_data)
        project_root: 프로젝트 루트 (예: /home/ktaemin/tm_project)
        n_spk: 화자 수 (예: "2spk", "3spk", "4spk")
        ratio: 중첩률 (예: "10", "20", "30")
    """

    # 소스 경로
    source_kaldi = sim_data_root / "kaldi" / "all" / n_spk / ratio

    if not source_kaldi.exists():
        raise ValueError(f"Kaldi 디렉토리 없음: {source_kaldi}")

    print(f"\n{'=' * 60}")
    print(f"Kaldi 데이터 복사")
    print(f"{'=' * 60}")
    print(f"소스: {source_kaldi}")

    # 목적지 경로
    dest_kaldi = project_root / "data" / "audio_kaldi_format"
    dest_kaldi.mkdir(parents=True, exist_ok=True)

    print(f"목적지: {dest_kaldi}")
    print(f"{'=' * 60}\n")

    # Kaldi 파일 복사
    kaldi_files = ["wav.scp", "utt2spk", "spk2utt", "segments", "reco2dur"]
    copied_files = []

    for filename in kaldi_files:
        source_file = source_kaldi / filename
        dest_file = dest_kaldi / filename

        if source_file.exists():
            shutil.copy2(source_file, dest_file)

            # 파일 크기 및 라인 수 확인
            if filename.endswith('.scp') or filename in ['utt2spk', 'spk2utt']:
                with open(dest_file, 'r') as f:
                    lines = len(f.readlines())
                print(f"  ✓ {filename}: {lines} 라인")
            else:
                print(f"  ✓ {filename}")

            copied_files.append(filename)
        else:
            print(f"  ⚠️  {filename} 없음 (선택적)")

    # RTTM 파일 복사
    source_rttm = sim_data_root / "rttm" / "all" / n_spk / ratio
    dest_rttm = project_root / "data" / "ground_truth_rttm"
    dest_rttm.mkdir(parents=True, exist_ok=True)

    print(f"\nRTTM 파일 복사 중...")
    print(f"  소스: {source_rttm}")
    print(f"  목적지: {dest_rttm}")

    if source_rttm.exists():
        rttm_files = list(source_rttm.glob("*.rttm"))

        for rttm_file in rttm_files:
            dest_file = dest_rttm / rttm_file.name
            shutil.copy2(rttm_file, dest_file)

        print(f"  ✓ {len(rttm_files)}개 RTTM 파일 복사 완료")
    else:
        print(f"  ⚠️  RTTM 디렉토리 없음: {source_rttm}")

    # 메타데이터 생성
    with open(dest_kaldi / "wav.scp", 'r') as f:
        num_utterances = len(f.readlines())

    with open(dest_kaldi / "spk2utt", 'r') as f:
        num_speakers = len(f.readlines())

    metadata = {
        "source": str(sim_data_root),
        "config": {
            "n_spk": n_spk,
            "ratio": ratio
        },
        "num_utterances": num_utterances,
        "num_speakers": num_speakers,
        "copied_files": copied_files
    }

    with open(dest_kaldi / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"✅ 완료!")
    print(f"{'=' * 60}")
    print(f"설정: {n_spk} / {ratio}% 중첩")
    print(f"발화 수: {num_utterances}")
    print(f"화자 수: {num_speakers}")
    print(f"{'=' * 60}\n")

    # 다음 단계 안내
    print(f"📋 다음 단계:")
    print(f"")
    print(f"python infer_4_anon.py \\")
    print(f"    --config config/eendeda_config.yaml \\")
    print(f"    --infer-data-dir data/audio_kaldi_format \\")
    print(f"    --models-path models/eendeda/checkpoints \\")
    print(f"    --epochs \"28\" \\")
    print(f"    --out-dir output/eendeda \\")
    print(f"    --gpu 0")
    print(f"")

    return metadata


def list_available_configs(sim_data_root: Path):
    """사용 가능한 설정 목록 출력"""

    kaldi_root = sim_data_root / "kaldi" / "all"

    if not kaldi_root.exists():
        print(f"❌ Kaldi 루트 없음: {kaldi_root}")
        return

    print(f"\n{'=' * 60}")
    print(f"사용 가능한 데이터 설정")
    print(f"{'=' * 60}\n")

    configs = []

    for nspk_dir in sorted(kaldi_root.glob("*spk")):
        n_spk = nspk_dir.name

        for ratio_dir in sorted(nspk_dir.glob("*")):
            if ratio_dir.is_dir():
                ratio = ratio_dir.name

                # 파일 개수 확인
                wav_scp = ratio_dir / "wav.scp"
                if wav_scp.exists():
                    with open(wav_scp, 'r') as f:
                        num_utts = len(f.readlines())
                else:
                    num_utts = 0

                # RTTM 개수 확인
                rttm_dir = sim_data_root / "rttm" / "all" / n_spk / ratio
                if rttm_dir.exists():
                    num_rttm = len(list(rttm_dir.glob("*.rttm")))
                else:
                    num_rttm = 0

                configs.append({
                    'n_spk': n_spk,
                    'ratio': ratio,
                    'num_utts': num_utts,
                    'num_rttm': num_rttm,
                    'kaldi_path': ratio_dir,
                    'rttm_path': rttm_dir
                })

                print(f"{n_spk} / {ratio}% 중첩:")
                print(f"  - 발화 수: {num_utts}")
                print(f"  - RTTM 파일: {num_rttm}개")
                print(f"  - Kaldi 경로: {ratio_dir}")
                print(f"  - RTTM 경로: {rttm_dir}")
                print()

    print(f"{'=' * 60}")
    print(f"총 {len(configs)}개 설정")
    print(f"{'=' * 60}\n")

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="sim_data의 Kaldi 데이터를 프로젝트로 복사",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:

1. 사용 가능한 설정 확인:
   python copy_sim_data.py --sim-data /home/ktaemin/sim_data --list

2. 특정 설정 복사:
   python copy_sim_data.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project \\
       --n-spk 2spk \\
       --ratio 10

3. 다른 설정으로 복사:
   python copy_sim_data.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project \\
       --n-spk 3spk \\
       --ratio 20
        """
    )

    parser.add_argument(
        '--sim-data',
        required=True,
        type=Path,
        help='sim_data 루트 디렉토리'
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path('/home/ktaemin/tm_project'),
        help='프로젝트 루트 디렉토리'
    )

    parser.add_argument(
        '--n-spk',
        type=str,
        choices=['2spk', '3spk', '4spk'],
        help='화자 수'
    )

    parser.add_argument(
        '--ratio',
        type=str,
        choices=['10', '20', '30'],
        help='중첩률 (%)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='사용 가능한 설정 목록만 출력'
    )

    args = parser.parse_args()

    # 목록 출력
    if args.list:
        list_available_configs(args.sim_data)
        return

    # 설정 확인
    if not args.n_spk or not args.ratio:
        print("❌ --n-spk와 --ratio를 모두 지정해야 합니다.")
        print("   예: --n-spk 2spk --ratio 10")
        print("")
        print("또는 --list로 사용 가능한 설정을 확인하세요.")
        return

    # 복사 실행
    copy_kaldi_data(
        sim_data_root=args.sim_data,
        project_root=args.project_root,
        n_spk=args.n_spk,
        ratio=args.ratio
    )


if __name__ == "__main__":
    main()