#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sim_data 구조 전용 어댑터
기존 Kaldi 형식과 RTTM이 이미 있는 경우 사용
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SimDataAdapter:
    """
    sim_data 구조를 EEND-EDA 입력 형식으로 변환

    입력 구조:
        sim_data/
        ├── kaldi/all/
        │   ├── wav.scp
        │   ├── segments
        │   ├── utt2spk
        │   └── reco2dur
        ├── wav/all/{N}spk/{Ratio}/
        ├── rttm/all/{N}spk/{Ratio}/
        └── labels/all/{N}spk/{Ratio}/
    """

    def __init__(self, sim_data_root: Path):
        self.sim_data_root = Path(sim_data_root)
        self.kaldi_dir = self.sim_data_root / "kaldi" / "all"
        self.wav_dir = self.sim_data_root / "wav" / "all"
        self.rttm_dir = self.sim_data_root / "rttm" / "all"
        self.labels_dir = self.sim_data_root / "labels" / "all"
        self.meta_dir = self.sim_data_root / "meta" / "all"

        # 존재 확인
        if not self.kaldi_dir.exists():
            raise ValueError(f"Kaldi 디렉토리 없음: {self.kaldi_dir}")

        print(f"✓ sim_data 루트: {self.sim_data_root}")

    def get_available_configs(self) -> List[Dict]:
        """
        사용 가능한 데이터 설정 확인
        """
        configs = []

        for nspk_dir in self.wav_dir.glob("*spk"):
            n_spk = nspk_dir.name  # "2spk", "3spk", etc.

            for ratio_dir in nspk_dir.glob("*"):
                if ratio_dir.is_dir():
                    ratio = ratio_dir.name  # "10", "20", "30"

                    # wav 파일 개수 확인
                    wav_files = list(ratio_dir.glob("*.wav"))

                    # rttm 파일 개수 확인
                    rttm_path = self.rttm_dir / n_spk / ratio
                    rttm_files = list(rttm_path.glob("*.rttm")) if rttm_path.exists() else []

                    configs.append({
                        'n_spk': n_spk,
                        'ratio': ratio,
                        'num_wav': len(wav_files),
                        'num_rttm': len(rttm_files),
                        'wav_dir': ratio_dir,
                        'rttm_dir': rttm_path
                    })

        return sorted(configs, key=lambda x: (x['n_spk'], x['ratio']))

    def filter_kaldi_by_config(
            self,
            output_dir: Path,
            n_spk: Optional[str] = None,
            ratio: Optional[str] = None,
            max_utterances: Optional[int] = None
    ):
        """
        특정 설정(화자 수, 중첩률)에 맞는 데이터만 필터링

        Args:
            output_dir: 출력 디렉토리
            n_spk: 화자 수 필터 (예: "2spk", "3spk")
            ratio: 중첩률 필터 (예: "10", "20")
            max_utterances: 최대 발화 수 제한
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Kaldi 데이터 필터링")
        print(f"{'=' * 60}")
        print(f"필터: n_spk={n_spk}, ratio={ratio}, max={max_utterances}")

        # wav.scp 읽기
        wav_scp = {}
        with open(self.kaldi_dir / "wav.scp", 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    utt_id, path = parts
                    wav_scp[utt_id] = path

        print(f"원본 wav.scp: {len(wav_scp)}개")

        # utt2spk 읽기
        utt2spk = {}
        if (self.kaldi_dir / "utt2spk").exists():
            with open(self.kaldi_dir / "utt2spk", 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        utt2spk[parts[0]] = parts[1]

        # segments 읽기 (있는 경우)
        segments = {}
        if (self.kaldi_dir / "segments").exists():
            with open(self.kaldi_dir / "segments", 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # utt_id reco_id start end
                        segments[parts[0]] = {
                            'reco_id': parts[1],
                            'start': float(parts[2]),
                            'end': float(parts[3])
                        }

        # 필터링
        filtered_utts = []

        for utt_id, wav_path in wav_scp.items():
            # 경로 기반 필터링
            path_parts = Path(wav_path).parts

            # n_spk 필터
            if n_spk:
                if n_spk not in path_parts:
                    continue

            # ratio 필터
            if ratio:
                if ratio not in path_parts:
                    continue

            filtered_utts.append(utt_id)

            # 최대 개수 제한
            if max_utterances and len(filtered_utts) >= max_utterances:
                break

        print(f"필터링 후: {len(filtered_utts)}개")

        # 필터링된 데이터 저장
        with open(output_dir / "wav.scp", 'w') as f:
            for utt_id in filtered_utts:
                f.write(f"{utt_id} {wav_scp[utt_id]}\n")

        if utt2spk:
            with open(output_dir / "utt2spk", 'w') as f:
                for utt_id in filtered_utts:
                    if utt_id in utt2spk:
                        f.write(f"{utt_id} {utt2spk[utt_id]}\n")

        if segments:
            with open(output_dir / "segments", 'w') as f:
                for utt_id in filtered_utts:
                    if utt_id in segments:
                        seg = segments[utt_id]
                        f.write(f"{utt_id} {seg['reco_id']} {seg['start']} {seg['end']}\n")

        # spk2utt 생성
        spk2utt = {}
        for utt_id in filtered_utts:
            if utt_id in utt2spk:
                spk = utt2spk[utt_id]
                if spk not in spk2utt:
                    spk2utt[spk] = []
                spk2utt[spk].append(utt_id)

        with open(output_dir / "spk2utt", 'w') as f:
            for spk in sorted(spk2utt.keys()):
                utts = ' '.join(sorted(spk2utt[spk]))
                f.write(f"{spk} {utts}\n")

        # 메타데이터 저장
        metadata = {
            'source': str(self.sim_data_root),
            'filter': {
                'n_spk': n_spk,
                'ratio': ratio,
                'max_utterances': max_utterances
            },
            'num_utterances': len(filtered_utts),
            'num_speakers': len(spk2utt),
            'utterances': filtered_utts[:100]  # 처음 100개만 저장
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 출력: {output_dir}")
        print(f"  - wav.scp: {len(filtered_utts)}개")
        print(f"  - utt2spk: {len(filtered_utts)}개")
        print(f"  - spk2utt: {len(spk2utt)}개 화자")
        print(f"{'=' * 60}\n")

        return metadata

    def copy_corresponding_rttm(
            self,
            filtered_kaldi_dir: Path,
            output_rttm_dir: Path
    ):
        """
        필터링된 발화에 해당하는 RTTM 파일만 복사
        """
        output_rttm_dir = Path(output_rttm_dir)
        output_rttm_dir.mkdir(parents=True, exist_ok=True)

        print(f"RTTM 파일 복사 중...")

        # 필터링된 발화 목록 읽기
        with open(filtered_kaldi_dir / "wav.scp", 'r') as f:
            filtered_utts = [line.split()[0] for line in f]

        copied = 0
        not_found = []

        for utt_id in filtered_utts:
            # RTTM 파일 찾기
            found = False

            for nspk_dir in self.rttm_dir.glob("*spk"):
                for ratio_dir in nspk_dir.glob("*"):
                    rttm_file = ratio_dir / f"{utt_id}.rttm"
                    if rttm_file.exists():
                        # 복사
                        dest = output_rttm_dir / f"{utt_id}.rttm"
                        shutil.copy2(rttm_file, dest)
                        copied += 1
                        found = True
                        break
                if found:
                    break

            if not found:
                not_found.append(utt_id)

        print(f"  ✓ 복사: {copied}개")
        if not_found:
            print(f"  ⚠️  RTTM 없음: {len(not_found)}개")
            if len(not_found) <= 10:
                for utt in not_found:
                    print(f"    - {utt}")

        return copied

    def create_project_structure(
            self,
            project_root: Path,
            n_spk: Optional[str] = None,
            ratio: Optional[str] = None,
            max_utterances: Optional[int] = None
    ):
        """
        프로젝트 전체 구조 생성
        """
        project_root = Path(project_root)

        print(f"\n{'=' * 60}")
        print(f"프로젝트 구조 생성: {project_root}")
        print(f"{'=' * 60}\n")

        # 1. Kaldi 데이터 필터링
        kaldi_output = project_root / "data" / "audio_kaldi_format"
        metadata = self.filter_kaldi_by_config(
            kaldi_output,
            n_spk=n_spk,
            ratio=ratio,
            max_utterances=max_utterances
        )

        # 2. RTTM 파일 복사
        rttm_output = project_root / "data" / "ground_truth_rttm"
        self.copy_corresponding_rttm(kaldi_output, rttm_output)

        # 3. 기타 디렉토리 생성
        dirs = [
            "config",
            "scripts",
            "models/eendeda/checkpoints",
            "models/msa",
            "output/eendeda/rttm",
            "output/eendeda/spkvec",
            "output/anonymized"
        ]

        for d in dirs:
            (project_root / d).mkdir(parents=True, exist_ok=True)

        print(f"\n✅ 프로젝트 구조 생성 완료!\n")

        # 요약 출력
        print(f"{'=' * 60}")
        print(f"📊 데이터 요약")
        print(f"{'=' * 60}")
        print(f"발화 수: {metadata['num_utterances']}")
        print(f"화자 수: {metadata['num_speakers']}")
        if n_spk:
            print(f"화자 수 필터: {n_spk}")
        if ratio:
            print(f"중첩률 필터: {ratio}%")
        print(f"{'=' * 60}\n")

        # 다음 단계 안내
        print(f"📋 다음 단계:")
        print(f"")
        print(f"1. EEND-EDA 추론:")
        print(f"   python infer_4_anon.py \\")
        print(f"       --config config/eendeda_config.yaml \\")
        print(f"       --infer-data-dir {kaldi_output} \\")
        print(f"       --models-path models/eendeda/checkpoints \\")
        print(f"       --out-dir output/eendeda \\")
        print(f"       --gpu 0")
        print(f"")
        print(f"2. 결과 평가 (ground truth와 비교):")
        print(f"   python scripts/evaluate_diarization.py \\")
        print(f"       --hypothesis output/eendeda/rttm \\")
        print(f"       --reference {rttm_output}")
        print(f"")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="sim_data 구조를 EEND-EDA 입력 형식으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:

1. 사용 가능한 데이터 확인:
   python sim_data_adapter.py --sim-data /home/ktaemin/sim_data --list

2. 2화자, 10% 중첩 데이터만 사용:
   python sim_data_adapter.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project \\
       --n-spk 2spk \\
       --ratio 10

3. 3화자, 모든 중첩률, 최대 100개 발화:
   python sim_data_adapter.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project \\
       --n-spk 3spk \\
       --max-utterances 100

4. 모든 데이터 사용:
   python sim_data_adapter.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project
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
        help='화자 수 필터'
    )

    parser.add_argument(
        '--ratio',
        type=str,
        choices=['10', '20', '30'],
        help='중첩률 필터 (%)'
    )

    parser.add_argument(
        '--max-utterances',
        type=int,
        help='최대 발화 수 제한'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='사용 가능한 데이터 설정 목록만 출력'
    )

    args = parser.parse_args()

    # Adapter 초기화
    adapter = SimDataAdapter(args.sim_data)

    # 목록만 출력
    if args.list:
        configs = adapter.get_available_configs()

        print(f"\n{'=' * 60}")
        print(f"사용 가능한 데이터 설정")
        print(f"{'=' * 60}")

        for cfg in configs:
            print(f"\n{cfg['n_spk']} / {cfg['ratio']}% 중첩:")
            print(f"  - WAV 파일: {cfg['num_wav']}개")
            print(f"  - RTTM 파일: {cfg['num_rttm']}개")
            print(f"  - WAV 경로: {cfg['wav_dir']}")
            print(f"  - RTTM 경로: {cfg['rttm_dir']}")

        print(f"\n{'=' * 60}")
        print(f"총 {len(configs)}개 설정")
        print(f"{'=' * 60}\n")

        return

    # 프로젝트 구조 생성
    adapter.create_project_structure(
        project_root=args.project_root,
        n_spk=args.n_spk,
        ratio=args.ratio,
        max_utterances=args.max_utterances
    )


if __name__ == "__main__":
    main()