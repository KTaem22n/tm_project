#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_all_experiments.py
모든 테스트 데이터셋에 대해 자동으로 실험 실행
"""

import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse


class ExperimentRunner:
    """전체 실험 자동화"""

    def __init__(
            self,
            sim_data_root: Path,
            project_root: Path,
            gpu: int = 0
    ):
        self.sim_data_root = Path(sim_data_root)
        self.project_root = Path(project_root)
        self.gpu = gpu

        # 실험 설정 리스트
        self.configs = [
            {'n_spk': '2spk', 'ratio': '10'},
            {'n_spk': '2spk', 'ratio': '20'},
            {'n_spk': '2spk', 'ratio': '30'},
            {'n_spk': '3spk', 'ratio': '10'},
            {'n_spk': '3spk', 'ratio': '20'},
            {'n_spk': '3spk', 'ratio': '30'},
            {'n_spk': '4spk', 'ratio': '10'},
            {'n_spk': '4spk', 'ratio': '20'},
            {'n_spk': '4spk', 'ratio': '30'},
        ]

        # 결과 저장 디렉토리
        self.results_root = self.project_root / "results"
        self.results_root.mkdir(exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"전체 실험 자동화")
        print(f"{'=' * 70}")
        print(f"sim_data: {self.sim_data_root}")
        print(f"project: {self.project_root}")
        print(f"GPU: {self.gpu}")
        print(f"실험 설정: {len(self.configs)}개")
        print(f"{'=' * 70}\n")

    def run_single_experiment(self, n_spk: str, ratio: str):
        """단일 실험 실행"""

        exp_name = f"{n_spk}_{ratio}"
        print(f"\n{'=' * 70}")
        print(f"실험: {exp_name}")
        print(f"{'=' * 70}\n")

        # 결과 디렉토리
        exp_result_dir = self.results_root / exp_name
        exp_result_dir.mkdir(exist_ok=True)

        # 로그 파일
        log_file = exp_result_dir / "experiment.log"
        log = open(log_file, 'w')

        try:
            # Step 1: Kaldi 데이터 복사
            print(f"[1/4] Kaldi 데이터 복사 중... ({n_spk}/{ratio})")
            cmd = [
                "python", "scripts/copy_sim_data.py",
                "--sim-data", str(self.sim_data_root),
                "--project-root", str(self.project_root),
                "--n-spk", n_spk,
                "--ratio", ratio
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            log.write("=== STEP 1: Copy Kaldi Data ===\n")
            log.write(result.stdout)
            log.write(result.stderr)

            if result.returncode != 0:
                print(f"  ❌ Kaldi 데이터 복사 실패")
                return False
            print(f"  ✓ 완료")

            # Step 2: EEND-EDA 추론
            print(f"[2/4] EEND-EDA 추론 중...")

            # 출력 디렉토리 설정
            eendeda_out = self.project_root / "output" / "eendeda" / exp_name
            eendeda_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "infer_4_anon.py",
                "--config", "config/eendeda_config.yaml",
                "--infer-data-dir", "data/audio_kaldi_format",
                "--models-path", "models/eendeda/checkpoints",
                "--epochs", "28",
                "--out-dir", str(eendeda_out),
                "--gpu", str(self.gpu)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            log.write("\n=== STEP 2: EEND-EDA Inference ===\n")
            log.write(result.stdout)
            log.write(result.stderr)

            if result.returncode != 0:
                print(f"  ❌ EEND-EDA 추론 실패")
                return False
            print(f"  ✓ 완료")

            # Step 3: MSA 익명화
            print(f"[3/4] MSA 익명화 중...")

            # 익명화 출력 디렉토리
            anon_out = self.project_root / "output" / "anonymized" / exp_name
            anon_out.mkdir(parents=True, exist_ok=True)

            # run_msa_anonymization.py 수정된 버전 실행
            # 원본 오디오 경로 동적 설정
            audio_data_dir = self.sim_data_root / "wav" / "all" / n_spk / ratio

            cmd = [
                "python", "scripts/run_msa_anonymization_full.py",  # 새 스크립트 사용
                "--eendeda-output", str(eendeda_out),
                "--audio-dir", str(audio_data_dir),
                "--output-dir", str(anon_out),
                "--rttm-dir", str(eendeda_out / "rttm"),
                "--method", "select",  # 'select', 'as', 'ds' 중 선택
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            log.write("\n=== STEP 3: MSA Anonymization ===\n")
            log.write(result.stdout)
            log.write(result.stderr)

            if result.returncode != 0:
                print(f"  ❌ MSA 익명화 실패")
                return False
            print(f"  ✓ 완료")

            # Step 4: 평가
            print(f"[4/4] 평가 중...")

            # DER 평가
            print(f"  - DER 계산...")
            cmd = [
                "python", "scripts/evaluate_der.py",
                "--reference", "data/ground_truth_rttm",
                "--hypothesis", str(eendeda_out / "rttm")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            log.write("\n=== STEP 4a: DER Evaluation ===\n")
            log.write(result.stdout)
            log.write(result.stderr)

            # DER 결과 저장
            der_result_file = exp_result_dir / "der_results.txt"
            with open(der_result_file, 'w') as f:
                f.write(result.stdout)

            # Privacy 평가
            print(f"  - Privacy (FAR) 계산...")
            cmd = [
                "python", "scripts/evaluate_privacy.py",
                "--original-audio", str(audio_data_dir),
                "--anonymized-audio", str(anon_out),
                "--rttm-dir", str(eendeda_out / "rttm")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            log.write("\n=== STEP 4b: Privacy Evaluation ===\n")
            log.write(result.stdout)
            log.write(result.stderr)

            # Privacy 결과 저장
            privacy_result_file = exp_result_dir / "privacy_results.txt"
            with open(privacy_result_file, 'w') as f:
                f.write(result.stdout)

            print(f"  ✓ 완료")

            # 메타데이터 저장
            metadata = {
                'experiment': exp_name,
                'n_spk': n_spk,
                'ratio': ratio,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'audio_data_dir': str(audio_data_dir),
                'eendeda_output': str(eendeda_out),
                'anonymized_output': str(anon_out),
                'results': {
                    'der': str(der_result_file),
                    'privacy': str(privacy_result_file)
                }
            }

            with open(exp_result_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n✅ {exp_name} 완료!")
            print(f"   결과: {exp_result_dir}\n")

            return True

        except Exception as e:
            print(f"\n❌ {exp_name} 실패: {e}\n")
            log.write(f"\n=== ERROR ===\n{str(e)}\n")
            return False

        finally:
            log.close()

    def run_all_experiments(self, skip_existing: bool = False):
        """모든 실험 실행"""

        results = []

        for config in self.configs:
            n_spk = config['n_spk']
            ratio = config['ratio']
            exp_name = f"{n_spk}_{ratio}"

            # 이미 실행된 실험 스킵
            if skip_existing:
                exp_result_dir = self.results_root / exp_name
                if (exp_result_dir / "metadata.json").exists():
                    print(f"\n⏭️  {exp_name} 스킵 (이미 완료)")
                    results.append({
                        'experiment': exp_name,
                        'status': 'skipped'
                    })
                    continue

            # 실험 실행
            success = self.run_single_experiment(n_spk, ratio)

            results.append({
                'experiment': exp_name,
                'n_spk': n_spk,
                'ratio': ratio,
                'status': 'success' if success else 'failed'
            })

        # 전체 결과 요약
        self.print_summary(results)

        # 결과 저장
        summary_file = self.results_root / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(results),
                'results': results
            }, f, indent=2)

        print(f"\n전체 결과 저장: {summary_file}")

    def print_summary(self, results):
        """결과 요약 출력"""

        print(f"\n{'=' * 70}")
        print(f"전체 실험 결과 요약")
        print(f"{'=' * 70}\n")

        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')

        print(f"총 실험: {len(results)}개")
        print(f"  - 성공: {success_count}개")
        print(f"  - 실패: {failed_count}개")
        print(f"  - 스킵: {skipped_count}개")
        print()

        # 상세 결과
        for result in results:
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️'
            }.get(result['status'], '?')

            print(f"{status_icon} {result['experiment']}: {result['status']}")

        print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="모든 테스트 데이터셋에 대해 자동 실험 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:

1. 모든 실험 실행:
   python scripts/run_all_experiments.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project

2. 특정 GPU 지정:
   python scripts/run_all_experiments.py \\
       --sim-data /home/ktaemin/sim_data \\
       --gpu 1

3. 이미 완료된 실험 스킵:
   python scripts/run_all_experiments.py \\
       --sim-data /home/ktaemin/sim_data \\
       --skip-existing

4. 특정 설정만 실행:
   python scripts/run_all_experiments.py \\
       --sim-data /home/ktaemin/sim_data \\
       --configs 2spk/10 2spk/20 3spk/10
        """
    )

    parser.add_argument(
        '--sim-data',
        type=Path,
        default=Path('/home/ktaemin/sim_data'),
        help='sim_data 루트 디렉토리'
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path('/home/ktaemin/tm_project'),
        help='프로젝트 루트 디렉토리'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU 번호 (-1: CPU)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='이미 완료된 실험 스킵'
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        help='특정 설정만 실행 (예: 2spk/10 3spk/20)'
    )

    args = parser.parse_args()

    # Runner 초기화
    runner = ExperimentRunner(
        sim_data_root=args.sim_data,
        project_root=args.project_root,
        gpu=args.gpu
    )

    # 특정 설정만 실행
    if args.configs:
        original_configs = runner.configs
        runner.configs = []

        for config_str in args.configs:
            parts = config_str.split('/')
            if len(parts) == 2:
                n_spk = parts[0] if 'spk' in parts[0] else f"{parts[0]}spk"
                ratio = parts[1]
                runner.configs.append({'n_spk': n_spk, 'ratio': ratio})

        print(f"\n선택된 설정: {len(runner.configs)}개")
        for cfg in runner.configs:
            print(f"  - {cfg['n_spk']}/{cfg['ratio']}")
        print()

    # 모든 실험 실행
    runner.run_all_experiments(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
