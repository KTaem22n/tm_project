#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_comprehensive_evaluation.py
전체 실험 자동화 및 종합 평가
- 3가지 익명화 방법: select, as, ds
- 3가지 화자 수: 2spk, 3spk, 4spk
- 3가지 중첩률: 10%, 20%, 30%
- 총 27개 실험 자동 실행 및 결과 종합
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np


class ComprehensiveEvaluator:
    """종합 평가 시스템"""

    def __init__(
            self,
            sim_data_root: Path,
            project_root: Path,
            gpu: int = 0,
            use_oracle: bool = False
    ):
        self.sim_data_root = Path(sim_data_root)
        self.project_root = Path(project_root)
        self.gpu = gpu
        self.use_oracle = use_oracle

        # 실험 설정
        self.methods = ['select', 'as', 'ds']
        self.n_spks = ['2spk', '3spk', '4spk']
        self.ratios = ['10', '20', '30']

        # 결과 저장
        suffix = "_oracle" if use_oracle else "_estimated"
        self.results_root = self.project_root / f"comprehensive_results{suffix}"
        self.results_root.mkdir(exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"종합 평가 시스템")
        print(f"{'=' * 80}")
        print(f"화자 수 설정: {'Oracle (알고 있음)' if use_oracle else 'Estimated (추정)'}")
        print(
            f"총 실험 수: {len(self.methods)} × {len(self.n_spks)} × {len(self.ratios)} = {len(self.methods) * len(self.n_spks) * len(self.ratios)}")
        print(f"익명화 방법: {', '.join(self.methods)}")
        print(f"화자 수: {', '.join(self.n_spks)}")
        print(f"중첩률: {', '.join(self.ratios)}%")
        print(f"{'=' * 80}\n")

    def run_single_experiment(
            self,
            method: str,
            n_spk: str,
            ratio: str
    ) -> dict:
        """단일 실험 실행"""

        exp_id = f"{method}_{n_spk}_{ratio}"
        print(f"\n{'=' * 80}")
        print(f"실험: {exp_id}")
        print(f"{'=' * 80}")

        exp_dir = self.results_root / exp_id
        exp_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Kaldi 데이터 복사
            print(f"[1/4] Kaldi 데이터 준비...")
            cmd = [
                "python", "scripts/copy_sim_data.py",
                "--sim-data", str(self.sim_data_root),
                "--project-root", str(self.project_root),
                "--n-spk", n_spk,
                "--ratio", ratio
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"  ❌ 데이터 준비 실패")
                return {'status': 'failed', 'stage': 'data_prep'}

            # Step 2: EEND-EDA 추론
            print(f"[2/4] EEND-EDA 추론...")
            eendeda_out = self.project_root / "output" / "eendeda" / f"{n_spk}_{ratio}"
            if self.use_oracle:
                eendeda_out = self.project_root / "output" / "eendeda_oracle" / f"{n_spk}_{ratio}"
            eendeda_out.mkdir(parents=True, exist_ok=True)

            # Config 파일 수정 (Oracle 모드)
            if self.use_oracle:
                # 화자 수 추출 (2spk -> 2)
                num_spk = int(n_spk.replace('spk', ''))
                self.update_config_for_oracle(num_spk)
            else:
                self.restore_config_to_estimated()

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

            if result.returncode != 0:
                print(f"  ❌ EEND-EDA 추론 실패")
                return {'status': 'failed', 'stage': 'eendeda'}

            # Step 3: MSA 익명화
            print(f"[3/4] MSA 익명화 ({method.upper()})...")
            anon_out = self.project_root / "output" / "anonymized" / exp_id
            anon_out.mkdir(parents=True, exist_ok=True)

            audio_data_dir = self.sim_data_root / "wav" / "all" / n_spk / ratio

            cmd = [
                "python", "scripts/run_msa_anonymization_complete.py",
                "--eendeda-output", str(eendeda_out),
                "--audio-dir", str(audio_data_dir),
                "--output-dir", str(anon_out),
                "--rttm-dir", str(eendeda_out / "rttm"),
                "--method", method,
                "--skip-synthesis"  # 빠른 실행
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"  ❌ 익명화 실패")
                return {'status': 'failed', 'stage': 'anonymization'}

            # Step 4: 평가
            print(f"[4/4] 평가...")

            # DER 평가
            cmd = [
                "python", "scripts/evaluate_der.py",
                "--reference", "data/ground_truth_rttm",
                "--hypothesis", str(eendeda_out / "rttm")
            ]
            der_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            # DER 파싱
            der_value = self.parse_der(der_result.stdout)

            # Privacy 평가
            cmd = [
                "python", "scripts/evaluate_privacy.py",
                "--original-audio", str(audio_data_dir),
                "--anonymized-audio", str(anon_out),
                "--rttm-dir", str(eendeda_out / "rttm")
            ]
            privacy_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            # FAR 파싱
            far_value = self.parse_far(privacy_result.stdout)

            # 결과 저장
            results = {
                'status': 'success',
                'method': method,
                'n_spk': n_spk,
                'ratio': ratio,
                'der': der_value,
                'far': far_value,
                'eendeda_output': str(eendeda_out),
                'anonymized_output': str(anon_out),
                'timestamp': datetime.now().isoformat()
            }

            # 개별 결과 저장
            with open(exp_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)

            with open(exp_dir / "der_output.txt", 'w') as f:
                f.write(der_result.stdout)

            with open(exp_dir / "privacy_output.txt", 'w') as f:
                f.write(privacy_result.stdout)

            print(f"  ✅ 완료: DER={der_value:.2f}%, FAR={far_value:.2f}%")

            return results

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            return {'status': 'failed', 'error': str(e)}

    def update_config_for_oracle(self, num_spk: int):
        """Config 파일을 Oracle 모드로 수정"""
        config_path = self.project_root / "config" / "eendeda_config.yaml"

        # 백업
        backup_path = config_path.with_suffix('.yaml.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)

        # 파일 읽기
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 수정
        config['estimate_spk_qty'] = num_spk
        config['estimate_spk_qty_thr'] = -1

        # 쓰기
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"  ✓ Oracle 모드: estimate_spk_qty={num_spk}")

        # 확인
        with open(config_path, 'r') as f:
            print(f"  확인: {f.read()}")

    def restore_config_to_estimated(self):
        """Config 파일을 Estimated 모드로 복원 (화자 수 추정)"""
        config_path = self.project_root / "config" / "eendeda_config.yaml"

        # Config 읽기
        with open(config_path, 'r') as f:
            lines = f.readlines()

        # estimate-spk-qty 수정
        new_lines = []
        for line in lines:
            if 'estimate-spk-qty:' in line or 'estimate_spk_qty:' in line:
                # Estimated 모드: -1
                new_lines.append("estimate_spk_qty: -1\n")
            elif 'estimate-spk-qty-thr:' in line or 'estimate_spk_qty_thr:' in line:
                # threshold 사용
                new_lines.append("estimate_spk_qty_thr: 0.5\n")
            else:
                new_lines.append(line)

        # Config 쓰기
        with open(config_path, 'w') as f:
            f.writelines(new_lines)

        print(f"  ✓ Config 업데이트: Estimated 모드 (화자 수 추정)")

    def parse_der(self, output: str) -> float:
        """DER 값 파싱"""
        try:
            for line in output.split('\n'):
                if '평균 DER' in line or 'Average DER' in line:
                    # "평균 DER: 15.23%" 형식에서 숫자 추출
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value_str = parts[1].strip().replace('%', '')
                        return float(value_str)
            return -1.0  # 파싱 실패
        except:
            return -1.0

    def parse_far(self, output: str) -> float:
        """FAR 값 파싱"""
        try:
            for line in output.split('\n'):
                if 'FAR' in line and ':' in line:
                    # "FAR (False Acceptance Rate): 2.35%" 형식
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value_str = parts[-1].strip().replace('%', '')
                        return float(value_str)
            return -1.0
        except:
            return -1.0

    def run_all_experiments(self, skip_existing: bool = False):
        """모든 실험 실행"""

        all_results = []

        total = len(self.methods) * len(self.n_spks) * len(self.ratios)

        with tqdm(total=total, desc="전체 진행") as pbar:
            for method in self.methods:
                for n_spk in self.n_spks:
                    for ratio in self.ratios:
                        exp_id = f"{method}_{n_spk}_{ratio}"

                        # 기존 결과 스킵
                        if skip_existing:
                            result_file = self.results_root / exp_id / "results.json"
                            if result_file.exists():
                                with open(result_file) as f:
                                    results = json.load(f)
                                all_results.append(results)
                                pbar.update(1)
                                print(f"\n⏭️  {exp_id} 스킵 (기존 결과 사용)")
                                continue

                        # 실험 실행
                        results = self.run_single_experiment(method, n_spk, ratio)
                        all_results.append(results)
                        pbar.update(1)

        # 결과 분석 및 저장
        self.analyze_and_save_results(all_results)

        return all_results

    def analyze_and_save_results(self, all_results: list):
        """결과 분석 및 저장"""

        print(f"\n{'=' * 80}")
        print(f"결과 분석 및 저장")
        print(f"{'=' * 80}\n")

        # DataFrame 생성
        df = pd.DataFrame(all_results)

        # 성공한 실험만 필터
        df_success = df[df['status'] == 'success'].copy()

        if len(df_success) == 0:
            print("⚠️  성공한 실험이 없습니다.")
            return

        # CSV 저장 (상세)
        csv_path = self.results_root / "detailed_results.csv"
        df_success.to_csv(csv_path, index=False)
        print(f"✓ 상세 결과 저장: {csv_path}")

        # Overall 통계 계산
        overall_stats = self.compute_overall_statistics(df_success)

        # Overall 결과 저장 (JSON)
        overall_json = self.results_root / "overall_results.json"
        with open(overall_json, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"✓ Overall 결과 저장: {overall_json}")

        # Overall 결과 저장 (CSV)
        self.save_overall_csv(overall_stats)

        # 결과 출력
        self.print_summary(df_success, overall_stats)

    def compute_overall_statistics(self, df: pd.DataFrame) -> dict:
        """Overall 통계 계산"""

        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(df),
            'by_method': {},
            'by_n_spk': {},
            'by_ratio': {},
            'overall': {}
        }

        # 방법별 통계
        for method in self.methods:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                stats['by_method'][method] = {
                    'count': len(method_df),
                    'der_mean': float(method_df['der'].mean()),
                    'der_std': float(method_df['der'].std()),
                    'far_mean': float(method_df['far'].mean()),
                    'far_std': float(method_df['far'].std())
                }

        # 화자 수별 통계
        for n_spk in self.n_spks:
            nspk_df = df[df['n_spk'] == n_spk]
            if len(nspk_df) > 0:
                stats['by_n_spk'][n_spk] = {
                    'count': len(nspk_df),
                    'der_mean': float(nspk_df['der'].mean()),
                    'der_std': float(nspk_df['der'].std()),
                    'far_mean': float(nspk_df['far'].mean()),
                    'far_std': float(nspk_df['far'].std())
                }

        # 중첩률별 통계
        for ratio in self.ratios:
            ratio_df = df[df['ratio'] == ratio]
            if len(ratio_df) > 0:
                stats['by_ratio'][ratio] = {
                    'count': len(ratio_df),
                    'der_mean': float(ratio_df['der'].mean()),
                    'der_std': float(ratio_df['der'].std()),
                    'far_mean': float(ratio_df['far'].mean()),
                    'far_std': float(ratio_df['far'].std())
                }

        # Overall 통계
        stats['overall'] = {
            'der_mean': float(df['der'].mean()),
            'der_std': float(df['der'].std()),
            'der_min': float(df['der'].min()),
            'der_max': float(df['der'].max()),
            'far_mean': float(df['far'].mean()),
            'far_std': float(df['far'].std()),
            'far_min': float(df['far'].min()),
            'far_max': float(df['far'].max())
        }

        return stats

    def save_overall_csv(self, stats: dict):
        """Overall 통계를 CSV로 저장"""

        # 방법별 결과
        method_rows = []
        for method, data in stats['by_method'].items():
            method_rows.append({
                'Category': 'Method',
                'Name': method.upper(),
                'DER_Mean': f"{data['der_mean']:.2f}",
                'DER_Std': f"{data['der_std']:.2f}",
                'FAR_Mean': f"{data['far_mean']:.2f}",
                'FAR_Std': f"{data['far_std']:.2f}"
            })

        # 화자 수별 결과
        nspk_rows = []
        for n_spk, data in stats['by_n_spk'].items():
            nspk_rows.append({
                'Category': 'Speakers',
                'Name': n_spk,
                'DER_Mean': f"{data['der_mean']:.2f}",
                'DER_Std': f"{data['der_std']:.2f}",
                'FAR_Mean': f"{data['far_mean']:.2f}",
                'FAR_Std': f"{data['far_std']:.2f}"
            })

        # 중첩률별 결과
        ratio_rows = []
        for ratio, data in stats['by_ratio'].items():
            ratio_rows.append({
                'Category': 'Overlap',
                'Name': f"{ratio}%",
                'DER_Mean': f"{data['der_mean']:.2f}",
                'DER_Std': f"{data['der_std']:.2f}",
                'FAR_Mean': f"{data['far_mean']:.2f}",
                'FAR_Std': f"{data['far_std']:.2f}"
            })

        # Overall
        overall_row = [{
            'Category': 'Overall',
            'Name': 'All',
            'DER_Mean': f"{stats['overall']['der_mean']:.2f}",
            'DER_Std': f"{stats['overall']['der_std']:.2f}",
            'FAR_Mean': f"{stats['overall']['far_mean']:.2f}",
            'FAR_Std': f"{stats['overall']['far_std']:.2f}"
        }]

        # 합치기
        all_rows = method_rows + nspk_rows + ratio_rows + overall_row
        df_overall = pd.DataFrame(all_rows)

        # 저장
        csv_path = self.results_root / "overall_summary.csv"
        df_overall.to_csv(csv_path, index=False)
        print(f"✓ Overall 요약 저장: {csv_path}")

    def print_summary(self, df: pd.DataFrame, stats: dict):
        """결과 요약 출력"""

        print(f"\n{'=' * 80}")
        print(f"종합 결과 요약")
        print(f"{'=' * 80}\n")

        # Overall
        print(f"📊 Overall (전체 {len(df)}개 실험)")
        print(f"  DER: {stats['overall']['der_mean']:.2f} ± {stats['overall']['der_std']:.2f}%")
        print(f"       (범위: {stats['overall']['der_min']:.2f}% ~ {stats['overall']['der_max']:.2f}%)")
        print(f"  FAR: {stats['overall']['far_mean']:.2f} ± {stats['overall']['far_std']:.2f}%")
        print(f"       (범위: {stats['overall']['far_min']:.2f}% ~ {stats['overall']['far_max']:.2f}%)")

        # 방법별 상세 결과
        for method in self.methods:
            method_df = df[df['method'] == method]
            if len(method_df) == 0:
                continue

            print(f"\n{'=' * 80}")
            print(f"📈 {method.upper()} 방법")
            print(f"{'=' * 80}")
            print(f"{'설정':15s} {'DER':>10s} {'FAR':>10s}")
            print(f"{'-' * 40}")

            for n_spk in self.n_spks:
                for ratio in self.ratios:
                    row = method_df[
                        (method_df['n_spk'] == n_spk) &
                        (method_df['ratio'] == ratio)
                        ]

                    if len(row) > 0:
                        setting = f"{n_spk} {ratio}%"
                        der = row.iloc[0]['der']
                        far = row.iloc[0]['far']
                        print(f"{setting:15s} {der:>9.2f}% {far:>9.2f}%")

        print(f"\n{'=' * 80}")
        print(f"요약 통계")
        print(f"{'=' * 80}\n")

        # 방법별 평균
        print(f"📊 익명화 방법별 평균")
        print(f"{'방법':10s} {'DER':>12s} {'FAR':>12s}")
        print(f"{'-' * 40}")
        for method in self.methods:
            if method in stats['by_method']:
                data = stats['by_method'][method]
                print(f"{method.upper():10s} {data['der_mean']:>11.2f}% {data['far_mean']:>11.2f}%")

        # 화자 수별 평균
        print(f"\n👥 화자 수별 평균")
        print(f"{'화자수':10s} {'DER':>12s} {'FAR':>12s}")
        print(f"{'-' * 40}")
        for n_spk in self.n_spks:
            if n_spk in stats['by_n_spk']:
                data = stats['by_n_spk'][n_spk]
                print(f"{n_spk:10s} {data['der_mean']:>11.2f}% {data['far_mean']:>11.2f}%")

        # 중첩률별 평균
        print(f"\n🔀 중첩률별 평균")
        print(f"{'중첩률':10s} {'DER':>12s} {'FAR':>12s}")
        print(f"{'-' * 40}")
        for ratio in self.ratios:
            if ratio in stats['by_ratio']:
                data = stats['by_ratio'][ratio]
                print(f"{ratio + '%':10s} {data['der_mean']:>11.2f}% {data['far_mean']:>11.2f}%")

        print(f"\n{'=' * 80}\n")

        # 최고 성능
        best_der_idx = df['der'].idxmin()
        best_far_idx = df['far'].idxmin()

        print(f"🏆 최고 성능")
        print(f"  최저 DER: {df.loc[best_der_idx, 'method'].upper()} "
              f"({df.loc[best_der_idx, 'n_spk']}/{df.loc[best_der_idx, 'ratio']}%) "
              f"= {df.loc[best_der_idx, 'der']:.2f}%")
        print(f"  최저 FAR: {df.loc[best_far_idx, 'method'].upper()} "
              f"({df.loc[best_far_idx, 'n_spk']}/{df.loc[best_far_idx, 'ratio']}%) "
              f"= {df.loc[best_far_idx, 'far']:.2f}%")

        print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="종합 평가 자동화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:

1. 전체 실험 실행:
   python scripts/run_comprehensive_evaluation.py \\
       --sim-data /home/ktaemin/sim_data \\
       --project-root /home/ktaemin/tm_project

2. 기존 결과 스킵하고 실행:
   python scripts/run_comprehensive_evaluation.py \\
       --sim-data /home/ktaemin/sim_data \\
       --skip-existing

3. CPU 사용:
   python scripts/run_comprehensive_evaluation.py \\
       --sim-data /home/ktaemin/sim_data \\
       --gpu -1
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
        '--use-oracle',
        action='store_true',
        help='Oracle 모드: 화자 수를 모델에 알려줌 (estimate_spk_qty 설정)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='기존 결과 스킵'
    )

    args = parser.parse_args()

    # 평가 시작
    evaluator = ComprehensiveEvaluator(
        sim_data_root=args.sim_data,
        project_root=args.project_root,
        gpu=args.gpu,
        use_oracle=args.use_oracle
    )

    evaluator.run_all_experiments(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()