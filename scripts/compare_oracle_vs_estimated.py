#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/compare_oracle_vs_estimated.py
Oracle 모드와 Estimated 모드 결과 비교
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def compare_results(project_root: Path):
    """두 모드 결과 비교"""

    estimated_dir = project_root / "comprehensive_results_estimated"
    oracle_dir = project_root / "comprehensive_results_oracle"

    # 결과 로드
    try:
        df_est = pd.read_csv(estimated_dir / "detailed_results.csv")
        df_oracle = pd.read_csv(oracle_dir / "detailed_results.csv")
    except FileNotFoundError as e:
        print(f"❌ 결과 파일 없음: {e}")
        print("   먼저 두 모드 모두 실행하세요:")
        print("   1. python scripts/run_comprehensive_evaluation.py")
        print("   2. python scripts/run_comprehensive_evaluation.py --use-oracle")
        return

    # 성공한 실험만
    df_est = df_est[df_est['status'] == 'success']
    df_oracle = df_oracle[df_oracle['status'] == 'success']

    print(f"\n{'=' * 80}")
    print(f"Oracle vs Estimated 모드 비교")
    print(f"{'=' * 80}\n")

    # Overall 비교
    print(f"📊 Overall 비교")
    print(f"{'':15s} {'Estimated':>15s} {'Oracle':>15s} {'차이':>15s}")
    print(f"{'-' * 65}")

    est_der = df_est['der'].mean()
    oracle_der = df_oracle['der'].mean()
    diff_der = est_der - oracle_der
    print(f"{'DER (평균)':15s} {est_der:>14.2f}% {oracle_der:>14.2f}% {diff_der:>14.2f}%")

    est_far = df_est['far'].mean()
    oracle_far = df_oracle['far'].mean()
    diff_far = est_far - oracle_far
    print(f"{'FAR (평균)':15s} {est_far:>14.2f}% {oracle_far:>14.2f}% {diff_far:>14.2f}%")

    # 방법별 비교
    print(f"\n📈 익명화 방법별 비교")
    print(f"{'방법':10s} {'Estimated DER':>15s} {'Oracle DER':>15s} {'개선':>10s}")
    print(f"{'-' * 60}")

    for method in ['select', 'as', 'ds']:
        est_m = df_est[df_est['method'] == method]['der'].mean()
        oracle_m = df_oracle[df_oracle['method'] == method]['der'].mean()
        improve = est_m - oracle_m
        print(f"{method.upper():10s} {est_m:>14.2f}% {oracle_m:>14.2f}% {improve:>9.2f}%")

    # 화자 수별 비교
    print(f"\n👥 화자 수별 비교")
    print(f"{'화자수':10s} {'Estimated DER':>15s} {'Oracle DER':>15s} {'개선':>10s}")
    print(f"{'-' * 60}")

    for n_spk in ['2spk', '3spk', '4spk']:
        est_n = df_est[df_est['n_spk'] == n_spk]['der'].mean()
        oracle_n = df_oracle[df_oracle['n_spk'] == n_spk]['der'].mean()
        improve = est_n - oracle_n
        print(f"{n_spk:10s} {est_n:>14.2f}% {oracle_n:>14.2f}% {improve:>9.2f}%")

    # 중첩률별 비교
    print(f"\n🔀 중첩률별 비교")
    print(f"{'중첩률':10s} {'Estimated DER':>15s} {'Oracle DER':>15s} {'개선':>10s}")
    print(f"{'-' * 60}")

    for ratio in ['10', '20', '30']:
        est_r = df_est[df_est['ratio'] == ratio]['der'].mean()
        oracle_r = df_oracle[df_oracle['ratio'] == ratio]['der'].mean()
        improve = est_r - oracle_r
        print(f"{ratio + '%':10s} {est_r:>14.2f}% {oracle_r:>14.2f}% {improve:>9.2f}%")

    print(f"\n{'=' * 80}")
    print(f"💡 결론")
    print(f"{'=' * 80}")
    print(f"Oracle 모드 사용 시 평균 {diff_der:.2f}% DER 개선")
    print(f"→ 화자 수 추정의 중요성: {'높음' if diff_der > 2 else '보통' if diff_der > 1 else '낮음'}")
    print(f"{'=' * 80}\n")

    # 비교 표 저장
    comparison = []
    for _, row_est in df_est.iterrows():
        key = f"{row_est['method']}_{row_est['n_spk']}_{row_est['ratio']}"
        row_oracle = df_oracle[
            (df_oracle['method'] == row_est['method']) &
            (df_oracle['n_spk'] == row_est['n_spk']) &
            (df_oracle['ratio'] == row_est['ratio'])
            ]

        if len(row_oracle) > 0:
            comparison.append({
                'method': row_est['method'],
                'n_spk': row_est['n_spk'],
                'ratio': row_est['ratio'],
                'der_estimated': row_est['der'],
                'der_oracle': row_oracle.iloc[0]['der'],
                'der_improvement': row_est['der'] - row_oracle.iloc[0]['der'],
                'far_estimated': row_est['far'],
                'far_oracle': row_oracle.iloc[0]['far']
            })

    df_comparison = pd.DataFrame(comparison)
    output_path = project_root / "oracle_vs_estimated_comparison.csv"
    df_comparison.to_csv(output_path, index=False)
    print(f"✓ 비교 표 저장: {output_path}")

    # 그래프 생성 (optional)
    try:
        plot_comparison(df_comparison, project_root)
    except Exception as e:
        print(f"⚠️  그래프 생성 실패: {e}")


def plot_comparison(df: pd.DataFrame, project_root: Path):
    """비교 그래프 생성"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Oracle vs Estimated Mode Comparison', fontsize=16)

    # 1. 방법별 비교
    ax = axes[0, 0]
    methods = df.groupby('method')[['der_estimated', 'der_oracle']].mean()
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, methods['der_estimated'], width, label='Estimated', alpha=0.8)
    ax.bar(x + width / 2, methods['der_oracle'], width, label='Oracle', alpha=0.8)
    ax.set_xlabel('Method')
    ax.set_ylabel('DER (%)')
    ax.set_title('DER by Anonymization Method')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods.index])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. 화자 수별 비교
    ax = axes[0, 1]
    n_spks = df.groupby('n_spk')[['der_estimated', 'der_oracle']].mean()
    x = np.arange(len(n_spks))
    ax.bar(x - width / 2, n_spks['der_estimated'], width, label='Estimated', alpha=0.8)
    ax.bar(x + width / 2, n_spks['der_oracle'], width, label='Oracle', alpha=0.8)
    ax.set_xlabel('Number of Speakers')
    ax.set_ylabel('DER (%)')
    ax.set_title('DER by Number of Speakers')
    ax.set_xticks(x)
    ax.set_xticklabels(n_spks.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. 중첩률별 비교
    ax = axes[1, 0]
    ratios = df.groupby('ratio')[['der_estimated', 'der_oracle']].mean()
    x = np.arange(len(ratios))
    ax.bar(x - width / 2, ratios['der_estimated'], width, label='Estimated', alpha=0.8)
    ax.bar(x + width / 2, ratios['der_oracle'], width, label='Oracle', alpha=0.8)
    ax.set_xlabel('Overlap Ratio (%)')
    ax.set_ylabel('DER (%)')
    ax.set_title('DER by Overlap Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in ratios.index])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. 개선 정도 분포
    ax = axes[1, 1]
    improvements = df['der_improvement']
    ax.hist(improvements, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(improvements.mean(), color='red', linestyle='--',
               label=f'Mean: {improvements.mean():.2f}%')
    ax.set_xlabel('DER Improvement (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of DER Improvement\n(Estimated - Oracle)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # 저장
    output_path = project_root / "oracle_vs_estimated_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 그래프 저장: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path,
                        default=Path('/home/ktaemin/tm_project'))
    args = parser.parse_args()

    compare_results(args.project_root)