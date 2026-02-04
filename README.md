/home/ktaemin/tm_project/
├── README.md
├── requirements.txt
│
├── config/
│   ├── eendeda_config.yaml           # EEND-EDA 추론 설정
│   └── eendeda_config.yaml.backup    # 백업
│
├── scripts/
│   ├── copy_sim_data.py              # sim_data → Kaldi 변환
│   ├── run_msa_anonymization_complete.py  # MSA 익명화 (HuBERT+HiFi-GAN)
│   ├── run_comprehensive_evaluation.py    # 전체 실험 자동화 (27개)
│   ├── compare_oracle_vs_estimated.py     # Oracle vs Estimated 비교
│   ├── evaluate_der.py               # DER 평가
│   └── evaluate_privacy.py           # Privacy (FAR) 평가
│
├── utils/
│   ├── audio_utils.py                # 오디오 처리 유틸
│   ├── rttm_utils.py                 # RTTM 파싱 유틸
│   └── vector_utils.py               # 화자 벡터 유틸
│
├── data/
│   ├── audio_kaldi_format/           # Kaldi 형식 데이터
│   │   ├── wav.scp
│   │   ├── utt2spk
│   │   ├── spk2utt
│   │   ├── segments
│   │   └── metadata.json
│   └── ground_truth_rttm/            # Ground Truth RTTM
│       └── *.rttm
│
├── models/
│   ├── eendeda/
│   │   └── checkpoints/
│   │       └── checkpoint_28.tar     # EEND-EDA 모델
│   └── msa/
│       ├── hubert/                   # HuBERT (자동 다운로드)
│       ├── hifigan/                  # HiFi-GAN (자동 다운로드)
│       └── ecapa_tdnn/               # ECAPA-TDNN (자동 다운로드)
│
├── output/
│   ├── eendeda/                      # EEND-EDA 출력 (Estimated)
│   │   ├── 2spk_10/
│   │   │   ├── rttm/
│   │   │   ├── spkvec/
│   │   │   └── manifest.json
│   │   ├── 2spk_20/
│   │   └── ...
│   ├── eendeda_oracle/               # EEND-EDA 출력 (Oracle)
│   │   └── ...
│   └── anonymized/                   # 익명화된 오디오
│       ├── select_2spk_10/
│       ├── as_2spk_10/
│       └── ...
│
├── comprehensive_results_estimated/   # Estimated 모드 결과
│   ├── select_2spk_10/
│   │   ├── results.json
│   │   ├── der_output.txt
│   │   └── privacy_output.txt
│   ├── ... (27개 디렉토리)
│   ├── detailed_results.csv          # ⭐ 모든 실험 상세
│   └── overall_summary.csv           # ⭐ Overall 요약
│
├── comprehensive_results_oracle/     # Oracle 모드 결과
│   ├── ... (27개 디렉토리)
│   ├── detailed_results.csv
│   └── overall_summary.csv
│
├── oracle_vs_estimated_comparison.csv   # ⭐ 비교 결과
├── oracle_vs_estimated_comparison.png   # ⭐ 비교 그래프
│
├── eendeda_repo/                     # EEND-EDA-VIB 저장소
│   └── eendedavib/
│       ├── backend/
│       └── common_utils/
│
├── infer_4_anon.py                   # EEND-EDA 추론 스크립트
│
└── .gitignore
