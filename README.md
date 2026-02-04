tm_project/
├── README.md
├── requirements.txt
│
├── config/
│   ├── eendeda_config.yaml              # EEND-EDA inference config
│   └── eendeda_config.yaml.backup       # backup
│
├── scripts/                             # main runnable scripts
│   ├── copy_sim_data.py                 # sim_data → Kaldi format conversion
│   ├── run_msa_anonymization_complete.py# MSA anonymization (HuBERT + HiFi-GAN)
│   ├── run_comprehensive_evaluation.py  # full automation (27 experiments)
│   ├── compare_oracle_vs_estimated.py   # Oracle vs Estimated comparison
│   ├── evaluate_der.py                  # DER evaluation
│   └── evaluate_privacy.py              # Privacy (FAR) evaluation
│
├── utils/                               # shared utilities
│   ├── audio_utils.py                   # audio processing utils
│   ├── rttm_utils.py                    # RTTM parsing utils
│   └── vector_utils.py                  # speaker vector utils
│
├── data/
│   ├── audio_kaldi_format/              # Kaldi-style data
│   │   ├── wav.scp
│   │   ├── utt2spk
│   │   ├── spk2utt
│   │   ├── segments
│   │   └── metadata.json
│   └── ground_truth_rttm/               # ground-truth RTTM
│       └── *.rttm
│
├── models/
│   ├── eendeda/
│   │   └── checkpoints/
│   │       └── checkpoint_28.tar        # EEND-EDA model checkpoint
│   └── msa/
│       ├── hubert/                      # auto-downloaded HuBERT
│       ├── hifigan/                     # auto-downloaded HiFi-GAN
│       └── ecapa_tdnn/                  # auto-downloaded ECAPA-TDNN
│
├── output/
│   ├── eendeda/                         # diarization outputs (Estimated)
│   │   ├── 2spk_10/
│   │   │   ├── rttm/
│   │   │   ├── spkvec/
│   │   │   └── manifest.json
│   │   ├── 2spk_20/
│   │   └── ...
│   ├── eendeda_oracle/                  # diarization outputs (Oracle)
│   │   └── ...
│   └── anonymized/                      # anonymized audio outputs
│       ├── select_2spk_10/
│       ├── as_2spk_10/
│       └── ...
│
├── comprehensive_results_estimated/     # results (Estimated mode)
│   ├── select_2spk_10/
│   │   ├── results.json
│   │   ├── der_output.txt
│   │   └── privacy_output.txt
│   ├── ... (27 dirs)
│   ├── detailed_results.csv             # all experiment details
│   └── overall_summary.csv              # overall summary
│
├── comprehensive_results_oracle/        # results (Oracle mode)
│   ├── ... (27 dirs)
│   ├── detailed_results.csv
│   └── overall_summary.csv
│
├── oracle_vs_estimated_comparison.csv   # final comparison table
├── oracle_vs_estimated_comparison.png   # final comparison plot
│
├── eendeda_repo/                        # vendored EEND-EDA-VIB repo
│   └── eendedavib/
│       ├── backend/
│       └── common_utils/
│
├── infer_4_anon.py                      # EEND-EDA inference wrapper
└── .gitignore
