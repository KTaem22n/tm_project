flowchart TD

    A[Input Audio] --> B[EEND-EDA Diarization]
    B --> C1[Estimated RTTM]
    B --> C2[Oracle RTTM]

    C1 --> D1[Speaker Vector Extraction]
    C2 --> D2[Speaker Vector Extraction]

    D1 --> E1[MSA Anonymization<br>(HuBERT + HiFi-GAN)]
    D2 --> E2[MSA Anonymization<br>(HuBERT + HiFi-GAN)]

    E1 --> F1[Anonymized Audio<br>(Estimated Mode)]
    E2 --> F2[Anonymized Audio<br>(Oracle Mode)]

    F1 --> G1[DER Evaluation]
    F1 --> H1[Privacy Evaluation (FAR)]

    F2 --> G2[DER Evaluation]
    F2 --> H2[Privacy Evaluation (FAR)]

    G1 --> I[Comprehensive Results]
    H1 --> I
    G2 --> I
    H2 --> I
