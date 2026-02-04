# scripts/verify_rttm.py
from pathlib import Path
from utils.rttm_utils import RTTMParser

rttm_dir = Path("output/eendeda/rttm")

for rttm_file in rttm_dir.glob("*.rttm"):
    segments = RTTMParser.parse_file(rttm_file)
    speakers = set(seg['speaker_id'] for seg in segments)

    print(f"\n{rttm_file.name}:")
    print(f"  - 전체 세그먼트: {len(segments)}")
    print(f"  - 화자 수: {len(speakers)}")
    print(f"  - 화자: {sorted(speakers)}")

    total_duration = sum(seg['duration'] for seg in segments)
    print(f"  - 총 발화 시간: {total_duration:.2f}초")