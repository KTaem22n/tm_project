# scripts/verify_attractors.py
import numpy as np
from pathlib import Path

spkvec_dir = Path("output/eendeda/spkvec")

for meta_file in spkvec_dir.glob("*.meta.json"):
    import json

    with open(meta_file) as f:
        meta = json.load(f)

    utt = meta['utt']
    z_mu = np.load(spkvec_dir / f"{utt}.z_mu.npy")

    print(f"\n{utt}:")
    print(f"  - 화자 수: {meta['num_speakers']}")
    print(f"  - z_mu shape: {z_mu.shape}")
    print(f"  - z_mu 범위: [{z_mu.min():.3f}, {z_mu.max():.3f}]")