#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# infer_4_anon.py
# - EEND-EDA-VIB inference
# - outputs: RTTM + attractor vectors per utterance

import sys
from pathlib import Path

# CRITICAL: eendedavib 모듈 경로 추가
# tm_project/eendeda_repo/eendedavib/ 를 Python path에 추가
script_dir = Path(__file__).resolve().parent
eendedavib_path = script_dir / "eendeda_repo" / "eendedavib"
sys.path.insert(0, str(eendedavib_path))

from backend.models import average_checkpoints, get_model
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from pathlib import Path
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import TextIO, Tuple, List, Dict, Any
import argparse
import json
import logging
import numpy as np
import os
import random
import torch
import yamlargparse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    if args.infer_data_dir is None:
        raise ValueError("infer_data_dir is not defined. Check your config or arguments.")
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )
    Y, _, _, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dim {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader


def hard_labels_to_rttm(labels: np.ndarray, id_file: str, rttm_file: TextIO, frameshift: float = 10) -> None:
    # Remove speakers that do not speak
    if len(labels.shape) > 1:
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    n_spks = labels.shape[1] if len(labels.shape) > 1 else 1

    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]

        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([end_indices, labels.shape[0] - 1])

        assert len(ini_indices) == len(end_indices), "Start/end mismatch. Check labels."

        for ini, end in zip(ini_indices, end_indices):
            spk_list.append(spk)
            ini_list.append(ini)
            end_list.append(end)

    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 "
            f"{round(ini * frameshift / 1000, 3)} "
            f"{round((end - ini) * frameshift / 1000, 3)} "
            f"<NA> <NA> spk{spk} <NA> <NA>\n"
        )


def postprocess_output(probabilities, subsampling: int, threshold: float, median_window_length: int) -> np.ndarray:
    thresholded = probabilities > threshold
    filtered = np.zeros(thresholded.shape)
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk].int().cpu().numpy(),
            kernel_size=median_window_length
        )
    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended


def estimate_speakers(att_probs_1d: np.ndarray, thr: float) -> int:
    """
    Conservative rule:
    - If att_probs is length (K+1), we count how many >=thr from the beginning.
    - If it is length K, same rule.
    """
    k = 0
    for v in att_probs_1d:
        if v >= thr:
            k += 1
        else:
            break
    return max(k, 1)


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description="EEND-EDA-VIB inference for anon export")
    parser.add_argument("-c", "--config", help="config file path", action=yamlargparse.ActionConfigFile)

    # keep compatibility with existing infer.py args
    parser.add_argument("--temp", default=1.0, type=float, help="temperature parameter for inference")
    parser.add_argument("--vad-loss-weight", default=1.0, type=float)
    parser.add_argument("--context-size", default=0, type=int)
    parser.add_argument("--encoder-units", type=int)
    parser.add_argument("--epochs", type=str, help="epochs to average, e.g., '82' or '10,11,12' or '10-20'")
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--frame-size", type=int)
    parser.add_argument("--frame-shift", type=int)
    parser.add_argument("--gpu", "-g", default=-1, type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--infer-data-dir", required=True, help="inference data directory (kaldi style).")
    parser.add_argument("--input-transform", default="", choices=["logmel", "logmel_meannorm", "logmel_meanvarnorm"])
    parser.add_argument("--median-window-length", default=11, type=int)
    parser.add_argument("--model-type", default="TransformerEDA", type=str)
    parser.add_argument("--models-path", required=True, type=str, help="directory with model(s) to evaluate")
    parser.add_argument("--num-frames", default=-1, type=int)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--subsampling", default=10, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)

    parser.add_argument("--transformer-encoder-n-heads", type=int)
    parser.add_argument("--transformer-encoder-n-layers", type=int)
    parser.add_argument("--transformer-encoder-dropout", type=float)

    # vib / inference options (same as infer.py)
    parser.add_argument("--use-mu-z", type=bool, default=False)
    parser.add_argument("--use-mu-e", type=bool, default=False)
    parser.add_argument("--zkld-loss-weight", default=[0.000001], type=float, nargs="+")
    parser.add_argument("--ekld-loss-weight", default=[0.000001], type=float, nargs="+")
    parser.add_argument("--kld-weight-type", type=str, default="none",
                        choices=["none", "cyclical_des", "cyclical_asc", "un", "dwa"])
    parser.add_argument("--num-esamples", type=int, default=0)
    parser.add_argument("--num-zsamples", type=int, default=0)
    parser.add_argument("--average-type", type=str, default="sample", choices=["loss", "prob", "sample"])
    parser.add_argument("--infer-type", default="sample", choices=["mu", "sample", "logprob", "avgprob"])

    attractor_args = parser.add_argument_group("attractor")
    attractor_args.add_argument("--time-shuffle", action="store_true")
    attractor_args.add_argument("--attractor-loss-ratio", default=1.0, type=float)
    attractor_args.add_argument("--attractor-encoder-dropout", default=0.1, type=float)
    attractor_args.add_argument("--attractor-decoder-dropout", default=0.1, type=float)
    attractor_args.add_argument("--estimate-spk-qty", default=-1, type=int)
    attractor_args.add_argument("--estimate-spk-qty-thr", default=-1, type=float)
    attractor_args.add_argument("--detach-attractor-loss", default=False, type=bool)

    # NEW: unified output root
    parser.add_argument("--out-dir", required=True, type=str, help="output root directory (rttm/spkvec saved here)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 0:
        gpuid = use_single_gpu(1)
        logging.info("GPU device %s is used", gpuid)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # speaker qty estimation config must be set
    assert args.estimate_spk_qty_thr != -1 or args.estimate_spk_qty != -1, \
        "Either --estimate-spk-qty or --estimate-spk-qty-thr must be defined."

    # build & load model
    model = get_model(args)
    model = average_checkpoints(args.device, model, args.models_path, args.epochs)
    model.eval()
    print("Trainable parameters: %.4f" % count_parameters(model))

    # output dirs
    out_root = Path(args.out_dir)
    rttm_dir = out_root / "rttm"
    spkvec_dir = out_root / "spkvec"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    spkvec_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i, batch in enumerate(infer_loader):
        name = batch["names"][0]
        print(f"Processing {i}: {name}")

        input_x = torch.stack(batch["xs"]).to(args.device)

        with torch.no_grad():
            # IMPORTANT: same API as your infer.py
            y_pred, y_probs, att_probs, z_mu, z_logvar, e_mu, e_logvar = \
                model.estimate_sequential(input_x, args, return_ys=True)

        # y_pred: list with one sequence -> [T, S]
        y_pred = y_pred[0].cpu()  # [T, S]
        post_y = postprocess_output(y_pred, args.subsampling, args.threshold, args.median_window_length)

        # RTTM
        rttm_path = rttm_dir / f"{name}.rttm"
        with open(rttm_path, "w", encoding="utf-8") as f:
            # frameshift should be 10ms if frame_shift=160 at 16k; keep consistent with original infer.py usage
            hard_labels_to_rttm(post_y, name, f, frameshift=args.frame_shift / (args.sampling_rate / 1000.0))

        # attractors
        # z_mu[0]: [K(+1), D]
        z_mu_ = z_mu[0].detach().cpu().numpy()
        z_logvar_ = z_logvar[0].detach().cpu().numpy()

        # att_probs can be tensor (K or K+1) - make 1D
        attp = att_probs.detach().cpu().numpy().reshape(-1)

        # determine S
        if args.estimate_spk_qty != -1:
            S = int(args.estimate_spk_qty)
        else:
            S = int(estimate_speakers(attp, float(args.estimate_spk_qty_thr)))

        # slice to S
        z_mu_S = z_mu_[:S]
        z_logvar_S = z_logvar_[:S]
        attp_S = attp[:S]

        # save per-utt
        zmu_path = spkvec_dir / f"{name}.z_mu.npy"
        zlv_path = spkvec_dir / f"{name}.z_logvar.npy"
        ap_path = spkvec_dir / f"{name}.att_probs.npy"
        meta_path = spkvec_dir / f"{name}.meta.json"

        np.save(zmu_path, z_mu_S)
        np.save(zlv_path, z_logvar_S)
        np.save(ap_path, attp_S)

        meta = {
            "utt": name,
            "num_speakers": S,
            "z_mu_shape": list(z_mu_S.shape),
            "z_logvar_shape": list(z_logvar_S.shape),
            "att_probs_shape": list(attp_S.shape),
            "threshold": float(args.threshold),
            "median_window_length": int(args.median_window_length),
            "subsampling": int(args.subsampling),
            "frame_shift": int(args.frame_shift),
            "sampling_rate": int(args.sampling_rate),
            "models_path": str(args.models_path),
            "epochs": str(args.epochs),
            "infer_type": str(args.infer_type),
            "average_type": str(args.average_type),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        manifest.append({
            "utt": name,
            "rttm": str(rttm_path),
            "z_mu": str(zmu_path),
            "z_logvar": str(zlv_path),
            "att_probs": str(ap_path),
            "meta": str(meta_path),
        })

    # manifest
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done.\n- RTTM: {rttm_dir}\n- SPKVEC: {spkvec_dir}\n- manifest: {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()
