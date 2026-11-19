"""Evaluate anatomy-preserving gates for the V12 fine-tuned direct model."""

from __future__ import annotations

import os

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from epg_synth.metrics import correlation
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.enhancement_predictor import EnhancementPredictor
from guided_diffusion.wfm import apply_dwt


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
VAL = PROJECT / "data/model_192x224x192/val"
BASE = PROJECT / "runs/direct_v12_prior_blend050_predictions_20261028"
FT = PROJECT / "runs/direct_v12_finetune_best_predictions_20261028"
OLD_ENHANCER = (
    PROJECT
    / "runs/enh_predictor_v6_refined_20261027/checkpoints/enh_pred_v5_003500.pt"
)
V12_ENHANCER = (
    PROJECT
    / "runs/enh_predictor_v12_smooth_20261028/checkpoints/enh_pred_v5_003875.pt"
)
OUTPUT = PROJECT / "runs/direct_v12_finetune_gating_20261028"
FOCUSED_ROOT = PROJECT / "runs/direct_v12_finetune_global075_predictions_20261028"
BALANCED_ROOT = PROJECT / "runs/direct_v12_finetune_gated050_predictions_20261028"


def load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def load_enhancer(path, mode, device):
    model = EnhancementPredictor(output_mode=mode)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def save_prediction(prediction, reference_path, destination):
    reference = nib.load(str(reference_path))
    image = nib.Nifti1Image(
        np.clip(prediction, 0.0, 1.0).astype(np.float32),
        reference.affine,
    )
    image.set_qform(None)
    image.set_sform(reference.affine, code=1)
    image.set_data_dtype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(destination))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FOCUSED_ROOT.mkdir(parents=True, exist_ok=True)
    BALANCED_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    old_model = load_enhancer(OLD_ENHANCER, "soft_mask", device)
    v12_model = load_enhancer(V12_ENHANCER, "smooth_mask", device)
    dataset = BRATSVolumes(
        str(VAL),
        mode="eval_target",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    alphas = (0.25, 0.50, 0.75, 1.00)
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            subject = sample["subj"]
            t1 = sample["t1n"].unsqueeze(0).to(device)
            t2 = sample["t2w"].unsqueeze(0).to(device)
            flair = sample["t2f"].unsqueeze(0).to(device)
            full_target = sample["t1c"].unsqueeze(0).to(device)
            target_enh = torch.clamp(full_target - t1, 0.0, 1.0)
            condition = torch.cat(
                [apply_dwt(t1), apply_dwt(t2), apply_dwt(flair)],
                dim=1,
            )
            common = {"t1n": t1, "t2w": t2, "t2f": flair}
            old_output = old_model(
                condition,
                torch.zeros(1, device=device),
                **common,
            )
            v12_output = v12_model(
                condition,
                torch.zeros(1, device=device),
                **common,
            )
            gate = 0.5 * old_output["mask"] + 0.5 * v12_output["mask"]
            gate = torch.nn.functional.max_pool3d(
                gate,
                kernel_size=3,
                stride=1,
                padding=1,
            ).clamp(0.0, 1.0)
            base = torch.from_numpy(
                load(BASE / subject / "sample_t1c.nii.gz")
            )[None, None].to(device)
            ft = torch.from_numpy(
                load(FT / subject / "sample_t1c.nii.gz")
            )[None, None].to(device)
            roi = (target_enh > 0.15) & (t1 > 0.05)
            brain = t1 > 0
            full_target_np = full_target[0, 0].cpu().numpy()
            target_np = target_enh[0, 0].cpu().numpy()
            t1_np = t1[0, 0].cpu().numpy()
            brain_np = brain[0, 0].cpu().numpy()
            gt_mask = target_np > 0.15
            candidates = {"baseline": base}
            for alpha in alphas:
                candidates[f"global_{alpha:.2f}"] = (
                    base + alpha * (ft - base)
                ).clamp(0.0, 1.0)
                candidates[f"gated_{alpha:.2f}"] = (
                    base + alpha * gate * (ft - base)
                ).clamp(0.0, 1.0)
            for method, prediction in candidates.items():
                prediction[~brain] = 0.0
                pred_enh = torch.clamp(prediction - t1, 0.0, 1.0)
                prediction_np = prediction[0, 0].cpu().numpy()
                pred_mask = pred_enh[0, 0].cpu().numpy() > 0.15
                inter = (pred_mask & gt_mask).sum()
                dice = (2.0 * inter + 1e-6) / (
                    pred_mask.sum() + gt_mask.sum() + 1e-6
                )
                mse = float(
                    ((prediction_np - full_target_np) ** 2)[brain_np].mean()
                )
                rows.append(
                    {
                        "method": method,
                        "subject": subject,
                        "r_native": correlation(
                            pred_enh[roi].cpu().numpy(),
                            target_enh[roi].cpu().numpy(),
                        ),
                        "mask_dice": float(dice),
                        "ROI_ratio": float(
                            prediction_np[gt_mask].mean()
                            / max(full_target_np[gt_mask].mean(), 1e-8)
                        ),
                        "PSNR_FG": float(
                            10.0 * np.log10(1.0 / max(mse, 1e-12))
                        ),
                    }
                )
                if method == "global_0.75":
                    save_prediction(
                        prediction_np,
                        VAL / subject / f"{subject}_T1.nii.gz",
                        FOCUSED_ROOT / subject / "sample_t1c.nii.gz",
                    )
                if method == "gated_0.50":
                    save_prediction(
                        prediction_np,
                        VAL / subject / f"{subject}_T1.nii.gz",
                        BALANCED_ROOT / subject / "sample_t1c.nii.gz",
                    )
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(f"evaluated={index + 1}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "per_case_metrics.csv", index=False, float_format="%.6f")
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            r_native_mean=("r_native", "mean"),
            mask_dice_mean=("mask_dice", "mean"),
            ROI_ratio_mean=("ROI_ratio", "mean"),
            PSNR_FG_mean=("PSNR_FG", "mean"),
            n=("subject", "count"),
        )
    )
    summary["score"] = (
        0.55 * summary["r_native_mean"]
        + 0.30 * (summary["PSNR_FG_mean"] / 30.0)
        + 0.15 * (1.0 - (summary["ROI_ratio_mean"] - 1.0).abs())
    )
    summary = summary.sort_values("score", ascending=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, float_format="%.6f")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
