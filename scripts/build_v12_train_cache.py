"""Cache the frozen V12 balanced prediction for V13-R training.

The script first verifies that its reconstruction reproduces an existing
validation prediction. Only after that check passes can it populate caches for
the training split. The test split is never read.
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from epg_synth.model_factory import build_direct_model
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.enhancement_predictor import EnhancementPredictor
from guided_diffusion.wfm import apply_dwt, apply_idwt


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA = PROJECT / "data/model_192x224x192"
OUTPUT = PROJECT / "data/v12_cache_20261029"
OLD_ENHANCER = (
    PROJECT
    / "runs/enh_predictor_v6_refined_20261027/checkpoints/enh_pred_v5_003500.pt"
)
V12_ENHANCER = (
    PROJECT
    / "runs/enh_predictor_v12_smooth_20261028/checkpoints/enh_pred_v5_003875.pt"
)
BASE_MODEL = (
    PROJECT
    / "runs/direct_residual_unet_20261028/checkpoints/direct_unet_002500.pt"
)
FINETUNED_MODEL = (
    PROJECT
    / "runs/direct_v12_finetune_20261028/checkpoints/direct_unet_000500.pt"
)
REFERENCE_VAL = (
    PROJECT
    / "runs/direct_v12_finetune_gated050_predictions_20261028"
)


def load_enhancer(path: Path, mode: str, device: torch.device):
    model = EnhancementPredictor(output_mode=mode)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def predict_v12(
    sample: dict,
    old_enhancer: torch.nn.Module,
    new_enhancer: torch.nn.Module,
    base_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    t1 = sample["t1n"].unsqueeze(0).to(device)
    t2 = sample["t2w"].unsqueeze(0).to(device)
    flair = sample["t2f"].unsqueeze(0).to(device)
    target = sample["t1c"].unsqueeze(0).to(device)
    condition = torch.cat(
        [apply_dwt(t1), apply_dwt(t2), apply_dwt(flair)],
        dim=1,
    )
    common = {"t1n": t1, "t2w": t2, "t2f": flair}
    old = old_enhancer(
        condition,
        torch.zeros(1, device=device),
        **common,
    )
    new = new_enhancer(
        condition,
        torch.zeros(1, device=device),
        **common,
    )
    old_enhancement = old["intensity"] * old["mask"]
    new_enhancement = new["enhancement"]

    source_base = torch.clamp(t1 + 0.5 * old_enhancement + 0.5 * new_enhancement, 0.0, 1.0)
    source_base_dwt = apply_dwt(source_base)
    y = torch.tensor([1], device=device, dtype=torch.long)
    base_residual = base_model(
        torch.cat([source_base_dwt, condition], dim=1),
        torch.zeros(1, device=device),
        y=y,
    )
    base_prediction = apply_idwt(
        source_base_dwt + base_residual
    ).clamp(0.0, 1.0)
    base_prediction[t1 == 0] = 0.0

    # The fine-tuned model was trained with the same 50/50 enhancement prior
    # used by the base direct model; the "V12" aspect is the fine-tuning,
    # not a change to a V12-only source.
    source_ft = source_base
    source_ft_dwt = apply_dwt(source_ft)
    ft_residual = finetuned_model(
        torch.cat([source_ft_dwt, condition], dim=1),
        torch.zeros(1, device=device),
        y=y,
    )
    ft_prediction = apply_idwt(
        source_ft_dwt + ft_residual
    ).clamp(0.0, 1.0)
    ft_prediction[t1 == 0] = 0.0

    gate = 0.5 * old["mask"] + 0.5 * new["mask"]
    gate = F.max_pool3d(
        gate,
        kernel_size=3,
        stride=1,
        padding=1,
    ).clamp(0.0, 1.0)
    balanced = (
        base_prediction + 0.5 * gate * (ft_prediction - base_prediction)
    ).clamp(0.0, 1.0)
    balanced[t1 == 0] = 0.0
    return {
        "source": t1,
        "target": target,
        "v12_base": base_prediction,
        "v12_finetuned": ft_prediction,
        "v12_balanced": balanced,
        "gate": gate,
        "old_enhancement": old_enhancement,
        "new_enhancement": new_enhancement,
    }


def run_precheck(
    old_enhancer,
    new_enhancer,
    base_model,
    finetuned_model,
    device,
) -> int:
    dataset = BRATSVolumes(
        str(DATA / "val"),
        mode="eval_target",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    sample = dataset[0]
    subject = sample["subj"]
    with torch.no_grad():
        result = predict_v12(
            sample,
            old_enhancer,
            new_enhancer,
            base_model,
            finetuned_model,
            device,
        )
    prediction = result["v12_balanced"][0, 0].cpu().numpy()
    reference_path = REFERENCE_VAL / subject / "sample_t1c.nii.gz"
    reference = np.asarray(
        nib.load(str(reference_path)).dataobj,
        dtype=np.float32,
    )
    difference = np.abs(prediction - reference)
    report = {
        "status": (
            "PASS"
            if float(difference.max()) <= 1e-3
            and float(difference.mean()) <= 1e-5
            else "FAIL"
        ),
        "subject": subject,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "prediction_range": [
            float(prediction.min()),
            float(prediction.max()),
        ],
        "reference_range": [
            float(reference.min()),
            float(reference.max()),
        ],
        "test_set_used": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "precheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


def build_cache(
    old_enhancer,
    new_enhancer,
    base_model,
    finetuned_model,
    device,
) -> int:
    precheck_path = OUTPUT / "precheck.json"
    if not precheck_path.exists():
        raise FileNotFoundError("Run precheck successfully first")
    precheck = json.loads(precheck_path.read_text(encoding="utf-8"))
    if precheck["status"] != "PASS":
        raise RuntimeError("V12 cache precheck did not pass")

    dataset = BRATSVolumes(
        str(DATA / "train"),
        mode="train",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    train_root = OUTPUT / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            subject = Path(dataset.database[index]["t1n"]).parent.name
            result = predict_v12(
                sample,
                old_enhancer,
                new_enhancer,
                base_model,
                finetuned_model,
                device,
            )
            case_dir = train_root / subject
            case_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "v12_base",
                "v12_finetuned",
                "v12_balanced",
                "gate",
            ):
                np.save(
                    case_dir / f"{name}.npy",
                    result[name][0, 0].detach().cpu().numpy().astype(
                        np.float16
                    ),
                    allow_pickle=False,
                )
            rows.append(
                {
                    "subject": subject,
                    **{
                        f"{name}_min": float(
                            result[name].min().item()
                        )
                        for name in (
                            "v12_base",
                            "v12_finetuned",
                            "v12_balanced",
                            "gate",
                        )
                    },
                    **{
                        f"{name}_max": float(
                            result[name].max().item()
                        )
                        for name in (
                            "v12_base",
                            "v12_finetuned",
                            "v12_balanced",
                            "gate",
                        )
                    },
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(f"cached={index + 1}/{len(dataset)}", flush=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(OUTPUT / "train_manifest.csv", index=False)
    (OUTPUT / "cache_summary.json").write_text(
        json.dumps(
            {
                "train_cases": len(rows),
                "test_cases_read": False,
                "format": "float16, model space (D,H,W) = 192x224x192",
                "precheck_subject": precheck["subject"],
                "precheck_max_abs_difference": precheck[
                    "max_abs_difference"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("precheck", "build"),
        required=True,
    )
    args = parser.parse_args()
    device = torch.device("cuda:0")
    old_enhancer = load_enhancer(OLD_ENHANCER, "soft_mask", device)
    new_enhancer = load_enhancer(V12_ENHANCER, "smooth_mask", device)
    base_model = build_direct_model()
    base_model.load_state_dict(
        torch.load(BASE_MODEL, map_location="cpu", weights_only=True)
    )
    base_model.to(device)
    base_model.eval()
    finetuned_model = build_direct_model()
    finetuned_model.load_state_dict(
        torch.load(FINETUNED_MODEL, map_location="cpu", weights_only=True)
    )
    finetuned_model.to(device)
    finetuned_model.eval()
    if args.mode == "precheck":
        return run_precheck(
            old_enhancer,
            new_enhancer,
            base_model,
            finetuned_model,
            device,
        )
    return build_cache(
        old_enhancer,
        new_enhancer,
        base_model,
        finetuned_model,
        device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
