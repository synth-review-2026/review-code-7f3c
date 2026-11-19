"""Frozen single-entry inference for the V12-Balanced model.

Inputs are preprocessed T1n, T2 and FLAIR tensors/volumes. Internally the
module runs the old and V12 enhancement predictors, the base direct residual
model, and the existing V12 fine-tuned model. The HF correction branch was
retired after paired internal and downstream evaluation showed no material
benefit.
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
CONFIG_PATH = PROJECT / "configs" / "v12_core_config.json"
CHECKPOINT_ROOT = Path(
    os.environ.get("EPG_SYNTH_CHECKPOINTS", PROJECT / "checkpoints")
).expanduser().resolve()
OLD_ENHANCER = CHECKPOINT_ROOT / "enh_pred_v5_003500.pt"
V12_ENHANCER = CHECKPOINT_ROOT / "enh_pred_v5_003875.pt"
BASE_DIRECT = CHECKPOINT_ROOT / "direct_unet_002500.pt"
V12_FINETUNED = CHECKPOINT_ROOT / "direct_unet_000500.pt"
EXPECTED_BALANCED = Path(
    os.environ.get(
        "EPG_SYNTH_PRECHECK_REFERENCE",
        PROJECT / "examples" / "precheck_predictions",
    )
).expanduser().resolve()


def load_enhancer(path: Path, mode: str, device: torch.device):
    model = EnhancementPredictor(output_mode=mode)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_direct_model(path: Path, device: torch.device):
    model = build_direct_model()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class V12CoreInference:
    """Frozen V12-Balanced inference contract."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.old_enhancer = load_enhancer(
            OLD_ENHANCER,
            "soft_mask",
            self.device,
        )
        self.v12_enhancer = load_enhancer(
            V12_ENHANCER,
            "smooth_mask",
            self.device,
        )
        self.base_model = load_direct_model(BASE_DIRECT, self.device)
        self.v12_model = load_direct_model(V12_FINETUNED, self.device)

    @torch.no_grad()
    def predict(
        self,
        t1n: torch.Tensor,
        t2w: torch.Tensor,
        t2f: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if t1n.ndim != 5:
            raise ValueError(
                f"Expected T1n shape [B,1,D,H,W], got {tuple(t1n.shape)}"
            )
        if t1n.shape != t2w.shape or t1n.shape != t2f.shape:
            raise ValueError("T1n/T2/FLAIR shape mismatch")
        if t1n.shape[1] != 1 or tuple(t1n.shape[2:]) != (192, 224, 192):
            raise ValueError(
                "Expected input shape [B,1,192,224,192], got "
                f"{tuple(t1n.shape)}"
            )
        for name, tensor in (("T1n", t1n), ("T2", t2w), ("FLAIR", t2f)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
            if tensor.max() > 1.01 or tensor.min() < -0.01:
                raise ValueError(
                    f"{name} must use preprocessed [0,1] scale"
                )

        t1n = t1n.to(self.device)
        t2w = t2w.to(self.device)
        t2f = t2f.to(self.device)
        with torch.profiler.record_function("v12:input_wavelets"):
            condition = torch.cat(
                [apply_dwt(t1n), apply_dwt(t2w), apply_dwt(t2f)],
                dim=1,
            )
        common = {"t1n": t1n, "t2w": t2w, "t2f": t2f}
        zeros = torch.zeros(t1n.shape[0], device=self.device)
        with torch.profiler.record_function("v12:enhancement_priors"):
            old = self.old_enhancer(condition, zeros, **common)
            new = self.v12_enhancer(condition, zeros, **common)
            old_map = old["intensity"] * old["mask"]
            new_map = new["enhancement"]
        with torch.profiler.record_function("v12:source_prior"):
            source = torch.clamp(
                t1n
                + self.config["source_prior"]["old_enhancer_weight"] * old_map
                + self.config["source_prior"]["v12_enhancer_weight"] * new_map,
                0.0,
                1.0,
            )
            source_dwt = apply_dwt(source)
            model_input = torch.cat([source_dwt, condition], dim=1)
        y = torch.ones(
            t1n.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        with torch.profiler.record_function("v12:base_residual"):
            base_residual = self.base_model(
                model_input,
                zeros,
                y=y,
            )
            base_prediction = torch.clamp(
                apply_idwt(source_dwt + base_residual),
                0.0,
                1.0,
            )
        with torch.profiler.record_function("v12:v12_residual"):
            v12_residual = self.v12_model(
                model_input,
                zeros,
                y=y,
            )
            v12_prediction = torch.clamp(
                apply_idwt(source_dwt + v12_residual),
                0.0,
                1.0,
            )
        with torch.profiler.record_function("v12:gated_fusion"):
            gate = 0.5 * old["mask"] + 0.5 * new["mask"]
            gate = F.max_pool3d(
                gate,
                kernel_size=3,
                stride=1,
                padding=1,
            ).clamp(0.0, 1.0)
            balanced = torch.clamp(
                base_prediction
                + self.config["v12_gate_alpha"]
                * gate
                * (v12_prediction - base_prediction),
                0.0,
                1.0,
            )
        prediction = balanced.clone()
        prediction[t1n == 0] = 0.0
        return {
            "prediction": prediction,
            "v12_balanced": balanced,
            "gate": gate,
            "old_enhancement": old_map,
            "new_enhancement": new_map,
            "source": source,
        }

    def predict_volume(
        self,
        t1n: np.ndarray,
        t2w: np.ndarray,
        t2f: np.ndarray,
    ) -> np.ndarray:
        tensors = [
            torch.from_numpy(
                np.asarray(value, dtype=np.float32)[None, None]
            )
            for value in (t1n, t2w, t2f)
        ]
        output = self.predict(*tensors)
        return output["prediction"][0, 0].cpu().numpy()


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def precheck_cases(device: str) -> int:
    dataset = BRATSVolumes(
        str(
            Path(
                os.environ.get(
                    "EPG_SYNTH_VAL_DATA",
                    PROJECT / "data" / "model_192x224x192" / "val",
                )
            ).expanduser().resolve()
        ),
        mode="eval_target",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    sample = dataset[0]
    subject = sample["subj"]
    bundle = V12CoreInference(device=device)
    result = bundle.predict(
        sample["t1n"].unsqueeze(0),
        sample["t2w"].unsqueeze(0),
        sample["t2f"].unsqueeze(0),
    )
    prediction = result["prediction"][0, 0].cpu().numpy()
    reference = load_nifti(
        EXPECTED_BALANCED / subject / "sample_t1c.nii.gz"
    )
    difference = np.abs(prediction - reference)
    fraction_large_difference = float(
        np.mean(difference > 0.01)
    )
    report = {
        "status": (
            "PASS"
            if float(difference.max()) <= 0.05
            and float(difference.mean()) <= 1e-5
            and fraction_large_difference <= 1e-4
            else "FAIL"
        ),
        "subject": subject,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "fraction_abs_difference_gt_0.01": fraction_large_difference,
        "prediction_range": [
            float(prediction.min()),
            float(prediction.max()),
        ],
        "test_set_used": False,
    }
    output = PROJECT / "runs/v12_balanced_integration_precheck_20261102"
    output.mkdir(parents=True, exist_ok=True)
    (output / "precheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


def run_nifti(
    input_dir: Path,
    output_dir: Path,
    device: str,
    subjects: list[str] | None = None,
) -> int:
    bundle = V12CoreInference(device=device)
    selected = set(subjects) if subjects else None
    cases = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir()
        and (selected is None or path.name in selected)
    )
    if selected is not None and {
        path.name for path in cases
    } != selected:
        raise ValueError(
            "Requested subjects are missing from input_dir: "
            + ", ".join(sorted(selected - {path.name for path in cases}))
        )
    for index, case_dir in enumerate(cases, start=1):
        subject = case_dir.name
        t1_path = case_dir / f"{subject}_T1.nii.gz"
        t2_path = case_dir / f"{subject}_T2.nii.gz"
        flair_path = case_dir / f"{subject}_FLAIR.nii.gz"
        t1_image = nib.load(str(t1_path))
        prediction = bundle.predict_volume(
            load_nifti(t1_path),
            load_nifti(t2_path),
            load_nifti(flair_path),
        )
        image = nib.Nifti1Image(prediction, t1_image.affine)
        image.set_qform(None)
        image.set_sform(t1_image.affine, code=1)
        image.set_data_dtype(np.float32)
        destination = output_dir / subject / "sample_t1c.nii.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, str(destination))
        print(f"predicted {index}/{len(cases)} {subject}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("precheck", "run"), required=True)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "precheck":
        return precheck_cases(args.device)
    if args.input_dir is None or args.output_dir is None:
        parser.error("--input_dir and --output_dir are required for run")
    return run_nifti(
        args.input_dir,
        args.output_dir,
        args.device,
        args.subjects,
    )


if __name__ == "__main__":
    raise SystemExit(main())
