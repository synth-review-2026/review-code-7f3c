"""Train a T1C-RFlow baseline on the project's internal splits."""

from __future__ import annotations

import os

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.schedulers import RFlowScheduler


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
LATENTS = PROJECT / "data/t1c_rflow/latents_192x224x192"
DEFAULT_OUTPUT = PROJECT / "runs/t1c_rflow_192x224x192_20261103"


class LatentCaseDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        cache_in_memory: bool,
        limit: int | None = None,
    ) -> None:
        self.cases = []
        for case_dir in sorted(
            path
            for path in split_dir.iterdir()
            if path.is_dir()
        ):
            subject = case_dir.name
            paths = {}
            for modality in ("t1n", "t2f", "t1c"):
                paths[f"{modality}_mean"] = (
                    case_dir
                    / f"{subject}-{modality}_z_mu.pt"
                )
                paths[f"{modality}_sigma"] = (
                    case_dir
                    / f"{subject}-{modality}_z_sigma.pt"
                )
            if not all(path.exists() for path in paths.values()):
                raise FileNotFoundError(
                    f"Incomplete latents for {subject}"
                )
            self.cases.append(
                {
                    "subject": subject,
                    "paths": paths,
                }
            )
        if not self.cases:
            raise ValueError(
                f"No latent cases found under {split_dir}"
            )
        if limit is not None:
            self.cases = self.cases[:limit]
        self.cache = {}
        if cache_in_memory:
            for case in tqdm(
                self.cases,
                desc=f"cache {split_dir.name}",
                unit="case",
            ):
                self.cache[case["subject"]] = {
                    key: torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    for key, path in case["paths"].items()
                }

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]
        if self.cache:
            values = self.cache[case["subject"]]
        else:
            values = {
                key: torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
                for key, path in case["paths"].items()
            }
        return {"subject": case["subject"], **values}


def sample_latents(
    batch: dict,
    modality: str,
    stochastic: bool,
) -> torch.Tensor:
    mean = batch[f"{modality}_mean"]
    if not stochastic:
        return mean
    sigma = batch[f"{modality}_sigma"]
    return mean + sigma * torch.randn_like(sigma)


def make_model(device: torch.device) -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=12,
        out_channels=4,
        num_channels=(128, 128, 256),
        attention_levels=(False, True, True),
        num_head_channels=(0, 32, 32),
        num_res_blocks=2,
        norm_num_groups=32,
        resblock_updown=True,
        include_fc=True,
        use_flash_attention=True,
    ).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = LatentCaseDataset(
        LATENTS / "train",
        cache_in_memory=True,
        limit=args.limit_train,
    )
    val_dataset = LatentCaseDataset(
        LATENTS / "val",
        cache_in_memory=True,
        limit=args.limit_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = make_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scaler = GradScaler("cuda")
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=True,
        sample_method="logit-normal",
        use_timestep_transform=True,
        base_img_size_numel=48 * 56 * 48,
        spatial_dim=3,
    )

    start_epoch = 0
    best_val = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(
            str(args.resume),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint.get("unet", checkpoint),
            strict=True,
        )
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
        start_epoch = int(
            checkpoint.get("epoch", -1)
        ) + 1
        best_val = float(
            checkpoint.get("val_l1", best_val)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        global_step = int(
            checkpoint.get("global_step", 0)
        )

    history_path = output / "loss_history.csv"
    if not history_path.exists():
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "global_step",
                    "train_l1",
                    "val_l1",
                    "seconds",
                ]
            )
    metadata = {
        "model": "T1C-RFlow",
        "source_commit": (
            "fc8314f60d877f9ee55996f960f89b17b269200f"
        ),
        "implementation": (
            "paper-faithful reimplementation using official "
            "MAISI VAE and MONAI RFlow scheduler"
        ),
        "training_split": "internal train (410 cases)",
        "validation_split": "internal validation (87 cases)",
        "test_split_locked": "internal test (89 cases)",
        "input_conditions": ["T1n", "T2-FLAIR"],
        "target": "T1c",
        "input_shape": [192, 224, 192],
        "latent_shape": [4, 48, 56, 48],
        "model_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
            )
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "loss": "L1 velocity loss",
        "sampling_steps": 200,
        "test_tuning": False,
        "pretrained_rflow_checkpoint_available": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        losses = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            leave=False,
        )
        for batch in progress:
            target_clean = sample_latents(
                batch,
                "t1c",
                True,
            ).to(
                device,
                non_blocking=True,
                dtype=torch.float16,
            )
            cond_t1 = sample_latents(
                batch,
                "t1n",
                True,
            ).to(
                device,
                non_blocking=True,
                dtype=torch.float16,
            )
            cond_flair = sample_latents(
                batch,
                "t2f",
                True,
            ).to(
                device,
                non_blocking=True,
                dtype=torch.float16,
            )
            noise = torch.randn_like(target_clean)
            timesteps = scheduler.sample_timesteps(
                target_clean
            ).long().to(device)
            noisy = scheduler.add_noise(
                target_clean,
                noise,
                timesteps,
            )
            # MONAI's RFlow scheduler advances from noise to the clean
            # sample with a positive time decrement.
            target_velocity = target_clean - noise
            model_input = torch.cat(
                [noisy, cond_t1, cond_flair],
                dim=1,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                prediction = model(
                    model_input,
                    timesteps,
                )
                loss = F.l1_loss(
                    prediction.float(),
                    target_velocity.float(),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            global_step += 1
            progress.set_postfix(
                loss=f"{losses[-1]:.5f}"
            )

        model.eval()
        validation_losses = []
        validation_generator = torch.Generator(
            device=device
        )
        validation_generator.manual_seed(
            args.seed + epoch
        )
        with torch.inference_mode():
            for batch in val_loader:
                target_clean = sample_latents(
                    batch,
                    "t1c",
                    False,
                ).to(
                    device,
                    dtype=torch.float16,
                )
                cond_t1 = sample_latents(
                    batch,
                    "t1n",
                    False,
                ).to(
                    device,
                    dtype=torch.float16,
                )
                cond_flair = sample_latents(
                    batch,
                    "t2f",
                    False,
                ).to(
                    device,
                    dtype=torch.float16,
                )
                timesteps = torch.full(
                    (target_clean.shape[0],),
                    500,
                    dtype=torch.long,
                    device=device,
                )
                noise = torch.randn(
                    target_clean.shape,
                    generator=validation_generator,
                    device=device,
                    dtype=torch.float16,
                )
                noisy = scheduler.add_noise(
                    target_clean,
                    noise,
                    timesteps,
                )
                target_velocity = target_clean - noise
                with autocast("cuda", dtype=torch.float16):
                    prediction = model(
                        torch.cat(
                            [noisy, cond_t1, cond_flair],
                            dim=1,
                        ),
                        timesteps,
                    )
                    validation_losses.append(
                        float(
                            F.l1_loss(
                                prediction.float(),
                                target_velocity.float(),
                            ).item()
                        )
                    )
        train_loss = float(np.mean(losses))
        val_loss = float(
            np.mean(validation_losses)
        )
        elapsed = time.time() - started
        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss,
                    elapsed,
                ]
            )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "unet": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_l1": val_loss,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "metadata": metadata,
        }
        torch.save(
            checkpoint,
            output / "unet_latest.pt",
        )
        if improved:
            best_checkpoint = {
                "unet": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_l1": val_loss,
                "metadata": metadata,
            }
            torch.save(
                best_checkpoint,
                output / "unet_best.pt",
            )
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f} "
            f"best={best_val:.6f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                "early stopping: validation L1 has not improved "
                f"for {args.patience} epochs",
                flush=True,
            )
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
