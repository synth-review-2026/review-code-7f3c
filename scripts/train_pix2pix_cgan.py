"""3D conditional Pix2Pix baseline for meningioma T1c synthesis."""

from __future__ import annotations

import os

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion.bratsloader import BRATSVolumes
from train_conditional_wgan_gp import (
    PATCH_SHAPE,
    PatchDiscriminator3D,
    build_generator,
    random_crop,
)


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA = PROJECT / "data/model_192x224x192"
DEFAULT_OUTPUT = PROJECT / "runs/pix2pix_cgan_3d_20261030"


def zero_timesteps(condition: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        condition.shape[0],
        device=condition.device,
        dtype=torch.long,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("precheck", "train"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_final", type=float, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    dataset = BRATSVolumes(
        str(DATA / "train"),
        mode="train",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    generator = build_generator(args.use_checkpoint)
    generator.to(device)
    discriminator = PatchDiscriminator3D().to(device)
    resume_state = None
    start_step = 0
    if args.resume_checkpoint is not None:
        if args.mode != "train":
            raise ValueError("--resume_checkpoint requires --mode train")
        resume_state = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        generator.load_state_dict(resume_state["generator"])
        discriminator.load_state_dict(resume_state["discriminator"])
        start_step = int(resume_state.get("step", 0))
    optimizer_g = Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
    )
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
    )
    if resume_state is not None:
        if resume_state.get("optimizer_g") is not None:
            optimizer_g.load_state_dict(resume_state["optimizer_g"])
        if resume_state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(resume_state["optimizer_d"])
    adversarial_loss = nn.BCEWithLogitsLoss()
    iterator = iter(loader)

    if args.mode == "precheck":
        started = time.time()
        for step in range(1, 3):
            batch = next(iterator)
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            roi = (
                (batch["t1c"] - batch["t1n"] > 0.15)
                & (batch["t1n"] > 0.05)
            )
            batch = random_crop(batch, roi)
            condition = torch.cat(
                [batch["t1n"], batch["t2w"], batch["t2f"]],
                dim=1,
            )
            residual = generator(condition, zero_timesteps(condition))
            fake = torch.clamp(batch["t1n"] + residual, 0.0, 1.0)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(condition, batch["t1c"])
            fake_logits = discriminator(condition, fake.detach())
            loss_d = 0.5 * (
                adversarial_loss(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                + adversarial_loss(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
            )
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(condition, fake)
            loss_g = adversarial_loss(
                fake_logits,
                torch.ones_like(fake_logits),
            ) + args.lambda_l1 * F.l1_loss(fake, batch["t1c"])
            loss_g.backward()
            optimizer_g.step()
            print(
                f"precheck step={step} d={float(loss_d):.6f} "
                f"g={float(loss_g):.6f}",
                flush=True,
            )
        report = {
            "status": "PASS",
            "batch_size": args.batch_size,
            "patch_shape": PATCH_SHAPE,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "seconds": time.time() - started,
            "test_set_used": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "precheck.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    precheck = json.loads(
        (args.output_dir / "precheck.json").read_text(encoding="utf-8")
    )
    if precheck["status"] != "PASS":
        raise RuntimeError("Pix2Pix precheck did not pass")
    writer = SummaryWriter(args.output_dir / "tb")
    started = time.time()
    if args.max_iterations <= start_step:
        raise ValueError(
            f"--max_iterations={args.max_iterations} must exceed "
            f"resume step {start_step}"
        )
    for step in range(start_step + 1, args.max_iterations + 1):
        if args.lr_final is not None and step > args.lr_decay_start:
            decay_span = max(
                args.max_iterations - args.lr_decay_start,
                1,
            )
            progress = min(
                (step - args.lr_decay_start) / decay_span,
                1.0,
            )
            current_lr_g = args.lr_g + progress * (
                args.lr_final - args.lr_g
            )
            current_lr_d = args.lr_d + progress * (
                args.lr_final - args.lr_d
            )
            for group in optimizer_g.param_groups:
                group["lr"] = current_lr_g
            for group in optimizer_d.param_groups:
                group["lr"] = current_lr_d
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        roi = (
            (batch["t1c"] - batch["t1n"] > 0.15)
            & (batch["t1n"] > 0.05)
        )
        batch = random_crop(batch, roi)
        condition = torch.cat(
            [batch["t1n"], batch["t2w"], batch["t2f"]],
            dim=1,
        )

        generator.eval()
        with torch.no_grad():
            fake_for_discriminator = torch.clamp(
                batch["t1n"]
                + generator(condition, zero_timesteps(condition)),
                0.0,
                1.0,
            )
        discriminator.train()
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(condition, batch["t1c"])
        fake_logits = discriminator(
            condition,
            fake_for_discriminator,
        )
        loss_d = 0.5 * (
            adversarial_loss(
                real_logits,
                torch.ones_like(real_logits),
            )
            + adversarial_loss(
                fake_logits,
                torch.zeros_like(fake_logits),
            )
        )
        loss_d.backward()
        optimizer_d.step()

        generator.train()
        optimizer_g.zero_grad(set_to_none=True)
        residual = generator(condition, zero_timesteps(condition))
        fake = torch.clamp(batch["t1n"] + residual, 0.0, 1.0)
        reconstruction = F.l1_loss(fake, batch["t1c"])
        if step <= args.warmup_steps:
            loss_g = args.lambda_l1 * reconstruction
            adversarial_value = 0.0
        else:
            fake_logits = discriminator(condition, fake)
            adversarial = adversarial_loss(
                fake_logits,
                torch.ones_like(fake_logits),
            )
            loss_g = adversarial + args.lambda_l1 * reconstruction
            adversarial_value = float(adversarial.detach())
        loss_g.backward()
        optimizer_g.step()

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - started
            writer.add_scalar("train/g", float(loss_g), step)
            writer.add_scalar("train/l1", float(reconstruction), step)
            writer.add_scalar("train/adversarial", adversarial_value, step)
            writer.add_scalar("train/d", float(loss_d), step)
            print(
                f"step={step} g={float(loss_g):.6f} "
                f"l1={float(reconstruction):.6f} "
                f"adv={adversarial_value:.6f} "
                f"d={float(loss_d):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if step % args.save_interval == 0:
            checkpoint = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "step": step,
            }
            path = args.output_dir / "checkpoints" / f"pix2pix_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)

    summary = {
        "status": "complete",
        "iterations": args.max_iterations,
        "start_step": start_step,
        "resume_checkpoint": (
            str(args.resume_checkpoint)
            if args.resume_checkpoint is not None
            else None
        ),
        "patch_shape": PATCH_SHAPE,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "loss": "conditional BCE adversarial + 100 * L1",
        "test_set_used": False,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    writer.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
