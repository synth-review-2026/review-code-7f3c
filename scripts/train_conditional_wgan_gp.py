"""3D conditional WGAN-GP baseline for meningioma T1c synthesis."""

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
from guided_diffusion.script_util import create_model


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA = PROJECT / "data/model_192x224x192"
DEFAULT_OUTPUT = PROJECT / "runs/conditional_wgan_gp_3d_20261029"
PATCH_SHAPE = (96, 96, 96)


class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels: int = 4, width: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels, width, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, width * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width * 2, width * 4, 4, stride=2, padding=1),
            nn.GroupNorm(16, width * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width * 4, 1, 3, stride=1, padding=1),
        )

    def forward(self, condition, image):
        return self.network(torch.cat([condition, image], dim=1))


def build_generator(use_checkpoint: bool):
    return create_model(
        image_size=224,
        num_channels=32,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 4),
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=use_checkpoint,
        attention_resolutions="",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        dims=3,
        num_groups=16,
        in_channels=3,
        out_channels=1,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
    )


def random_crop(batch, roi):
    target_shape = roi.shape[2:]
    positive = roi[0, 0].nonzero()
    center = None
    if len(positive) and random.random() < 0.85:
        selected = positive[torch.randint(len(positive), (1,))]
        selected = selected[0]
        jitter = torch.randint(
            -8,
            9,
            (3,),
            device=selected.device,
        )
        center = torch.clamp(
            selected + jitter,
            torch.zeros(3, dtype=torch.long, device=selected.device),
            torch.tensor(
                target_shape,
                dtype=torch.long,
                device=selected.device,
            )
            - 1,
        )
    starts = []
    for dimension, patch_size in enumerate(PATCH_SHAPE):
        maximum = target_shape[dimension] - patch_size
        start = (
            random.randint(0, maximum)
            if center is None
            else int(
                max(
                    0,
                    min(
                        maximum,
                        int(center[dimension]) - patch_size // 2,
                    ),
                )
            )
        )
        starts.append(start)
    slices = tuple(
        slice(start, start + patch_size)
        for start, patch_size in zip(starts, PATCH_SHAPE)
    )
    selected = {
        key: value[(slice(None), slice(None), *slices)]
        for key, value in batch.items()
        if torch.is_tensor(value) and value.ndim == 5
    }
    return selected


def gradient_penalty(discriminator, condition, real, fake, device):
    alpha = torch.rand(real.shape[0], 1, 1, 1, 1, device=device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    score = discriminator(condition, interpolated)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradient.reshape(gradient.shape[0], -1).norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("precheck", "train"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--critic_steps", type=int, default=3)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--lambda_l1", type=float, default=50.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261029)
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
        betas=(0.5, 0.9),
    )
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.9),
    )
    if resume_state is not None:
        if resume_state.get("optimizer_g") is not None:
            optimizer_g.load_state_dict(resume_state["optimizer_g"])
        if resume_state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(resume_state["optimizer_d"])
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
            residual = generator(
                condition,
                torch.zeros(
                    condition.shape[0],
                    device=device,
                    dtype=torch.long,
                ),
            )
            fake = torch.clamp(batch["t1n"] + residual, 0.0, 1.0)

            optimizer_d.zero_grad(set_to_none=True)
            real_score = discriminator(condition, batch["t1c"]).mean()
            fake_score = discriminator(condition, fake.detach()).mean()
            gp = gradient_penalty(
                discriminator,
                condition,
                batch["t1c"],
                fake.detach(),
                device,
            )
            loss_d = fake_score - real_score + args.lambda_gp * gp
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            adversarial = -discriminator(condition, fake).mean()
            reconstruction = F.l1_loss(fake, batch["t1c"])
            loss_g = adversarial + args.lambda_l1 * reconstruction
            loss_g.backward()
            optimizer_g.step()
            print(
                f"precheck step={step} d={float(loss_d):.6f} "
                f"g={float(loss_g):.6f} gp={float(gp):.6f}",
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
        raise RuntimeError("WGAN-GP precheck did not pass")
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
            fake_for_critic = torch.clamp(
                batch["t1n"]
                + generator(
                    condition,
                    torch.zeros(
                        condition.shape[0],
                        device=device,
                        dtype=torch.long,
                    ),
                ),
                0.0,
                1.0,
            )
        d_losses = []
        if step > args.warmup_steps:
            discriminator.train()
            for _ in range(args.critic_steps):
                optimizer_d.zero_grad(set_to_none=True)
                real_score = discriminator(condition, batch["t1c"]).mean()
                fake_score = discriminator(
                    condition,
                    fake_for_critic,
                ).mean()
                gp = gradient_penalty(
                    discriminator,
                    condition,
                    batch["t1c"],
                    fake_for_critic,
                    device,
                )
                loss_d = fake_score - real_score + args.lambda_gp * gp
                loss_d.backward()
                optimizer_d.step()
                d_losses.append(float(loss_d.detach()))

        generator.train()
        optimizer_g.zero_grad(set_to_none=True)
        residual = generator(
            condition,
            torch.zeros(
                condition.shape[0],
                device=device,
                dtype=torch.long,
            ),
        )
        fake = torch.clamp(batch["t1n"] + residual, 0.0, 1.0)
        reconstruction = F.l1_loss(fake, batch["t1c"])
        if step <= args.warmup_steps:
            loss_g = args.lambda_l1 * reconstruction
            adversarial_value = 0.0
        else:
            adversarial = -discriminator(condition, fake).mean()
            loss_g = adversarial + args.lambda_l1 * reconstruction
            adversarial_value = float(adversarial.detach())
        loss_g.backward()
        optimizer_g.step()
        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - started
            writer.add_scalar("train/g", float(loss_g), step)
            writer.add_scalar("train/l1", float(reconstruction), step)
            writer.add_scalar("train/adversarial", adversarial_value, step)
            print(
                f"step={step} g={float(loss_g):.6f} "
                f"l1={float(reconstruction):.6f} "
                f"adv={adversarial_value:.6f} "
                f"d={sum(d_losses)/max(len(d_losses),1):.6f} "
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
            path = args.output_dir / "checkpoints" / f"wgan_gp_{step:06d}.pt"
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
