"""Train a direct residual 3D U-Net with the same inputs as the WFM model."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.enhancement_predictor import (
    EnhancementPredictor,
    select_enhancement,
)
from guided_diffusion.script_util import create_model
from guided_diffusion.wfm import apply_dwt, apply_idwt


def finite_difference_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (2, 3, 4):
        losses.append(
            F.smooth_l1_loss(
                prediction.diff(dim=dimension),
                target.diff(dim=dimension),
                beta=0.02,
            )
        )
    return torch.stack(losses).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--enh_predictor_path", required=True)
    parser.add_argument("--new_enh_predictor_path", default="")
    parser.add_argument("--new_prior_weight", type=float, default=0.0)
    parser.add_argument(
        "--new_enh_head",
        choices=["intensity", "soft_mask", "offset_mask", "smooth_mask"],
        default="soft_mask",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--roi_loss_weight", type=float, default=0.0)
    parser.add_argument("--detail_loss_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--use_checkpoint", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    teacher = EnhancementPredictor(output_mode="soft_mask")
    teacher.load_state_dict(torch.load(args.enh_predictor_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    new_teacher = None
    if args.new_enh_predictor_path:
        if not 0.0 <= args.new_prior_weight <= 1.0:
            raise ValueError("--new_prior_weight must be in [0, 1]")
        new_teacher = EnhancementPredictor(output_mode="soft_mask")
        new_teacher.load_state_dict(
            torch.load(args.new_enh_predictor_path, map_location="cpu")
        )
        new_teacher.to(device)
        new_teacher.eval()
        for parameter in new_teacher.parameters():
            parameter.requires_grad_(False)
    elif args.new_prior_weight != 0.0:
        raise ValueError(
            "--new_prior_weight requires --new_enh_predictor_path"
        )

    model = create_model(
        image_size=224,
        num_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 4, 4),
        learn_sigma=False,
        class_cond=True,
        use_checkpoint=args.use_checkpoint,
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
        num_groups=32,
        in_channels=32,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
    )
    if args.resume_checkpoint:
        model.load_state_dict(
            torch.load(args.resume_checkpoint, map_location="cpu")
        )
        print(f"resumed direct model from {args.resume_checkpoint}", flush=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    dataset = BRATSVolumes(
        args.data_dir,
        mode="train",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    iterator = iter(loader)
    started = time.time()
    model.train()
    for step in range(1, args.max_iterations + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            cond_dwt = torch.cat(
                [
                    apply_dwt(batch[key])
                    for key in ("t1n", "t2w", "t2f")
                ],
                dim=1,
            )
            enh_outputs = teacher(
                cond_dwt,
                torch.zeros(args.batch_size, device=device),
                t1n=batch["t1n"],
                t2w=batch["t2w"],
                t2f=batch["t2f"],
            )
            enh_map = select_enhancement(enh_outputs, "soft_mask")
            if new_teacher is not None:
                new_outputs = new_teacher(
                    cond_dwt,
                    torch.zeros(args.batch_size, device=device),
                    t1n=batch["t1n"],
                    t2w=batch["t2w"],
                    t2f=batch["t2f"],
                )
                new_enh_map = select_enhancement(
                    new_outputs,
                    args.new_enh_head,
                )
                enh_map = (
                    (1.0 - args.new_prior_weight) * enh_map
                    + args.new_prior_weight * new_enh_map
                )
            source_dwt = apply_dwt(batch["t1n"] + enh_map)
            target_dwt = apply_dwt(batch["t1c"])
            residual_target = target_dwt - source_dwt
        y = torch.full(
            (args.batch_size,),
            1,
            dtype=torch.long,
            device=device,
        )
        output = model(
            torch.cat([source_dwt, cond_dwt], dim=1),
            torch.zeros(args.batch_size, device=device),
            y=y,
        )
        residual_loss = F.mse_loss(output, residual_target)
        prediction = apply_idwt(source_dwt + output).clamp(0.0, 1.0)
        target = batch["t1c"]
        roi = (
            (target - batch["t1n"] > 0.15)
            & (batch["t1n"] > 0.05)
        )
        roi_loss = (
            ((prediction - target) ** 2 * roi).sum()
            / roi.sum().clamp(min=1.0)
        )
        detail_loss = finite_difference_loss(prediction, target)
        loss = (
            residual_loss
            + args.roi_loss_weight * roi_loss
            + args.detail_loss_weight * detail_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - started
            writer.add_scalar("loss/train", float(loss), step)
            print(
                f"step {step}: loss={float(loss):.6f} "
                f"residual={float(residual_loss):.6f} "
                f"roi={float(roi_loss):.6f} "
                f"detail={float(detail_loss):.6f} "
                f"elapsed={elapsed:.1f}s iter/s={step / elapsed:.2f}",
                flush=True,
            )
        if step % args.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_dir, "checkpoints", f"direct_unet_{step:06d}.pt"
                ),
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            args.output_dir,
            "checkpoints",
            f"direct_unet_{args.max_iterations:06d}.pt",
        ),
    )
    writer.close()
    print("TRAINING COMPLETE", flush=True)
    print(
        f"PEAK_GPU_MEMORY_GB={torch.cuda.max_memory_allocated() / 1024**3:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
