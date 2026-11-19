# -*- coding: utf-8 -*-
"""Enhancement predictor v5 with Tversky mask loss.

Outputs:
  ch0: continuous enhancement intensity
  ch1: enhancement mask logits
  ch2: log-variance for heteroscedastic uncertainty

The model is trained in the same T1 intensity space as the WFM model. With
--target_calibration t1_robust, only the T1c label is mapped into the T1n
intensity space. Model inputs remain target-free.
"""
import argparse
import csv
import math
import os
import random
import time

import nibabel as nib
import numpy as np
import torch as th
import torch.nn.functional as F

import sys
sys.path.append('.')
from guided_diffusion.script_util import create_model
from guided_diffusion.bratsloader import BRATSVolumes, shared_t1_calibrate_and_normalize
from guided_diffusion.enhancement_predictor import EnhancementPredictor
from guided_diffusion.enhancement_boundary import BoundaryAwareEnhancementPredictor
from guided_diffusion.target_calibration import calibrate_t1c_target
from guided_diffusion.wfm import apply_dwt


def soft_dice(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return ((2 * inter + eps) / (denom + eps)).mean()


def soft_tversky(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    # alpha weights FP, beta weights FN.  beta > alpha gives large-lesion recall.
    dims = (1, 2, 3, 4)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    return ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def focal_bce(logits, target, gamma=2.0, alpha=0.25):
    p = th.sigmoid(logits)
    pt = th.where(target > 0.5, p, 1.0 - p)
    w = th.where(target > 0.5, alpha, 1.0 - alpha)
    return (-w * (1.0 - pt) ** gamma * th.log(pt.clamp(min=1e-8))).mean()


def finite_difference_detail_loss(prediction, target):
    """Penalize missed or invented local enhancement edges in all 3 axes."""
    losses = []
    for dimension in (2, 3, 4):
        pred_diff = prediction.diff(dim=dimension)
        target_diff = target.diff(dim=dimension)
        losses.append(F.smooth_l1_loss(pred_diff, target_diff, beta=0.02))
    return th.stack(losses).mean()


def _corr_numpy(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.std() <= 1e-8 or second.std() <= 1e-8:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def validate_predictor(model, val_dir, device, target_depth, target_h, target_w,
                       threshold=0.15):
    dataset = BRATSVolumes(
        val_dir,
        mode='eval_target',
        normalization='preprocessed_dhw',
        target_calibration='t1_robust',
        target_depth=target_depth,
        target_h=target_h,
        target_w=target_w,
    )
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    rows = []
    model.eval()
    with th.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if th.is_tensor(value) else value
                for key, value in batch.items()
            }
            cond = th.cat(
                [apply_dwt(batch[key]) for key in ('t1n', 't2w', 't2f')],
                dim=1,
            )
            outputs = model(
                cond,
                th.zeros(1, device=device),
                t1n=batch['t1n'],
                t2w=batch['t2w'],
                t2f=batch['t2f'],
            )
            target = th.clamp(batch['t1c'] - batch['t1n'], 0.0, 1.0)
            target_mask = target > threshold
            prediction = outputs['intensity'] * outputs['mask']
            pred_mask = outputs['mask'] > threshold
            inter = (pred_mask & target_mask).sum().item()
            dice = (2.0 * inter + 1e-6) / (
                pred_mask.sum().item() + target_mask.sum().item() + 1e-6
            )
            subject = batch['subj']
            if isinstance(subject, (list, tuple)):
                subject = subject[0]
            row = {'subject': subject}
            row['mask_dice'] = float(dice)
            for mode, enhancement in (
                ('intensity', outputs['intensity']),
                ('soft_mask', outputs['intensity'] * outputs['mask']),
                ('offset_mask', outputs['intensity'] * (0.2 + 0.8 * outputs['mask'])),
                ('boundary', outputs['enhancement']),
            ):
                selected = target_mask[0, 0]
                pred_values = enhancement[0, 0][selected].detach().cpu().numpy()
                target_values = target[0, 0][selected].detach().cpu().numpy()
                row[f'{mode}_r'] = _corr_numpy(pred_values, target_values)
            selected = target_mask[0, 0]
            if selected.any():
                pred_selected = prediction[0, 0][selected]
                target_selected = target[0, 0][selected]
                row['roi_ratio'] = float(
                    pred_selected.mean()
                    / target_selected.mean().clamp(min=1e-8)
                )
                row['roi_p90_ratio'] = float(
                    th.quantile(pred_selected, 0.9)
                    / th.quantile(target_selected, 0.9).clamp(min=1e-8)
                )
            else:
                row['roi_ratio'] = np.nan
                row['roi_p90_ratio'] = np.nan
            rows.append(row)
    model.train()
    output = {}
    for mode in ('intensity', 'soft_mask', 'offset_mask', 'boundary'):
        values = np.asarray([row[f'{mode}_r'] for row in rows], dtype=np.float64)
        output[f'{mode}_r_mean'] = float(np.nanmean(values))
        output[f'{mode}_r_median'] = float(np.nanmedian(values))
    output['mask_dice_mean'] = float(
        np.mean([row['mask_dice'] for row in rows])
    )
    output['roi_ratio_mean'] = float(
        np.nanmean([row['roi_ratio'] for row in rows])
    )
    output['roi_p90_ratio_mean'] = float(
        np.nanmean([row['roi_p90_ratio'] for row in rows])
    )
    output['n'] = len(rows)
    return output, rows


def robust_case_weights(ds, normalization, target_calibration, out_dir):
    cache = os.path.join(out_dir, 'case_weights.npy')
    if os.path.exists(cache):
        return np.load(cache)
    print('computing per-case enhancement richness...', flush=True)
    weights = []
    for i, fp in enumerate(ds.database):
        try:
            c = nib.load(fp['t1c']).get_fdata().astype(np.float32)
            n = nib.load(fp['t1n']).get_fdata().astype(np.float32)
            if normalization == 't1_shared':
                n, c = shared_t1_calibrate_and_normalize(n, c)
            elif normalization == 'preprocessed_dhw':
                # The corrected data is already normalized inside the tissue
                # mask. Do not apply a second intensity transform.
                pass
            else:
                # Fallback for backward-compatible datasets.
                from guided_diffusion.bratsloader import clip_and_normalize
                c = clip_and_normalize(c)
                n = clip_and_normalize(n)
            if target_calibration == 't1_robust':
                c, _ = calibrate_t1c_target(n, c)
            weights.append(float(((c - n) > 0.15).mean()))
        except Exception as exc:
            print('  weight failed:', fp.get('t1c', '?'), exc, flush=True)
            weights.append(0.0)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(ds.database)}', flush=True)
    weights = np.asarray(weights, dtype=np.float64)
    np.save(cache, weights)
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_iterations', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_interval', type=int, default=500)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume_checkpoint', default='')
    p.add_argument('--resume_step', type=int, default=0)
    p.add_argument('--normalization', default='per_modality')
    p.add_argument(
        '--target_calibration',
        choices=['none', 't1_robust'],
        default='t1_robust',
    )
    p.add_argument('--enh_thresh', type=float, default=0.15)
    p.add_argument('--target_depth', type=int, default=192)
    p.add_argument('--target_h', type=int, default=224)
    p.add_argument('--target_w', type=int, default=192)
    p.add_argument(
        '--enh_output_mode',
        choices=['intensity', 'soft_mask', 'offset_mask', 'smooth_mask'],
        default='soft_mask',
    )
    p.add_argument('--mask_weight', type=float, default=1.0)
    p.add_argument('--tversky_alpha', type=float, default=0.3)
    p.add_argument('--tversky_beta', type=float, default=0.7)
    p.add_argument('--inten_weight', type=float, default=0.5)
    p.add_argument(
        '--roi_fused_weight',
        type=float,
        default=0.0,
        help='loss on intensity*mask inside the target enhancement ROI',
    )
    p.add_argument(
        '--amplitude_weight',
        type=float,
        default=0.0,
        help='casewise ROI mean-amplitude calibration loss',
    )
    p.add_argument(
        '--casewise_mask_weight',
        type=float,
        default=0.0,
        help='case-balanced Dice loss to protect small lesions',
    )
    p.add_argument(
        '--detail_weight',
        type=float,
        default=0.0,
        help='3-axis finite-difference enhancement-edge loss',
    )
    p.add_argument(
        '--false_positive_weight',
        type=float,
        default=0.0,
        help='penalize enhancement outside target-positive tissue',
    )
    p.add_argument(
        '--boundary_weight',
        type=float,
        default=0.0,
        help='penalize errors on the target enhancement boundary',
    )
    p.add_argument(
        '--casewise_area_weight',
        type=float,
        default=0.0,
        help='casewise relative-area calibration loss',
    )
    p.add_argument(
        '--soft_shape_weight',
        type=float,
        default=0.0,
        help='continuous weak annotation loss around the enhancement threshold',
    )
    p.add_argument(
        '--hard_negative_weight',
        type=float,
        default=0.0,
        help='penalize the highest predicted background activations',
    )
    p.add_argument(
        '--hard_negative_topk',
        type=int,
        default=2048,
        help='number of hard-negative voxels retained per prediction',
    )
    p.add_argument(
        '--boundary_aware',
        action='store_true',
        help='use the V11 learned boundary-transition refiner',
    )
    p.add_argument(
        '--transition_smooth_weight',
        type=float,
        default=0.0,
        help='smooth enhancement jumps inside the target boundary band',
    )
    p.add_argument(
        '--teacher_v8_checkpoint',
        default='',
        help='frozen V8 teacher for ensemble distillation',
    )
    p.add_argument(
        '--teacher_v10_checkpoint',
        default='',
        help='frozen V10 teacher for ensemble distillation',
    )
    p.add_argument(
        '--ensemble_distill_weight',
        type=float,
        default=0.0,
    )
    p.add_argument('--unc_weight', type=float, default=0.05)
    p.add_argument('--sampler_max_weight', type=float, default=3.0)
    p.add_argument(
        '--sampler_mode',
        choices=['richness', 'balanced_size'],
        default='richness',
    )
    p.add_argument('--uniform_sampling', action='store_true')
    p.add_argument('--val_dir', default='')
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--augment', action='store_true',
                        help='geometric augmentation (flips + 90deg in-plane rotation) applied identically to all modalities')
    args = p.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if getattr(args, 'normalization', 'per_modality') == 't1_shared':
        raise ValueError('t1_shared is target-leaky and disabled for new training; use target_free')
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)
    device = th.device('cuda:0')

    ds = BRATSVolumes(
        args.data_dir,
        mode='train',
        normalization=args.normalization,
        target_calibration=args.target_calibration,
        target_depth=args.target_depth,
        target_h=args.target_h,
        target_w=args.target_w,
    )
    if args.uniform_sampling:
        sampler = None
    else:
        raw_w = robust_case_weights(
            ds,
            args.normalization,
            args.target_calibration,
            args.out_dir,
        )
        mean_w = float(raw_w.mean()) if raw_w.size else 1.0
        rel = raw_w / max(mean_w, 1e-8)
        if args.sampler_mode == 'balanced_size':
            sw = 1.0 / np.sqrt(np.clip(rel, 0.05, 4.0))
        else:
            sw = 1.0 + (args.sampler_max_weight - 1.0) * np.clip(rel, 0.0, 2.0)
        sampler = th.utils.data.WeightedRandomSampler(
            th.as_tensor(sw, dtype=th.double), len(sw), replacement=True)

    if args.boundary_aware:
        model = BoundaryAwareEnhancementPredictor()
    else:
        model = EnhancementPredictor(output_mode=args.enh_output_mode)
    if args.resume_checkpoint:
        state = th.load(args.resume_checkpoint, map_location='cpu')
        if args.boundary_aware:
            load_result = model.load_state_dict(state, strict=False)
            unexpected = [
                key for key in load_result.unexpected_keys
                if 'boundary_head' not in key
            ]
            if unexpected:
                raise RuntimeError(f'Unexpected checkpoint keys: {unexpected}')
        else:
            model.load_state_dict(state)
        print(f'resumed predictor from {args.resume_checkpoint} at step {args.resume_step}', flush=True)
    model.to(device)
    print(f'predictor v5 parameters: {sum(x.numel() for x in model.parameters()):,}', flush=True)
    teachers = []
    if args.ensemble_distill_weight > 0.0:
        if not args.teacher_v8_checkpoint or not args.teacher_v10_checkpoint:
            raise ValueError(
                'Ensemble distillation requires both teacher checkpoints'
            )
        for teacher_path in (
            args.teacher_v8_checkpoint,
            args.teacher_v10_checkpoint,
        ):
            teacher = EnhancementPredictor(output_mode='soft_mask')
            teacher.load_state_dict(
                th.load(teacher_path, map_location='cpu')
            )
            teacher.to(device)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            teachers.append(teacher)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    dl = th.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, drop_last=True)
    it = iter(dl)
    t0 = time.time()
    validation_history = []

    for step in range(args.resume_step + 1, args.max_iterations + 1):
        schedule_step = step - args.resume_step
        schedule_total = max(1, args.max_iterations - args.resume_step)
        lr = args.lr * 0.5 * (
            1.0
            + math.cos(math.pi * schedule_step / schedule_total)
        )
        for g in opt.param_groups:
            g['lr'] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(device) if th.is_tensor(v) else v for k, v in batch.items()}
        if args.augment:
            # Geometric augmentation: the SAME transform is applied to every
            # modality so the enhancement target (t1c - t1n) stays consistent.
            _k = random.randint(0, 3)
            if _k:
                batch = {m: (th.rot90(v, _k, dims=(3, 4)) if th.is_tensor(v) else v)
                         for m, v in batch.items()}
            for _d in (2, 3, 4):
                if random.random() < 0.5:
                    batch = {m: (th.flip(v, dims=(_d,)) if th.is_tensor(v) else v)
                             for m, v in batch.items()}

        cond = th.cat([apply_dwt(batch[m]) for m in ('t1n', 't2w', 't2f')], dim=1)
        enh = th.clamp(batch['t1c'] - batch['t1n'], 0.0, 1.0)
        tb = (enh > args.enh_thresh).float()
        fg = (enh > 0.03).float()

        outputs = model(
            cond,
            th.zeros(cond.shape[0], device=device),
            t1n=batch['t1n'],
            t2w=batch['t2w'],
            t2f=batch['t2f'],
        )
        inten = outputs['intensity']
        mlogit = outputs['mask_logit']

        # Mask branch: primary localisation objective.
        mask_prob = th.sigmoid(mlogit)
        dice = soft_dice(mask_prob, tb)  # monitoring only
        tversky = soft_tversky(mask_prob, tb, args.tversky_alpha, args.tversky_beta)
        focal = focal_bce(mlogit, tb)
        l_mask = 2.0 * (1.0 - tversky) + 0.5 * focal

        # This is the tensor actually consumed downstream by WFM.
        fused_enh = (
            outputs['enhancement']
            if args.boundary_aware or args.enh_output_mode == 'smooth_mask'
            else inten * mask_prob
        )
        l_distill = th.zeros((), device=device)
        if teachers:
            with th.no_grad():
                teacher_maps = []
                for teacher in teachers:
                    teacher_output = teacher(
                        cond,
                        th.zeros(cond.shape[0], device=device),
                        t1n=batch['t1n'],
                        t2w=batch['t2w'],
                        t2f=batch['t2f'],
                    )
                    teacher_map = (
                        teacher_output['intensity']
                        * F.avg_pool3d(
                            teacher_output['mask'],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    teacher_maps.append(teacher_map)
                ensemble_teacher = (
                    0.75 * teacher_maps[0]
                    + 0.25 * teacher_maps[1]
                )
            l_distill = F.smooth_l1_loss(
                fused_enh,
                ensemble_teacher,
                beta=0.02,
            )
        l_casewise = th.zeros((), device=device)
        if args.casewise_mask_weight > 0.0:
            case_intersection = (mask_prob * tb).sum(dim=(1, 2, 3, 4))
            case_denominator = (
                mask_prob.sum(dim=(1, 2, 3, 4))
                + tb.sum(dim=(1, 2, 3, 4))
            )
            case_dice = (
                2.0 * case_intersection + 1e-6
            ) / (case_denominator + 1e-6)
            l_casewise = (1.0 - case_dice).mean()

        # Intensity branch: dense smooth-L1, with modest extra weight inside
        # the predicted/target enhancement region.  This avoids the v25
        # failure mode of rewarding global brightness.
        err = F.smooth_l1_loss(inten, enh, beta=0.02, reduction='none')
        w_int = 0.1 + 4.0 * fg
        l_int = (err * w_int).sum() / w_int.sum().clamp(min=1.0)

        roi = tb
        l_roi_fused = th.zeros((), device=device)
        l_amplitude = th.zeros((), device=device)
        if args.roi_fused_weight > 0.0:
            fused_error = F.smooth_l1_loss(
                fused_enh,
                enh,
                beta=0.02,
                reduction='none',
            )
            l_roi_fused = (
                fused_error * roi
            ).sum() / roi.sum().clamp(min=1.0)
        if args.amplitude_weight > 0.0:
            pred_mean = (fused_enh * roi).sum(dim=(1, 2, 3, 4))
            target_mean = (enh * roi).sum(dim=(1, 2, 3, 4))
            roi_count = roi.sum(dim=(1, 2, 3, 4))
            valid_case = (roi_count > 0) & (target_mean > 1e-5)
            pred_mean = pred_mean / roi_count.clamp(min=1.0)
            target_mean = target_mean / roi_count.clamp(min=1.0)
            ratio_error = (
                pred_mean / target_mean.clamp(min=1e-5) - 1.0
            ) ** 2
            l_amplitude = (
                ratio_error * valid_case
            ).sum() / valid_case.sum().clamp(min=1.0)

        l_detail = th.zeros((), device=device)
        if args.detail_weight > 0.0:
            l_detail = finite_difference_detail_loss(fused_enh, enh)

        l_false_positive = th.zeros((), device=device)
        if args.false_positive_weight > 0.0:
            negative = (enh < 0.03) & (batch['t1n'] > 0.01)
            l_false_positive = (
                fused_enh * negative
            ).sum() / negative.sum().clamp(min=1.0)

        l_boundary = th.zeros((), device=device)
        if args.boundary_weight > 0.0:
            target_max = F.max_pool3d(
                tb,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            target_min = -F.max_pool3d(
                -tb,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            target_boundary = (target_max - target_min).clamp(0.0, 1.0)
            pred_max = F.max_pool3d(
                mask_prob,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            pred_min = -F.max_pool3d(
                -mask_prob,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            pred_boundary = (pred_max - pred_min).clamp(0.0, 1.0)
            boundary_error = F.smooth_l1_loss(
                pred_boundary,
                target_boundary,
                beta=0.05,
                reduction='none',
            )
            boundary_mask = 1.0 + 4.0 * target_boundary
            l_boundary = (
                boundary_error * boundary_mask
            ).sum() / boundary_mask.sum().clamp(min=1.0)

        l_area = th.zeros((), device=device)
        if args.casewise_area_weight > 0.0:
            pred_area = mask_prob.sum(dim=(1, 2, 3, 4))
            target_area = tb.sum(dim=(1, 2, 3, 4))
            valid_area = target_area > 1.0
            area_ratio_error = (
                (pred_area - target_area).abs()
                / target_area.clamp(min=1.0)
            )
            l_area = (
                area_ratio_error * valid_area
            ).sum() / valid_area.sum().clamp(min=1.0)

        l_soft_shape = th.zeros((), device=device)
        if args.soft_shape_weight > 0.0:
            soft_target = th.sigmoid(
                (enh - args.enh_thresh) / 0.04
            )
            l_soft_shape = F.binary_cross_entropy(
                mask_prob.clamp(1e-6, 1.0 - 1e-6),
                soft_target,
            )

        l_hard_negative = th.zeros((), device=device)
        if args.hard_negative_weight > 0.0:
            negative = (
                (enh < 0.03)
                & (batch['t1n'] > 0.01)
            )
            candidate_values = mask_prob * negative.float()
            flattened = candidate_values.flatten(start_dim=1)
            k = min(args.hard_negative_topk, flattened.shape[1])
            hard_values = flattened.topk(k, dim=1).values
            l_hard_negative = hard_values.mean()

        l_transition_smooth = th.zeros((), device=device)
        if args.transition_smooth_weight > 0.0:
            target_max = F.max_pool3d(
                tb,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            target_min = -F.max_pool3d(
                -tb,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            boundary_band = (
                target_max - target_min
            ).clamp(0.0, 1.0)
            local_mean = F.avg_pool3d(
                fused_enh,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            l_transition_smooth = (
                (fused_enh - local_mean).abs() * boundary_band
            ).sum() / boundary_band.sum().clamp(min=1.0)

        # Heteroscedastic uncertainty: the model can down-weight ambiguous
        # voxels rather than hallucinating high-confidence enhancement.
        # The uncertainty head is retained in the coarse predictor for
        # experimental compatibility but excluded from the main loss because
        # it is not consumed by the WFM synthesis path.
        l_unc = th.zeros((), device=device)

        loss = (
            args.mask_weight * l_mask
            + args.inten_weight * l_int
            + args.roi_fused_weight * l_roi_fused
            + args.amplitude_weight * l_amplitude
            + args.casewise_mask_weight * l_casewise
            + args.detail_weight * l_detail
            + args.false_positive_weight * l_false_positive
            + args.boundary_weight * l_boundary
            + args.casewise_area_weight * l_area
            + args.soft_shape_weight * l_soft_shape
            + args.hard_negative_weight * l_hard_negative
            + args.transition_smooth_weight * l_transition_smooth
            + args.ensemble_distill_weight * l_distill
        )
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_interval == 0 or step == 1:
            with th.no_grad():
                pred_mask = (th.sigmoid(mlogit) > args.enh_thresh).float()
                inter = (pred_mask * tb).sum()
                dice_bin = ((2 * inter + 1e-6) / (pred_mask.sum() + tb.sum() + 1e-6)).item()
            print(f'step {step}: loss={loss.item():.4f} mask={l_mask.item():.4f} '
                  f'int={l_int.item():.5f} roi={l_roi_fused.item():.5f} '
                  f'amp={l_amplitude.item():.5f} case={l_casewise.item():.5f} '
                  f'detail={l_detail.item():.5f} fp={l_false_positive.item():.5f} '
                  f'boundary={l_boundary.item():.5f} '
                  f'area={l_area.item():.5f} '
                  f'softshape={l_soft_shape.item():.5f} '
                  f'hardneg={l_hard_negative.item():.5f} '
                  f'transition={l_transition_smooth.item():.5f} '
                  f'distill={l_distill.item():.5f} '
                  f'unc={l_unc.item():.4f} '
                  f'dice={dice_bin:.4f} lr={lr:.2e} iter/s={step / (time.time() - t0):.2f}',
                  flush=True)
        if step % args.save_interval == 0:
            th.save(model.state_dict(),
                    os.path.join(args.out_dir, 'checkpoints', f'enh_pred_v5_{step:06d}.pt'))
        if args.val_dir and step % args.eval_interval == 0:
            val_summary, val_rows = validate_predictor(
                model,
                args.val_dir,
                device,
                args.target_depth,
                args.target_h,
                args.target_w,
                args.enh_thresh,
            )
            val_summary['step'] = step
            validation_history.append(val_summary)
            with open(
                os.path.join(args.out_dir, 'validation_metrics.csv'),
                'w',
                newline='',
                encoding='utf-8-sig',
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(val_summary))
                writer.writeheader()
                writer.writerows(validation_history)
            print(
                f'validation step={step} '
                f"intensity_r={val_summary['intensity_r_mean']:.4f} "
                f"soft_mask_r={val_summary['soft_mask_r_mean']:.4f} "
                f"offset_mask_r={val_summary['offset_mask_r_mean']:.4f} "
                f"boundary_r={val_summary['boundary_r_mean']:.4f} "
                f"dice={val_summary['mask_dice_mean']:.4f} "
                f"roi_ratio={val_summary['roi_ratio_mean']:.4f} "
                f"p90_ratio={val_summary['roi_p90_ratio_mean']:.4f}",
                flush=True,
            )
    th.save(model.state_dict(),
            os.path.join(args.out_dir, 'checkpoints', f'enh_pred_v5_{args.max_iterations:06d}.pt'))
    print('TRAINING COMPLETE', flush=True)
    print(
        f'PEAK_GPU_MEMORY_GB={th.cuda.max_memory_allocated() / 1024**3:.3f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
