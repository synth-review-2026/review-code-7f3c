import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
import os.path
import warnings
import nibabel
from scipy.ndimage import zoom
from .target_calibration import apply_t1c_calibration, calibrate_t1c_target


def clip_and_normalize(img):
    """Legacy per-modality min-max normalization (kept for backward compatibility)."""
    img = np.asarray(img, dtype=np.float32)
    low = np.quantile(img, 0.001)
    high = np.quantile(img, 0.999)
    if high - low < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img_clipped = np.clip(img, low, high)
    img_normalized = (img_clipped - low) / (high - low)
    return np.clip(img_normalized, 0.0, 1.0).astype(np.float32)


def shared_t1_calibrate_and_normalize(t1n, t1c, hi_percentile=99.5, return_stats=False):
    """DEPRECATED / TARGET-LEAKY.

    This function uses T1c to compute the shared scale hi, which makes the
    normalized T1n input depend on the target.  It is retained only for
    reproducibility of old exploratory runs.  New experiments must use
    BRATSVolumes(normalization='target_free').

    Original docstring:
    Calibrate T1c into the T1n intensity space and normalize both jointly.

    T1n and T1c are the same T1-weighted sequence.  Independent per-image
    normalization destroys their shared intensity reference, especially in
    the strongest enhancement voxels.  Here we fit a robust affine map from
    T1c to T1n on central brain tissue, apply that map to T1c, and divide both
    images by the same upper percentile.  The background is preserved as zero.
    """
    n = np.asarray(t1n, dtype=np.float32)
    c = np.asarray(t1c, dtype=np.float32)
    # A few legacy cases have slightly different grids.  Resample T1n to T1c
    # before subtraction/calibration so boolean masks always share one grid.
    if n.shape != c.shape:
        n = zoom(n, [t / s for t, s in zip(c.shape, n.shape)], order=1).astype(np.float32)
    mask = n > 0
    if mask.sum() < 100:
        stats = dict(fallback=1, mask_vox=int(mask.sum()))
        out = (clip_and_normalize(n), clip_and_normalize(c))
        return out + (stats,) if return_stats else out

    nv = n[mask].astype(np.float64, copy=False)
    cv = c[mask].astype(np.float64, copy=False)
    p05, p95 = np.percentile(nv, [5.0, 95.0])
    sel = (nv >= p05) & (nv <= p95)
    if sel.sum() >= 100:
        try:
            a, b = np.polyfit(cv[sel], nv[sel], 1)
        except Exception:
            a, b = 1.0, float(np.median(nv - cv))
    else:
        a, b = 1.0, float(np.median(nv - cv))

    # Guard against pathological fits caused by a dominant lesion or empty data.
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or a > 10.0:
        a, b = 1.0, float(np.median(nv - cv))

    c_cal = np.clip(a * c + b, 0.0, None).astype(np.float32)
    c_cal = np.where(mask, c_cal, 0.0).astype(np.float32)
    n_cal = np.where(mask, n, 0.0).astype(np.float32)

    vals = np.concatenate([n_cal[mask], c_cal[mask]])
    hi = float(np.percentile(vals, hi_percentile))
    if not np.isfinite(hi) or hi <= 1e-8:
        stats = dict(fallback=2, mask_vox=int(mask.sum()), a=float(a), b=float(b), hi=hi)
        out = (clip_and_normalize(n_cal), clip_and_normalize(c_cal))
        return out + (stats,) if return_stats else out

    n_norm = np.clip(n_cal / hi, 0.0, 1.0).astype(np.float32)
    c_norm = np.clip(c_cal / hi, 0.0, 1.0).astype(np.float32)
    stats = dict(fallback=0, mask_vox=int(mask.sum()), a=float(a), b=float(b), hi=hi)
    return (n_norm, c_norm, stats) if return_stats else (n_norm, c_norm)


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None, target_depth=128,
                 target_h=256, target_w=256, normalization='per_modality',
                 target_calibration='none'):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1n, t1c, t2w, t2f, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.target_depth = target_depth
        self.target_h = target_h
        self.target_w = target_w
        if normalization not in (
            'per_modality', 'target_free', 't1_shared', 'preprocessed_dhw'
        ):
            raise ValueError(f'Unknown normalization mode: {normalization}')
        if normalization == 't1_shared':
            warnings.warn(
                't1_shared is DEPRECATED and leaks target T1c information into '
                'input T1n normalization. Use normalization=target_free instead.',
                RuntimeWarning, stacklevel=2)
        self.normalization = normalization
        if target_calibration not in ('none', 't1_robust'):
            raise ValueError(f'Unknown target calibration: {target_calibration}')
        self.target_calibration = target_calibration
        self._target_calibration_cache = {}
        self.preprocessed_dhw = normalization == 'preprocessed_dhw'
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)

        self.database = []

        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                datapoint = dict()
                for f in files:
                    low = f.lower()
                    if self.preprocessed_dhw:
                        if low.endswith('_t1.nii.gz'):
                            seqtype = 't1n'
                        elif low.endswith('_t1ce.nii.gz'):
                            seqtype = 't1c'
                        elif low.endswith('_t2.nii.gz'):
                            seqtype = 't2w'
                        elif low.endswith('_flair.nii.gz'):
                            seqtype = 't2f'
                        else:
                            # The corrected training directory also contains
                            # mask and metadata files that the model does not use.
                            continue
                    # Legacy raw-volume filename support.
                    elif 't1n' in f:
                        seqtype = 't1n'
                    elif 't1c' in f:
                        seqtype = 't1c'
                    elif 't2w' in f:
                        seqtype = 't2w'
                    elif 't2f' in f:
                        seqtype = 't2f'
                    elif 'seg' in f:
                        seqtype = 'seg'
                    else:
                        parts = f.split('-')
                        if len(parts) >= 5:
                            seqtype = parts[4].split('.')[0]
                        else:
                            print(f"警告: 无法解析文件名 {f}，跳过")
                            continue
                    datapoint[seqtype] = os.path.join(root, f)
                if datapoint:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'

        TARGET_DEPTH = self.target_depth
        TARGET_H = self.target_h
        TARGET_W = self.target_w

        def interpolate_depth(data_np):
            """Interpolate (H, W, D) to (1, target_depth, target_h, target_w)."""
            tensor = torch.from_numpy(np.asarray(data_np, dtype=np.float32)).float()
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(TARGET_DEPTH, TARGET_H, TARGET_W),
                mode='trilinear',
                align_corners=False
            )
            return tensor.squeeze(0)

        def load_raw(key):
            if key not in filedict:
                return None
            return nibabel.load(filedict[key]).get_fdata().astype(np.float32)

        if self.preprocessed_dhw:
            expected_shape = (TARGET_DEPTH, TARGET_H, TARGET_W)

            def load_preprocessed(key):
                data_np = load_raw(key)
                if data_np is None:
                    return None
                if data_np.shape != expected_shape:
                    raise ValueError(
                        f'{key} has shape {data_np.shape}, expected {expected_shape}: '
                        f'{filedict[key]}'
                    )
                if not np.isfinite(data_np).all():
                    raise ValueError(f'Non-finite values in {filedict[key]}')
                if float(data_np.min()) < -1e-6 or float(data_np.max()) > 1.0 + 1e-6:
                    raise ValueError(
                        f'{key} values are outside [0,1]: {filedict[key]}'
                    )
                # Corrected preprocessing already yields (D,H,W). No interpolation
                # or a second intensity normalization is allowed here.
                return torch.from_numpy(data_np).unsqueeze(0)

            t1n = load_preprocessed('t1n')
            if t1n is None:
                missing = 't1n'
                t1n = torch.zeros(1, *expected_shape)

            if self.mode in ('eval', 'auto'):
                t1c = None
            else:
                t1c = load_preprocessed('t1c')
            if t1c is None:
                missing = 't1c'
                t1c = torch.zeros(1, *expected_shape)

            t2w = load_preprocessed('t2w')
            if t2w is None:
                missing = 't2w'
                t2w = torch.zeros(1, *expected_shape)

            t2f = load_preprocessed('t2f')
            if t2f is None:
                missing = 't2f'
                t2f = torch.zeros(1, *expected_shape)

            if (
                self.target_calibration == 't1_robust'
                and t1c.abs().sum().item() > 0
            ):
                t1n_np = t1n[0].detach().cpu().numpy()
                t1c_np = t1c[0].detach().cpu().numpy()
                cache_key = filedict['t1c']
                if cache_key not in self._target_calibration_cache:
                    _, stats = calibrate_t1c_target(t1n_np, t1c_np)
                    self._target_calibration_cache[cache_key] = (
                        stats['slope'],
                        stats['intercept'],
                    )
                slope, intercept = self._target_calibration_cache[cache_key]
                calibrated = apply_t1c_calibration(t1c_np, slope, intercept)
                t1c = torch.from_numpy(calibrated).unsqueeze(0)

            if self.mode in ('eval', 'auto', 'eval_target', 'eval_oracle'):
                for key in ('t1n', 't1c', 't2w', 't2f'):
                    if key in filedict:
                        subj = os.path.basename(os.path.dirname(filedict[key]))
                        break
                else:
                    subj = 'dummy_string'
            else:
                subj = 'dummy_string'

            return {
                't1n': t1n.float(),
                't1c': t1c.float(),
                't2w': t2w.float(),
                't2f': t2f.float(),
                'missing': missing,
                'subj': subj,
                'normalization': self.normalization,
            }

        # target_free: T1n normalization depends ONLY on T1n.  T1c is loaded
        # only in training and is used only as the supervised target.  In
        # eval/auto mode T1c is not loaded at all, so no target information can
        # enter the inference input or its normalization.
        t1n_raw = load_raw('t1n')
        t2w_raw = load_raw('t2w')
        t2f_raw = load_raw('t2f')
        if self.normalization == 'target_free' and self.mode in ('eval', 'auto'):
            # Strict inference mode: T1c is not loaded at all.
            t1c_raw = None
        else:
            # Training or offline eval_target: T1c is loaded only as a label.
            t1c_raw = load_raw('t1c')

        if self.normalization == 't1_shared' and t1n_raw is not None and t1c_raw is not None:
            t1n_np, t1c_np = shared_t1_calibrate_and_normalize(t1n_raw, t1c_raw)
        elif self.normalization == 'target_free':
            # Independent canonical normalization.  T1n is target-free; T1c is
            # a label-space transform used only during training.
            t1n_np = clip_and_normalize(t1n_raw) if t1n_raw is not None else None
            t1c_np = clip_and_normalize(t1c_raw) if t1c_raw is not None else None
        else:
            t1n_np = clip_and_normalize(t1n_raw) if t1n_raw is not None else None
            t1c_np = clip_and_normalize(t1c_raw) if t1c_raw is not None else None

        if (
            self.target_calibration == 't1_robust'
            and t1n_np is not None
            and t1c_np is not None
        ):
            cache_key = filedict['t1c']
            if cache_key not in self._target_calibration_cache:
                _, stats = calibrate_t1c_target(t1n_np, t1c_np)
                self._target_calibration_cache[cache_key] = (
                    stats['slope'],
                    stats['intercept'],
                )
            slope, intercept = self._target_calibration_cache[cache_key]
            t1c_np = apply_t1c_calibration(t1c_np, slope, intercept)

        if t1n_np is not None:
            t1n = interpolate_depth(t1n_np)
        else:
            missing = 't1n'
            t1n = torch.zeros(1, TARGET_DEPTH, TARGET_H, TARGET_W)

        if t1c_np is not None:
            t1c = interpolate_depth(t1c_np)
        else:
            missing = 't1c'
            t1c = torch.zeros(1, TARGET_DEPTH, TARGET_H, TARGET_W)

        if t2w_raw is not None:
            t2w = interpolate_depth(clip_and_normalize(t2w_raw))
        else:
            missing = 't2w'
            t2w = torch.zeros(1, TARGET_DEPTH, TARGET_H, TARGET_W)

        if t2f_raw is not None:
            t2f = interpolate_depth(clip_and_normalize(t2f_raw))
        else:
            missing = 't2f'
            t2f = torch.zeros(1, TARGET_DEPTH, TARGET_H, TARGET_W)

        if self.mode in ('eval', 'auto', 'eval_target', 'eval_oracle'):
            if 't1n' in filedict:
                subj = filedict['t1n']
            elif 't1c' in filedict:
                subj = filedict['t1c']
            elif 't2w' in filedict:
                subj = filedict['t2w']
            elif 't2f' in filedict:
                subj = filedict['t2f']
            else:
                subj = 'dummy_string'
            subj = os.path.basename(os.path.dirname(subj))
        else:
            subj = 'dummy_string'

        return {'t1n': t1n.float(),
                't1c': t1c.float(),
                't2w': t2w.float(),
                't2f': t2f.float(),
                'missing': missing,
                'subj': subj,
                'normalization': self.normalization}

    def __len__(self):
        return len(self.database)
