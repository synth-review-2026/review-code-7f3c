# EPG-Synth

Reference implementation for contrast-enhanced T1-weighted MRI synthesis from
noncontrast T1-weighted, T2-weighted, and FLAIR MRI in meningioma.

This repository contains the frozen V12-Balanced EPG-Synth inference path,
model definitions, training entry points, preprocessing utilities, and selected
evaluation code. Patient images, case mappings, generated predictions, model
weights, and clinical data are not included.

## Method summary

EPG-Synth combines two enhancement-prior predictors with two residual U-Net
branches:

```text
source = T1n + 0.5 * old_enhancement + 0.5 * V12_enhancement
base   = IDWT(source_dwt + BaseResidual)
fine   = IDWT(source_dwt + FineTunedResidual)
gate   = 0.5 * old_prior_mask + 0.5 * V12_prior_mask
output = base + 0.5 * gate * (fine - base)
```

The final model contains approximately 188.4 million parameters. A
validation-stage high-frequency branch was removed because it provided no
material paired benefit.

## Repository layout

```text
configs/                  Frozen model configuration
checkpoints/              Expected checkpoint filenames; weights not committed
docs/                     Model card, data contract and reproduction notes
epg_synth/                Frozen inference package and model factory
guided_diffusion/         Network, data loader, wavelet and loss utilities
DWT_IDWT/                 Vendored 3D Haar wavelet transform
scripts/                  Training, preprocessing and evaluation entry points
tests/                    Static repository checks
```

## Installation

Use Python 3.9-3.12 in a CUDA-capable environment. The released experiments
used PyTorch 2.6 with CUDA 12.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu126 `
  torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install -e .
```

`Pillow` and `openpyxl` are included for figure generation and reader-score
analysis. The optional TensorBoard image-summary path in
`guided_diffusion/logger.py` requires a separate TensorFlow installation.

## Checkpoints

Place the four frozen EPG-Synth checkpoints in `checkpoints/` using
[CHECKPOINTS.md](docs/CHECKPOINTS.md). Weights are intentionally excluded from
Git history. A controlled-access route must be published before journal
submission.

## Inference

Inputs must already be resampled onto the common `192 x 224 x 192` grid,
oriented consistently, normalized to `[0, 1]`, and use the `preprocessed_dhw`
layout expected by the data loader.

Expected case layout:

```text
input_root/
  subject_001/
    subject_001_T1.nii.gz
    subject_001_T2.nii.gz
    subject_001_FLAIR.nii.gz
```

Run:

```bash
python -m epg_synth.inference \
  --mode run \
  --input_dir /path/to/preprocessed_cases \
  --output_dir /path/to/predictions
```

The result is written as:

```text
output_dir/subject_001/sample_t1c.nii.gz
```

Override defaults when needed:

```bash
export EPG_SYNTH_ROOT=/path/to/epg-synth
export EPG_SYNTH_CHECKPOINTS=/path/to/checkpoints
```

## Training entry points

The principal scripts are:

- `scripts/train_enhancer_predictor.py`: enhancement-prior predictor.
- `scripts/train_direct_residual_unet.py`: residual synthesis network.
- `scripts/train_pix2pix_cgan.py`: 3D Pix2Pix cGAN comparator.
- `scripts/train_conditional_wgan_gp.py`: conditional WGAN-GP comparator.
- `scripts/train_t1c_rflow_same_data.py`: same-data T1C-RFlow reimplementation.

Training commands and environment details are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Scripts that do not expose every
path on the command line can be configured through `EPG_SYNTH_ROOT` and the
dataset-specific environment variables documented in each script.

## Data availability

The clinical data are not part of this repository. Access is subject to
institutional ethics approval, data-use agreements, and de-identification
requirements.

Model weights are not part of the public Git history. Their SHA-256 values are
listed in `docs/CHECKPOINTS.md`.

## License status

No project-level open-source license has been selected. Adding code to GitHub
does not grant reuse rights. See `LICENSE_NOTE.md` and
`THIRD_PARTY_NOTICES.md` before making the repository public.

## Intended use and limitations

This is research software. It is not a medical device and must not be used for
clinical diagnosis, treatment planning, or replacement of contrast-enhanced
MRI without independent validation and regulatory approval.

The completed reader study did not demonstrate a statistically significant
advantage over the in-house WFM-Adapted comparator on the prespecified primary
reader endpoint. External whole-tumour Dice was numerically highest but did not
differ significantly from WFM-Adapted. The repository must not be presented as
evidence of clinical equivalence or replacement.

## Double-anonymized review

This copy omits the author list, institutional affiliation, and citation
metadata. During anonymized review, the repository URL and commit identity
should also remain non-identifying. The identified repository can be restored
after manuscript acceptance.

## Smoke test

After installing dependencies, verify the vendored wavelet transform:

```bash
python scripts/smoke_test.py
```

This check uses synthetic tensors and requires no patient data or weights.
Regenerate the release checksums after changing any tracked file:

```bash
python scripts/make_sha256sums.py
```

## Release checklist

Before making the repository public:

1. Add the authors' chosen `LICENSE`.
2. Confirm redistribution rights for all third-party code.
3. Add the final weight-access route.
4. Add the manuscript DOI when available.
