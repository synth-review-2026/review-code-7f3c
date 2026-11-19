# Reproduction notes

## Environment

The experiments used a single NVIDIA A100 40 GB GPU and PyTorch 2.6 with CUDA
12.6.

## Training sequence

1. Preprocess all modalities to the common `192 x 224 x 192` grid.
2. Train the initial enhancement-prior predictor.
3. Train the V12 smooth-mask enhancement predictor.
4. Train the base direct residual U-Net using the source prior.
5. Fine-tune the residual U-Net with the V12 prior.
6. Build the validation gate and fit the fixed fusion coefficient.
7. Freeze the model and evaluate the locked internal and external cohorts.

The final fusion coefficient is `0.5`. The development-stage high-frequency
correction branch is intentionally absent from the final model.

## Splits

- Training: 410 patients.
- Validation and model selection: 87 patients.
- Held-out internal test: 89 patients.
- External expert-mask evaluation: 118 patients.

Do not use the 89 internal test patients for early stopping, threshold
selection, or architecture selection.

## Evaluation hierarchy

Image endpoints include whole-brain PSNR, 3D SSIM, expert-ROI PSNR, ROI MAE,
enhancement-pattern Dice, and native-enhancement correlation. Downstream
endpoints include whole-tumour Dice, raw tumour detection, overlap-conditioned
Dice and HD95, and all-case censored HD95.

Paired confidence intervals used patient-level bootstrap resampling. Multiple
endpoint comparisons used Benjamini-Hochberg correction.
