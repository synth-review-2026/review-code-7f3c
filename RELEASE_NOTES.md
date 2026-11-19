# EPG-Synth GitHub release

Prepared: 2026-11-19

## Included

- Frozen EPG-Synth inference code and model factory.
- Network, wavelet, data-loader, loss, and preprocessing utilities.
- Vendored `DWT_IDWT` Haar-transform implementation.
- Training entry points for EPG-Synth components and selected comparators.
- Selected image, ROI, segmentation, orientation, reader-score, and ablation
  evaluation code.
- Model card, data contract, checkpoint manifest, provenance, tests, and
  GitHub upload instructions.

## Explicitly excluded

- Model checkpoints and weights.
- Patient MRI volumes and DICOM files.
- Expert masks and segmentation derivatives.
- Generated predictions and per-case outputs.
- Case mappings and reader-study image files.
- Local paths, caches, logs, and generated results.

## Verification completed

- Python syntax parsing passed for all 45 Python files.
- Static import resolution passed with no unresolved modules.
- Six repository tests passed.
- No `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, `.nii`, `.dcm`, `.nrrd`,
  `.xlsx`, `.pyc`, or `__pycache__` files are included.
- SHA-256 manifest matches every packaged file.

## Before public release

- Select and add a project-level license.
- Add full third-party license texts after redistribution review.
- Add the final model-weight access URL or controlled-download instructions.
- Keep the repository private during double-anonymized review if a public
  account or citation metadata would reveal author identity.
- Run `scripts/smoke_test.py` and an inference smoke test in the CUDA
  environment with the four final checkpoints.
