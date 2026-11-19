# Data format and preprocessing contract

## Case layout

```text
case_root/
  SUBJECT_ID/
    SUBJECT_ID_T1.nii.gz
    SUBJECT_ID_T2.nii.gz
    SUBJECT_ID_FLAIR.nii.gz
```

Use de-identified identifiers in public-facing examples.

## Geometry

- Resample all modalities to the common `192 x 224 x 192` grid.
- Use 1 mm isotropic spacing.
- Center the volume on the brain mask.
- Preserve or explicitly normalize anatomical orientation.
- Co-register T1n, T2, and FLAIR to the same physical space.

The external analysis found that source orientation must be audited. Do not
mix in-plane orientations silently.

## Intensity normalization

- Normalize each modality within the brain mask using the 0.1st and 99.9th
  percentiles, then map to `[0, 1]`.
- Set voxels outside the brain mask to zero.
- Apply robust reference calibration to T1c targets during training only.
- The inference interface expects preprocessed volumes and does not apply a
  second normalization.

## Expert masks

The study used whole-tumour masks initialized by a segmentation model and
subsequently corrected by experts. They were not independent annotations
created from zero.

