# EPG-Synth model card

## Model

EPG-Synth synthesizes a contrast-enhanced T1-weighted brain MRI volume from
noncontrast T1-weighted, T2-weighted, and FLAIR inputs. The released final
configuration is named V12-Balanced.

The model uses two enhancement-prior predictors, two residual U-Nets, a
spatial fusion gate, and fixed coefficient `0.5`. It operates on a common
`192 x 224 x 192`, 1 mm isotropic, brain-centered grid and contains
approximately 188.4 million parameters.

## Inputs

Three spatially registered volumes:

- noncontrast T1-weighted MRI;
- T2-weighted MRI;
- FLAIR MRI.

## Output

A single synthetic contrast-enhanced T1-weighted volume with values in
`[0, 1]`. Voxels outside the noncontrast brain support are set to zero.

## Cohorts

Internal cohort: 586 patients with meningioma, divided into 410 training,
87 validation/model-selection, and 89 test patients.

External evaluation: 118 patients from two centers (Center 1, n = 106;
Center 2, n = 12).

## Evaluation and limitations

EPG-Synth achieved the highest external image and expert-ROI metrics among the
six compared methods. External whole-tumour Dice was numerically highest, but
the paired difference versus WFM-Adapted was not statistically significant.
The formal three-reader study also did not show a statistically significant
advantage on its primary endpoint.

This is research software, not a medical device. It must not be used as a
replacement for contrast-enhanced MRI or as the sole input to clinical
diagnosis, treatment planning, or surgical decisions.
