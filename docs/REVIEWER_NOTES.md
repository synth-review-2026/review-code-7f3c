# Reviewer reproduction checklist

## Included

- frozen EPG-Synth inference implementation;
- network definitions and wavelet operations;
- preprocessing and split utilities;
- enhancement-prior and residual-network training entry points;
- comparator training entry points;
- external image, ROI, downstream, and orientation-audit scripts;
- formal reader-score analysis code;
- frozen configuration and checkpoint manifest.

## Not included

- DICOM, NIfTI, or other patient-level images;
- expert or automatic segmentation masks;
- case identifiers or hospital mapping files;
- reader-study images or identifying information;
- model checkpoint binaries;
- generated predictions and downstream outputs.

## Suggested response

State that model code, preprocessing, training, and evaluation logic are
provided; clinical data remain under institutional ethics and data-use
restrictions; model weights can be released through a controlled download
after the authors select a licence and redistribution policy.

