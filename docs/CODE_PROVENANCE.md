# Code provenance

This release was curated on 2026-11-19 from the project handoff maintained for
the EPG-Synth manuscript.

- Source code handoff: `01_CODE_REPORTS_ENVIRONMENT_20261105.tar.gz`
- Source archive SHA-256:
  `6134e4156bb22d8e5b43676d38828bd9cd328e5d8c5c36a74c6d8cdca4ba44e3`
- Final checkpoint hashes and model configuration:
  `EPG-Synth_FINAL_SUPPLEMENT_20261114.zip`

The vendored `DWT_IDWT` implementation was extracted from:

```text
02_code/DWT_IDWT/DWT_IDWT_layer.py
02_code/DWT_IDWT/__init__.py
```

The released checkpoint manifest contains four EPG-Synth components. Model
weights, patient data, masks, predictions, case maps, reader-study images,
and local output directories are intentionally excluded.
