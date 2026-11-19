# Script data contracts

The repository does not include patient-level images, masks, predictions, or
reader-study files. Authorized users can provide those files through controlled
storage and configure the following environment variables:

- `EPG_SYNTH_ROOT`: root containing `data/`, `runs/`, and `checkpoints/`.
- `EPG_SYNTH_DATA_ROOT`: root containing prepared de-identified imaging data.
- `EPG_SYNTH_CHECKPOINTS`: directory containing the four released model files.
- `EPG_SYNTH_READER_ROOT`: root containing `BLIND_REVIEW/` and
  `COORDINATOR_ONLY/` for reader-score analysis.
- `EPG_SYNTH_READER_OUTPUT`: output directory for reader-score analysis.

Do not commit files from these locations to Git. The `.gitignore` file blocks
common imaging, tabular, weight, prediction, cache, and log formats.

Typical command sequence after installing dependencies:

```bash
python scripts/smoke_test.py
python scripts/evaluate_image_roi_118.py
python scripts/evaluate_downstream_118.py
```

The evaluation commands require locally authorized datasets and checkpoint
locations and will stop if those files are absent.
