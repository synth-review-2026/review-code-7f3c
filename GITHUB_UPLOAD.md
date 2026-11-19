# GitHub upload steps

1. Create an empty GitHub repository, for example `EPG-Synth`.
2. Open a terminal in this directory.
3. Confirm that the repository contains no files matching the prohibited
   patient-data or weight patterns.
4. Install the test dependency and run:

```bash
python -m pip install pytest
python -m pytest tests
```

5. Initialise and upload:

```bash
git init
git add .
git commit -m "Release EPG-Synth reference implementation"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

## Before making the repository public

- Add the licence selected by the corresponding author.
- Confirm third-party redistribution terms.
- Do not add the checkpoint files under `checkpoints/`.
- Do not add clinical images, masks, case mappings, or reader-study files.
- Do not add `data/`, `runs/`, `outputs/`, `__pycache__/`, or generated logs.
- Keep the repository private during double-anonymized peer review if a public
  account, author metadata, or institution would reveal author identity.
- Replace the weight-access placeholder before advertising reproducibility.
- Confirm that the manuscript's data-availability and code-availability
  statements match the repository and controlled-access plan.
