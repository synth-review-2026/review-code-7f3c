# Examples

No patient images or synthetic NIfTI volumes are distributed here.

For the optional `precheck` inference mode, an authorized user must provide:

- the private validation dataset through `EPG_SYNTH_VAL_DATA`;
- the private reference prediction directory through
  `EPG_SYNTH_PRECHECK_REFERENCE`.

The normal `run` mode only requires preprocessed T1n, T2, and FLAIR volumes
and does not require a reference T1CE.
