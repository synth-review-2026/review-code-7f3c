# Frozen checkpoint manifest

The code expects four EPG-Synth weights in `checkpoints/`.

| Role | Filename | SHA-256 |
| --- | --- | --- |
| Old enhancement prior | `enh_pred_v5_003500.pt` | `78b93cdb24be1033a6d47b3b6ab52f4b84eb74950c9144346b3ea7d87fa3ef73` |
| V12 enhancement prior | `enh_pred_v5_003875.pt` | `1a35c6da8b3a8b9273e2a6858825d992420eefb9b4759577913a1decdb6f0ba1` |
| Base residual U-Net | `direct_unet_002500.pt` | `40f1d3fc58ce5ba2666e80d334a5b1f2bcee644e49dc0b884a67062e9c17f83f` |
| V12 fine-tuned residual U-Net | `direct_unet_000500.pt` | `fef4eb796463791ec9dcb94fd7d1999b9281d9ff11c8565c2f6650b865689ba8` |

These hashes match the frozen model manifest used for the 2026 external
evaluation. Comparison weights are not required for EPG-Synth inference.

