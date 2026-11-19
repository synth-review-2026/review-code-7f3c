"""Model factory matching the frozen EPG-Synth V12-Balanced backbone."""

from __future__ import annotations

import torch

from guided_diffusion.script_util import create_model


def build_direct_model() -> torch.nn.Module:
    """Build the 3D residual U-Net used by both synthesis branches."""
    return create_model(
        image_size=224,
        num_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 4, 4),
        learn_sigma=False,
        class_cond=True,
        use_checkpoint=False,
        attention_resolutions="",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        dims=3,
        num_groups=32,
        in_channels=32,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
    )
