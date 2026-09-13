"""
model_finetune.py

Architecture surgery for fine-tuning TCIRModel (trained on global TCIR,
4 channels, no metadata, 3 heads) onto INSAT-3D/3DR India-basin data
(6 channels, 4 ERA5 meta-features, 6 heads).

This module does NOT retrain the backbone from scratch. It builds a new
model with the modified input/trunk/head shapes and transfers every
weight that matches shape from a pretrained checkpoint, leaving only the
genuinely new parameters randomly initialized.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tcir_model import ConvBNAct, SqueezeExcite, make_stage  # noqa: F401 (SqueezeExcite/ConvBNAct re-exported for clarity)


# --------------------------------------------------------------------------
# INSAT channel handling
# --------------------------------------------------------------------------

# INSAT-3D/3DR raw channel order as delivered in the .h5 frames (per the
# uploaded sample 3DIMG_*_L1C_*.h5): SWIR, MIR, TIR1, TIR2, VIS, WV.
INSAT_RAW_ORDER = ["SWIR", "MIR", "TIR1", "TIR2", "VIS", "WV"]

# Order the model actually consumes, chosen so the first 3 slots line up
# with the pretrained model's [IR1, WV, VIS] stem channels:
#   IR1 (pretrained) <-> TIR1 (INSAT)
MODEL_CHANNEL_ORDER = ["TIR1", "WV", "VIS", "SWIR", "MIR", "TIR2"]

# Number of pretrained input-channel slots we can transfer into the new
# 6-channel stem (IR1->TIR1, WV->WV, VIS->VIS). Pretrained slot 3 was PMW,
# which has no INSAT equivalent, so only the first 3 slots transfer.
N_TRANSFERABLE_STEM_CHANNELS = 3


def reorder_insat_channels(raw_stack: torch.Tensor) -> torch.Tensor:
    """Reorder a (C, H, W) or (B, C, H, W) INSAT stack from
    INSAT_RAW_ORDER into MODEL_CHANNEL_ORDER."""
    idx = [INSAT_RAW_ORDER.index(ch) for ch in MODEL_CHANNEL_ORDER]
    channel_dim = 0 if raw_stack.dim() == 3 else 1
    return raw_stack.index_select(channel_dim, torch.tensor(idx, device=raw_stack.device))


# --------------------------------------------------------------------------
# Fine-tune model
# --------------------------------------------------------------------------

class TCIRFineTuneModel(nn.Module):
    """TCIRModel variant for the India fine-tuning run.

    Differences from the pretrained TCIRModel:
      - 6-channel stem instead of 4.
      - meta_dim=4 (ERA5 u10, v10, msl, sst) -> trunk input is 256+32=288.
      - 6 output heads instead of 3: vmax, mslp, r34avg (transferred) plus
        rmw, r50avg, r64avg (new).
    """

    def __init__(self, in_channels: int = 6, meta_dim: int = 4):
        super().__init__()
        self.meta_dim = meta_dim

        # --- Stem (new: 6 input channels) ---
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # --- Backbone stages (structurally identical to the pretrained
        # TCIRBackbone; weights transfer 1:1 by name where shapes match) ---
        self.stage1 = make_stage(32, 32, num_blocks=2, stride=1)
        self.stage2 = make_stage(32, 64, num_blocks=2, stride=2)
        self.stage3 = make_stage(64, 128, num_blocks=3, stride=2)
        self.stage4 = make_stage(128, 192, num_blocks=3, stride=2)
        self.stage5 = make_stage(192, 256, num_blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Metadata branch (new) ---
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
        )

        # --- Trunk: first layer is new (288 in), second layer transfers ---
        self.trunk_fc1 = nn.Linear(256 + 32, 128)   # NEW - shape mismatch vs pretrained (256,128)
        self.trunk_act1 = nn.SiLU(inplace=True)
        self.trunk_drop = nn.Dropout(0.1)
        self.trunk_fc2 = nn.Linear(128, 64)         # TRANSFERS from pretrained trunk[3]

        # --- Heads: first 3 transfer, last 3 are new ---
        self.vmax_head = nn.Linear(64, 1)
        self.mslp_head = nn.Linear(64, 1)
        self.r34avg_head = nn.Linear(64, 1)
        self.rmw_head = nn.Linear(64, 1)
        self.r50avg_head = nn.Linear(64, 1)
        self.r64avg_head = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        img_feat = torch.flatten(self.gap(x), 1)          # (B, 256)

        meta_feat = self.meta_branch(meta)                 # (B, 32)
        combined = torch.cat([img_feat, meta_feat], dim=1)  # (B, 288)

        h = self.trunk_fc1(combined)
        h = self.trunk_act1(h)
        h = self.trunk_drop(h)
        h = self.trunk_fc2(h)
        h = self.trunk_act1(h)  # trunk's second SiLU (same module reused, matches original Sequential)

        return {
            "vmax": self.vmax_head(h),
            "mslp": self.mslp_head(h),
            "r34avg": self.r34avg_head(h),
            "rmw": self.rmw_head(h),
            "r50avg": self.r50avg_head(h),
            "r64avg": self.r64avg_head(h),
        }

    # ---------------------------------------------------------------- #
    # Parameter groups for freezing / differential LR (spec section 7)
    # ---------------------------------------------------------------- #
    def get_param_groups(self, lr_low: float = 1e-4, lr_high: float = 1e-3) -> List[dict]:
        # Freeze stage1, stage2 entirely.
        for p in self.stage1.parameters():
            p.requires_grad = False
        for p in self.stage2.parameters():
            p.requires_grad = False

        low_lr_modules = [self.stage3, self.stage4, self.stage5,
                           self.vmax_head, self.mslp_head, self.r34avg_head]
        low_lr_params: List[nn.Parameter] = []
        for m in low_lr_modules:
            low_lr_params += [p for p in m.parameters() if p.requires_grad]
        low_lr_params += [p for p in self.trunk_fc2.parameters() if p.requires_grad]

        high_lr_modules = [self.stem, self.meta_branch, self.trunk_fc1,
                            self.rmw_head, self.r50avg_head, self.r64avg_head]
        high_lr_params: List[nn.Parameter] = []
        for m in high_lr_modules:
            high_lr_params += [p for p in m.parameters() if p.requires_grad]

        return [
            {"params": low_lr_params, "lr": lr_low, "name": "pretrained_unfrozen"},
            {"params": high_lr_params, "lr": lr_high, "name": "new_random_init"},
        ]


# --------------------------------------------------------------------------
# Weight transfer
# --------------------------------------------------------------------------

def build_and_load_finetune_model(
    checkpoint_path: str,
    meta_dim: int = 4,
    device: str = "cpu",
) -> Tuple[TCIRFineTuneModel, dict]:
    """Build the fine-tune model and transfer everything possible from the
    pretrained checkpoint. Prints exactly which keys were transferred,
    randomly initialized, or skipped, per spec section 7.

    Returns (model, checkpoint_dict) so the caller can also pull
    label_mean/label_std etc. out of the checkpoint.
    """
    model = TCIRFineTuneModel(in_channels=6, meta_dim=meta_dim)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        pretrained_state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        pretrained_state = ckpt["model_state_dict"]
    else:
        pretrained_state = ckpt

    # Build a state dict for the new model, remapping names where the
    # pretrained model used a Sequential ("trunk.0", "trunk.3", ...) but
    # the new model uses named submodules (trunk_fc1, trunk_fc2, ...).
    # Pretrained TCIRModel.trunk = Sequential(Linear, SiLU, Dropout, Linear, SiLU)
    #   -> trunk.0 = fc1 (256->128)   [SHAPE MISMATCH, do not transfer]
    #   -> trunk.3 = fc2 (128->64)    [transfers]
    remapped = {}
    for k, v in pretrained_state.items():
        if k.startswith("trunk.3."):
            remapped[k.replace("trunk.3.", "trunk_fc2.")] = v
        elif k.startswith("trunk.0."):
            continue  # shape mismatch (256,128) vs (288,128); skip on purpose
        elif k.startswith("backbone.stem."):
            continue  # shape mismatch (32,4,7,7) vs (32,6,7,7); handled manually below
        elif k.startswith("backbone."):
            remapped[k.replace("backbone.", "")] = v
        else:
            remapped[k] = v

    new_state = model.state_dict()
    transferred, skipped_shape, random_init = [], [], []
    for name, param in new_state.items():
        if name in remapped and remapped[name].shape == param.shape:
            new_state[name] = remapped[name]
            transferred.append(name)
        elif name in remapped:
            skipped_shape.append(name)
            random_init.append(name)
        else:
            random_init.append(name)

    model.load_state_dict(new_state, strict=False)

    # --- Manual stem-conv surgery ---
    pretrained_stem_conv_key = "backbone.stem.0.conv.weight"  # ConvBNAct.conv
    if pretrained_stem_conv_key in pretrained_state:
        old_w = pretrained_state[pretrained_stem_conv_key]  # (32, 4, 7, 7)
        new_conv = model.stem[0].conv  # Conv2d(6, 32, 7, ...)
        with torch.no_grad():
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            new_conv.weight[:, :N_TRANSFERABLE_STEM_CHANNELS, :, :] = old_w[:, :N_TRANSFERABLE_STEM_CHANNELS, :, :]
        transferred.append("stem.0.conv.weight (channels 0-2 only)")
        random_init.append("stem.0.conv.weight (channels 3-5 only)")
    else:
        print(f"WARNING: could not find '{pretrained_stem_conv_key}' in checkpoint; "
              f"stem conv is fully randomly initialized. Check checkpoint key names.")
        random_init.append("stem.0.conv.weight (ALL channels - pretrained key not found)")

    # --- Report ---
    print("=" * 70)
    print("WEIGHT TRANSFER REPORT")
    print("=" * 70)
    print(f"\nTransferred ({len(transferred)} tensors):")
    for n in sorted(transferred):
        print(f"  [OK]      {n}")
    print(f"\nRandomly initialized ({len(random_init)} tensors):")
    for n in sorted(random_init):
        print(f"  [RANDOM]  {n}")
    if skipped_shape:
        print(f"\nSkipped due to shape mismatch ({len(skipped_shape)} tensors):")
        for n in sorted(skipped_shape):
            print(f"  [SKIP]    {n}")
    print("=" * 70)

    return model, ckpt