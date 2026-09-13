"""
tcir_model.py

A ResNet-style CNN for tropical cyclone intensity regression on the TCIR
dataset. Takes 4-channel satellite imagery (201x201) plus optional scalar
metadata (basin one-hot, day-of-year sin/cos, etc.) and predicts three
storm-intensity targets: VMAX, MSLP, and R34avg.

Architecture summary
---------------------
Stem            : Conv7x7/s2 -> BN -> SiLU -> MaxPool3x3/s2      (201 -> 51)
Stage 1 (x2)    : 32  filters, stride 1                          (51 -> 51)
Stage 2 (x2)    : 64  filters, stride 2 on first block            (51 -> 26)
Stage 3 (x3)    : 128 filters, stride 2 on first block            (26 -> 13)
Stage 4 (x3)    : 192 filters, stride 2 on first block            (13 -> 7)
Stage 5 (x2)    : 256 filters, stride 2 on first block            (7  -> 4)
Each stage ends with a Squeeze-and-Excite channel-attention block.
Global Average Pool -> 256-d image embedding.
Optional metadata MLP -> 32-d embedding, concatenated -> 288-d.
Shared trunk -> 64-d representation -> three independent linear heads.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU. The basic conv unit used everywhere."""

    def __init__(self,in_channels: int,out_channels: int,kernel_size: int,stride: int = 1,padding: int = 0,activation: bool = True,):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Lightweight Squeeze-and-Excite channel-attention block.

    Squeezes spatial dims via global average pooling, learns a per-channel
    gate through a small bottleneck MLP, and rescales the input channels.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class ResidualBlock(nn.Module):
    """Standard 2-conv residual block (3x3 -> 3x3), BN+SiLU after each conv.

    Only the first block in a stage carries the stride (for spatial
    downsampling / channel expansion); all others use stride 1. A 1x1
    projection shortcut is used whenever shape/channels change.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        # Second conv has no activation yet -- activation happens after the
        # residual addition.
        self.conv2 = ConvBNAct(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            activation=False,
        )

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        return self.act(out)


def make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    """Build one residual stage: `num_blocks` ResidualBlocks followed by a
    Squeeze-Excite block. Only the first block applies `stride`."""

    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    layers.append(SqueezeExcite(out_channels))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------


class TCIRBackbone(nn.Module):
    """Convolutional feature extractor: stem + 5 residual stages + GAP.

    Input:  (batch, 4, 201, 201)
    Output: (batch, 256) global-average-pooled feature vector.
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # Stem: 201x201 -> 101x101 -> 51x51
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages. Channel counts / block counts / strides per spec.
        self.stage1 = make_stage(32, 32, num_blocks=2, stride=1)    # 51 -> 51
        self.stage2 = make_stage(32, 64, num_blocks=2, stride=2)    # 51 -> 26
        self.stage3 = make_stage(64, 128, num_blocks=3, stride=2)   # 26 -> 13
        self.stage4 = make_stage(128, 192, num_blocks=3, stride=2)  # 13 -> 7
        self.stage5 = make_stage(192, 256, num_blocks=2, stride=2)  # 7 -> 4

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gap(x)
        return torch.flatten(x, 1)  # (batch, 256)


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------


class TCIRModel(nn.Module):
    """Full tropical-cyclone intensity regression model.

    Combines the image backbone with an optional metadata MLP branch, feeds
    the combined embedding through a shared trunk, and predicts VMAX, MSLP,
    and R34avg via three independent linear heads.
    """

    def __init__(self, in_channels: int = 4, meta_dim: int = 0):
        super().__init__()
        self.meta_dim = meta_dim

        self.backbone = TCIRBackbone(in_channels=in_channels)

        if meta_dim > 0:
            self.meta_branch = nn.Sequential(
                nn.Linear(meta_dim, 32),
                nn.SiLU(inplace=True),
                nn.Linear(32, 32),
            )
            trunk_in = 256 + 32  # 288
        else:
            self.meta_branch = None
            trunk_in = 256

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
        )

        # Independent output heads
        self.vmax_head = nn.Linear(64, 1)
        self.mslp_head = nn.Linear(64, 1)
        self.r34avg_head = nn.Linear(64, 1)

    def forward(
        self, image: torch.Tensor, meta: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        img_feat = self.backbone(image)  # (batch, 256)

        if self.meta_branch is not None:
            if meta is None:
                raise ValueError(
                    "Model was built with meta_dim > 0 but no metadata "
                    "tensor was provided."
                )
            meta_feat = self.meta_branch(meta)  # (batch, 32)
            combined = torch.cat([img_feat, meta_feat], dim=1)  # (batch, 288)
        else:
            combined = img_feat  # (batch, 256)

        shared = self.trunk(combined)  # (batch, 64)

        return {
            "vmax": self.vmax_head(shared),
            "mslp": self.mslp_head(shared),
            "r34avg": self.r34avg_head(shared),
        }


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


class MultiTaskHuberLoss(nn.Module):
    """Multi-task Smooth L1 (Huber) loss over the three intensity heads.

    Each head gets its own SmoothL1Loss term; terms are combined into a
    weighted sum via configurable per-head weights. `forward` returns a
    dict with each individual loss plus the weighted "total" loss.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weights = weights or {"vmax": 1.0, "mslp": 1.0, "r34avg": 1.0}
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = 0.0
        for key in ("vmax", "mslp", "r34avg"):
            pred = preds[key]
            target = targets[key].view_as(pred)
            l = self.loss_fn(pred, target)
            losses[key] = l
            total = total + self.weights.get(key, 1.0) * l
        losses["total"] = total
        return losses


# --------------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 8
    meta_dim = 6  # e.g. 4 basin one-hot dims + sin/cos day-of-year

    model = TCIRModel(in_channels=4, meta_dim=meta_dim)

    dummy_images = torch.randn(batch_size, 4, 201, 201)
    dummy_meta = torch.randn(batch_size, meta_dim)

    outputs = model(dummy_images, dummy_meta)

    print("Output shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {tuple(v.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Loss sanity check
    dummy_targets = {
        "vmax": torch.randn(batch_size, 1),
        "mslp": torch.randn(batch_size, 1),
        "r34avg": torch.randn(batch_size, 1),
    }
    criterion = MultiTaskHuberLoss(weights={"vmax": 1.0, "mslp": 0.5, "r34avg": 0.5})
    losses = criterion(outputs, dummy_targets)
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Also sanity-check the no-metadata path
    model_no_meta = TCIRModel(in_channels=4, meta_dim=0)
    outputs_no_meta = model_no_meta(dummy_images)
    print("\nOutput shapes (no metadata branch):")
    for k, v in outputs_no_meta.items():
        print(f"  {k}: {tuple(v.shape)}")