# -*- coding: utf-8 -*-
"""FER System - ASPP Module

Atrous Spatial Pyramid Pooling for multi-scale feature extraction.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Uses parallel branches with different dilation rates to capture
    multi-scale context, plus a global average pooling branch.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dilations: List of dilation rates for the atrous convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 6, 12, 18]

        num_branches = len(dilations) + 1  # dilated convs + global pool
        branch_channels = out_channels // num_branches
        # Remainder channels added to the last branch
        remainder = out_channels - branch_channels * num_branches

        # Atrous convolution branches
        self.convs = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            # Last dilated conv branch gets the remainder channels
            out_ch = branch_channels + (remainder if i == len(dilations) - 1 else 0)
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_ch, out_ch, 3,
                        padding=dilation, dilation=dilation, bias=False,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map [B, C, H, W].

        Returns:
            Multi-scale feature map [B, out_channels, H, W].
        """
        size = x.size()[2:]
        features = []

        for conv in self.convs:
            features.append(conv(x))

        # Global pooling branch with upsample
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat, size=size, mode="bilinear", align_corners=False
        )
        features.append(global_feat)

        # Concatenate and project
        x = torch.cat(features, dim=1)
        return self.project(x)
