# -*- coding: utf-8 -*-
"""FER System - Attention Modules

SE (Squeeze-and-Excitation) and CBAM (Convolutional Block Attention Module)
implementations for channel and spatial attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    Squeezes spatial dimensions via global average pooling, then excites
    channels through a bottleneck FC layer to compute channel-wise weights.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck layer.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Channel-attention-weighted tensor [B, C, H, W].
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.sigmoid()


class ChannelAttention(nn.Module):
    """CBAM channel attention sub-module.

    Uses shared MLP on both average-pooled and max-pooled features.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the shared MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Channel attention map [B, C, 1, 1].
        """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM spatial attention sub-module.

    Concatenates channel-wise average and max pooling then applies convolution.

    Args:
        kernel_size: Convolution kernel size (typically 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Spatial attention map [B, 1, H, W].
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.

    Args:
        channels: Number of input channels.
        reduction: Channel attention reduction ratio.
        spatial_kernel: Kernel size for spatial attention convolution.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attention-refined tensor [B, C, H, W].
        """
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
