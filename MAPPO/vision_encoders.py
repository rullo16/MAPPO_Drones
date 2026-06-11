"""
CNN encoder for visual observations.
"""

import numpy as np
import torch
import torch.nn as nn


class EfficientVisionEncoder(nn.Module):
    """
    Compact CNN encoder: four strided conv blocks followed by global average
    pooling, with an optional linear projection to `output_dim`.

    Note: BatchNorm is kept for compatibility with previously pretrained
    checkpoints, but this encoder is intended to be used FROZEN (eval mode)
    inside the RL agent — the MAPPO pipeline stores encoded features in the
    rollout buffer, so policy gradients cannot reach the encoder. Pretrain it
    with PretrainFeatureExtraction.ipynb first.

    input_shape: Tuple of (channels, height, width), e.g., (4, 84, 84)
    output_dim: Output feature dimension (default 256, the conv stack's
                native dimension; other values add a linear projection)
    """

    def __init__(self, input_shape, output_dim=256):
        super().__init__()

        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Discover the conv stack's output dimension instead of hardcoding it
        with torch.no_grad():
            conv_dim = self.features(torch.zeros(1, *input_shape)).shape[1]
        self.conv_dim = conv_dim

        if output_dim != conv_dim:
            self.projection = nn.Sequential(
                nn.Linear(conv_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Tanh()
            )
        else:
            self.projection = nn.Identity()

        self.output_dim = output_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, channels, height, width]

        returns features: Output tensor of shape [batch_size, output_dim]
        """
        x = self.features(x)
        x = self.projection(x)
        return x
