"""
Hourglass Network backbone for MEPNet
Based on "Stacked Hourglass Networks for Human Pose Estimation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ResidualBlock(nn.Module):
    """Basic residual block with optional shape conditioning"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shape_cond_dim: int = 0
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)

        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        # Shape conditioning
        self.shape_cond_dim = shape_cond_dim
        if shape_cond_dim > 0:
            self.shape_film = FiLMLayer(shape_cond_dim, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, shape_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply shape conditioning if available
        if self.shape_cond_dim > 0 and shape_cond is not None:
            out = self.shape_film(out, shape_cond)

        out = out + residual
        out = self.relu(out)

        return out


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for conditioning"""

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.gamma_layer = nn.Linear(cond_dim, num_features)
        self.beta_layer = nn.Linear(cond_dim, num_features)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) feature maps
            condition: (B, cond_dim) conditioning vector
        Returns:
            Modulated features
        """
        B, C, H, W = features.shape

        # Generate scaling and shifting parameters
        gamma = self.gamma_layer(condition).view(B, C, 1, 1)
        beta = self.beta_layer(condition).view(B, C, 1, 1)

        # Apply modulation
        return gamma * features + beta


class HourglassModule(nn.Module):
    """Single hourglass module"""

    def __init__(
        self,
        depth: int,
        num_features: int,
        num_blocks: int = 1,
        shape_cond_dim: int = 0
    ):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.shape_cond_dim = shape_cond_dim

        # Upper branch (skip connection)
        self._make_up_layers()

        # Lower branch
        self._make_low_layers()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, stride=2)

    def _make_up_layers(self):
        """Create upper branch layers"""
        self.up_blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            self.up_blocks.append(
                ResidualBlock(
                    self.num_features,
                    self.num_features,
                    shape_cond_dim=self.shape_cond_dim
                )
            )

    def _make_low_layers(self):
        """Create lower branch layers"""
        self.low1_blocks = nn.ModuleList()
        self.low2_blocks = nn.ModuleList()
        self.low3_blocks = nn.ModuleList()

        # Going down
        for i in range(self.num_blocks):
            self.low1_blocks.append(
                ResidualBlock(
                    self.num_features,
                    self.num_features,
                    shape_cond_dim=self.shape_cond_dim
                )
            )

        # Recursive or bottom layer
        if self.depth > 1:
            self.low2 = HourglassModule(
                self.depth - 1,
                self.num_features,
                self.num_blocks,
                self.shape_cond_dim
            )
        else:
            low2_blocks = []
            for i in range(self.num_blocks):
                low2_blocks.append(
                    ResidualBlock(
                        self.num_features,
                        self.num_features,
                        shape_cond_dim=self.shape_cond_dim
                    )
                )
            self.low2 = nn.Sequential(*low2_blocks)

        # Coming up
        for i in range(self.num_blocks):
            self.low3_blocks.append(
                ResidualBlock(
                    self.num_features,
                    self.num_features,
                    shape_cond_dim=self.shape_cond_dim
                )
            )

    def forward(self, x: torch.Tensor, shape_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upper branch
        up_out = x
        for block in self.up_blocks:
            up_out = block(up_out, shape_cond)

        # Lower branch - going down
        low_out = self.pool(x)
        for block in self.low1_blocks:
            low_out = block(low_out, shape_cond)

        # Recursive or bottom
        if isinstance(self.low2, HourglassModule):
            low_out = self.low2(low_out, shape_cond)
        else:
            for block in self.low2:
                low_out = block(low_out, shape_cond)

        # Coming up
        for block in self.low3_blocks:
            low_out = block(low_out, shape_cond)

        # Upsample
        low_out = F.interpolate(low_out, scale_factor=2, mode='nearest')

        # Merge branches
        return up_out + low_out


class HourglassNet(nn.Module):
    """Stacked Hourglass Network"""

    def __init__(
        self,
        num_stacks: int = 2,
        num_blocks: int = 1,
        num_features: int = 256,
        input_channels: int = 3,
        shape_cond_dim: int = 0,
        hourglass_depth: int = 4
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_features = num_features
        self.shape_cond_dim = shape_cond_dim

        # Initial processing
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = ResidualBlock(64, 128, stride=1, shape_cond_dim=shape_cond_dim)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.layer2 = ResidualBlock(128, 128, stride=1, shape_cond_dim=shape_cond_dim)
        self.layer3 = ResidualBlock(128, num_features, stride=1, shape_cond_dim=shape_cond_dim)

        # Hourglass stacks
        self.hourglasses = nn.ModuleList()
        self.features_after_hg = nn.ModuleList()
        self.features_before_out = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.merge_features = nn.ModuleList()
        self.merge_predictions = nn.ModuleList()

        for i in range(num_stacks):
            # Hourglass module
            self.hourglasses.append(
                HourglassModule(
                    hourglass_depth,
                    num_features,
                    num_blocks,
                    shape_cond_dim
                )
            )

            # Post-hourglass residual blocks
            res_blocks = []
            for j in range(num_blocks):
                res_blocks.append(
                    ResidualBlock(
                        num_features,
                        num_features,
                        shape_cond_dim=shape_cond_dim
                    )
                )
            self.features_after_hg.append(nn.Sequential(*res_blocks))

            # Feature layers before output
            self.features_before_out.append(
                nn.Conv2d(num_features, num_features, 1, bias=False)
            )

            # Output layer (will be specialized in task-specific heads)
            self.out_layers.append(
                nn.Conv2d(num_features, num_features, 1, bias=False)
            )

            # For intermediate supervision
            if i < num_stacks - 1:
                self.merge_features.append(
                    nn.Conv2d(num_features, num_features, 1, bias=False)
                )
                self.merge_predictions.append(
                    nn.Conv2d(num_features, num_features, 1, bias=False)
                )

    def forward(
        self,
        x: torch.Tensor,
        shape_cond: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
            shape_cond: Shape conditioning vector (B, shape_cond_dim)
        Returns:
            List of feature maps from each stack
        """
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x, shape_cond)
        x = self.pool(x)
        x = self.layer2(x, shape_cond)
        x = self.layer3(x, shape_cond)

        # Process through stacks
        outputs = []
        for i in range(self.num_stacks):
            # Hourglass
            y = self.hourglasses[i](x, shape_cond)

            # Post-processing
            # features_after_hg[i] is a Sequential of ResidualBlocks
            # We need to apply each block with shape_cond
            for block in self.features_after_hg[i]:
                y = block(y, shape_cond)

            y = self.features_before_out[i](y)

            # Store output features
            outputs.append(y)

            # Prepare for next stack
            if i < self.num_stacks - 1:
                # Merge features for next stack
                x = x + self.merge_predictions[i](y) + self.merge_features[i](y)

        return outputs


class LightweightHourglass(nn.Module):
    """
    Lightweight version of Hourglass for faster inference
    Uses depthwise separable convolutions
    """

    def __init__(
        self,
        num_stacks: int = 1,
        num_features: int = 128,
        input_channels: int = 3,
        shape_cond_dim: int = 0
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_features = num_features

        # Initial layers
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, num_features, stride=1)
        )

        # Hourglass stacks
        self.stacks = nn.ModuleList()
        for i in range(num_stacks):
            self.stacks.append(
                LightweightHourglassModule(num_features, shape_cond_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        shape_cond: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        x = self.stem(x)

        outputs = []
        for stack in self.stacks:
            x = stack(x, shape_cond)
            outputs.append(x)

        return outputs


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class LightweightHourglassModule(nn.Module):
    """Single lightweight hourglass module"""

    def __init__(self, num_features: int, shape_cond_dim: int = 0):
        super().__init__()
        self.num_features = num_features

        # Encoder (downsampling)
        self.encoder = nn.ModuleList([
            DepthwiseSeparableConv(num_features, num_features, stride=2),
            DepthwiseSeparableConv(num_features, num_features, stride=2),
            DepthwiseSeparableConv(num_features, num_features, stride=2)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(num_features, num_features * 2),
            DepthwiseSeparableConv(num_features * 2, num_features)
        )

        # Decoder (upsampling)
        self.decoder = nn.ModuleList([
            DepthwiseSeparableConv(num_features * 2, num_features),
            DepthwiseSeparableConv(num_features * 2, num_features),
            DepthwiseSeparableConv(num_features * 2, num_features)
        ])

        # Shape conditioning
        if shape_cond_dim > 0:
            self.shape_film = FiLMLayer(shape_cond_dim, num_features)
        else:
            self.shape_film = None

    def forward(
        self,
        x: torch.Tensor,
        shape_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encoder path with skip connections
        skip_connections = []
        enc = x

        for encoder_block in self.encoder:
            skip_connections.append(enc)
            enc = encoder_block(enc)

        # Bottleneck
        enc = self.bottleneck(enc)

        # Decoder path
        dec = enc
        for i, decoder_block in enumerate(self.decoder):
            # Upsample
            dec = F.interpolate(dec, scale_factor=2, mode='nearest')

            # Skip connection
            skip = skip_connections[-(i + 1)]
            dec = torch.cat([dec, skip], dim=1)

            # Process
            dec = decoder_block(dec)

        # Apply shape conditioning if available
        if self.shape_film is not None and shape_cond is not None:
            dec = self.shape_film(dec, shape_cond)

        return dec