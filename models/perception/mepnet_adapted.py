"""
Adapted MEPNet for IKEA Assembly
Modifies LEGO's HourglassShapeCondModel for continuous 6D pose prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

# Import from adapted hourglass module
from .hourglass import HourglassNet, ResidualBlock


@dataclass
class MEPNetConfig:
    """Configuration for MEPNet model"""
    # Input/Output dimensions
    input_channels: int = 3
    image_size: Tuple[int, int] = (512, 512)

    # Architecture
    num_stacks: int = 2  # Number of hourglass stacks
    num_blocks: int = 1  # Number of residual blocks
    num_features: int = 256  # Feature dimension

    # Shape conditioning
    shape_encoding_dim: int = 512
    use_shape_condition: bool = True
    shape_encoder_type: str = "pointnet"  # "pointnet" or "voxel"

    # Output heads
    num_keypoints: int = 16  # Keypoints per part
    num_rotation_bins: int = 24  # For discrete rotation (LEGO compatibility)
    use_continuous_rotation: bool = True  # Use 6D rotation representation

    # Part detection
    max_parts: int = 10
    detect_parts: bool = True

    # Additional outputs
    predict_mask: bool = True
    predict_depth: bool = False

    # Training
    intermediate_supervision: bool = True


class ShapeEncoder(nn.Module):
    """
    Encode 3D shape information for conditioning
    Replaces LEGO's brick-specific encoding
    """

    def __init__(self, config: MEPNetConfig):
        super().__init__()
        self.config = config

        if config.shape_encoder_type == "pointnet":
            self.encoder = PointNetEncoder(
                output_dim=config.shape_encoding_dim,
                num_points=1024
            )
        elif config.shape_encoder_type == "voxel":
            self.encoder = VoxelEncoder(
                output_dim=config.shape_encoding_dim,
                voxel_size=32
            )
        else:
            raise ValueError(f"Unknown shape encoder: {config.shape_encoder_type}")

    def forward(self, shape_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shape_input: Point cloud (B, N, 3) or voxel grid (B, 1, D, D, D)
        Returns:
            Shape encoding (B, shape_encoding_dim)
        """
        return self.encoder(shape_input)


class PointNetEncoder(nn.Module):
    """Simple PointNet encoder for 3D shapes"""

    def __init__(self, output_dim: int = 512, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points

        # Point-wise MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        # Global feature extraction
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(0.3)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point cloud
        Returns:
            Global feature (B, output_dim)
        """
        B, N, _ = points.shape

        # Transpose for Conv1d (B, 3, N)
        x = points.transpose(1, 2)

        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 512)

        # Final MLPs
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class VoxelEncoder(nn.Module):
    """3D CNN encoder for voxel grids"""

    def __init__(self, output_dim: int = 512, voxel_size: int = 32):
        super().__init__()
        self.voxel_size = voxel_size

        # 3D convolutions
        self.conv1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, 3, stride=2, padding=1)

        # Calculate flattened size
        final_size = (voxel_size // 8) ** 3 * 256

        self.fc1 = nn.Linear(final_size, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: (B, 1, D, D, D) voxel grid
        Returns:
            Global feature (B, output_dim)
        """
        x = F.relu(self.conv1(voxels))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # MLPs
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Rotation6DHead(nn.Module):
    """
    Continuous 6D rotation representation head
    Based on "On the Continuity of Rotation Representations in Neural Networks"
    """

    def __init__(self, in_features: int, num_parts: int = 1):
        super().__init__()
        self.num_parts = num_parts

        # Predict 6D rotation (first two columns of rotation matrix)
        self.fc = nn.Linear(in_features, num_parts * 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor (B, in_features)
        Returns:
            Rotation matrices (B, num_parts, 3, 3)
        """
        B = x.shape[0]

        # Predict 6D representation
        rot_6d = self.fc(x).view(B, self.num_parts, 6)

        # Convert to rotation matrix
        rot_mat = self.six_d_to_rotation_matrix(rot_6d)

        return rot_mat

    @staticmethod
    def six_d_to_rotation_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to 3x3 rotation matrix
        Args:
            rot_6d: (B, N, 6) tensor
        Returns:
            Rotation matrices (B, N, 3, 3)
        """
        B, N = rot_6d.shape[:2]

        # Reshape to get two vectors
        a1 = rot_6d[..., :3]  # (B, N, 3)
        a2 = rot_6d[..., 3:]  # (B, N, 3)

        # Gram-Schmidt process
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        # Stack to form rotation matrix
        rot_mat = torch.stack([b1, b2, b3], dim=-1)  # (B, N, 3, 3)

        return rot_mat


class MEPNetAdapted(nn.Module):
    """
    Main MEPNet model adapted for IKEA assembly
    """

    def __init__(self, config: MEPNetConfig):
        super().__init__()
        self.config = config

        # Shape encoder
        if config.use_shape_condition:
            self.shape_encoder = ShapeEncoder(config)
            shape_cond_dim = config.shape_encoding_dim
        else:
            shape_cond_dim = 0

        # Hourglass backbone
        self.hourglass = HourglassNet(
            num_stacks=config.num_stacks,
            num_blocks=config.num_blocks,
            num_features=config.num_features,
            input_channels=config.input_channels,
            shape_cond_dim=shape_cond_dim
        )

        # Output heads
        out_features = config.num_features

        # Part detection head
        if config.detect_parts:
            self.part_detector = nn.Sequential(
                nn.Conv2d(out_features, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, config.max_parts, 1)
            )

        # Keypoint detection head
        self.keypoint_head = nn.ModuleList([
            nn.Conv2d(out_features, config.num_keypoints, 1)
            for _ in range(config.num_stacks)
        ])

        # Rotation prediction head
        if config.use_continuous_rotation:
            # 6D continuous rotation
            self.rotation_head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_features, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    Rotation6DHead(256, config.max_parts)
                )
                for _ in range(config.num_stacks)
            ])
        else:
            # Discrete rotation bins (LEGO style)
            self.rotation_head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_features, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, config.num_rotation_bins * config.max_parts, 1)
                )
                for _ in range(config.num_stacks)
            ])

        # Translation offset head
        self.translation_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_features, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, config.max_parts * 3, 1)  # (x, y, z) offsets
            )
            for _ in range(config.num_stacks)
        ])

        # Mask prediction head
        if config.predict_mask:
            self.mask_head = nn.Sequential(
                nn.Conv2d(out_features, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, config.max_parts, 1)
            )

        # Depth prediction head
        if config.predict_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(out_features, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        image: torch.Tensor,
        shape: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            image: Input image (B, 3, H, W)
            shape: 3D shape input (point cloud or voxel)
            return_intermediate: Whether to return intermediate stack outputs

        Returns:
            Dictionary of predictions
        """
        B = image.shape[0]

        # Shape encoding
        shape_encoding = None
        if self.config.use_shape_condition and shape is not None:
            # Handle list of shapes (batch)
            if isinstance(shape, list):
                # If list is empty or contains None, skip shape encoding
                if not shape or all(s is None for s in shape):
                    shape_encoding = None
                else:
                    # For now, disable shape encoding when meshes are lists
                    # TODO: Implement proper batched shape encoding
                    shape_encoding = None
            elif isinstance(shape, torch.Tensor) and shape.numel() > 0:
                shape_encoding = self.shape_encoder(shape)
            else:
                shape_encoding = None

        # Hourglass forward pass
        features_list = self.hourglass(image, shape_encoding)

        outputs = {}

        # Process each hourglass stack output
        keypoints_list = []
        rotations_list = []
        translations_list = []

        for i, features in enumerate(features_list):
            # Keypoint heatmaps
            keypoints = self.keypoint_head[i](features)
            keypoints_list.append(keypoints)

            # Rotation prediction
            if self.config.use_continuous_rotation:
                # Returns (B, max_parts, 3, 3)
                rotations = self.rotation_head[i](features)
            else:
                # Discrete rotation bins
                rot_logits = self.rotation_head[i](features)
                rot_logits = rot_logits.view(B, self.config.max_parts, -1)
                rotations = F.softmax(rot_logits, dim=-1)
            rotations_list.append(rotations)

            # Translation offsets
            translations = self.translation_head[i](features)
            translations = translations.view(B, self.config.max_parts, 3, -1)
            translations = translations.mean(dim=-1)  # Average over spatial dims
            translations_list.append(translations)

        # Use last stack's outputs as primary
        outputs['keypoints'] = keypoints_list[-1]
        outputs['rotations'] = rotations_list[-1]
        outputs['translations'] = translations_list[-1]

        # Part detection (from last features)
        if self.config.detect_parts:
            outputs['part_scores'] = self.part_detector(features_list[-1])

        # Mask prediction
        if self.config.predict_mask:
            outputs['masks'] = torch.sigmoid(self.mask_head(features_list[-1]))

        # Depth prediction
        if self.config.predict_depth:
            outputs['depth'] = torch.sigmoid(self.depth_head(features_list[-1]))

        # Intermediate supervision
        if return_intermediate or self.config.intermediate_supervision:
            outputs['intermediate'] = {
                'keypoints': keypoints_list[:-1],
                'rotations': rotations_list[:-1],
                'translations': translations_list[:-1]
            }

        return outputs

    def decode_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        camera: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Decode network outputs to 6D poses

        Args:
            outputs: Network predictions
            camera: Camera parameters (K, R, t)
            threshold: Detection threshold

        Returns:
            List of detected parts with 6D poses
        """
        B = outputs['keypoints'].shape[0]
        results = []

        for b in range(B):
            batch_results = []

            # Get part scores
            if 'part_scores' in outputs:
                part_scores = torch.sigmoid(outputs['part_scores'][b])
                part_scores = part_scores.max(dim=1)[0].max(dim=1)[0]  # Max over spatial
                valid_parts = part_scores > threshold
            else:
                valid_parts = torch.ones(self.config.max_parts, dtype=torch.bool)

            # Process each valid part
            for i in range(self.config.max_parts):
                if not valid_parts[i]:
                    continue

                # Extract keypoints
                kp_heatmap = outputs['keypoints'][b, i]
                kp_coords = self._extract_keypoints_from_heatmap(kp_heatmap)

                # Get rotation
                if self.config.use_continuous_rotation:
                    R = outputs['rotations'][b, i].cpu().numpy()
                else:
                    # Decode discrete rotation
                    rot_idx = outputs['rotations'][b, i].argmax()
                    R = self._decode_discrete_rotation(rot_idx)

                # Get translation
                t = outputs['translations'][b, i].cpu().numpy()

                # Convert to world coordinates using camera
                if camera is not None:
                    K = camera['K'][b].cpu().numpy()
                    t_world = self._unproject_translation(t, kp_coords, K)
                else:
                    t_world = t

                part_result = {
                    'part_id': i,
                    'score': part_scores[i].item() if 'part_scores' in outputs else 1.0,
                    'R': R,
                    't': t_world,
                    'keypoints_2d': kp_coords,
                    'keypoints_3d': None  # Would need shape model for 3D keypoints
                }

                # Add mask if available
                if 'masks' in outputs:
                    part_result['mask'] = outputs['masks'][b, i].cpu().numpy()

                batch_results.append(part_result)

            results.append(batch_results)

        return results

    def _extract_keypoints_from_heatmap(self, heatmap: torch.Tensor) -> np.ndarray:
        """Extract keypoint coordinates from heatmap"""
        # Simple argmax approach
        flat_idx = heatmap.view(-1).argmax()
        h, w = heatmap.shape
        y = (flat_idx // w).float()
        x = (flat_idx % w).float()

        # Refine with local weighted average
        if x > 0 and x < w - 1 and y > 0 and y < h - 1:
            # 3x3 window around peak
            window = heatmap[int(y)-1:int(y)+2, int(x)-1:int(x)+2]

            # Weighted average
            xs, ys = torch.meshgrid(
                torch.arange(3, dtype=torch.float32),
                torch.arange(3, dtype=torch.float32)
            )
            x_offset = (xs * window).sum() / window.sum() - 1
            y_offset = (ys * window).sum() / window.sum() - 1

            x = x + x_offset
            y = y + y_offset

        return np.array([x.item(), y.item()])

    def _decode_discrete_rotation(self, rot_idx: int) -> np.ndarray:
        """Decode discrete rotation index to matrix"""
        # Simple uniform sampling of SO(3)
        # In practice, would use predefined rotation set
        angle = (rot_idx / self.config.num_rotation_bins) * 2 * np.pi

        # Rotation around vertical axis as example
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        return R

    def _unproject_translation(
        self,
        t_image: np.ndarray,
        keypoint_2d: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """Unproject translation from image space to world space"""
        # Simple unprojection using assumed depth
        # In practice, would use depth prediction or known part size

        # Assume depth (would be predicted or estimated)
        depth = 1.0

        # Unproject keypoint
        x, y = keypoint_2d
        X = (x - K[0, 2]) * depth / K[0, 0]
        Y = (y - K[1, 2]) * depth / K[1, 1]
        Z = depth

        # Add image-space offset
        t_world = np.array([X, Y, Z]) + t_image

        return t_world


def create_mepnet_model(config: Optional[MEPNetConfig] = None) -> MEPNetAdapted:
    """Factory function to create MEPNet model"""
    if config is None:
        config = MEPNetConfig()

    return MEPNetAdapted(config)