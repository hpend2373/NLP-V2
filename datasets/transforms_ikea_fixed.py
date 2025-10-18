"""
Data transformations and augmentations for IKEA dataset (FIXED VERSION)
Fixes image tensor scaling consistency issues
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, List, Optional, Any
import random
from scipy.spatial.transform import Rotation

class IKEATransform:
    """Base class for IKEA data transformations"""
    def __call__(self, sample: Dict) -> Dict:
        raise NotImplementedError

class ImageNormalize(IKEATransform):
    """Normalize image with ImageNet stats or custom values"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

    def __call__(self, sample: Dict) -> Dict:
        if 'manual_step_image' in sample:
            if isinstance(sample['manual_step_image'], np.ndarray):
                # FIX: Ensure image is in [0, 1] range before normalizing
                if sample['manual_step_image'].max() > 1.0:
                    sample['manual_step_image'] = sample['manual_step_image'] / 255.0
                # Image should be in CHW format
                sample['manual_step_image'] = (sample['manual_step_image'] - self.mean) / self.std
            elif isinstance(sample['manual_step_image'], torch.Tensor):
                # FIX: Ensure tensor is in [0, 1] range
                if sample['manual_step_image'].max() > 1.0:
                    sample['manual_step_image'] = sample['manual_step_image'] / 255.0
                mean = torch.tensor(self.mean).to(sample['manual_step_image'].device)
                std = torch.tensor(self.std).to(sample['manual_step_image'].device)
                sample['manual_step_image'] = (sample['manual_step_image'] - mean) / std
        return sample

class RandomCrop(IKEATransform):
    """Random crop augmentation with camera adjustment"""
    def __init__(self, crop_size: Tuple[int, int], preserve_center: bool = True):
        self.crop_size = crop_size
        self.preserve_center = preserve_center

    def __call__(self, sample: Dict) -> Dict:
        if 'manual_step_image' not in sample:
            return sample

        image = sample['manual_step_image']
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            h, w, _ = image.shape

        crop_h, crop_w = self.crop_size

        if h <= crop_h or w <= crop_w:
            return sample

        # Compute crop region
        if self.preserve_center:
            # Crop around center with small random offset
            max_offset = 0.1
            offset_h = int((h - crop_h) * (0.5 + random.uniform(-max_offset, max_offset)))
            offset_w = int((w - crop_w) * (0.5 + random.uniform(-max_offset, max_offset)))
        else:
            offset_h = random.randint(0, h - crop_h)
            offset_w = random.randint(0, w - crop_w)

        offset_h = max(0, min(offset_h, h - crop_h))
        offset_w = max(0, min(offset_w, w - crop_w))

        # Crop image
        if isinstance(image, torch.Tensor):
            sample['manual_step_image'] = image[:, offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
        else:
            sample['manual_step_image'] = image[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]

        # Adjust camera intrinsics
        if 'camera' in sample and 'K' in sample['camera']:
            K = sample['camera']['K'].clone() if isinstance(sample['camera']['K'], torch.Tensor) else sample['camera']['K'].copy()
            # Adjust principal point
            K[0, 2] -= offset_w
            K[1, 2] -= offset_h
            sample['camera']['K'] = K

        # Crop masks if present
        if 'masks_2d' in sample and sample['masks_2d'] is not None:
            cropped_masks = []
            for mask in sample['masks_2d']:
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        cropped_masks.append(mask[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w])
                    else:
                        cropped_masks.append(mask[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w])
                else:
                    cropped_masks.append(None)
            sample['masks_2d'] = cropped_masks

        return sample

class RandomScale(IKEATransform):
    """Random scale augmentation"""
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, sample: Dict) -> Dict:
        scale = random.uniform(*self.scale_range)

        # Scale image
        if 'manual_step_image' in sample:
            image = sample['manual_step_image']
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                sample['manual_step_image'] = cv2.resize(image, (new_w, new_h))

        # Scale camera intrinsics
        if 'camera' in sample and 'K' in sample['camera']:
            sample['camera']['K'][:2] *= scale

        # Scale translations
        if 'gt_poses' in sample:
            for pose in sample['gt_poses']:
                if 't' in pose:
                    pose['t'] *= scale

        return sample

class ColorJitter(IKEATransform):
    """Color jittering for manual images"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample: Dict) -> Dict:
        if 'manual_step_image' not in sample:
            return sample

        image = sample['manual_step_image']

        # Apply color jittering
        if isinstance(image, np.ndarray):
            # FIX: Handle both [0,1] and [0,255] ranges properly
            was_float = image.dtype in [np.float32, np.float64]
            input_range = image.max()

            # Convert to uint8 for processing
            if was_float:
                if input_range <= 1.0:
                    img_uint8 = (image * 255).astype(np.uint8)
                else:
                    img_uint8 = image.astype(np.uint8)
            else:
                img_uint8 = image

            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Brightness and saturation
            hsv[:, :, 2] *= random.uniform(1 - self.brightness, 1 + self.brightness)
            hsv[:, :, 1] *= random.uniform(1 - self.saturation, 1 + self.saturation)

            # Hue
            hsv[:, :, 0] += random.uniform(-self.hue, self.hue) * 180

            # Clip values
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # FIX: Always return in [0, 1] range for consistency
            sample['manual_step_image'] = image.astype(np.float32) / 255.0

        return sample

class ManualStyleAugment(IKEATransform):
    """
    Augmentations specific to instruction manual style
    (line drawings, diagrams, etc.)
    """
    def __init__(self, sketch_prob=0.3, edge_prob=0.2, binarize_prob=0.1):
        self.sketch_prob = sketch_prob
        self.edge_prob = edge_prob
        self.binarize_prob = binarize_prob

    def __call__(self, sample: Dict) -> Dict:
        if 'manual_step_image' not in sample:
            return sample

        image = sample['manual_step_image']

        if isinstance(image, np.ndarray):
            # FIX: Handle both [0,1] and [0,255] ranges properly
            was_float = image.dtype in [np.float32, np.float64]
            input_range = image.max()

            # Convert to uint8 for processing
            if was_float:
                if input_range <= 1.0:
                    img_uint8 = (image * 255).astype(np.uint8)
                else:
                    img_uint8 = image.astype(np.uint8)
            else:
                img_uint8 = image

            # Apply sketch effect
            if random.random() < self.sketch_prob:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                img_uint8 = cv2.addWeighted(img_uint8, 0.7, edges_colored, 0.3, 0)

            # Apply edge enhancement
            if random.random() < self.edge_prob:
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                img_uint8 = cv2.filter2D(img_uint8, -1, kernel)

            # Apply binarization (black and white)
            if random.random() < self.binarize_prob:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                img_uint8 = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

            # FIX: Always return in [0, 1] range for consistency
            sample['manual_step_image'] = img_uint8.astype(np.float32) / 255.0

        return sample

class RandomPoseNoise(IKEATransform):
    """Add noise to ground truth poses for robustness"""
    def __init__(self, rot_noise_deg=5.0, trans_noise_ratio=0.05):
        self.rot_noise_deg = rot_noise_deg
        self.trans_noise_ratio = trans_noise_ratio

    def __call__(self, sample: Dict) -> Dict:
        if 'gt_poses' not in sample:
            return sample

        for pose in sample['gt_poses']:
            # Add rotation noise
            if 'R' in pose:
                noise_euler = np.random.uniform(
                    -self.rot_noise_deg,
                    self.rot_noise_deg,
                    size=3
                )
                noise_rot = Rotation.from_euler('xyz', noise_euler, degrees=True)

                if isinstance(pose['R'], torch.Tensor):
                    R_np = pose['R'].numpy()
                    R_noisy = noise_rot.as_matrix() @ R_np
                    pose['R'] = torch.from_numpy(R_noisy).float()
                else:
                    pose['R'] = noise_rot.as_matrix() @ pose['R']

            # Add translation noise
            if 't' in pose:
                if isinstance(pose['t'], torch.Tensor):
                    noise = torch.randn_like(pose['t']) * self.trans_noise_ratio
                    pose['t'] += noise * torch.norm(pose['t'])
                else:
                    noise = np.random.randn(*pose['t'].shape) * self.trans_noise_ratio
                    pose['t'] += noise * np.linalg.norm(pose['t'])

        return sample

class Compose(IKEATransform):
    """Compose multiple transforms"""
    def __init__(self, transforms: List[IKEATransform]):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class ToTensor(IKEATransform):
    """Convert numpy arrays to PyTorch tensors"""
    def __call__(self, sample: Dict) -> Dict:
        # Image
        if 'manual_step_image' in sample and isinstance(sample['manual_step_image'], np.ndarray):
            # FIX: Ensure consistent scaling to [0, 1] range
            image = sample['manual_step_image']

            # Check if image needs scaling to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

            # HWC to CHW
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)

            sample['manual_step_image'] = torch.from_numpy(image.copy()).float()

        # Camera
        if 'camera' in sample:
            for key in ['K', 'R', 't']:
                if key in sample['camera'] and isinstance(sample['camera'][key], np.ndarray):
                    sample['camera'][key] = torch.from_numpy(sample['camera'][key]).float()

        # Poses
        if 'gt_poses' in sample:
            for pose in sample['gt_poses']:
                for key in ['R', 't']:
                    if key in pose and isinstance(pose[key], np.ndarray):
                        pose[key] = torch.from_numpy(pose[key]).float()

        # Masks
        if 'masks_2d' in sample and sample['masks_2d'] is not None:
            masks = []
            for mask in sample['masks_2d']:
                if mask is not None and isinstance(mask, np.ndarray):
                    # FIX: Ensure masks are also in [0, 1] range
                    if mask.max() > 1.0:
                        mask = mask / 255.0
                    masks.append(torch.from_numpy(mask).float())
                else:
                    masks.append(None)
            sample['masks_2d'] = masks

        return sample

def get_train_transforms(config: Dict) -> Compose:
    """Get training augmentation pipeline"""
    transforms = []

    # Basic augmentations
    if config.get('use_crop', True):
        transforms.append(RandomCrop(
            crop_size=config.get('crop_size', (480, 480)),
            preserve_center=config.get('preserve_center', True)
        ))

    if config.get('use_scale', True):
        transforms.append(RandomScale(
            scale_range=config.get('scale_range', (0.8, 1.2))
        ))

    if config.get('use_color_jitter', True):
        transforms.append(ColorJitter(
            brightness=config.get('brightness', 0.2),
            contrast=config.get('contrast', 0.2),
            saturation=config.get('saturation', 0.2),
            hue=config.get('hue', 0.1)
        ))

    # Manual-specific augmentations
    if config.get('use_manual_style', True):
        transforms.append(ManualStyleAugment(
            sketch_prob=config.get('sketch_prob', 0.3),
            edge_prob=config.get('edge_prob', 0.2),
            binarize_prob=config.get('binarize_prob', 0.1)
        ))

    # Pose noise for robustness
    if config.get('use_pose_noise', False):
        transforms.append(RandomPoseNoise(
            rot_noise_deg=config.get('rot_noise_deg', 5.0),
            trans_noise_ratio=config.get('trans_noise_ratio', 0.05)
        ))

    # Convert to tensor
    transforms.append(ToTensor())

    # Normalize
    if config.get('normalize', True):
        transforms.append(ImageNormalize())

    return Compose(transforms)

def get_val_transforms(config: Dict) -> Compose:
    """Get validation/test transformation pipeline"""
    transforms = []

    # Only essential transforms for validation
    transforms.append(ToTensor())

    if config.get('normalize', True):
        transforms.append(ImageNormalize())

    return Compose(transforms)