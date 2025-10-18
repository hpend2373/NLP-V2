#!/usr/bin/env python3
"""Debug image sizes in the dataset"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.ikea_dataset import IKEADataset
from datasets.transforms_ikea import get_train_transforms

# Load dataset without transforms
dataset_no_transform = IKEADataset(
    root_dir="./IKEA-Manuals-at-Work-Sample",
    split='train',
    transform=None,
    image_size=(512, 512)
)

print(f"Dataset size: {len(dataset_no_transform)}")

# Check each sample's image size
for i in range(min(3, len(dataset_no_transform))):
    sample = dataset_no_transform[i]
    img = sample['manual_step_image']
    print(f"Sample {i} - Image shape before transform: {img.shape}")

# Now with transforms
config = {
    'use_crop': False,
    'use_scale': False,
    'use_color_jitter': False,
    'use_manual_style': False,
    'use_pose_noise': False,
    'normalize': True
}

transform = get_train_transforms(config)

dataset_with_transform = IKEADataset(
    root_dir="./IKEA-Manuals-at-Work-Sample",
    split='train',
    transform=transform,
    image_size=(512, 512)
)

print("\nWith transforms:")
for i in range(min(3, len(dataset_with_transform))):
    sample = dataset_with_transform[i]
    img = sample['manual_step_image']
    print(f"Sample {i} - Image shape after transform: {img.shape}")