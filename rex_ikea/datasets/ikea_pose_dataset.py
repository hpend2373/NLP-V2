"""
Stage-B Pose Refinement Dataset Loader

Loads IKEA part crops with 6DoF GT poses, optionally using Stage-A predictions
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

from .ikea_gt_loader import create_gt_loader


class IKEAPoseDataset(Dataset):
    """
    IKEA 6DoF Pose Dataset for Stage-B training
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        crop_size: int = 224,
        full_image_size: int = 512,
        use_stage_a_predictions: bool = False,
        stage_a_checkpoint: Optional[str] = None,
        augmentation: Optional[Dict] = None,
        normalization: Optional[Dict] = None
    ):
        """
        Args:
            data_root: Root directory (IKEA-Manuals-at-Work)
            split: "train" or "val"
            crop_size: Size for part crops
            full_image_size: Size for full assembly image
            use_stage_a_predictions: Use Stage-A predicted boxes instead of GT
            stage_a_checkpoint: Path to Stage-A model checkpoint
            augmentation: Data augmentation config
            normalization: Image normalization config
        """
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size
        self.full_image_size = full_image_size
        self.use_stage_a_predictions = use_stage_a_predictions
        self.stage_a_checkpoint = stage_a_checkpoint

        # Load 6DoF GT
        print(f"Loading 6DoF GT from {data_root}...")
        self.gt_loader = create_gt_loader(str(data_root))
        print(f"✓ Loaded {len(self.gt_loader.pose_index)} poses")

        # Load dataset metadata
        self.samples = self._load_samples()

        # Augmentation
        self.augmentation = augmentation or {}
        self.setup_transforms(normalization)

        # Load Stage-A model if needed
        if use_stage_a_predictions and stage_a_checkpoint:
            self.stage_a_model = self._load_stage_a_model(stage_a_checkpoint)
        else:
            self.stage_a_model = None

    def _load_samples(self) -> List[Dict]:
        """
        Load dataset samples from grounding_train.jsonl

        Uses pre-processed grounding data with correct image/mask paths
        """
        samples = []

        # Determine which file to load
        if self.use_stage_a_predictions:
            # Load generated Stage A predictions
            # Prioritize fine-tuned data
            data_file = self.data_root / "data" / "stage_a" / "stage_a_rex_finetuned.jsonl"
            if not data_file.exists():
                data_file = self.data_root / "data" / "stage_a" / "stage_a_rex_generated.jsonl"
            
            if not data_file.exists():
                # Fallback location
                data_file = Path("/Users/minyeop/NLP/NLP-V2/data/stage_a/stage_a_rex_finetuned.jsonl")
                if not data_file.exists():
                    data_file = Path("/Users/minyeop/NLP/NLP-V2/data/stage_a/stage_a_rex_generated.jsonl")
            
            if not data_file.exists():
                print(f"Error: Stage A predictions file not found at {data_file}")
                return samples
                
            print(f"Loading Stage A predictions from {data_file}...")
            with open(data_file, 'r') as f:
                raw_samples = [json.loads(line) for line in f]
                
            print(f"Found {len(raw_samples)} Stage A predictions")
            
            for r_sample in raw_samples:
                furniture_id = r_sample['furniture_id']
                step_idx = r_sample['step_idx']
                part_id = r_sample['part_id']
                
                # Rex-Omni prediction format:
                # "rex_prediction": {"object_1": [{"type": "box", "coords": [x1, y1, x2, y2]}]}
                # We need to extract the best box.
                # For now, take the first box if available.
                
                preds = r_sample.get('rex_prediction', {})
                # Assuming single object label like "object_1" or similar
                # The generation script output format: {"object_1": [...]}
                
                best_box = None
                # Iterate over all keys (e.g. object_1)
                for label, boxes in preds.items():
                    if boxes:
                        # Take the first box
                        # box is {'type': 'box', 'coords': [x1, y1, x2, y2]}
                        best_box = boxes[0]['coords']
                        break
                
                if best_box is None:
                    continue
                    
                # Load 6DoF GT
                gt_pose = self.gt_loader.get_pose(
                    furniture_id=furniture_id,
                    step_idx=step_idx,
                    part_id=part_id,
                    frame_id=0
                )
                
                if gt_pose is None:
                    continue
                    
                gt_rotation, gt_translation = gt_pose
                
                # Determine category (not explicitly in record, but can be inferred or we can add it to generation script)
                # Generation script *did* have category in loop but didn't save it to record.
                # We can try to infer from path: data/manual_img/{Category}/...
                image_path_str = r_sample['image_path']
                parts = image_path_str.split('/')
                # Expected: data/manual_img/Category/FurnitureID/...
                if 'manual_img' in parts:
                    idx = parts.index('manual_img')
                    if idx + 1 < len(parts):
                        category = parts[idx+1]
                    else:
                        category = "Unknown"
                else:
                    category = "Unknown"

                sample = {
                    'image_path': str(self.data_root / image_path_str),
                    'bbox': best_box, # [x1, y1, x2, y2]
                    'furniture_id': furniture_id,
                    'category': category,
                    'step_idx': step_idx,
                    'part_id': part_id,
                    'gt_rotation': gt_rotation,
                    'gt_translation': gt_translation
                }
                samples.append(sample)

        else:
            # Original logic for Grounding/GT Masks
            # Find grounding JSONL file - try multiple locations
            grounding_jsonl = None
            possible_paths = [
                # Inside IKEA dataset (if preprocessing was done there)
                self.data_root / "data" / "grounding" / "grounding_train.jsonl",
                # Project root (NLP-V2/data/grounding/)
                self.data_root.parent / "data" / "grounding" / "grounding_train.jsonl",
                # Relative to this script
                Path(__file__).parent.parent.parent / "data" / "grounding" / "grounding_train.jsonl",
                # Absolute fallback
                Path("/Users/minyeop/NLP/NLP-V2/data/grounding/grounding_train.jsonl")
            ]
    
            for path in possible_paths:
                if path.exists():
                    grounding_jsonl = path
                    break
    
            if grounding_jsonl is None:
                print(f"Warning: grounding_train.jsonl not found in any of these locations:")
                for p in possible_paths:
                    print(f"  - {p}")
                return samples
    
            print(f"Loading samples from {grounding_jsonl}...")
    
            # Load grounding data
            with open(grounding_jsonl, 'r') as f:
                grounding_samples = [json.loads(line) for line in f]
    
            print(f"Found {len(grounding_samples)} grounding samples")
    
            # Convert to pose dataset samples
            for g_sample in grounding_samples:
                furniture_id = g_sample['furniture_id']
                step_idx = g_sample['step_idx']
                category = g_sample['category']
    
                # Extract part IDs from the sample
                parts = g_sample.get('parts', [])
    
                if not parts:
                    continue
    
                # Get image path (relative to data_root)
                image_path = self.data_root / g_sample['image_path']
    
                # For each part in this step
                for part_id in parts:
                    # Construct mask path based on IKEA structure
                    # Mask format: {category}/{furniture_id}/step_{step_idx}_mask.png
                    # But we need per-part masks, which should be in grounding data
    
                    # Try to find mask path
                    # Standard format: Table/applaro/step_0_mask.png
                    mask_path = self.data_root / "data" / "manual_masks" / category / furniture_id / f"step_{step_idx}_mask.png"
    
                    if not mask_path.exists():
                        # Try lowercase
                        mask_path = self.data_root / "data" / "manual_masks" / category.lower() / furniture_id.lower() / f"step_{step_idx}_mask.png"
    
                    if not mask_path.exists():
                        continue
    
                    # Load 6DoF GT from data.json
                    gt_pose = self.gt_loader.get_pose(
                        furniture_id=furniture_id,
                        step_idx=step_idx,
                        part_id=part_id,
                        frame_id=0  # Default frame
                    )
    
                    if gt_pose is None:
                        # GT not available, skip this sample
                        continue
    
                    gt_rotation, gt_translation = gt_pose
    
                    sample = {
                        'image_path': str(image_path),
                        'mask_path': str(mask_path),
                        'furniture_id': furniture_id,
                        'category': category,
                        'step_idx': step_idx,
                        'part_id': part_id,
                        'gt_rotation': gt_rotation,
                        'gt_translation': gt_translation
                    }
    
                    samples.append(sample)

        print(f"Loaded {len(samples)} pose samples with GT")
        return samples

    def _load_stage_a_model(self, checkpoint_path: str):
        """Load Stage-A model for predictions"""
        # Placeholder - actual implementation depends on Rex-Omni checkpoint format
        return None

    def setup_transforms(self, normalization: Optional[Dict] = None):
        """Setup image transforms"""
        if normalization is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = normalization.get('mean', [0.485, 0.456, 0.406])
            std = normalization.get('std', [0.229, 0.224, 0.225])

        # Crop transform
        crop_transforms = [
            transforms.Resize((self.crop_size, self.crop_size)),
        ]

        if self.split == "train" and self.augmentation:
            if self.augmentation.get('color_jitter', False):
                crop_transforms.append(
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    )
                )
            if self.augmentation.get('random_rotation', 0) > 0:
                crop_transforms.append(
                    transforms.RandomRotation(self.augmentation['random_rotation'])
                )

        crop_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.crop_transform = transforms.Compose(crop_transforms)

        # Full image transform
        full_transforms = [
            transforms.Resize((self.full_image_size, self.full_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        self.full_transform = transforms.Compose(full_transforms)

    def extract_crop_from_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        padding: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract crop from image using mask

        Args:
            image: (H, W, 3) RGB image
            mask: (H, W) binary mask
            padding: Relative padding around bbox

        Returns:
            crop: (H', W', 3) cropped image
            bbox_info: Dict with bbox coordinates
        """
        # Find bounding box from mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # Return full image if no contour
            return image, {'x1': 0, 'y1': 0, 'x2': image.shape[1], 'y2': image.shape[0]}

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)

        # Crop
        crop = image[y1:y2, x1:x2]

        bbox_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        return crop, bbox_info

    def extract_crop_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[float],
        padding: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract crop from image using bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1_p = max(0, int(x1 - pad_w))
        y1_p = max(0, int(y1 - pad_h))
        x2_p = min(image.shape[1], int(x2 + pad_w))
        y2_p = min(image.shape[0], int(y2 + pad_h))
        
        # Crop
        crop = image[y1_p:y2_p, x1_p:x2_p]
        
        # If crop is empty (e.g. bad bbox), return full image
        if crop.size == 0:
            return image, {'x1': 0, 'y1': 0, 'x2': image.shape[1], 'y2': image.shape[0]}
            
        bbox_info = {'x1': x1_p, 'y1': y1_p, 'x2': x2_p, 'y2': y2_p}
        return crop, bbox_info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Handle missing image gracefully?
            # Create black image
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract crop
        if self.use_stage_a_predictions and 'bbox' in sample:
            # Use pre-loaded Stage-A prediction bbox
            crop, bbox_info = self.extract_crop_from_bbox(image, sample['bbox'])
            confidence = 1.0 # We could use actual confidence if we saved it
        elif self.use_stage_a_predictions and self.stage_a_model is not None:
            # Use Stage-A predictions (online inference)
            # TODO: Implement Stage-A inference
            # For now fallback to GT if mask exists
            if 'mask_path' in sample:
                mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                crop, bbox_info = self.extract_crop_from_mask(image, mask)
            else:
                crop = image
            confidence = 1.0
        else:
            # Use GT mask
            if 'mask_path' in sample:
                mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                crop, bbox_info = self.extract_crop_from_mask(image, mask)
            else:
                # Should not happen if _load_samples filters correctly
                crop = image
            confidence = 1.0  # GT has full confidence

        # Convert to PIL for transforms
        crop_pil = Image.fromarray(crop)
        full_pil = Image.fromarray(image)

        # Apply transforms
        crop_tensor = self.crop_transform(crop_pil)
        full_tensor = self.full_transform(full_pil)

        # Get GT pose
        gt_rotation = torch.from_numpy(sample['gt_rotation']).float()
        gt_translation = torch.from_numpy(sample['gt_translation']).float()

        # Part ID to integer (simple hash for now)
        part_id_int = hash(sample['part_id']) % 100

        # Text embedding placeholder (from BERT/RoBERTa)
        text_embed = torch.zeros(768)  # Placeholder

        return {
            'crop_image': crop_tensor,
            'full_image': full_tensor,
            'part_id': part_id_int,
            'confidence': confidence,
            'text_embed': text_embed,
            'gt_rotation': gt_rotation,
            'gt_translation': gt_translation,
            'furniture_id': sample['furniture_id'],
            'category': sample['category'],
            'step_idx': sample['step_idx'],
            'part_id_str': sample['part_id']
        }


def pose_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for pose dataset
    """
    crop_images = torch.stack([item['crop_image'] for item in batch])
    full_images = torch.stack([item['full_image'] for item in batch])
    part_ids = torch.tensor([item['part_id'] for item in batch], dtype=torch.long)
    confidences = torch.tensor([item['confidence'] for item in batch], dtype=torch.float32)
    text_embeds = torch.stack([item['text_embed'] for item in batch])
    gt_rotations = torch.stack([item['gt_rotation'] for item in batch])
    gt_translations = torch.stack([item['gt_translation'] for item in batch])

    return {
        'crop_image': crop_images,
        'full_image': full_images,
        'part_id': part_ids,
        'confidence': confidences,
        'text_embed': text_embeds,
        'gt_rotation': gt_rotations,
        'gt_translation': gt_translations,
        'metadata': {
            'furniture_ids': [item['furniture_id'] for item in batch],
            'categories': [item['category'] for item in batch],
            'step_indices': [item['step_idx'] for item in batch],
            'part_id_strs': [item['part_id_str'] for item in batch]
        }
    }


def create_pose_dataloader(
    config: Dict,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Factory function to create pose dataloader

    Args:
        config: Dataset configuration
        split: "train" or "val"
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader instance
    """
    dataset = IKEAPoseDataset(
        data_root=config['data_root'],
        split=split,
        crop_size=config['visual_encoder']['crop_size'],
        full_image_size=config['visual_encoder']['full_image_size'],
        use_stage_a_predictions=config.get('use_stage_a_predictions', False),
        stage_a_checkpoint=config.get('stage_a_checkpoint'),
        augmentation=config.get('augmentation'),
        normalization=config.get('normalization')
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=True
    )

    return dataloader
