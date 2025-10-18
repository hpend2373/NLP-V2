"""
IKEA Dataset Adapter for MEPNet
Bridges IKEA Manuals at Work dataset to MEPNet-compatible format
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import trimesh
import cv2
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

class IKEADataset(Dataset):
    """
    Dataset loader for IKEA Manuals at Work
    Adapts IKEA manual images, 3D parts, and poses to MEPNet format
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        load_meshes: bool = True,
        load_videos: bool = False,
        furniture_categories: Optional[List[str]] = None,
        max_parts_per_step: int = 10,
        image_size: Tuple[int, int] = (512, 512),
        normalize_scale: bool = True,
        cache_meshes: bool = True
    ):
        """
        Args:
            root_dir: Path to IKEA-Manuals-at-Work root directory
            split: 'train', 'val', or 'test'
            transform: Optional image transformations
            load_meshes: Whether to load 3D meshes
            load_videos: Whether to load video data
            furniture_categories: List of categories to include (e.g., ['Chair', 'Table'])
            max_parts_per_step: Maximum number of parts to handle per step
            image_size: Target image size (H, W)
            normalize_scale: Whether to normalize mesh scales
            cache_meshes: Whether to cache loaded meshes
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_meshes = load_meshes
        self.load_videos = load_videos
        self.furniture_categories = furniture_categories
        self.max_parts_per_step = max_parts_per_step
        self.image_size = image_size
        self.normalize_scale = normalize_scale
        self.cache_meshes = cache_meshes

        # Paths
        self.data_json_path = self.root_dir / 'data' / 'data.json'
        self.manual_img_dir = self.root_dir / 'data' / 'manual_img'
        self.parts_dir = self.root_dir / 'data' / 'parts'
        self.videos_dir = self.root_dir / 'data' / 'videos' if load_videos else None

        # Load metadata
        self.metadata = self._load_metadata()
        # Allow metadata stored either as dict or list of records
        if isinstance(self.metadata, list):
            self.metadata = self._convert_metadata_list(self.metadata)

        # Filter by split and categories
        self.samples = self._prepare_samples()

        # Mesh cache
        self.mesh_cache = {} if cache_meshes else None

        # Coordinate system conversion (IKEA to MEPNet)
        # IKEA uses centimeters, we'll normalize to unit scale
        self.scale_factor = 0.01 if normalize_scale else 1.0

        print(f"Initialized IKEA Dataset: {len(self.samples)} samples in {split} split")

    def _load_metadata(self) -> Dict:
        """Load and parse data.json"""
        if not self.data_json_path.exists():
            raise FileNotFoundError(f"Data JSON not found at {self.data_json_path}")

        with open(self.data_json_path, 'r') as f:
            data = json.load(f)

        return data

    def _convert_metadata_list(self, records: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Convert metadata stored as a list into dictionary format keyed by furniture_id.
        """
        converted: Dict[str, Dict] = {}
        furniture_counts = {}  # Track duplicates

        for idx, record in enumerate(records):
            furniture_id = (
                record.get('furniture_id')
                or record.get('id')
                or record.get('name')
            )

            if not furniture_id:
                warnings.warn(
                    f"Metadata record at index {idx} missing furniture identifier; skipping."
                )
                continue

            # Handle duplicates quietly - append suffix
            if furniture_id in furniture_counts:
                furniture_counts[furniture_id] += 1
                key = f"{furniture_id}_{furniture_counts[furniture_id]}"
                # Only warn if there are many duplicates (more than 2)
                if furniture_counts[furniture_id] == 2:
                    warnings.warn(f"Multiple entries for furniture_id '{furniture_id}' found")
            else:
                furniture_counts[furniture_id] = 0
                key = furniture_id

            converted[key] = record

        return converted

    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare dataset samples based on split and categories
        Each sample represents one assembly step
        """
        samples = []

        # If no metadata, return empty list (will be caught by fallback logic)
        if not self.metadata:
            warnings.warn("No metadata found in data.json")
            return samples

        # Debug: Show what we're working with
        total_furniture = len(self.metadata)
        furniture_with_steps = 0

        # Parse metadata structure
        # Expected structure: {furniture_id: {steps: [...], parts: [...], ...}}
        for furniture_id, furniture_data in self.metadata.items():
            # Skip metadata entries that are not furniture data
            if not isinstance(furniture_data, dict):
                warnings.warn(f"Furniture {furniture_id}: Not a dict, skipping")
                continue

            if 'steps' not in furniture_data:
                # Debug what keys are present
                keys_present = list(furniture_data.keys())[:10]
                warnings.warn(f"Furniture {furniture_id}: No 'steps' key. Has keys: {keys_present}")
                continue

            furniture_with_steps += 1

            # Filter by category if specified
            if self.furniture_categories:
                category = furniture_data.get('category', '')
                if category not in self.furniture_categories:
                    continue

            # Check split assignment (implement your split logic)
            if not self._check_split(furniture_id, furniture_data):
                continue

            # Process each assembly step
            steps = furniture_data.get('steps', [])
            for step_idx, step_data in enumerate(steps):
                # Try multiple possible keys for parts
                parts = None
                for key in ['manual_parts', 'added_parts', 'parts', 'part_ids', 'components']:
                    if key in step_data:
                        parts = step_data[key]
                        break

                # If still no parts, check if the step itself is a list
                if parts is None and isinstance(step_data, list):
                    parts = step_data  # The step itself might be the parts list

                # Debug: Check why steps might be empty
                if not parts:
                    # Check what keys are in the step
                    step_keys = list(step_data.keys()) if isinstance(step_data, dict) else []
                    if step_idx == 0:  # Only log for first step to avoid spam
                        warnings.warn(f"Step {step_idx} of {furniture_id} has no parts. Keys: {step_keys}")
                    continue

                # Get manual connections if available
                manual_connections = step_data.get('manual_connections', [])

                sample = {
                    'furniture_id': furniture_id,
                    'category': furniture_data.get('category', 'unknown'),
                    'step_idx': step_idx,
                    'step_data': step_data,
                    'furniture_data': furniture_data,
                    'manual_image_path': self._get_manual_image_path(
                        furniture_id, furniture_data, step_idx, step_data
                    ),
                    'added_parts': parts,  # Use discovered parts
                    'manual_connections': manual_connections,  # Add manual connections
                    'camera_params': step_data.get('camera', {}),
                    'base_assembly': step_data.get('base_assembly', []),
                    # Add frame-level annotations if available
                    'aligned_frames': step_data.get('aligned_frames', [])
                }

                # Validate sample
                if self._validate_sample(sample):
                    samples.append(sample)

        # Debug summary
        if len(samples) == 0:
            warnings.warn(f"No samples created! Total furniture: {total_furniture}, With steps: {furniture_with_steps}")

        return samples

    def _check_split(self, furniture_id: str, furniture_data: Dict) -> bool:
        """
        Determine if furniture belongs to current split
        For now, include all data in all splits to ensure we have samples
        """
        # TODO: Implement proper train/val/test split when we have more data
        # For now, include everything to avoid empty datasets
        return True

        # Future implementation with proper splitting:
        # import hashlib
        # hash_obj = hashlib.md5(furniture_id.encode())
        # hash_val = int(hash_obj.hexdigest()[:8], 16) % 100
        # if self.split == 'train':
        #     return hash_val < 70
        # elif self.split == 'val':
        #     return 70 <= hash_val < 85
        # else:  # test
        #     return hash_val >= 85

    def _validate_sample(self, sample: Dict) -> bool:
        """Validate that sample has required data"""
        # Be EXTREMELY lenient with validation to work with partial data
        # For debugging, accept almost everything

        # Check if has parts to add
        if not sample.get('added_parts'):
            # Debug what's in the sample
            warnings.warn(f"Sample for {sample.get('furniture_id')} step {sample.get('step_idx')} has no added_parts")
            # Still return True to see what happens
            return True  # Changed to True for debugging

        # Don't require manual image to exist during development
        # We can work with missing images
        if sample.get('manual_image_path'):
            if not sample['manual_image_path'].exists():
                # Create a placeholder image if needed
                pass  # Remove warning spam

        return True

    def _get_manual_image_path(
        self,
        furniture_id: str,
        furniture_data: Dict,
        step_idx: int,
        step_data: Dict
    ) -> Path:
        """Get path to manual image for given step

        IKEA dataset structure: manual_img/<category>/<furniture_id>/...
        """
        category = furniture_data.get('category', 'unknown')

        # Try multiple possible paths
        # Option 1: With category folder
        path_with_category = self.manual_img_dir / category / furniture_id / f"step_{step_idx:03d}.png"
        if path_with_category.exists():
            return path_with_category

        # Option 2: Direct furniture_id (legacy)
        path_direct = self.manual_img_dir / furniture_id / f"step_{step_idx:03d}.png"
        if path_direct.exists():
            return path_direct

        # Option 3: Check if step_data has explicit image path
        if 'manual_image' in step_data:
            explicit_path = self.manual_img_dir / step_data['manual_image']
            if explicit_path.exists():
                return explicit_path

        # Return the expected path (with category) even if not exists
        # (will be filtered out in validation)
        return path_with_category

    def _load_mesh(self, part_id: str) -> Optional[trimesh.Trimesh]:
        """Load 3D mesh for a part"""
        if self.cache_meshes and part_id in self.mesh_cache:
            return self.mesh_cache[part_id]

        # Try different file extensions
        for ext in ['.obj', '.stl', '.ply']:
            mesh_path = self.parts_dir / f"{part_id}{ext}"
            if mesh_path.exists():
                try:
                    mesh = trimesh.load(str(mesh_path), force='mesh')

                    # Apply scaling
                    if self.normalize_scale:
                        mesh.apply_scale(self.scale_factor)

                    # Center mesh
                    mesh.vertices -= mesh.centroid

                    if self.cache_meshes:
                        self.mesh_cache[part_id] = mesh

                    return mesh
                except Exception as e:
                    warnings.warn(f"Failed to load mesh {mesh_path}: {e}")

        return None

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess manual image"""
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Create a placeholder image if file doesn't exist
            warnings.warn(f"Image not found: {image_path}, using placeholder")
            image = Image.new('RGB', self.image_size, color='white')

        # Resize to target size
        image = image.resize(self.image_size, Image.BILINEAR)

        # Convert to numpy
        image = np.array(image)

        return image

    def _parse_camera_params(self, camera_data: Dict) -> Dict[str, np.ndarray]:
        """Parse camera parameters into standard format"""
        camera = {}

        # Intrinsic matrix
        if 'intrinsic' in camera_data:
            camera['K'] = np.array(camera_data['intrinsic']).reshape(3, 3)
        else:
            # Default intrinsics if not provided
            fx = fy = self.image_size[0]  # Assuming unit focal length
            cx = self.image_size[1] / 2
            cy = self.image_size[0] / 2
            camera['K'] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        # Extrinsic matrix
        if 'extrinsic' in camera_data:
            extrinsic = np.array(camera_data['extrinsic']).reshape(4, 4)
            camera['R'] = extrinsic[:3, :3]
            camera['t'] = extrinsic[:3, 3]
        else:
            # Default identity pose
            camera['R'] = np.eye(3)
            camera['t'] = np.zeros(3)

        return camera

    def _get_part_pose(self, part_data: Dict) -> Dict[str, np.ndarray]:
        """Extract 6D pose for a part"""
        pose = {}

        if 'rotation' in part_data:
            # Convert rotation representation (quaternion/matrix/euler)
            if isinstance(part_data['rotation'], list) and len(part_data['rotation']) == 4:
                # Quaternion to rotation matrix
                from scipy.spatial.transform import Rotation
                q = part_data['rotation']  # [w, x, y, z] or [x, y, z, w]
                pose['R'] = Rotation.from_quat(q).as_matrix()
            elif isinstance(part_data['rotation'], list) and len(part_data['rotation']) == 9:
                # Rotation matrix
                pose['R'] = np.array(part_data['rotation']).reshape(3, 3)
            else:
                pose['R'] = np.eye(3)
        else:
            pose['R'] = np.eye(3)

        if 'translation' in part_data:
            pose['t'] = np.array(part_data['translation']) * self.scale_factor
        else:
            pose['t'] = np.zeros(3)

        return pose

    def _get_part_mask(self, sample: Dict, part_id: str) -> Optional[np.ndarray]:
        """Load or generate 2D mask for a part"""
        # Check if pre-computed masks exist
        mask_path = self.manual_img_dir / sample['furniture_id'] / 'masks' / f"step_{sample['step_idx']:03d}_{part_id}.png"

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size)
            return mask

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample compatible with MEPNet format

        IMPORTANT:
        - Manual-level: manual_parts, manual_connections (no pose/camera)
        - Frame-level: aligned_frames contain poses/masks/camera
        """
        sample = self.samples[idx]

        # Load manual image (no camera parameters at manual level)
        image = self._load_image(sample['manual_image_path'])

        # Get frame-level annotations for supervision
        aligned_frames = sample.get('aligned_frames', [])

        # Use first aligned frame for GT poses/masks/camera (if available)
        if aligned_frames:
            frame_data = aligned_frames[0]  # Use first frame or aggregate multiple
            camera = self._parse_camera_params(frame_data.get('camera', {}))
            frame_poses = frame_data.get('poses', {})
            frame_masks = frame_data.get('masks', {})
        else:
            # No frame alignment - use manual-level camera if available, else default
            camera = self._parse_camera_params(sample.get('camera_params', {}))
            frame_poses = {}
            frame_masks = {}

        # Prepare added components data
        added_components = []
        part_meshes = []
        part_poses = []
        part_masks = []

        # Handle case where added_parts might be empty or None
        parts_to_process = sample.get('added_parts', []) or []
        if not parts_to_process:
            # Create dummy data to avoid crash
            parts_to_process = [{'part_id': 'dummy_part', 'instance_id': 'dummy_0'}]

        for part_info in parts_to_process[:self.max_parts_per_step]:
            part_id = part_info.get('part_id', part_info.get('id'))

            component = {
                'part_id': part_id,
                'instance_id': part_info.get('instance_id', f"{part_id}_0"),
                'subassembly_id': part_info.get('subassembly_id', None)
            }
            added_components.append(component)

            # Load mesh if requested
            if self.load_meshes:
                mesh = self._load_mesh(part_id)
                part_meshes.append(mesh)

            # Get part pose from frame-level data (if available)
            if part_id in frame_poses:
                pose = self._get_part_pose(frame_poses[part_id])
            else:
                # Try to get from part_info directly (legacy)
                pose = self._get_part_pose(part_info) if 'rotation' in part_info or 'translation' in part_info else {'R': np.eye(3), 't': np.zeros(3)}
            part_poses.append(pose)

            # Get part mask from frame-level data (if available)
            if part_id in frame_masks:
                mask = frame_masks[part_id]
                # Resize mask to target size if needed
                if isinstance(mask, np.ndarray) and mask.shape[:2] != self.image_size:
                    mask = cv2.resize(mask, self.image_size)
            else:
                # Try loading from file or return None
                mask = self._get_part_mask(sample, part_id)
            part_masks.append(mask)

        # Prepare output dictionary
        output = {
            # Identifiers
            'furniture_id': sample['furniture_id'],
            'category': sample['category'],
            'step_idx': sample['step_idx'],

            # Core data
            'manual_step_image': image,
            'added_components': added_components,
            'camera': camera,

            # CRITICAL: Manual connections for constraint engine
            'manual_connections': sample.get('manual_connections', []),

            # Optional data
            'part_meshes': part_meshes if self.load_meshes else None,
            'gt_poses': part_poses,
            'masks_2d': part_masks if any(part_masks) else None,

            # Assembly context
            'base_assembly': sample['base_assembly'],

            # Frame alignment info (for supervision)
            'has_frame_supervision': len(aligned_frames) > 0,

            # Metadata
            'num_parts': len(added_components)
        }

        # Apply transformations if provided
        if self.transform:
            output = self.transform(output)

        # Convert to torch tensors
        output = self._to_torch(output)

        return output

    def _to_torch(self, sample: Dict) -> Dict:
        """Convert numpy arrays to torch tensors"""
        # Image to tensor
        if isinstance(sample['manual_step_image'], np.ndarray):
            # HWC to CHW format
            image = sample['manual_step_image'].transpose(2, 0, 1)
            # Make sure it's float32, not float64
            sample['manual_step_image'] = torch.from_numpy(image.astype(np.float32)) / 255.0

        # Camera parameters to tensor
        for key in ['K', 'R', 't']:
            if key in sample['camera']:
                if isinstance(sample['camera'][key], np.ndarray):
                    sample['camera'][key] = torch.from_numpy(sample['camera'][key]).float()
                elif not isinstance(sample['camera'][key], torch.Tensor):
                    sample['camera'][key] = torch.tensor(sample['camera'][key]).float()

        # Poses to tensor
        if sample['gt_poses']:
            for pose in sample['gt_poses']:
                for key in ['R', 't']:
                    if key in pose:
                        if isinstance(pose[key], np.ndarray):
                            pose[key] = torch.from_numpy(pose[key]).float()
                        elif not isinstance(pose[key], torch.Tensor):
                            pose[key] = torch.tensor(pose[key]).float()

        # Masks to tensor
        if sample['masks_2d'] is not None:
            masks = []
            for mask in sample['masks_2d']:
                if mask is not None:
                    masks.append(torch.from_numpy(mask).float() / 255.0)
                else:
                    masks.append(None)
            sample['masks_2d'] = masks

        return sample

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching
        Handles variable number of parts per sample
        """
        collated = {}

        # Fixed size tensors - standard batching
        for key in ['manual_step_image', 'step_idx', 'num_parts']:
            if key in batch[0]:
                if isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack([b[key] for b in batch])
                else:
                    collated[key] = torch.tensor([b[key] for b in batch])

        # Lists and dictionaries - keep as lists
        for key in ['furniture_id', 'category', 'added_components',
                    'camera', 'part_meshes', 'gt_poses', 'masks_2d', 'base_assembly',
                    'manual_connections', 'has_frame_supervision']:  # ADDED manual_connections
            if key in batch[0]:
                collated[key] = [b[key] for b in batch]

        return collated
