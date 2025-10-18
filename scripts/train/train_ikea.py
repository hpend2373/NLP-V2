"""
Training script for IKEA MEPNet
Main training loop with multi-task losses and logging
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
    warnings.warn("wandb not installed, logging will be disabled")
from typing import Dict, List, Optional, Any
import warnings
from datetime import datetime
from collections import defaultdict

# Ensure project root (and its parent) are on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_PARENT = ROOT_DIR.parent
for path in (ROOT_DIR, ROOT_PARENT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Always use fully-qualified imports for reliability
from ikea_mepnet_adapter.datasets.ikea_dataset import IKEADataset
from ikea_mepnet_adapter.datasets.transforms_ikea import (
    get_train_transforms,
    get_val_transforms,
)
from ikea_mepnet_adapter.models.perception.mepnet_adapted import create_mepnet_model, MEPNetConfig
from ikea_mepnet_adapter.models.perception.hourglass import HourglassNet
from ikea_mepnet_adapter.assets.registry import AssetsRegistry
from ikea_mepnet_adapter.eval.eval_ikea import IKEAEvaluator


class MEPNetLoss(nn.Module):
    """
    Multi-task loss for MEPNet training
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Loss weights
        self.weights = {
            'keypoint': config.get('keypoint_weight', 1.0),
            'rotation': config.get('rotation_weight', 1.0),
            'translation': config.get('translation_weight', 1.0),
            'mask': config.get('mask_weight', 0.5),
            'part_detection': config.get('part_detection_weight', 0.5),
            'depth': config.get('depth_weight', 0.3)
        }

        # Loss functions
        self.keypoint_loss = nn.MSELoss()
        self.translation_loss = nn.L1Loss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        self.part_loss = nn.BCEWithLogitsLoss()
        self.depth_loss = nn.L1Loss()

        # For continuous rotation (6D representation)
        self.use_continuous_rotation = config.get('use_continuous_rotation', True)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task losses

        Args:
            outputs: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0

        # Keypoint loss
        if 'keypoints' in outputs and 'keypoints_gt' in targets:
            kp_loss = self.keypoint_loss(outputs['keypoints'], targets['keypoints_gt'])
            losses['keypoint_loss'] = kp_loss
            total_loss += self.weights['keypoint'] * kp_loss

        # Rotation loss
        if 'rotations' in outputs and 'rotations_gt' in targets:
            if self.use_continuous_rotation:
                # Geodesic loss for rotation matrices
                rot_loss = self.rotation_geodesic_loss(
                    outputs['rotations'],
                    targets['rotations_gt']
                )
            else:
                # Cross-entropy for discrete rotations
                rot_loss = F.cross_entropy(
                    outputs['rotations'].view(-1, outputs['rotations'].size(-1)),
                    targets['rotations_gt'].view(-1)
                )
            losses['rotation_loss'] = rot_loss
            total_loss += self.weights['rotation'] * rot_loss

        # Translation loss
        if 'translations' in outputs and 'translations_gt' in targets:
            trans_loss = self.translation_loss(
                outputs['translations'],
                targets['translations_gt']
            )
            losses['translation_loss'] = trans_loss
            total_loss += self.weights['translation'] * trans_loss

        # Mask loss
        if 'masks' in outputs and 'masks_gt' in targets:
            mask_loss = self.mask_loss(outputs['masks'], targets['masks_gt'])
            losses['mask_loss'] = mask_loss
            total_loss += self.weights['mask'] * mask_loss

        # Part detection loss
        if 'part_scores' in outputs and 'part_labels_gt' in targets:
            part_loss = self.part_loss(outputs['part_scores'], targets['part_labels_gt'])
            losses['part_detection_loss'] = part_loss
            total_loss += self.weights['part_detection'] * part_loss

        # Depth loss
        if 'depth' in outputs and 'depth_gt' in targets:
            depth_loss = self.depth_loss(outputs['depth'], targets['depth_gt'])
            losses['depth_loss'] = depth_loss
            total_loss += self.weights['depth'] * depth_loss

        # Intermediate supervision
        if 'intermediate' in outputs and self.config.get('intermediate_supervision', True):
            inter_loss = self.intermediate_supervision_loss(
                outputs['intermediate'],
                targets
            )
            losses['intermediate_loss'] = inter_loss
            total_loss += 0.5 * inter_loss  # Weight intermediate loss less

        losses['total_loss'] = total_loss
        return losses

    def rotation_geodesic_loss(
        self,
        pred_rotations: torch.Tensor,
        gt_rotations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic distance between rotation matrices
        """
        # Compute relative rotation
        R_rel = torch.matmul(pred_rotations.transpose(-2, -1), gt_rotations)

        # Extract angle from trace
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace = torch.clamp(trace, -1, 3)
        angle = torch.acos((trace - 1) / 2)

        return angle.mean()

    def intermediate_supervision_loss(
        self,
        intermediate_outputs: Dict,
        targets: Dict
    ) -> torch.Tensor:
        """Compute loss for intermediate hourglass outputs"""
        inter_loss = 0
        num_stacks = len(intermediate_outputs.get('keypoints', []))

        if num_stacks == 0:
            return torch.tensor(0.0)

        # Keypoint intermediate loss
        if 'keypoints' in intermediate_outputs and 'keypoints_gt' in targets:
            for kp_inter in intermediate_outputs['keypoints']:
                inter_loss += self.keypoint_loss(kp_inter, targets['keypoints_gt'])

        # Rotation intermediate loss
        if 'rotations' in intermediate_outputs and 'rotations_gt' in targets:
            for rot_inter in intermediate_outputs['rotations']:
                if self.use_continuous_rotation:
                    inter_loss += self.rotation_geodesic_loss(rot_inter, targets['rotations_gt'])
                else:
                    inter_loss += F.cross_entropy(
                        rot_inter.view(-1, rot_inter.size(-1)),
                        targets['rotations_gt'].view(-1)
                    )

        # Translation intermediate loss
        if 'translations' in intermediate_outputs and 'translations_gt' in targets:
            for trans_inter in intermediate_outputs['translations']:
                inter_loss += self.translation_loss(trans_inter, targets['translations_gt'])

        return inter_loss / max(num_stacks, 1)


class Trainer:
    """
    Main trainer class for IKEA MEPNet
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Setup paths
        self.exp_dir = Path(config['exp_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimization()
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

    def _setup_data(self):
        """Setup datasets and dataloaders"""
        data_config = self.config['data']

        # Create transforms
        train_transform = get_train_transforms(data_config.get('augmentation', {}))
        val_transform = get_val_transforms(data_config.get('augmentation', {}))

        dataset_args = data_config.get('dataset_args', {})

        # Create training dataset
        self.train_dataset = IKEADataset(
            root_dir=data_config['root_dir'],
            split='train',
            transform=train_transform,
            **dataset_args
        )

        if len(self.train_dataset) == 0:
            warnings.warn("Training split empty; falling back to full dataset without category filtering.")
            fallback_args = dict(dataset_args)
            fallback_args.pop("furniture_categories", None)
            self.train_dataset = IKEADataset(
                root_dir=data_config['root_dir'],
                split='train',
                transform=train_transform,
                **fallback_args
            )

            # If still empty, try to create sample data
            if len(self.train_dataset) == 0:
                warnings.warn("Dataset still empty after fallback. Attempting to create sample data...")

                # Try to import and create sample data
                try:
                    # Add scripts directory to path temporarily
                    import sys
                    scripts_path = Path(__file__).parent.parent

                    # Check if create_sample_data.py exists
                    sample_data_script = scripts_path / "create_sample_data.py"
                    if sample_data_script.exists():
                        sys.path.insert(0, str(scripts_path))
                        try:
                            from create_sample_data import create_sample_ikea_dataset
                            # Create sample data
                            root = Path(data_config['root_dir'])
                            print(f"Creating sample data at: {root}")
                            create_sample_ikea_dataset(str(root))
                            # Retry loading
                            self.train_dataset = IKEADataset(
                                root_dir=data_config['root_dir'],
                                split='train',
                                transform=train_transform,
                                **fallback_args
                            )
                            print(f"After sample data creation: {len(self.train_dataset)} samples")
                        finally:
                            # Clean up path
                            if str(scripts_path) in sys.path:
                                sys.path.remove(str(scripts_path))
                    else:
                        warnings.warn(f"create_sample_data.py not found at {sample_data_script}")
                        # Create minimal dataset to avoid crash
                        warnings.warn("Creating minimal dataset to allow training to proceed...")

                except Exception as e:
                    warnings.warn(f"Failed to create sample data: {e}")
                    warnings.warn("Training may fail due to empty dataset")

        # Create validation dataset
        self.val_dataset = IKEADataset(
            root_dir=data_config['root_dir'],
            split='val',
            transform=val_transform,
            **dataset_args
        )

        if len(self.val_dataset) == 0:
            warnings.warn("Validation split empty; using training dataset as fallback for validation.")
            self.val_dataset = self.train_dataset

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('val_batch_size', self.config['batch_size']),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def _setup_model(self):
        """Setup model and loss"""
        model_config = MEPNetConfig(**self.config['model'])
        self.model = create_mepnet_model(model_config)
        self.model = self.model.to(self.device)

        # Setup loss
        self.criterion = MEPNetLoss(self.config.get('loss', {}))

        # Load pretrained weights if specified
        if self.config.get('pretrained_path'):
            self.load_pretrained(self.config['pretrained_path'])

        # Multi-GPU training
        if torch.cuda.device_count() > 1 and self.config.get('use_multi_gpu', True):
            self.model = nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs")

    def _setup_optimization(self):
        """Setup optimizer and scheduler"""
        opt_config = self.config['optimization']

        # Optimizer
        if opt_config['optimizer'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['optimizer'] == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")

        # Scheduler
        if opt_config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=opt_config.get('min_lr', 1e-6)
            )
        elif opt_config.get('scheduler') == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=opt_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None

    def _setup_logging(self):
        """Setup logging and monitoring"""
        # Wandb
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'ikea-mepnet'),
                name=self.config.get('exp_name', f"exp_{datetime.now():%Y%m%d_%H%M%S}"),
                config=self.config
            )

        # Tensorboard
        if self.config.get('use_tensorboard', True):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.exp_dir / 'tensorboard')
        else:
            self.writer = None

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch

            # Train epoch
            train_losses = self.train_epoch()

            # Validation
            if (epoch + 1) % self.config.get('val_frequency', 1) == 0:
                val_losses = self.validate()

                # Log metrics
                self.log_metrics(train_losses, val_losses, epoch)

                # Save checkpoint
                if (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                    self.save_checkpoint(
                        f'checkpoint_epoch_{epoch+1}.pth',
                        is_best=val_losses['total_loss'] < self.best_metric
                    )

                    if val_losses['total_loss'] < self.best_metric:
                        self.best_metric = val_losses['total_loss']

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_losses.get('total_loss', 0))
                else:
                    self.scheduler.step()

        print("Training completed!")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self.move_to_device(batch)

            # Prepare targets
            targets = self.prepare_targets(batch)

            # Forward pass
            outputs = self.model(
                batch['manual_step_image'],
                batch.get('part_meshes'),
                return_intermediate=True
            )

            # Compute loss
            losses = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            self.optimizer.step()

            # Update metrics
            for key, value in losses.items():
                epoch_losses[key] += value.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            self.global_step += 1

            # Log to wandb
            if self.config.get('use_wandb') and self.global_step % 10 == 0:
                wandb.log({
                    'train/' + k: v.item()
                    for k, v in losses.items()
                })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return dict(epoch_losses)

    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_losses = defaultdict(float)
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                batch = self.move_to_device(batch)

                # Prepare targets
                targets = self.prepare_targets(batch)

                # Forward pass
                outputs = self.model(
                    batch['manual_step_image'],
                    batch.get('part_meshes')
                )

                # Compute loss
                losses = self.criterion(outputs, targets)

                # Update metrics
                for key, value in losses.items():
                    val_losses[key] += value.item()

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return dict(val_losses)

    def move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        batch_device = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(self.device)
            elif isinstance(value, list):
                # Handle nested lists/dicts
                if value and isinstance(value[0], dict):
                    batch_device[key] = [
                        {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in item.items()}
                        for item in value
                    ]
                else:
                    batch_device[key] = value
            else:
                batch_device[key] = value

        return batch_device

    def prepare_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare ground truth targets from batch
        """
        targets = {}

        # Extract ground truth poses
        if 'gt_poses' in batch:
            # Collect rotations and translations
            rotations = []
            translations = []

            for batch_poses in batch['gt_poses']:
                batch_rots = []
                batch_trans = []

                for pose in batch_poses[:self.config['model'].get('max_parts', 10)]:
                    batch_rots.append(pose['R'])
                    batch_trans.append(pose['t'])

                # Pad if necessary
                while len(batch_rots) < self.config['model'].get('max_parts', 10):
                    batch_rots.append(torch.eye(3, device=self.device))
                    batch_trans.append(torch.zeros(3, device=self.device))

                rotations.append(torch.stack(batch_rots))
                translations.append(torch.stack(batch_trans))

            targets['rotations_gt'] = torch.stack(rotations)
            targets['translations_gt'] = torch.stack(translations)

        # Extract masks
        if 'masks_2d' in batch and batch['masks_2d'] is not None:
            # Handle variable number of masks
            # This is simplified - actual implementation would need proper padding
            pass

        # Generate keypoint targets (simplified - would need actual keypoint extraction)
        # targets['keypoints_gt'] = ...

        return targets

    def log_metrics(self, train_losses: Dict, val_losses: Dict, epoch: int):
        """Log metrics to tensorboard/wandb"""
        # Tensorboard
        if self.writer:
            for key, value in train_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            # Log learning rate
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

        # Wandb
        if self.config.get('use_wandb'):
            wandb.log({
                'epoch': epoch,
                **{f'train_epoch/{k}': v for k, v in train_losses.items()},
                **{f'val_epoch/{k}': v for k, v in val_losses.items()},
                'lr': self.optimizer.param_groups[0]['lr']
            })

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        print(f"Checkpoint loaded from: {checkpoint_path}")

    def load_pretrained(self, pretrained_path: str):
        """Load pretrained weights (e.g., from LEGO training)"""
        pretrained = torch.load(pretrained_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in pretrained:
            pretrained_dict = pretrained['model_state_dict']
        else:
            pretrained_dict = pretrained

        # Filter out mismatched keys
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # Load weights
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")


def main():
    parser = argparse.ArgumentParser(description='Train IKEA MEPNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train or evaluate
    if args.eval_only:
        val_losses = trainer.validate()
        print("Validation results:")
        for key, value in val_losses.items():
            print(f"{key}: {value:.4f}")
    else:
        trainer.train()


if __name__ == '__main__':
    main()
