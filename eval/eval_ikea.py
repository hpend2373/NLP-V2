"""
Evaluation pipeline for IKEA MEPNet
Implements metrics for pose accuracy, plan quality, and assembly feasibility
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from collections import defaultdict
import warnings
import pickle

# Metrics
try:
    from chamferdist import ChamferDistance
except ImportError:
    ChamferDistance = None
    warnings.warn("ChamferDistance not available, chamfer distance metrics will be disabled")
import trimesh


class IKEAEvaluator:
    """
    Main evaluation class for IKEA assembly predictions
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pose_solver: Any,
        plan_assembler: Any,
        constraint_engine: Any,
        device: str = 'cuda',
        output_dir: str = './eval_results'
    ):
        """
        Args:
            model: MEPNet perception model
            pose_solver: Pose solver instance
            plan_assembler: Plan assembler instance
            constraint_engine: Constraint engine instance
            device: Computing device
            output_dir: Directory for evaluation outputs
        """
        self.model = model.to(device)
        self.pose_solver = pose_solver
        self.plan_assembler = plan_assembler
        self.constraint_engine = constraint_engine
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.chamfer_dist = ChamferDistance() if (ChamferDistance is not None and torch.cuda.is_available()) else None

        # Results storage
        self.results = defaultdict(list)

    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: List[str] = None,
        save_predictions: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run complete evaluation

        Args:
            dataloader: Data loader with IKEA samples
            metrics: List of metrics to compute
            save_predictions: Whether to save predictions
            verbose: Print progress

        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = [
                'pose_accuracy',
                'chamfer_distance',
                'mask_iou',
                'plan_accuracy',
                'assembly_feasibility'
            ]

        self.model.eval()
        all_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, disable=not verbose, desc="Evaluating")):
                # Move to device
                batch = self._move_to_device(batch)

                # Get predictions
                predictions = self._get_predictions(batch)

                # Evaluate batch
                batch_results = self._evaluate_batch(batch, predictions, metrics)
                all_results.append(batch_results)

                # Save predictions if requested
                if save_predictions:
                    self._save_batch_predictions(batch_idx, batch, predictions, batch_results)

        # Aggregate metrics
        final_metrics = self._aggregate_metrics(all_results, metrics)

        # Save summary
        self._save_evaluation_summary(final_metrics)

        return final_metrics

    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        batch_device = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    batch_device[key] = [v.to(self.device) for v in value]
                else:
                    batch_device[key] = value
            else:
                batch_device[key] = value

        return batch_device

    def _get_predictions(self, batch: Dict) -> Dict:
        """Get model predictions for batch"""
        # Get images and shapes
        images = batch['manual_step_image']
        part_shapes = batch.get('part_meshes')

        # Forward through model
        outputs = self.model(images, part_shapes)

        # Use pose solver for refinement
        refined_poses = []
        for i in range(images.shape[0]):
            camera = {k: v[i] for k, v in batch['camera'][i].items()} if 'camera' in batch else None

            poses = self.pose_solver.solve(
                images[i:i+1],
                [part_shapes[i]] if part_shapes else None,
                camera,
                None,  # No assembly state for evaluation
                None   # No part hints
            )
            refined_poses.append(poses[0] if poses else [])

        predictions = {
            'raw_outputs': outputs,
            'refined_poses': refined_poses
        }

        return predictions

    def _evaluate_batch(
        self,
        batch: Dict,
        predictions: Dict,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Evaluate predictions for a batch"""
        results = {}

        for metric in metrics:
            if metric == 'pose_accuracy':
                results['pose_accuracy'] = self._compute_pose_accuracy(
                    predictions['refined_poses'],
                    batch['gt_poses']
                )

            elif metric == 'chamfer_distance':
                results['chamfer_distance'] = self._compute_chamfer_distance(
                    predictions['refined_poses'],
                    batch['gt_poses'],
                    batch.get('part_meshes')
                )

            elif metric == 'mask_iou':
                results['mask_iou'] = self._compute_mask_iou(
                    predictions['raw_outputs'].get('masks'),
                    batch.get('masks_2d')
                )

            elif metric == 'plan_accuracy':
                results['plan_accuracy'] = self._compute_plan_accuracy(
                    predictions['refined_poses'],
                    batch
                )

            elif metric == 'assembly_feasibility':
                results['assembly_feasibility'] = self._compute_assembly_feasibility(
                    predictions['refined_poses']
                )

        return results

    def _compute_pose_accuracy(
        self,
        pred_poses: List[List[Dict]],
        gt_poses: List[List[Dict]],
        rotation_threshold: float = 30.0,  # degrees
        translation_threshold: float = 0.05  # meters
    ) -> Dict[str, float]:
        """
        Compute pose accuracy metrics

        Args:
            pred_poses: Predicted poses
            gt_poses: Ground truth poses
            rotation_threshold: Threshold for rotation accuracy (degrees)
            translation_threshold: Threshold for translation accuracy (meters)

        Returns:
            Dictionary of pose metrics
        """
        results = {
            'rotation_error': [],
            'translation_error': [],
            'rotation_accuracy': [],
            'translation_accuracy': [],
            'combined_accuracy': []
        }

        for batch_pred, batch_gt in zip(pred_poses, gt_poses):
            # Match predictions to ground truth
            matches = self._match_poses(batch_pred, batch_gt)

            for pred, gt in matches:
                if pred is None or gt is None:
                    continue

                # Rotation error
                R_pred = pred['R']
                R_gt = gt['R'].cpu().numpy() if isinstance(gt['R'], torch.Tensor) else gt['R']
                rot_error = self._rotation_error(R_pred, R_gt)
                results['rotation_error'].append(rot_error)
                results['rotation_accuracy'].append(rot_error < rotation_threshold)

                # Translation error
                t_pred = pred['t']
                t_gt = gt['t'].cpu().numpy() if isinstance(gt['t'], torch.Tensor) else gt['t']
                trans_error = np.linalg.norm(t_pred - t_gt)
                results['translation_error'].append(trans_error)
                results['translation_accuracy'].append(trans_error < translation_threshold)

                # Combined accuracy
                results['combined_accuracy'].append(
                    rot_error < rotation_threshold and trans_error < translation_threshold
                )

        # Compute averages
        metrics = {}
        for key in results:
            if results[key]:
                if 'accuracy' in key:
                    metrics[key] = np.mean(results[key])
                else:
                    metrics[key] = np.mean(results[key])
                    metrics[f"{key}_std"] = np.std(results[key])
            else:
                metrics[key] = 0.0

        return metrics

    def _compute_chamfer_distance(
        self,
        pred_poses: List[List[Dict]],
        gt_poses: List[List[Dict]],
        part_meshes: Optional[List[List[Any]]]
    ) -> Dict[str, float]:
        """Compute Chamfer distance between predicted and GT assemblies"""
        if self.chamfer_dist is None or part_meshes is None:
            return {'chamfer_distance': 0.0}

        distances = []

        for batch_idx, (batch_pred, batch_gt) in enumerate(zip(pred_poses, gt_poses)):
            if not part_meshes[batch_idx]:
                continue

            # Create assembled point clouds
            pred_points = self._assemble_point_cloud(batch_pred, part_meshes[batch_idx])
            gt_points = self._assemble_point_cloud(batch_gt, part_meshes[batch_idx])

            if pred_points is not None and gt_points is not None:
                # Convert to torch tensors
                pred_points_torch = torch.from_numpy(pred_points).float().to(self.device)
                gt_points_torch = torch.from_numpy(gt_points).float().to(self.device)

                # Add batch dimension
                pred_points_torch = pred_points_torch.unsqueeze(0)
                gt_points_torch = gt_points_torch.unsqueeze(0)

                # Compute Chamfer distance
                dist1, dist2 = self.chamfer_dist(pred_points_torch, gt_points_torch)
                chamfer = (dist1.mean() + dist2.mean()).item() / 2
                distances.append(chamfer)

        return {
            'chamfer_distance': np.mean(distances) if distances else 0.0,
            'chamfer_distance_std': np.std(distances) if distances else 0.0
        }

    def _compute_mask_iou(
        self,
        pred_masks: Optional[torch.Tensor],
        gt_masks: Optional[List[List[Any]]]
    ) -> Dict[str, float]:
        """Compute mask IoU metrics"""
        if pred_masks is None or gt_masks is None:
            return {'mask_iou': 0.0}

        ious = []

        for batch_idx in range(len(gt_masks)):
            batch_gt_masks = gt_masks[batch_idx]
            if not batch_gt_masks:
                continue

            batch_pred_masks = pred_masks[batch_idx]

            for part_idx, gt_mask in enumerate(batch_gt_masks):
                if gt_mask is None or part_idx >= batch_pred_masks.shape[0]:
                    continue

                pred_mask = batch_pred_masks[part_idx]

                # Binarize masks
                pred_binary = (pred_mask > 0.5).float()
                gt_binary = (gt_mask > 0.5).float() if isinstance(gt_mask, torch.Tensor) else torch.from_numpy(gt_mask > 0.5).float()

                # Compute IoU
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum() - intersection
                iou = intersection / (union + 1e-6)
                ious.append(iou.item())

        return {
            'mask_iou': np.mean(ious) if ious else 0.0,
            'mask_iou_std': np.std(ious) if ious else 0.0
        }

    def _compute_plan_accuracy(
        self,
        pred_poses: List[List[Dict]],
        batch: Dict
    ) -> Dict[str, float]:
        """Compute plan-level accuracy metrics"""
        results = {
            'part_detection_precision': [],
            'part_detection_recall': [],
            'sequence_accuracy': []
        }

        for batch_idx, batch_pred in enumerate(pred_poses):
            # Get ground truth parts
            gt_parts = batch['added_components'][batch_idx]
            gt_part_ids = [comp['part_id'] for comp in gt_parts]

            # Get predicted parts
            pred_part_ids = [p['part_id'] for p in batch_pred]

            # Compute precision/recall
            if pred_part_ids:
                correct_predictions = sum(1 for p in pred_part_ids if p in gt_part_ids)
                precision = correct_predictions / len(pred_part_ids)
                results['part_detection_precision'].append(precision)

            if gt_part_ids:
                correct_detections = sum(1 for g in gt_part_ids if g in pred_part_ids)
                recall = correct_detections / len(gt_part_ids)
                results['part_detection_recall'].append(recall)

            # Check sequence order (simplified - just check if parts are in roughly correct order)
            if len(pred_part_ids) == len(gt_part_ids):
                sequence_correct = all(p == g for p, g in zip(pred_part_ids, gt_part_ids))
                results['sequence_accuracy'].append(float(sequence_correct))

        # Compute averages
        metrics = {}
        for key in results:
            if results[key]:
                metrics[key] = np.mean(results[key])
            else:
                metrics[key] = 0.0

        # F1 score
        if metrics.get('part_detection_precision') > 0 and metrics.get('part_detection_recall') > 0:
            metrics['part_detection_f1'] = 2 * (
                metrics['part_detection_precision'] * metrics['part_detection_recall']
            ) / (metrics['part_detection_precision'] + metrics['part_detection_recall'])
        else:
            metrics['part_detection_f1'] = 0.0

        return metrics

    def _compute_assembly_feasibility(
        self,
        pred_poses: List[List[Dict]]
    ) -> Dict[str, float]:
        """Check if predicted assembly is physically feasible"""
        results = {
            'collision_free': [],
            'stability_score': [],
            'connectivity_score': []
        }

        for batch_pred in pred_poses:
            if not batch_pred:
                continue

            # Convert to assembly sequence
            sequence = [(p['part_id'], p['R'], p['t']) for p in batch_pred]

            # Validate using constraint engine
            is_valid, errors = self.constraint_engine.validate_assembly_sequence(sequence)
            results['collision_free'].append(float(is_valid))

            # Simple stability check (parts should be connected and supported)
            stability = self._check_stability(batch_pred)
            results['stability_score'].append(stability)

            # Connectivity check
            connectivity = self._check_connectivity(batch_pred)
            results['connectivity_score'].append(connectivity)

        # Compute averages
        metrics = {}
        for key in results:
            if results[key]:
                metrics[key] = np.mean(results[key])
            else:
                metrics[key] = 0.0

        return metrics

    def _match_poses(
        self,
        pred_poses: List[Dict],
        gt_poses: List[Dict]
    ) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
        """Match predicted poses to ground truth"""
        matches = []

        # Simple nearest neighbor matching based on position
        used_gt = set()

        for pred in pred_poses:
            best_match = None
            min_dist = float('inf')

            for i, gt in enumerate(gt_poses):
                if i in used_gt:
                    continue

                # Distance between positions
                t_pred = pred['t']
                t_gt = gt['t'].cpu().numpy() if isinstance(gt['t'], torch.Tensor) else gt['t']
                dist = np.linalg.norm(t_pred - t_gt)

                if dist < min_dist:
                    min_dist = dist
                    best_match = i

            if best_match is not None and min_dist < 0.1:  # 10cm threshold
                matches.append((pred, gt_poses[best_match]))
                used_gt.add(best_match)
            else:
                matches.append((pred, None))

        # Add unmatched GT
        for i, gt in enumerate(gt_poses):
            if i not in used_gt:
                matches.append((None, gt))

        return matches

    def _rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation error in degrees"""
        # Compute relative rotation
        R_rel = R1.T @ R2

        # Extract angle from trace
        trace = np.trace(R_rel)
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)

        return np.degrees(angle)

    def _assemble_point_cloud(
        self,
        poses: List[Dict],
        part_meshes: List[Any],
        num_points: int = 1000
    ) -> Optional[np.ndarray]:
        """Create assembled point cloud from parts and poses"""
        if not poses or not part_meshes:
            return None

        all_points = []

        for pose in poses:
            part_id = pose.get('part_id')
            if isinstance(part_id, str):
                # Try to find mesh by ID
                mesh = None
                for m in part_meshes:
                    if hasattr(m, 'metadata') and m.metadata.get('part_id') == part_id:
                        mesh = m
                        break
            elif isinstance(part_id, int) and part_id < len(part_meshes):
                mesh = part_meshes[part_id]
            else:
                continue

            if mesh is None:
                continue

            # Sample points from mesh
            if isinstance(mesh, trimesh.Trimesh):
                points = mesh.sample(num_points // len(poses))
            else:
                continue

            # Transform points
            R = pose['R']
            t = pose['t']
            points_transformed = points @ R.T + t
            all_points.append(points_transformed)

        if all_points:
            return np.vstack(all_points)
        return None

    def _check_stability(self, poses: List[Dict]) -> float:
        """Simple stability check"""
        if not poses:
            return 0.0

        # Check if parts are generally supported (lower parts should come first)
        heights = [p['t'][2] for p in poses]
        sorted_heights = sorted(heights)

        # Compute correlation between order and height
        correlation = np.corrcoef(range(len(heights)), heights)[0, 1]

        # Convert to score (higher correlation = better stability)
        stability_score = (correlation + 1) / 2  # Normalize to [0, 1]

        return stability_score

    def _check_connectivity(self, poses: List[Dict]) -> float:
        """Check if parts form connected assembly"""
        if len(poses) < 2:
            return 1.0

        # Simple distance-based connectivity
        max_gap = 0.15  # Maximum gap between connected parts
        connected_components = []

        for i, pose in enumerate(poses):
            pos = pose['t']
            connected = False

            # Check if connected to any existing component
            for component in connected_components:
                for j in component:
                    other_pos = poses[j]['t']
                    dist = np.linalg.norm(pos - other_pos)
                    if dist < max_gap:
                        component.add(i)
                        connected = True
                        break

                if connected:
                    break

            # Create new component if not connected
            if not connected:
                connected_components.append({i})

        # Merge overlapping components
        merged = True
        while merged:
            merged = False
            for i in range(len(connected_components)):
                for j in range(i + 1, len(connected_components)):
                    if connected_components[i] & connected_components[j]:
                        connected_components[i] |= connected_components[j]
                        connected_components.pop(j)
                        merged = True
                        break
                if merged:
                    break

        # Connectivity score: ratio of parts in largest component
        if connected_components:
            largest_component = max(connected_components, key=len)
            connectivity = len(largest_component) / len(poses)
        else:
            connectivity = 0.0

        return connectivity

    def _aggregate_metrics(
        self,
        all_results: List[Dict],
        metrics: List[str]
    ) -> Dict[str, float]:
        """Aggregate metrics across all batches"""
        aggregated = defaultdict(list)

        # Collect all values
        for batch_results in all_results:
            for metric in metrics:
                if metric in batch_results:
                    metric_values = batch_results[metric]
                    if isinstance(metric_values, dict):
                        for key, value in metric_values.items():
                            if not np.isnan(value):
                                aggregated[key].append(value)
                    else:
                        if not np.isnan(metric_values):
                            aggregated[metric].append(metric_values)

        # Compute averages
        final_metrics = {}
        for key, values in aggregated.items():
            if values:
                final_metrics[f"mean_{key}"] = np.mean(values)
                if len(values) > 1:
                    final_metrics[f"std_{key}"] = np.std(values)

        return final_metrics

    def _save_batch_predictions(
        self,
        batch_idx: int,
        batch: Dict,
        predictions: Dict,
        results: Dict
    ):
        """Save predictions for a batch"""
        save_path = self.output_dir / f"batch_{batch_idx:04d}_predictions.pkl"

        data = {
            'batch_idx': batch_idx,
            'furniture_ids': batch.get('furniture_id'),
            'predictions': predictions,
            'results': results
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def _save_evaluation_summary(self, metrics: Dict[str, float]):
        """Save evaluation summary"""
        summary_path = self.output_dir / "evaluation_summary.json"

        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Also save as readable text
        text_path = self.output_dir / "evaluation_summary.txt"
        with open(text_path, 'w') as f:
            f.write("IKEA MEPNet Evaluation Results\n")
            f.write("=" * 50 + "\n\n")

            for key, value in sorted(metrics.items()):
                f.write(f"{key:30s}: {value:.4f}\n")


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "./eval_results"
):
    """
    Main evaluation script

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        data_path: Path to IKEA dataset
        output_dir: Output directory for results
    """
    import yaml
    from ikea_mepnet_adapter.datasets.ikea_dataset import IKEADataset
    from ikea_mepnet_adapter.models.perception.mepnet_adapted import create_mepnet_model, MEPNetConfig
    from ikea_mepnet_adapter.models.mepnet_plus.solver import PoseSolver
    from ikea_mepnet_adapter.models.constraints.constraint_engine_complete import ConstraintEngine
    from ikea_mepnet_adapter.assets.registry import AssetsRegistry
    from ikea_mepnet_adapter.planner.assembler import PlanAssembler

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = IKEADataset(
        root_dir=data_path,
        split='test',
        **config.get('dataset', {})
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=dataset.collate_fn
    )

    # Create model
    model_config = MEPNetConfig(**config.get('model', {}))
    model = create_mepnet_model(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create auxiliary components
    registry = AssetsRegistry(
        parts_dir=os.path.join(data_path, 'data', 'parts')
    )

    constraint_engine = ConstraintEngine(
        assets_registry=registry
    )

    pose_solver = PoseSolver(
        perception_model=model,
        constraint_engine=constraint_engine,
        assets_registry=registry
    )

    plan_assembler = PlanAssembler(
        constraint_engine=constraint_engine,
        assets_registry=registry
    )

    # Create evaluator
    evaluator = IKEAEvaluator(
        model=model,
        pose_solver=pose_solver,
        plan_assembler=plan_assembler,
        constraint_engine=constraint_engine,
        output_dir=output_dir
    )

    # Run evaluation
    metrics = evaluator.evaluate(
        dataloader,
        metrics=config.get('metrics'),
        save_predictions=config.get('save_predictions', True)
    )

    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        print(f"{key:30s}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IKEA MEPNet")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='IKEA dataset path')
    parser.add_argument('--output', type=str, default='./eval_results', help='Output directory')

    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output
    )
