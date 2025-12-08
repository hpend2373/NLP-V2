import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rex_ikea.models.stage_b_pose_refiner import create_pose_refiner
from rex_ikea.datasets.ikea_pose_dataset import create_pose_dataloader
from rex_ikea.datasets.ikea_gt_loader import create_gt_loader

def compute_rotation_error(R_pred, R_gt):
    """Compute rotation error in degrees"""
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.detach().cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.detach().cpu().numpy()

    R_pred = np.array(R_pred)
    R_gt = np.array(R_gt)

    if R_pred.ndim == 1 and R_pred.shape[0] == 9:
        R_pred = R_pred.reshape(3, 3)
    if R_gt.ndim == 1 and R_gt.shape[0] == 9:
        R_gt = R_gt.reshape(3, 3)

    R_diff = R_pred.T @ R_gt
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle)

def compute_translation_error(t_pred, t_gt):
    """Compute translation error in meters"""
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.detach().cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.detach().cpu().numpy()

    t_pred = np.array(t_pred)
    t_gt = np.array(t_gt)
    return np.linalg.norm(t_pred - t_gt)

def evaluate_rex_mepnet_style(config_path, model_path, device='cpu'):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Load GT Loader to get ALL expected parts
    gt_loader = create_gt_loader(config['data']['data_root'])
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = create_pose_refiner(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load Validation Dataset (using generated Stage A predictions)
    print("Loading validation dataset...")
    # Force use_stage_a_predictions = True for evaluation
    config['data']['use_stage_a_predictions'] = True
    val_dataloader = create_pose_dataloader(
        config=config['data'],
        split='val', # or 'train' if we want to eval on train set for now
        batch_size=32,
        num_workers=0
    )
    
    # Store results: results[(furniture_id, step_idx)][part_id] = is_correct
    results = defaultdict(lambda: defaultdict(bool))
    
    # Metrics
    rot_errors = []
    trans_errors = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Move to device
            crop_image = batch['crop_image'].to(device)
            full_image = batch['full_image'].to(device)
            part_id = batch['part_id'].to(device)
            confidence = batch['confidence'].to(device)
            text_embed = batch['text_embed'].to(device)
            gt_rotation = batch['gt_rotation'].to(device)
            gt_translation = batch['gt_translation'].to(device)
            
            # Metadata
            meta_furn_ids = batch['metadata']['furniture_ids']
            meta_step_idxs = batch['metadata']['step_indices']
            meta_part_strs = batch['metadata']['part_id_strs']
            
            # Forward
            predictions = model(crop_image, full_image, part_id, confidence, text_embed)
            
            pred_rot_matrix = model.get_rotation_matrix(predictions)
            pred_trans = predictions['translation']
            
            # Compute metrics for batch
            for i in range(len(meta_furn_ids)):
                r_err = compute_rotation_error(pred_rot_matrix[i], gt_rotation[i])
                t_err = compute_translation_error(pred_trans[i], gt_translation[i])
                
                rot_errors.append(r_err)
                trans_errors.append(t_err)
                
                # Check correctness (Thresholds: 15 deg, 10cm)
                is_correct = (r_err < 15.0) and (t_err < 0.10)
                
                fid = meta_furn_ids[i]
                sidx = meta_step_idxs[i]
                pid = meta_part_strs[i]
                
                results[(fid, sidx)][pid] = is_correct

    # Compute Aggregated Metrics
    print("\nComputing Aggregated Metrics...")
    
    # 1. Part Accuracy (Micro average over detected parts? Or over all expected parts?)
    # MEPNet likely computes over all expected parts.
    # If a part is missing from `results`, it means Stage A failed to detect it.
    # So it counts as incorrect.
    
    total_expected_parts = 0
    total_correct_parts = 0
    
    total_expected_steps = 0
    total_correct_steps = 0
    
    furniture_stats = defaultdict(lambda: {'steps_total': 0, 'steps_correct': 0})
    
    # Iterate over all expected parts in GT
    # We need to iterate over the validation set split of GT.
    # Use the dataset's split list if available, or hardcode/load it.
    # Since val_dataloader.dataset has the split logic, let's use that.
    
    # Evaluate on ALL furniture items that were processed in the generated file
    # We can get this from the dataloader's dataset samples
    val_furniture_ids = set()
    if hasattr(val_dataloader.dataset, 'samples'):
        for s in val_dataloader.dataset.samples:
            val_furniture_ids.add(s['furniture_id'])
            
    print(f"Evaluated on {len(val_furniture_ids)} furniture items (All Processed).")
    
    for fid in val_furniture_ids:
        # Get all steps for this furniture from GT Loader
        # We need to know max steps or iterate through data.json structure
        # GT Loader doesn't expose structure easily, let's look at `results` keys
        # But `results` only has detected steps.
        # We should look at `gt_loader.data`
        
        item_data = gt_loader.data.get(fid) or gt_loader.data.get(fid.lower())
        if not item_data:
            continue
            
        steps = item_data.get('steps', [])
        for step in steps:
            sidx = step['step_idx']
            
            # Get expected parts in this step
            # We can use `gt_loader.get_all_poses_in_step`
            # But we need to know frame_id. Default 0.
            gt_poses = gt_loader.get_all_poses_in_step(fid, sidx)
            if not gt_poses:
                continue
                
            step_expected_parts = len(gt_poses)
            total_expected_parts += step_expected_parts
            
            # Check how many were correct in our results
            step_correct_parts = 0
            for pid in gt_poses.keys():
                if results[(fid, sidx)].get(pid, False):
                    step_correct_parts += 1
            
            total_correct_parts += step_correct_parts
            
            # Step Accuracy: All parts must be correct
            is_step_correct = (step_correct_parts == step_expected_parts)
            
            total_expected_steps += 1
            if is_step_correct:
                total_correct_steps += 1
                
            furniture_stats[fid]['steps_total'] += 1
            if is_step_correct:
                furniture_stats[fid]['steps_correct'] += 1

    # Furniture Accuracy
    total_furniture = len(furniture_stats)
    correct_furniture = 0
    for fid, stats in furniture_stats.items():
        if stats['steps_total'] > 0 and stats['steps_correct'] == stats['steps_total']:
            correct_furniture += 1

    # Print Results
    print("=" * 60)
    print("Rex-Omni + Stage B Evaluation Results")
    print("=" * 60)
    
    print(f"Rotation Error:    {np.mean(rot_errors):.2f}° ± {np.std(rot_errors):.2f}°")
    print(f"Translation Error: {np.mean(trans_errors)*1000:.1f}mm ± {np.std(trans_errors)*1000:.1f}mm")
    
    part_acc = (total_correct_parts / total_expected_parts * 100) if total_expected_parts > 0 else 0
    print(f"Part Accuracy:     {part_acc:.1f}% ({total_correct_parts}/{total_expected_parts})")
    
    step_acc = (total_correct_steps / total_expected_steps * 100) if total_expected_steps > 0 else 0
    print(f"Step Accuracy:     {step_acc:.1f}% ({total_correct_steps}/{total_expected_steps})")
    
    furn_acc = (correct_furniture / total_furniture * 100) if total_furniture > 0 else 0
    print(f"Complete Furniture:{furn_acc:.1f}% ({correct_furniture}/{total_furniture})")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_rex_mepnet_style(args.config, args.model)
