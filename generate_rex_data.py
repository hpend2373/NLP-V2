import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.ops as ops

# Add Rex-Omni to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "Rex-Omni"))
from rex_omni.wrapper import RexOmniWrapper

def generate_rex_data(
    model_path: str,
    ikea_root: str,
    output_file: str,
    limit: int = None,
    lora_path: str = None
):
    print(f"Loading Rex-Omni model from {model_path}...")
    # Disable Flash Attention 2 for Mac (MPS)
    model = RexOmniWrapper(
        model_path, 
        attn_implementation="eager",
        torch_dtype=torch.float16 # Use float16 for MPS
    )
    
    # Load LoRA if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        from peft import PeftModel
        # Access the underlying Qwen2_5_VLForConditionalGeneration model
        # RexOmniWrapper stores it in self.model
        model.model = PeftModel.from_pretrained(model.model, lora_path)
        print("LoRA adapter loaded successfully.")
    
    ikea_root = Path(ikea_root)
    ikea_data_file = ikea_root / "data" / "data.json"
    
    print(f"Loading IKEA data from {ikea_data_file}...")
    with open(ikea_data_file, 'r') as f:
        ikea_items = json.load(f)
        
    records = []
    
    # Flatten items for processing
    items_list = list(ikea_items.values())
    if limit:
        items_list = items_list[:limit]
        
    # Prepare output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Clear file if exists
    with open(output_file, 'w') as f:
        pass
        
    for item in tqdm(items_list, desc="Processing"):
        furniture_id = item.get('furniture_id') or item.get('name', '').upper()
        category = item.get('category', 'Unknown')
        steps = item.get('steps', [])
        
        for step_idx, step in enumerate(steps):
            manual_parts = step.get('manual_parts', [])
            if not manual_parts:
                continue
                
            # Find manual image
            fid_lower = furniture_id.lower()
            step_dir = ikea_root / "data" / "manual_img" / category / fid_lower / f"step_{step_idx}"
            
            image_path = None
            if step_dir.exists():
                png_files = list(step_dir.glob("*.png"))
                if png_files:
                    image_path = png_files[0]
            
            if not image_path:
                # Try uppercase furniture_id
                step_dir = ikea_root / "data" / "manual_img" / category / furniture_id / f"step_{step_idx}"
                if step_dir.exists():
                    png_files = list(step_dir.glob("*.png"))
                    if png_files:
                        image_path = png_files[0]
            
            if not image_path:
                continue
                
            try:
                manual_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load manual image {image_path}: {e}")
                continue
                
            # For each part in this step
            for part in manual_parts:
                part_id = part['part_id']
                
                # Find rendered part images (Multi-View)
                prompt_images = []
                
                # Search paths
                search_dirs = [
                    ikea_root / "data" / "rendered_parts" / category / furniture_id,
                    ikea_root / "data" / "rendered_parts" / category / furniture_id.lower()
                ]
                
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                        
                    # Try multi-view first
                    views = sorted(list(search_dir.glob(f"{part_id}_v*.png")))
                    if views:
                        prompt_images.extend(views)
                        break # Found views in this dir
                    
                    # Try single view
                    single = search_dir / f"{part_id}.png"
                    if single.exists():
                        prompt_images.append(single)
                        break
                
                if not prompt_images:
                    # print(f"Rendered part not found for {furniture_id} {part_id}")
                    continue
                
                # Prepare batch for inference
                batch_images = []
                batch_prompt_boxes = []
                
                # Multi-Scale Strategy
                scales = [150, 200, 250]
                
                for prompt_path in prompt_images:
                    # Load prompt image
                    prompt_base = Image.open(prompt_path).convert("RGB")
                    
                    for scale in scales:
                        # Resize
                        prompt_image = prompt_base.copy()
                        prompt_image.thumbnail((scale, scale))
                        
                        # Apply Edge Detection (Style Matching)
                        prompt_np = np.array(prompt_image.convert("L"))
                        edges = cv2.Canny(prompt_np, 100, 200)
                        edges_inverted = 255 - edges
                        prompt_image = Image.fromarray(edges_inverted).convert("RGB")
                        
                        p_w, p_h = prompt_image.size
                        
                        # Create Canvas
                        canvas = manual_image.copy()
                        canvas.paste(prompt_image, (0, 0))
                        
                        # Prompt Box
                        prompt_box = [0.0, 0.0, float(p_w), float(p_h)]
                        
                        batch_images.append(canvas)
                        batch_prompt_boxes.append([prompt_box])
                    
                if not batch_images:
                    continue
                    
                # Run Batch Inference
                try:
                    results = model.inference(
                        images=batch_images,
                        task="visual_prompting", 
                        visual_prompt_boxes=batch_prompt_boxes
                    )
                    
                    # Collect results (Ensemble)
                    all_boxes = []
                    all_scores = []
                    
                    for i, res in enumerate(results):
                        # Determine which scale this result belongs to (for filtering)
                        scale_idx = i % len(scales)
                        current_scale = scales[scale_idx]
                        
                        # Filter limit based on current scale
                        p_w_limit = current_scale
                        p_h_limit = current_scale
                        
                        raw_preds = res.get('extracted_predictions', {})
                        for label, items in raw_preds.items():
                            for item in items:
                                box = item.get('coords')
                                score = item.get('score', 1.0)
                                
                                if box:
                                    # Ensure it's a box (length 4)
                                    if len(box) != 4:
                                        continue
                                        
                                    # Filter prompt box dynamically
                                    if box[0] < p_w_limit and box[1] < p_h_limit:
                                        continue
                                        
                                    all_boxes.append(box)
                                    all_scores.append(score)

                except Exception as e:
                    print(f"Batch inference failed: {e}")
                    continue

                # NMS (Non-Maximum Suppression)
                rex_prediction = {}
                if all_boxes:
                    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
                    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
                    
                    # NMS threshold 0.5
                    keep_indices = ops.nms(boxes_tensor, scores_tensor, 0.5)
                    
                    final_boxes = boxes_tensor[keep_indices].tolist()
                    final_scores = scores_tensor[keep_indices].tolist()
                    
                    # Format for output
                    rex_prediction = {
                        "part": [
                            {'type': 'box', 'coords': box, 'score': score}
                            for box, score in zip(final_boxes, final_scores)
                        ]
                    }

                # Create Record
                record = {
                    'image_path': str(Path(image_path).relative_to(ikea_root)),
                    'furniture_id': furniture_id,
                    'step_idx': step_idx,
                    'part_id': part_id,
                    'rex_prediction': rex_prediction,
                }
                
                # Write immediately
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                
                records.append(record)
    
    print(f"Done. Saved {len(records)} records to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="IDEA-Research/Rex-Omni")
    parser.add_argument("--ikea_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()
    
    generate_rex_data(args.model_path, args.ikea_root, args.output, args.limit, args.lora_path)
