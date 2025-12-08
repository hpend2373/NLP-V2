
import os
import json
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

# --- Dataset ---
class RexGroundingDataset(Dataset):
    def __init__(self, data_file, ikea_root, transform=None):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.ikea_root = Path(ikea_root)
        self.transform = transform
        
        # Cache prompt images paths
        # Structure: data/rendered_parts/{Category}/{FurnitureID}/{PartID}_v{0-3}.png
        self.rendered_root = self.ikea_root / "data" / "rendered_parts"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_retries = len(self)
        retries = 0
        
        while retries < max_retries:
            current_idx = (idx + retries) % len(self)
            sample = self.data[current_idx]
            
            # Load Manual Image
            image_path = self.ikea_root / sample['image_path']
            if not image_path.exists():
                print(f"[DEBUG] Image not found: {image_path}")
                retries += 1
                continue
                
            try:
                manual_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[DEBUG] Failed to open image {image_path}: {e}")
                retries += 1
                continue
            
            # Parse Part ID
            text_prompt = sample['text_prompt']
            part_id = None
            
            if "Locate part" in text_prompt:
                part_id = text_prompt.split("Locate part ")[-1].strip()
            elif "Find parts" in text_prompt or "Identify components" in text_prompt:
                # For general prompts, pick a random part from the list if available
                if 'parts' in sample and sample['parts']:
                    part_id = np.random.choice(sample['parts'])
            
            if not part_id:
                # print(f"[DEBUG] No part_id parsed from: {text_prompt}")
                retries += 1
                continue

            # Find rendered views
            furniture_id = sample['furniture_id']
            category = sample['category']
            
            # Handle case sensitivity (Mac vs Linux)
            # Try original, lower, and upper
            candidates = [furniture_id, furniture_id.lower(), furniture_id.upper()]
            part_dir = None
            
            for fid in candidates:
                p = self.rendered_root / category / fid
                if p.exists():
                    part_dir = p
                    break
            
            if part_dir is None:
                # Try checking category casing too if needed, but usually category is consistent
                # Fallback to original for error reporting
                part_dir = self.rendered_root / category / furniture_id
            
            # Try to find views
            views = list(part_dir.glob(f"{part_id}_v*.png"))
            if not views:
                # Fallback: Check for legacy single-view file (e.g. "00.png")
                legacy_path = part_dir / f"{part_id}.png"
                if legacy_path.exists():
                    views = [legacy_path]
                else:
                    # Debugging: Check if directory exists
                    if not part_dir.exists():
                        # Only print once per directory to avoid spam
                        # print(f"[DEBUG] Part directory not found: {part_dir}")
                        pass
                    else:
                        # print(f"[DEBUG] No views found for part {part_id} in {part_dir}")
                        pass
                        
                    retries += 1
                    continue
                
            # Randomly select one view
            prompt_path = np.random.choice(views)
            prompt_image = Image.open(prompt_path).convert("RGB")
            
            # Augmentation: Edge Detection (50% chance)
            if np.random.rand() > 0.5:
                prompt_np = np.array(prompt_image.convert("L"))
                edges = cv2.Canny(prompt_np, 100, 200)
                edges_inverted = 255 - edges
                prompt_image = Image.fromarray(edges_inverted).convert("RGB")
                
            # Resize prompt (Random scale 150-250)
            scale = np.random.randint(150, 250)
            prompt_image.thumbnail((scale, scale))
            
            # Create Canvas (Visual Prompting)
            # Paste prompt at 0,0
            canvas = manual_image.copy()
            canvas.paste(prompt_image, (0, 0))
            
            # Resize Canvas if too large (Max 1024) to avoid OOM or aspect ratio issues
            max_dim = 1024
            if max(canvas.size) > max_dim:
                canvas.thumbnail((max_dim, max_dim))
            
            # Normalize Target BBox [x1, y1, x2, y2] -> [0-1000]
            # Note: After thumbnail, bbox needs to be scaled? 
            # No, bbox is relative to original size in the dataset?
            # Wait, sample['bbox'] is absolute coordinates in original image.
            # If we resize the image, we must scale the bbox!
            
            w_orig, h_orig = manual_image.size
            w_new, h_new = canvas.size
            
            bbox = sample['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Scale bbox to new size
            x1 = x1 * (w_new / w_orig)
            y1 = y1 * (h_new / h_orig)
            x2 = x2 * (w_new / w_orig)
            y2 = y2 * (h_new / h_orig)
            
            norm_box = [x1, y1, x2, y2] # Keep absolute for now, convert_to_bins handles normalization
            
            # Prompt Box (The visual prompt itself)
            # Prompt was pasted at 0,0 on original image.
            # It scales with the image.
            p_w, p_h = prompt_image.size
            p_w = p_w * (w_new / w_orig)
            p_h = p_h * (h_new / h_orig)
            prompt_box = [0, 0, p_w, p_h]
            
            return {
                "image": canvas,
                "prompt_box": prompt_box,
                "target_box": norm_box,
                "text": text_prompt
            }
            
        raise RuntimeError("Could not find any valid sample in the dataset. Check data paths and rendered_parts.")

# --- Utils ---
def convert_to_bins(box, w, h):
    # box: [x1, y1, x2, y2] (absolute)
    
    x1, y1, x2, y2 = box
    
    # Normalize to 0-1
    x1 /= w
    y1 /= h
    x2 /= w
    y2 /= h
    
    # Clamp
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    # Convert to 0-999
    x1_bin = int(x1 * 999)
    y1_bin = int(y1 * 999)
    x2_bin = int(x2 * 999)
    y2_bin = int(y2 * 999)
    
    return f"<{x1_bin}><{y1_bin}><{x2_bin}><{y2_bin}>"

# --- Training ---
def train(args):
    # Imports inside function to avoid top-level dependency issues if packages missing
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    
    model_id = "IDEA-Research/Rex-Omni"
    
    print(f"Loading model: {model_id}")
    # Use use_fast=False as per Rex-Omni wrapper
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=16*28*28, max_pixels=1024*28*28, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        attn_implementation="eager", 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Debug: Check vocab size
    print(f"Model Vocab Size: {model.config.vocab_size}")
    try:
        embed_weight = model.get_input_embeddings().weight
        print(f"Embedding Layer Shape: {embed_weight.shape}")
        embed_size = embed_weight.shape[0]
    except Exception as e:
        print(f"Could not get embedding layer: {e}")
        embed_size = model.config.vocab_size
    
    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Data
    dataset = RexGroundingDataset(args.data_file, args.ikea_root)
    # Use batch_size=1 for simplicity with variable image sizes, or use custom collate
    # For VLM training, batch processing requires careful padding. 
    # We'll use batch_size=1 and gradient accumulation if needed.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model.train()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        steps = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            batch_loss = 0
            for sample in batch:
                # Prepare Prompt
                image = sample['image']
                w, h = image.size
                
                # Visual Prompt Box (Prompt Image is pasted at 0,0)
                # prompt_box is [0, 0, p_w, p_h] relative to 1000x1000 canvas
                # We need to convert it to bins for the prompt string
                p_box = sample['prompt_box'] # [0, 0, pw_norm, ph_norm] (1000 scale)
                p_box_bins = convert_to_bins(p_box, 1000, 1000)
                
                visual_prompt_dict = {"object_1": [p_box_bins]}
                visual_prompt_json = json.dumps(visual_prompt_dict)
                
                prompt_text = (
                    f"Given reference boxes {visual_prompt_json} indicating one or more objects, "
                    "find all similar objects in the image and output their bounding boxes."
                )
                
                # Prepare Target
                t_box = sample['target_box'] # [x1, y1, x2, y2] (1000 scale)
                t_box_bins = convert_to_bins(t_box, 1000, 1000)
                target_text = f"<|object_ref_start|>object_1<|object_ref_end|><|box_start|>{t_box_bins}<|box_end|>"
                
                # Construct Messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_text}],
                    },
                ]
                
                # Process Inputs
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                
                # Debug: Check inputs
                input_ids = inputs.input_ids
                max_id = input_ids.max().item()
                
                # Re-fetch embed size if needed or use cached
                # We can trust model.config.vocab_size usually, but let's be safe
                vocab_size = model.config.vocab_size
                
                if max_id >= vocab_size:
                    print(f"ERROR: Input ID {max_id} exceeds vocab size {vocab_size}!")
                    # Clamp for debug
                    inputs.input_ids = torch.clamp(inputs.input_ids, max=vocab_size-1)
                
                if 'pixel_values' in inputs:
                    if torch.isnan(inputs.pixel_values).any():
                        print("ERROR: NaNs in pixel_values!")
                        continue
                
                # Create Labels
                input_ids = inputs.input_ids
                labels = input_ids.clone()
                
                # Forward
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                batch_loss += loss.item()
            
            # Average loss over batch (since we accumulated gradients)
            # Wait, we did backward per sample. So we should step once per batch?
            # Or step per sample?
            # With batch_size > 1 in dataloader but manual loop, we are doing gradient accumulation effectively.
            # We should divide loss by batch size.
            
            # Actually, let's do optimizer step after the batch loop
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= len(batch)
            
            optimizer.step()
            
            total_loss += batch_loss / len(batch)
            steps += 1
            
            if steps % 10 == 0:
                print(f"Epoch {epoch}, Step {steps}, Loss: {batch_loss / len(batch):.4f}")
            
        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
    # Save
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--ikea_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="rex_lora_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
