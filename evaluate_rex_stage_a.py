import json
from pathlib import Path
import argparse

def evaluate_stage_a(data_root, generated_file):
    data_root = Path(data_root)
    data_json_path = data_root / "data" / "data.json"
    
    print(f"Loading data.json from {data_json_path}...")
    with open(data_json_path, 'r') as f:
        ikea_data = json.load(f)
        
    # Load generated predictions
    print(f"Loading generated predictions from {generated_file}...")
    detected_parts = set()
    processed_furniture = set()
    
    with open(generated_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                furniture_id = record['furniture_id']
                processed_furniture.add(furniture_id)
                
                step_idx = record['step_idx']
                part_id = record['part_id']
                
                # Check if prediction is non-empty
                preds = record.get('rex_prediction', {})
                has_prediction = False
                for label, boxes in preds.items():
                    if boxes:
                        has_prediction = True
                        break
                
                if has_prediction:
                    key = (furniture_id, step_idx, part_id)
                    detected_parts.add(key)
            except json.JSONDecodeError:
                continue
                
    print(f"Processed furniture items: {len(processed_furniture)}")
    
    # Count total expected parts ONLY for processed furniture
    total_parts = 0
    expected_parts = set()
    
    print("Counting expected parts for processed items...")
    for furniture_id, item in ikea_data.items():
        # Check if this furniture was processed (case insensitive check might be needed)
        # The generated file uses the ID from data.json usually.
        
        # Check if furniture_id is in processed_furniture
        # Note: generate_rex_data uses item.get('furniture_id') or name.upper()
        # data.json keys are usually furniture_id.
        
        if furniture_id not in processed_furniture and furniture_id.upper() not in processed_furniture:
            continue
            
        steps = item.get('steps', [])
        for step_idx, step in enumerate(steps):
            manual_parts = step.get('manual_parts', [])
            for part in manual_parts:
                part_id = part['part_id']
                # Key: (FurnitureID, StepIdx, PartID)
                key = (furniture_id, step_idx, part_id)
                expected_parts.add(key)
                
    total_parts = len(expected_parts)
    print(f"Total expected parts in processed subset: {total_parts}")
    
    detected_count = len(detected_parts)
    print(f"Total detected parts: {detected_count}")
    
    recall = (detected_count / total_parts) * 100 if total_parts > 0 else 0
    print(f"Stage A Recall (Part Detection Rate): {recall:.2f}% ({detected_count}/{total_parts})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/Users/minyeop/NLP/NLP-V2/IKEA-Manuals-at-Work")
    parser.add_argument("--generated_file", default="/Users/minyeop/NLP/NLP-V2/data/stage_a/stage_a_rex_generated.jsonl")
    args = parser.parse_args()
    
    evaluate_stage_a(args.data_root, args.generated_file)
