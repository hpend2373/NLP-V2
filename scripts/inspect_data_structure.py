#!/usr/bin/env python3
"""
Inspect actual IKEA dataset structure to understand why no samples load
"""

import json
import sys
from pathlib import Path

def inspect_data_json(root_dir: str):
    """Inspect the actual structure of data.json"""

    root = Path(root_dir)
    data_json = root / "data" / "data.json"

    print(f"=== Inspecting IKEA Dataset ===")
    print(f"Looking at: {data_json}")
    print(f"Exists: {data_json.exists()}")
    print()

    if not data_json.exists():
        print("data.json not found!")
        return

    with open(data_json, 'r') as f:
        data = json.load(f)

    print(f"Type of data: {type(data)}")

    if isinstance(data, list):
        print(f"Data is a list with {len(data)} items")
        if data:
            print("\nFirst item structure:")
            first_item = data[0]
            print(f"  Type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"  Keys: {list(first_item.keys())[:10]}")  # First 10 keys

                # Check for important fields
                for key in ['furniture_id', 'id', 'name', 'category', 'steps', 'manual_parts']:
                    if key in first_item:
                        value = first_item[key]
                        if isinstance(value, (list, dict)):
                            print(f"  {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"  {key}: {value}")

                # Check steps structure
                if 'steps' in first_item:
                    steps = first_item['steps']
                    if steps and isinstance(steps, list):
                        print(f"\n  First step structure:")
                        step = steps[0]
                        if isinstance(step, dict):
                            print(f"    Keys: {list(step.keys())}")

                            # Check for manual_parts
                            if 'manual_parts' in step:
                                parts = step['manual_parts']
                                print(f"    manual_parts: {type(parts).__name__}")
                                if isinstance(parts, list) and parts:
                                    print(f"      First part: {parts[0]}")

                            # Check for aligned_frames
                            if 'aligned_frames' in step:
                                frames = step['aligned_frames']
                                print(f"    aligned_frames: {type(frames).__name__} with {len(frames) if isinstance(frames, list) else 0} items")

    elif isinstance(data, dict):
        print(f"Data is a dict with {len(data)} keys")
        keys = list(data.keys())[:10]  # First 10 keys
        print(f"First keys: {keys}")

        if keys:
            first_key = keys[0]
            first_value = data[first_key]
            print(f"\nFirst item '{first_key}':")
            print(f"  Type: {type(first_value)}")

            if isinstance(first_value, dict):
                print(f"  Keys: {list(first_value.keys())}")

                # Check for important fields
                for key in ['category', 'steps', 'parts']:
                    if key in first_value:
                        value = first_value[key]
                        if isinstance(value, (list, dict)):
                            print(f"  {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"  {key}: {value}")

    # Check for manual images
    print("\n=== Manual Images ===")
    manual_img_dir = root / "data" / "manual_img"
    if manual_img_dir.exists():
        print(f"Manual image directory exists")
        # List subdirectories
        subdirs = [p for p in manual_img_dir.iterdir() if p.is_dir()]
        print(f"Subdirectories: {[p.name for p in subdirs[:5]]}")  # First 5

        # Check structure
        for subdir in subdirs[:2]:  # Check first 2
            print(f"\n  {subdir.name}/")
            sub_subdirs = [p for p in subdir.iterdir() if p.is_dir()]
            if sub_subdirs:
                print(f"    Has subdirs: {[p.name for p in sub_subdirs[:3]]}")

            images = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))
            if images:
                print(f"    Has {len(images)} images directly")
    else:
        print("Manual image directory not found")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/content/IKEA-Manuals-at-Work")
    args = parser.parse_args()

    inspect_data_json(args.root)