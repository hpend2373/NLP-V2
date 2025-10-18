#!/usr/bin/env python3
"""
Debug script to understand why dataset is empty
"""

import json
import sys
from pathlib import Path

def debug_dataset(root_dir: str):
    """Debug the IKEA dataset loading"""

    root = Path(root_dir)
    data_json = root / "data" / "data.json"

    print(f"=== Dataset Debug ===")
    print(f"Root: {root}")
    print(f"Data JSON: {data_json}")
    print(f"Exists: {data_json.exists()}")

    if not data_json.exists():
        print(f"\n❌ data.json not found!")

        # Check what's in the directories
        print(f"\nDirectory structure:")
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    print(f"  File: {p.relative_to(root)}")
        return

    # Load and inspect data.json
    with open(data_json, 'r') as f:
        data = json.load(f)

    print(f"\n✓ Loaded data.json")
    print(f"Number of furniture items: {len(data)}")

    # Analyze each furniture item
    for furniture_id, furniture_data in data.items():
        print(f"\nFurniture: {furniture_id}")

        if isinstance(furniture_data, dict):
            print(f"  Category: {furniture_data.get('category', 'N/A')}")
            print(f"  Has 'steps': {'steps' in furniture_data}")

            if 'steps' in furniture_data:
                steps = furniture_data['steps']
                print(f"  Number of steps: {len(steps)}")

                # Check first step
                if steps:
                    step = steps[0]
                    print(f"  First step keys: {list(step.keys())}")

                    # Check for manual image
                    if 'manual_image' in step:
                        img_path = root / "data" / "manual_img" / step['manual_image']
                        print(f"  Manual image exists: {img_path.exists()}")
                        if not img_path.exists():
                            print(f"    Expected at: {img_path}")

                    # Check for manual_parts
                    if 'manual_parts' in step:
                        print(f"  Manual parts: {len(step['manual_parts'])} parts")

                    # Check for aligned_frames
                    if 'aligned_frames' in step:
                        frames = step['aligned_frames']
                        print(f"  Aligned frames: {len(frames)} frames")
                        if frames:
                            frame = frames[0]
                            print(f"    Frame keys: {list(frame.keys())}")
                            if 'poses' in frame:
                                print(f"    Parts with poses: {list(frame['poses'].keys())}")
        else:
            print(f"  WARNING: Not a dict, type: {type(furniture_data)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/content/IKEA-Manuals-at-Work")
    args = parser.parse_args()

    debug_dataset(args.root)