#!/usr/bin/env python3
"""
Create sample IKEA dataset structure for testing
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh

def create_sample_ikea_dataset(root_dir: str = "IKEA-Manuals-at-Work"):
    """Create a minimal IKEA dataset structure with sample data"""

    root = Path(root_dir)

    # Create directory structure
    dirs_to_create = [
        root / "data",
        root / "data" / "manual_img" / "Chair" / "STEFAN",
        root / "data" / "parts",
        root / "data" / "videos",
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create sample data.json
    data = {
        "STEFAN": {
            "furniture_id": "STEFAN",
            "category": "Chair",
            "steps": [
                {
                    "step_idx": 0,
                    "manual_parts": [
                        {"part_id": "leg_001", "instance_id": "leg_001_0"},
                        {"part_id": "seat_001", "instance_id": "seat_001_0"}
                    ],
                    "manual_connections": [
                        ["leg_001", "seat_001", "dowel"],
                    ],
                    "manual_image": "Chair/STEFAN/step_000.png",
                    "base_assembly": [],
                    "aligned_frames": [
                        {
                            "frame_id": 0,
                            "poses": {
                                "leg_001": {
                                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Identity matrix flattened
                                    "translation": [0, 0, 0]
                                },
                                "seat_001": {
                                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                    "translation": [0, 0, 0.1]
                                }
                            },
                            "masks": {},
                            "camera": {
                                "intrinsic": [500, 0, 256, 0, 500, 256, 0, 0, 1],
                                "extrinsic": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
                            }
                        }
                    ]
                },
                {
                    "step_idx": 1,
                    "manual_parts": [
                        {"part_id": "leg_002", "instance_id": "leg_002_0"},
                        {"part_id": "leg_003", "instance_id": "leg_003_0"}
                    ],
                    "manual_connections": [
                        ["leg_002", "seat_001", "dowel"],
                        ["leg_003", "seat_001", "dowel"],
                    ],
                    "manual_image": "Chair/STEFAN/step_001.png",
                    "base_assembly": ["leg_001", "seat_001"],
                    "aligned_frames": []
                }
            ],
            "parts": {
                "leg_001": {"category": "leg", "symmetry": "rotational_4"},
                "leg_002": {"category": "leg", "symmetry": "rotational_4"},
                "leg_003": {"category": "leg", "symmetry": "rotational_4"},
                "seat_001": {"category": "seat", "symmetry": "planar"}
            }
        },
        "LACK": {
            "furniture_id": "LACK",
            "category": "Table",
            "steps": [
                {
                    "step_idx": 0,
                    "manual_parts": [
                        {"part_id": "tabletop_001", "instance_id": "tabletop_001_0"},
                        {"part_id": "leg_001", "instance_id": "leg_001_0"}
                    ],
                    "manual_connections": [
                        ["leg_001", "tabletop_001", "screw"],
                    ],
                    "manual_image": "Table/LACK/step_000.png",
                    "base_assembly": [],
                    "aligned_frames": []
                }
            ],
            "parts": {
                "tabletop_001": {"category": "top", "symmetry": "rotational_4"},
                "leg_001": {"category": "leg", "symmetry": "cylindrical"}
            }
        }
    }

    # Save data.json
    data_json_path = root / "data" / "data.json"
    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Created data.json at {data_json_path}")

    # Create sample manual images
    for furniture_id in ["STEFAN", "LACK"]:
        furniture_data = data[furniture_id]
        category = furniture_data["category"]

        for step in furniture_data["steps"]:
            # Create a simple colored image as placeholder
            img = Image.new('RGB', (512, 512), color='white')

            # Draw some simple shapes to represent manual instructions
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            # Draw a border
            draw.rectangle([10, 10, 502, 502], outline='black', width=2)

            # Add step number
            try:
                # Try to use a default font
                draw.text((256, 50), f"Step {step['step_idx']}", fill='black', anchor="mm")
            except:
                # If font fails, just continue
                pass

            # Draw some rectangles to represent parts
            y_offset = 100
            for i, part in enumerate(step["manual_parts"]):
                draw.rectangle([50, y_offset + i*60, 150, y_offset + i*60 + 40],
                              outline='blue', width=2)
                draw.text((200, y_offset + i*60 + 20), part["part_id"], fill='black')

            # Save image
            img_path = root / "data" / "manual_img" / category / furniture_id / f"step_{step['step_idx']:03d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
            print(f"Created manual image: {img_path}")

    # Create sample 3D parts (simple box meshes)
    parts_to_create = [
        ("leg_001", [0.05, 0.05, 0.3]),  # Leg dimensions
        ("leg_002", [0.05, 0.05, 0.3]),
        ("leg_003", [0.05, 0.05, 0.3]),
        ("seat_001", [0.4, 0.4, 0.05]),  # Seat dimensions
        ("tabletop_001", [0.8, 0.8, 0.03]),  # Table top
    ]

    for part_id, dimensions in parts_to_create:
        # Create a simple box mesh
        mesh = trimesh.creation.box(extents=dimensions)

        # Save as OBJ
        mesh_path = root / "data" / "parts" / f"{part_id}.obj"
        mesh.export(str(mesh_path))
        print(f"Created mesh: {mesh_path}")

    print("\n✅ Sample IKEA dataset created successfully!")
    print(f"Root directory: {root.absolute()}")

    return str(root.absolute())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create sample IKEA dataset")
    parser.add_argument("--root", type=str, default="IKEA-Manuals-at-Work",
                       help="Root directory for dataset")

    args = parser.parse_args()

    create_sample_ikea_dataset(args.root)