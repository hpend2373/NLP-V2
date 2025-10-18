#!/usr/bin/env python3
"""Check what files exist in Colab environment"""
import os
from pathlib import Path

def check_files():
    base_path = Path("/content/ikea_mepnet_adapter")

    print("=== Checking Colab Files ===")
    print(f"Base path exists: {base_path.exists()}")

    if base_path.exists():
        # Check scripts directory
        scripts_dir = base_path / "scripts"
        print(f"\nScripts directory exists: {scripts_dir.exists()}")

        if scripts_dir.exists():
            print("\nFiles in scripts/:")
            for f in scripts_dir.iterdir():
                if f.is_file():
                    print(f"  - {f.name}")

        # Check for create_sample_data specifically
        sample_data_file = scripts_dir / "create_sample_data.py"
        print(f"\ncreate_sample_data.py exists: {sample_data_file.exists()}")

        # Check train directory
        train_dir = scripts_dir / "train"
        if train_dir.exists():
            print("\nFiles in scripts/train/:")
            for f in train_dir.iterdir():
                if f.is_file():
                    print(f"  - {f.name}")

if __name__ == "__main__":
    check_files()