#!/usr/bin/env python3
"""
Setup script for Colab environment with IKEA dataset
Downloads actual data files from Git LFS
"""

import os
import subprocess
import sys

def setup_git_lfs():
    """Install and setup Git LFS"""
    print("Installing Git LFS...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)
    subprocess.run(["git", "lfs", "install"], check=True)
    print("✓ Git LFS installed")

def clone_with_lfs():
    """Clone repository with LFS files"""
    print("\nCloning repository with LFS files...")

    # Remove existing directory if any
    if os.path.exists("NLP-V2"):
        subprocess.run(["rm", "-rf", "NLP-V2"], check=True)

    # Clone with LFS
    subprocess.run([
        "git", "clone",
        "https://github.com/hpend2373/NLP-V2.git"
    ], check=True)

    os.chdir("NLP-V2")

    # Pull LFS files
    print("Downloading LFS files (this may take several minutes)...")
    subprocess.run(["git", "lfs", "pull"], check=True)

    print("✓ Repository cloned with data files")

def verify_data():
    """Verify data files are downloaded correctly"""
    print("\nVerifying data files...")

    data_json = "IKEA-Manuals-at-Work/data/data.json"

    if not os.path.exists(data_json):
        print("✗ data.json not found!")
        return False

    # Check if it's still a LFS pointer (should be actual JSON)
    with open(data_json, 'r') as f:
        first_line = f.readline()
        if "git-lfs" in first_line:
            print("✗ data.json is still a LFS pointer, not actual data!")
            return False

    size = os.path.getsize(data_json)
    print(f"✓ data.json size: {size / 1e6:.1f} MB")

    return True

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling Python dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio"
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "opencv-python", "pillow", "scipy", "scikit-image",
        "trimesh", "pyyaml", "tqdm", "wandb", "tensorboard"
    ], check=True)

    # Install ChamferDistance for GPU
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "chamferdist"
    ], check=True)

    print("✓ Dependencies installed")

def main():
    """Main setup function"""
    print("="*60)
    print("IKEA MEPNet Adapter - Colab Setup")
    print("="*60)

    try:
        # Setup Git LFS
        setup_git_lfs()

        # Clone with LFS
        clone_with_lfs()

        # Verify data
        if not verify_data():
            print("\n⚠️  Warning: Data verification failed!")
            print("You may need to manually download the dataset.")

        # Install dependencies
        install_dependencies()

        print("\n" + "="*60)
        print("✓ Setup complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. cd NLP-V2")
        print("2. python scripts/train/train_ikea.py --config configs/train_config_a100.yaml")

    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
