#!/usr/bin/env python3
"""
Setup script for running IKEA MEPNet in Google Colab
Creates sample data if real dataset is not available
"""

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup the environment for Colab"""

    print("=== IKEA MEPNet Colab Setup ===\n")

    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✓ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("✗ Not running in Google Colab")

    # Set up paths
    if IN_COLAB:
        # Add the adapter directory to Python path
        adapter_path = "/content/ikea_mepnet_adapter"
        if adapter_path not in sys.path:
            sys.path.insert(0, adapter_path)
            print(f"✓ Added {adapter_path} to Python path")

        # Check if IKEA dataset exists
        data_path = Path("/content/IKEA-Manuals-at-Work/data/data.json")

        if not data_path.exists():
            print("\n⚠️  IKEA dataset not found. Creating sample data...")

            # Import and run the sample data creator
            sys.path.append("/content/ikea_mepnet_adapter/scripts")
            from create_sample_data import create_sample_ikea_dataset

            root = create_sample_ikea_dataset("/content/IKEA-Manuals-at-Work")
            print(f"✓ Sample dataset created at: {root}")
        else:
            print("✓ IKEA dataset found")

        # Install required packages if needed
        required_packages = [
            "trimesh",
            "pytorch3d",
            "chamferdist",
            "wandb",
        ]

        print("\n📦 Checking required packages...")
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package} is installed")
            except ImportError:
                print(f"✗ {package} not found. Installing...")
                os.system(f"pip install -q {package}")

        # Special handling for PyTorch3D
        try:
            import pytorch3d
        except ImportError:
            print("Installing PyTorch3D (this may take a few minutes)...")
            import torch
            cuda_version = torch.version.cuda.replace(".", "")
            torch_version = torch.__version__.split("+")[0].replace(".", "")

            # Install PyTorch3D
            os.system(f"""
                pip install -q --no-index --no-cache-dir pytorch3d \
                    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu{cuda_version}_pyt{torch_version}/download.html
            """)

    else:
        print("Please run this script in Google Colab")

    print("\n=== Setup Complete ===")


def test_import():
    """Test if all imports work"""
    print("\n🧪 Testing imports...")

    try:
        from ikea_mepnet_adapter.datasets.ikea_dataset import IKEADataset
        print("✓ IKEADataset import successful")

        from ikea_mepnet_adapter.models.perception.mepnet_adapted import create_mepnet_model
        print("✓ MEPNet model import successful")

        from ikea_mepnet_adapter.models.constraints.constraint_engine_final import ConstraintEngine
        print("✓ ConstraintEngine import successful")

        print("\n✅ All imports successful! You can now run training.")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    setup_colab_environment()
    test_import()

    print("\n📝 To start training, run:")
    print("!python /content/ikea_mepnet_adapter/scripts/train/train_ikea.py \\")
    print("    --config /content/ikea_mepnet_adapter/configs/train_config.yaml")