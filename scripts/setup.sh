#!/bin/bash
# Setup script for IKEA MEPNet Adapter

echo "Setting up IKEA MEPNet Adapter..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install PyTorch3D (requires manual installation)
echo "Installing PyTorch3D..."
echo "Please follow the instructions at: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
echo "For PyTorch3D 0.5.0:"
echo "  pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.5.0'"

# Download IKEA dataset if not present
if [ ! -d "IKEA-Manuals-at-Work" ]; then
    echo "IKEA dataset not found. Please download from:"
    echo "  https://github.com/yunongLiu1/IKEA-Manuals-at-Work"
    echo "And extract to ./IKEA-Manuals-at-Work"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p experiments
mkdir -p logs
mkdir -p cache
mkdir -p outputs

# Set up data symlinks (optional)
# ln -s /path/to/ikea/data IKEA-Manuals-at-Work

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"