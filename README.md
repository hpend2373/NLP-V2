# IKEA MEPNet Adapter

## Translating IKEA Furniture Assembly Manuals to Machine-Executable Plans

This project adapts the LEGO MEPNet (Machine-Executable Plan Network) to work with IKEA furniture assembly data from the "IKEA Manuals at Work" dataset. The system converts 2D manual images into 6D pose predictions and assembly sequences for real furniture.

## 🎯 Key Features

- **Data Adaptation**: Bridges IKEA dataset format to MEPNet-compatible structure
- **Continuous 6D Pose**: Extends LEGO's discrete rotation to continuous 6D poses
- **Constraint Engine**: Generalizes LEGO's stud-based constraints to furniture connections (dowels, screws, cam locks)
- **Multi-modal Learning**: Leverages manual images, 3D parts, and optional video data
- **Analysis-by-Synthesis**: Differentiable rendering for pose refinement
- **Assembly Planning**: Generates complete assembly plans with dependencies

## 📁 Project Structure

```
ikea_mepnet_adapter/
├── datasets/
│   ├── ikea_dataset.py         # IKEA dataset loader
│   └── transforms_ikea.py      # Data augmentations
├── models/
│   ├── perception/
│   │   ├── mepnet_adapted.py   # Adapted MEPNet model
│   │   └── hourglass.py        # Hourglass backbone
│   ├── constraints/
│   │   └── constraint_engine.py # Assembly constraints
│   └── mepnet_plus/
│       └── solver.py            # Pose solver with refinement
├── assets/
│   └── registry.py              # 3D part management
├── planner/
│   └── assembler.py             # Assembly plan generation
├── eval/
│   └── eval_ikea.py            # Evaluation pipeline
├── scripts/
│   ├── train/
│   │   └── train_ikea.py       # Training script
│   ├── setup.sh                # Environment setup
│   └── train.sh                # Training launcher
├── configs/
│   └── train_config.yaml       # Training configuration
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd ikea_mepnet_adapter

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate environment
source venv/bin/activate

# Install PyTorch3D manually (follow instructions)
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.5.0'
```

### 2. Prepare Data

Download the IKEA Manuals at Work dataset:
```bash
# Clone IKEA dataset (already done in setup)
# The dataset should be in ./IKEA-Manuals-at-Work/
```

### 3. Train Model

```bash
# Single GPU training
./scripts/train.sh --config configs/train_config.yaml

# Multi-GPU training
./scripts/train.sh --config configs/train_config.yaml --gpus 0,1,2,3

# Resume training
./scripts/train.sh --config configs/train_config.yaml --resume experiments/ikea_mepnet_baseline/checkpoints/checkpoint_epoch_50.pth
```

### 4. Evaluate

```bash
python eval/eval_ikea.py \
    --config configs/train_config.yaml \
    --checkpoint experiments/ikea_mepnet_baseline/checkpoints/best_model.pth \
    --data ./IKEA-Manuals-at-Work \
    --output ./eval_results
```

## 📊 Model Architecture

### MEPNet+ Overview

1. **Perception Module** (Hourglass Network)
   - Input: Manual image + optional 3D shape conditioning
   - Output: Keypoint heatmaps, 6D rotations, translations, masks

2. **Constraint Engine**
   - Manages assembly constraints (dowels, screws, cam locks)
   - Generates valid pose candidates
   - Validates physical feasibility

3. **Pose Solver**
   - Coarse pose from perception
   - Constraint-based refinement
   - Analysis-by-synthesis with differentiable rendering

4. **Plan Assembler**
   - Converts per-step predictions to assembly plans
   - Detects subassemblies
   - Optimizes assembly sequence

## 🔧 Key Adaptations from LEGO

| Component | LEGO Version | IKEA Adaptation |
|-----------|--------------|-----------------|
| **Parts** | Fixed brick types (130³ voxel) | Arbitrary meshes with auto-detected symmetries |
| **Connections** | Stud/anti-stud grid | Continuous constraints (dowels, screws, surfaces) |
| **Rotations** | 8 discrete bins | Continuous 6D representation |
| **Assembly** | Bottom-up only | Subassemblies with dependencies |
| **Scale** | LEGO units | Real-world centimeters |

## 📈 Evaluation Metrics

- **Pose Accuracy**: Rotation (<30°) and translation (<5cm) thresholds
- **Chamfer Distance**: 3D shape reconstruction quality
- **Mask IoU**: 2D segmentation accuracy
- **Plan Accuracy**: Part detection F1 and sequence accuracy
- **Assembly Feasibility**: Collision-free, stable, connected

## 🔍 Configuration

Edit `configs/train_config.yaml` to customize:

```yaml
model:
  use_continuous_rotation: true  # 6D vs discrete
  max_parts: 10                  # Parts per step
  shape_encoder_type: "pointnet" # Shape encoding

data:
  furniture_categories: ["Chair", "Table"]  # Furniture types
  augmentation:
    use_manual_style: true  # Manual-specific augmentations

optimization:
  lr: 0.0001
  scheduler: "cosine"
```

## 🛠️ Development

### Adding New Connection Types

1. Define in `assets/registry.py`:
```python
class ConnectionType(Enum):
    YOUR_TYPE = "your_type"
```

2. Add constraint config in `constraint_engine.py`:
```python
'your_type': {
    'translation': DoFType.TRANSLATION_1D,
    'rotation': DoFType.FIXED,
    ...
}
```

### Custom Augmentations

Add to `datasets/transforms_ikea.py`:
```python
class YourAugmentation(IKEATransform):
    def __call__(self, sample):
        # Your augmentation logic
        return sample
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{wang2022translating,
  title={Translating a Visual LEGO Manual to a Machine-Executable Plan},
  author={Wang, Ruocheng and others},
  booktitle={ECCV},
  year={2022}
}

@dataset{liu2024ikea,
  title={IKEA Manuals at Work},
  author={Liu, Yunong and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2024}
}
```

## 📄 License

This project bridges two repositories with their respective licenses:
- LEGO MEPNet: [Check original repository]
- IKEA Manuals at Work: CC-BY-4.0

## 🤝 Acknowledgments

- Original LEGO MEPNet implementation
- IKEA Manuals at Work dataset creators
- PyTorch3D for differentiable rendering

## 📮 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This is an research adaptation bridging LEGO assembly planning to real furniture. Performance may vary based on furniture complexity and manual quality.