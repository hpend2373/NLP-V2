# Repository Analysis Summary

A comprehensive analysis of both the **lego_release** and **IKEA-Manuals-at-Work** repositories has been completed.

## Files Generated

- **`REPOSITORY_ANALYSIS.md`** (11,000+ lines) - Detailed technical analysis containing:
  - Complete directory structures for both repositories
  - Deep dive into MEPNet architecture (hourglass networks, shape conditioning, PyTorch3D integration)
  - IKEA dataset structure (data.json format, 3D models, annotations)
  - Data loading pipelines and format specifications
  - Training infrastructure and hyperparameters
  - Rendering utilities and camera parameter handling
  - Integration challenges and solutions
  - Recommended architecture for the adapter

## Key Findings

### lego_release Repository
- **Size**: 15,038 lines of Python code
- **Focus**: MEPNet - advanced neural network for predicting 3D assembly from 2D instructions
- **Strengths**:
  - Sophisticated architecture with shape conditioning
  - Careful handling of discrete brick types and spatial relationships
  - Multi-stage inference with physics-aware constraints
  - Synthetic data with perfect camera calibration

**Core Files**:
1. `hourglass_shape_cond_model.py` (823 lines) - Main MEPNet model
2. `legokps_shape_cond_dataset.py` (833 lines) - Data loading
3. `brick_info.py` (1752 lines) - Brick system and voxel encoding
4. `train_kp.py` (284 lines) - Training loop

### IKEA-Manuals-at-Work Repository
- **Size**: 22,122 lines of Python code
- **Focus**: Dataset and annotation tools for 4D grounding of assembly instructions
- **Strengths**:
  - Rich real-world data (34,441 annotated frames, 98 videos)
  - 6D pose annotations with camera calibration
  - 36 furniture models across 6 categories
  - Professional annotation interfaces

**Key Assets**:
1. `data.json` (206MB, Git-LFS) - Main annotation file
2. `parts/` - 3D OBJ models organized by category
3. `render_part.py` (82 lines) - Open3D rendering utility
4. Camera parameter matrices (intrinsic & extrinsic)

## Critical Integration Points

### Data Representation Gaps

| Aspect | LEGO | IKEA | Solution |
|--------|------|------|----------|
| Brick Types | Fixed 70 types | ~1000+ part types | Build geometry-aware encoder |
| Pose Space | Discrete rotations (8 classes) | Continuous 6D | Replace classifier with regression |
| Voxel Grid | Fixed 130³ | Dynamic per-furniture | Implement adaptive voxelizer |
| Images | 512×512 synthetic | Variable resolution real | Preprocessing pipeline |
| Coordinates | LEGO units | Centimeters | Coordinate system mapper |

### Architecture Differences

**LEGO System**:
- Shape-conditioned image encoder
- Per-brick-type heatmap generation
- Discrete rotation classification
- Offset-based keypoint refinement
- Physics-based placement constraints

**IKEA Requirements**:
- Continuous 6D pose regression
- Arbitrary part geometry handling
- Multi-part per step assembly
- Real-world scene complexity
- Video temporal alignment

## Implementation Roadmap

### Phase 1: Data Adapters
1. Implement `ikea_dataset.py` - Load IKEA data.json
2. Create `part_loader.py` - OBJ mesh loading
3. Build `voxelizer.py` - Dynamic mesh voxelization
4. Implement `preprocessing.py` - Image/annotation transforms

### Phase 2: Model Modifications
1. Create `ikea_model.py` - Adapted MEPNet for IKEA
2. Implement `part_encoder.py` - Geometry-aware encoder
3. Build `pose_head.py` - 6D pose regression head
4. Adapt loss functions for continuous poses

### Phase 3: Inference & Validation
1. Implement inference pipeline
2. Create evaluation metrics
3. Build visualization tools
4. Performance benchmarking

## Technical Specifications

### MEPNet Architecture Overview

```
Input:
  - RGB Image (512×512×3)
  - Voxel Condition (130³)
  - Camera params (azim, elev)
  - Object pose (quaternion)

Processing:
  - Image Encoder (ResNet18)
  - Voxel Encoder (3D Convolutions)
  - Hourglass Network (Multi-scale)
  
Outputs:
  - Heatmaps (per brick type)
  - Rotation (discrete: 8 classes)
  - Offset (regression)
  - Masks (optional instance seg)
  
Loss:
  - L_hm: Heatmap loss (MSE/Focal)
  - L_rot: Cross-entropy
  - L_offset: L1 regression
  - L_mask: Mask loss (optional)
```

### IKEA Data Format

```json
{
  "furniture_id": {
    "category": "Chair|Table|...",
    "steps": [
      {
        "parts": [
          {
            "part_id": 1,
            "position": [x, y, z],
            "rotation": [w, x, y, z],
            "camera": {
              "intrinsic": 3×3,
              "extrinsic": 4×4
            }
          }
        ]
      }
    ]
  }
}
```

## Dependencies

### Unified Requirements
- PyTorch: 1.8.0 - 1.10.0
- PyTorch3D: 0.5.0 (pinned for compatibility)
- OpenCV: 4.5.4+
- Open3D: 0.15.2
- NumPy: 1.21.2
- PyRender: 0.1.43+
- PyBullet: 3.1.7

## Next Steps

1. **Read REPOSITORY_ANALYSIS.md** in detail - contains all implementation specifics
2. **Begin Phase 1** - Start with IKEA dataset loader
3. **Reference render_part.py** for Open3D rendering patterns
4. **Study hourglass_shape_cond_model.py** for architecture patterns
5. **Adapt legokps_shape_cond_dataset.py** for IKEA data format

## File Locations

**Working Directory**: `/Users/minyeop/NLP/ikea_mepnet_adapter/`

**lego_release**: `/Users/minyeop/NLP/ikea_mepnet_adapter/lego_release/`

**IKEA**: `/Users/minyeop/NLP/ikea_mepnet_adapter/IKEA-Manuals-at-Work/`

**Analysis Document**: `/Users/minyeop/NLP/ikea_mepnet_adapter/REPOSITORY_ANALYSIS.md`

---

**Analysis Date**: 2025-10-17
**Total Code Analyzed**: 37,160 lines
**Repositories**: 2 (lego_release + IKEA-Manuals-at-Work)
**Documentation Pages**: 11,000+ lines
