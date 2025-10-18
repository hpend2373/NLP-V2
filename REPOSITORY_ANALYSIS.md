# IKEA-MEPNet Adapter: Comprehensive Repository Analysis

## Executive Summary

This document provides a detailed analysis of both the **lego_release** and **IKEA-Manuals-at-Work** repositories to understand their structures and identify integration points for the adapter project.

- **lego_release**: 15,038 lines of Python code - MEPNet architecture for 3D part assembly prediction from 2D instructions
- **IKEA-Manuals-at-Work**: 22,122 lines of Python code - Dataset and tools for 4D grounding of assembly instructions on videos
- **Goal**: Bridge these two systems to adapt MEPNet for real-world IKEA assembly data

---

## Part 1: LEGO Release Repository Structure

### 1.1 Directory Organization

```
lego_release/
├── bricks/                 # LEGO brick type definitions & utilities
│   ├── brick_info.py      # Core brick system: 1700+ lines
│   ├── bricks.py          # Brick class definitions
│   └── utils.py           # Geometric transformations
├── models/                # Neural network architecture
│   ├── hourglass_shape_cond_model.py  # Main MEPNet model (823 lines)
│   ├── hourglass_trans_model.py       # Transformer variant (740 lines)
│   ├── networks.py        # Backbone encoders & utilities
│   ├── heatmap/          # Heatmap generation & decoding
│   └── coordconv.py      # Coordinate convolution layers
├── datasets/             # Data loading pipeline
│   ├── legokps_shape_cond_dataset.py  # Main dataset class (833 lines)
│   ├── base_dataset.py   # Base dataset utilities
│   └── definition.py     # Dataset definitions
├── lego/                 # LEGO-specific utilities
│   └── utils/
│       ├── camera_utils.py
│       ├── inference_utils.py
│       └── ...
├── options/              # Command-line argument parsing
│   ├── base_options.py
│   ├── train_options.py
│   └── test_options.py
├── util/                 # General utilities
│   ├── util.py
│   ├── visualizer.py
│   └── html.py
├── train_kp.py          # Main training script
├── test_kp_shape_cond.py # Evaluation script
└── eval.py              # Evaluation metrics
```

### 1.2 Core Model Architecture: MEPNet

#### A. Main Model: HourglassShapeCondModel (47KB)

**Location**: `/lego_release/models/hourglass_shape_cond_model.py`

**Key Components**:
- **Input Processing**:
  - RGB image (512x512)
  - Previous object occupancy voxel (130x130x130, binary)
  - Camera parameters (azimuth, elevation)
  - Object pose (quaternion)
  - Brick shape condition (voxel encoding)

- **Architecture**:
  - Image encoder: ResNet18 backbone (SimpleResNetEncoder)
  - Voxel encoder: 3D convolutions on shape condition
  - Hourglass network: Stacked feature extraction
  - Multiple output heads:
    - Heatmap (per brick type)
    - Rotation classification (8 classes for symmetric rotations)
    - Offset regression (for keypoint refinement)
    - Optional mask prediction

- **Key Features**:
  - Shape-conditioned: Adapts to different brick types
  - Multi-brick handling: Groups bricks by type
  - Mask prediction for instance segmentation (optional)
  - 3D pose inference for connected brick assembly

- **Loss Functions**:
  - Heatmap loss (MSE or Focal loss)
  - Rotation cross-entropy loss
  - Offset L1 loss
  - Mask losses (optional)
  - Association embedding loss (optional)

#### B. PyTorch3D Integration

**File**: `/lego_release/lego/utils/camera_utils.py`

- **Camera Model**: Pinhole camera with OpenGL NDC (Normalized Device Coordinates)
- **Transformations**: PT3D Transform3d for coordinate transformations
- **Projection**: Screen space projection for 2D-3D correspondence

**Dependencies**:
```python
import pytorch3d.transforms as pt
import pytorch3d.rendering
```

#### C. Voxel Representation

**File**: `/lego_release/bricks/brick_info.py` (1752 lines)

**Key Functions**:
- `get_brick_enc_voxel(bid)`: Encode brick type as 3D voxel (130x130x130)
- `get_cbrick_enc_voxel(cbrick)`: Composite brick encoding
- `pts2voxel()`: Convert 3D point cloud to voxel grid

**Voxel Encoding Scheme**:
- Different voxel IDs for different brick shapes
- Studs marked with ID 2, value 0.25
- Regular bricks ID 3, value 1.0
- Round bricks ID 4, value 0.75
- Slope bricks ID 5, value 0.6

### 1.3 Dataset: LegoKpsShapeCondDataset

**File**: `/lego_release/datasets/legokps_shape_cond_dataset.py` (833 lines)

#### Data Format

**Input Structure**:
```
data/
├── dataset_name/
│   ├── 000.png              # RGB image
│   ├── 001.png
│   ├── info.json            # Metadata
│   ├── occs.pkl             # Occupancy voxels
│   ├── conns.pkl            # Connections
│   └── metadata/
```

**info.json Structure**:
```python
{
    "operations": {
        "0": {
            "bricks": [
                {
                    "brick_type": "3024",
                    "brick_transform": {
                        "position": [x, y, z],
                        "rotation": [euler_x, euler_y, euler_z],
                        "rotation_euler": tuple,
                        "keypoint": [2d_x, 2d_y, ...]
                    },
                    "bbox": [[x1, y1], [x2, y2]],
                    "mask": rle_encoded_mask,
                    "op_type": 0 | 1 | 2  # 0=insert on top, 1=insert below, 2=ground
                }
            ],
            "obj_rotation_quat": quaternion,
            "view_direction": [azimuth, elevation]
        },
        ...
    },
    "grid_size": [21, 21, 21],
    "obj_scale": 1.0,
    "obj_center": [x, y, z]
}
```

#### Data Loading Pipeline

1. **Initialization** (lines 180-450):
   - Load all assembly steps from JSON
   - Build lists of: translations, rotations, brick types, bboxes, masks
   - Filter by max objects per image
   - Create indices for valid samples

2. **__getitem__** Processing** (lines 476-779):
   - Load RGB image (512x512)
   - Load previous occupancy voxel
   - Build target dictionaries with grouped brick data
   - Generate heatmaps for each brick type
   - Create instance masks
   - Collate into batch format

3. **Output Format**:
   ```python
   {
       'img': tensor(3, 512, 512),           # RGB image
       'img_path': str,
       'obj_occ_prev': binary_voxel,         # Previous occupancy
       'obj_r': quaternion,                   # Object rotation
       'obj_q': normalized_quaternion,        # Normalized rotation
       'azim': float,                         # Camera azimuth
       'elev': float,                         # Camera elevation
       'obj_scale': float,
       'obj_center': [x, y, z],
       
       # Per-brick-type grouped data
       'hm': heatmap_list,                    # Heatmaps
       'reg': offset_list,                    # Offset regressions
       'reg_mask': mask_list,                 # Validity masks
       'bid': brick_id_list,                  # Brick type IDs
       'rot': rotation_id_list,               # Rotation IDs
       'kp': keypoint_list,                   # 2D keypoints
       
       # Optional
       'mask': instance_masks,
       'mask_instance': inst_mask_per_type,
       'assoc_emb_sample_inds': sample_indices
   }
   ```

### 1.4 Training Pipeline

**Main Script**: `/lego_release/train_kp.py` (284 lines)

#### Key Training Components

1. **Data Loading**:
   - Distributed data loader support
   - Batch size: 2 (default)
   - Multiple workers

2. **Model Training** (lines 177-190):
   ```python
   for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
       for i, data in enumerate(dataset):
           model.set_input(data)
           model.optimize_parameters()
           if opt.lr_policy in batch_lr_schedulers:
               model.update_learning_rate()
   ```

3. **Loss Tracking**:
   - Heatmap loss
   - Rotation cross-entropy loss
   - Offset L1 loss
   - Optional: Mask losses

4. **Validation** (lines 235-268):
   - Validation dataset evaluation
   - Loss tracking per batch
   - Model checkpointing (best + latest)

#### Hyperparameters (from hourglass_shape_cond_model.py)

```python
batch_size = 2
lr = 1e-3
niter = 200                  # Epochs
niter_decay = 0
lr_policy = 'step'
lr_decay_iters = 3

# Loss weights
lbd_t = 1.0                  # Translation loss
lbd_r = 1.0                  # Rotation loss
lbd_q = 1.0                  # Quaternion loss
lbd_h = 1.0                  # Heatmap loss
lbd_o = 1.0                  # Offset loss
```

### 1.5 Key Dependencies

```
torch==1.8.0
torchvision==0.9.0
pytorch3d==0.5.0
trimesh==3.9.34
numpy==1.21.2
opencv-python==4.5.4.58
scipy==1.7.1
h5py==3.2.1
```

---

## Part 2: IKEA-Manuals-at-Work Repository Structure

### 2.1 Directory Organization

```
IKEA-Manuals-at-Work/
├── data/                    # Dataset files
│   ├── data.json           # Main annotation file (Git-LFS, 206MB)
│   ├── parts/              # 3D model OBJ files
│   │   ├── Bench/
│   │   ├── Chair/          # 59 furniture models
│   │   ├── Desk/
│   │   ├── Misc/
│   │   ├── Shelf/
│   │   └── Table/
│   ├── manual_img/         # Instruction manual images
│   ├── manual_masks/       # Segmentation masks for manual images
│   └── pdfs/               # Original PDF manuals
├── render_part.py          # Part rendering utility (82 lines)
├── src/IKEAVideo/          # Source utilities
│   ├── featurizers/        # Feature extraction (CLIP, DINOv2, MAE, etc.)
│   └── ... other modules
├── annotation_tool/        # Annotation interfaces
│   ├── Annotation-Interface/
│   └── Pose-Refine-Interface-Release/
├── notebooks/
│   └── data_viz.ipynb
└── datasheet.md            # Detailed dataset documentation
```

### 2.2 Dataset Structure

#### A. data.json Format (Git-LFS)

**Size**: 206MB (version 1)
**Structure**: JSON-serialized assembly annotations

**Conceptual Structure**:
```python
{
    "furniture_id": {
        "model_id": str,
        "category": "Chair" | "Table" | "Shelf" | "Bench" | "Desk" | "Misc",
        "num_steps": int,
        "steps": [
            {
                "step_id": int,
                "manual_image_id": str,
                "description": str,
                "parts": [
                    {
                        "part_id": int,
                        "part_name": str,
                        "part_type": str,
                        "quantity": int,
                        "manual_bbox": [x1, y1, x2, y2],
                        "6d_pose": {
                            "position": [x, y, z],
                            "rotation": quaternion | euler_angles
                        },
                        "camera_matrix": {
                            "intrinsic": 3x3,
                            "extrinsic": 4x4
                        }
                    }
                ],
                "video_alignments": [
                    {
                        "video_id": str,
                        "frame_ranges": [[start, end], ...],
                        "temporal_step": int
                    }
                ]
            }
        ]
    }
}
```

**Key Features**:
- 36 furniture models (3000+ parts total)
- 98 assembly videos
- 34,441 annotated video frames
- Dense spatio-temporal alignments
- Multi-part per step with 6D poses

#### B. Parts Directory Structure

```
parts/
├── Bench/
│   ├── model_01/
│   │   ├── 01.obj          # Part 1 mesh
│   │   ├── 02.obj          # Part 2 mesh
│   │   └── ...
│   └── model_02/
├── Chair/
│   ├── model_01/
│   │   ├── 01.obj
│   │   └── ...
│   └── ... (59 models for Chair category)
└── ... other categories
```

**Format**: Wavefront OBJ files (3D mesh geometry)

#### C. Manual Images Organization

```
manual_img/
├── Bench/
│   ├── model_01_manual.png    # Full manual image
│   ├── model_01_step_01.png   # Cropped step image
│   ├── model_01_step_02.png
│   └── ...
├── Chair/
├── Desk/
├── Misc/
├── Shelf/
└── Table/
```

**Format**: 
- PNG images
- Various resolutions
- Assembly instruction illustrations

#### D. Manual Masks Organization

```
manual_masks/
├── Bench/
├── Chair/
│   ├── model_01_step_01_parts.png
│   ├── model_01_step_02_parts.png
│   └── ...
└── ... other categories
```

**Content**:
- Part segmentation masks
- One channel per part
- Aligned with manual_img

### 2.3 Rendering Utility: render_part.py

**File**: `/IKEA-Manuals-at-Work/render_part.py` (82 lines)

#### Function Signature

```python
def render_part(
    obj_path,        # Base directory with OBJ files
    part_ids,        # List of part IDs to render
    ext_mat,         # Extrinsic matrix (4x4)
    int_mat,         # Intrinsic matrix (3x3)
    img_width,       # Image width
    img_height,      # Image height
    save_path,       # Output path
    colors=colors    # RGBA colors per part
)
```

#### Implementation Details

1. **Camera Setup**:
   - Open3D PinholeCameraIntrinsic from intrinsic matrix
   - Extrinsic matrix defines camera pose (world-to-camera)

2. **Rendering Process**:
   - Load each OBJ mesh
   - Assign unique color per part
   - Add to rendering scene
   - Render with white background
   - Save as PNG

3. **Camera Intrinsics Format**:
   ```
   [[fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]]
   ```

4. **Camera Extrinsics Format** (4x4):
   ```
   [[R11, R12, R13, tx],
    [R21, R22, R23, ty],
    [R31, R32, R33, tz],
    [ 0,   0,   0,  1]]
   ```

#### Dependencies

```python
import open3d as o3d
from PIL import Image
import numpy as np
from pyvirtualdisplay import Display  # For headless rendering
```

### 2.4 Data Format Specifications

#### A. Camera Parameter Representation

**Intrinsic Matrix** (3x3):
- Focal lengths: fx, fy
- Principal point: cx, cy
- Typical range: fx ≈ 600-2000 for 512-2048px images

**Extrinsic Matrix** (4x4 Transformation):
- 3x3 Rotation matrix (world-to-camera)
- 3x1 Translation vector
- Bottom row: [0, 0, 0, 1]

#### B. 6D Pose Representation

**Position** (3D):
- World coordinates
- Typically normalized to [-1, 1] or object-relative

**Orientation** (3D):
- Quaternion: [w, x, y, z] or [x, y, z, w]
- OR Euler angles: [roll, pitch, yaw]
- Rotation order: ZYX or XYZ (need to verify from data.json)

#### C. Part ID Convention

**Part Numbering**: Sequential integers
- Typically 01.obj, 02.obj, ..., nn.obj
- Mapped in data.json with descriptive names

### 2.5 Dataset Statistics

- **Total Furniture Models**: 36
  - Bench: ~10 models
  - Chair: ~22 models (most diverse)
  - Desk: ~2-3 models
  - Misc: ~2-3 models
  - Shelf: ~2-3 models
  - Table: ~7-10 models

- **Total Frames**: 34,441 annotated
- **Total Videos**: 98 assembly videos
- **Average Parts per Model**: ~50-100

### 2.6 Key Dependencies

```
open3d==0.15.2
opencv-python >= 3.4.2.17
PyPDF2
nvisii==1.1.70            # Alternative rendering
pybullet==3.1.7           # Physics simulation
pyrender >= 0.1.43        # PBR rendering
pytorch_lightning==1.6.1
wandb===0.13.10
```

---

## Part 3: Key Integration Points & Challenges

### 3.1 Data Format Gaps

#### Challenge 1: Coordinate System Differences

| Aspect | LEGO | IKEA |
|--------|------|------|
| Grid Size | 21×21×21 (65×65×65 voxel) | Variable per furniture |
| Unit | LEGO stud (8mm) | Centimeters |
| Origin | Grid center | Model-specific |
| Up Direction | +Y (standard) | May vary |

**Action**: Need coordinate system mapper

#### Challenge 2: Brick vs Part Representation

| LEGO | IKEA |
|------|------|
| Fixed brick types (~70 types) | Variable part geometry (~1000s types) |
| Discrete snapping positions | Continuous 6D poses |
| Stud/antistu connection points | Arbitrary part geometries |
| Quaternion representation | May have specific conventions |

**Action**: Build part classifier & geometry-aware encoder

#### Challenge 3: Image Resolution & Camera

| Aspect | LEGO | IKEA |
|--------|------|------|
| Image Size | 512×512 fixed | Varies (likely 800-2048) |
| Camera Model | Synthetic (perfect) | Real world videos + estimated poses |
| Background | White | Cluttered real scenes |
| Lighting | Controlled | Variable |

**Action**: Preprocessing pipeline for resolution & augmentation

### 3.2 Voxel Encoding Differences

**Current LEGO System**:
- Fixed voxel grid (130×130×130)
- Discrete brick types (IDs 1-70)
- Pre-computed encodings with brick ID

**IKEA Requirements**:
- Dynamic grid sizing (per furniture)
- Continuous part geometry
- Need geometry-aware encoding (not just ID-based)

**Solution**: 
1. Use OBJ mesh directly in voxelizer
2. Create per-category encoders
3. Implement differentiable voxelization

### 3.3 Training Data Preprocessing

**Required Transforms**:
1. Image resizing (512×512)
2. Depth map generation (optional)
3. Part instance segmentation
4. Keypoint detection (part contact points)
5. Rotation normalization
6. Pose alignment with camera

**Tools Available**:
- `render_part.py`: OBJ to 2D projection
- Open3D: Mesh utilities
- OpenCV: Image processing

### 3.4 Loss Function Adaptations

**LEGO Specific**:
- Discrete rotation space (8 classes)
- Stud-based positioning
- Connection constraints

**IKEA Needs**:
- Continuous 6D pose regression
- Arbitrary part shapes
- Physics-based constraints (collision detection)

**Adaptation Strategy**:
1. Keep heatmap for part detection
2. Replace rotation classifier with continuous regression
3. Add pose refinement loss
4. Implement physics-aware loss (optional)

---

## Part 4: File Mapping Summary

### Critical LEGO Files for Adaptation

| File | Purpose | Adaptation Needed |
|------|---------|-------------------|
| `hourglass_shape_cond_model.py` | Main MEPNet architecture | Modify input/output heads |
| `legokps_shape_cond_dataset.py` | Data loading | Complete rewrite for IKEA data |
| `brick_info.py` | Brick type system | Replace with part geometry loader |
| `camera_utils.py` | Camera projections | Adapt focal length ranges |
| `train_kp.py` | Training loop | Minor modifications |

### Critical IKEA Files to Integrate

| File | Purpose | Usage |
|------|---------|-------|
| `data.json` | Annotations | Data source |
| `parts/*/` | 3D models | Shape conditioning |
| `manual_img/` | Assembly steps | Potential auxiliary input |
| `render_part.py` | Rendering | Part visualization & projection |
| `datasheet.md` | Specs | Reference implementation details |

---

## Part 5: Recommended Integration Architecture

### 5.1 New Module Structure

```
ikea_adapter/
├── data/
│   ├── ikea_dataset.py           # IKEA data loader
│   ├── part_loader.py            # OBJ mesh loading
│   ├── voxelizer.py              # Geometry voxelization
│   └── preprocessing.py          # Image/annotation transforms
├── models/
│   ├── ikea_model.py             # Modified MEPNet for IKEA
│   ├── part_encoder.py           # Geometry encoder
│   └── pose_head.py              # 6D pose regression
├── lego_compat/
│   ├── camera_utils.py           # Extended camera utilities
│   └── inference_utils.py        # Updated inference
├── utils/
│   ├── pose_utils.py             # Quaternion/Euler conversions
│   ├── mesh_utils.py             # OBJ processing
│   └── rendering.py              # Integration with render_part.py
└── config/
    ├── ikea_defaults.py          # Default hyperparameters
    └── category_configs.py       # Per-furniture-category configs
```

### 5.2 Data Flow

```
IKEA Dataset (data.json)
        ↓
[Part & Step Loader]
        ↓
[Geometry Voxelizer]
        ↓
[Voxel Encoder] ← OBJ meshes
        ↓
[Image Preprocessor]
        ↓
IKEADataset(Dataset)
        ↓
[DataLoader]
        ↓
[IKEAMEPNet]
        ├→ Image Encoder (ResNet)
        ├→ Voxel Encoder (3D Conv)
        ├→ Heatmap Head → Part Detection
        ├→ Rotation Head → Orientation  
        └→ Translation Head → 6D Pose
        ↓
[Loss Functions]
        ↓
[Training Loop]
```

---

## Part 6: Dependencies Alignment

### Conflicts to Resolve

1. **PyTorch3D Version**: 
   - LEGO requires: 0.5.0
   - IKEA uses: Not explicit, likely 0.6+
   - **Solution**: Pin to 0.5.0, verify API compatibility

2. **Open3D**:
   - LEGO: Not required
   - IKEA: 0.15.2
   - **Solution**: Add to lego_release requirements

3. **NumPy**:
   - LEGO: 1.21.2
   - IKEA: >= 1.18.5
   - **Solution**: Use 1.21.2 (backward compatible)

### Combined Requirements

```
torch==1.8.0 or 1.10.0
torchvision==0.9.0 or 0.11.0
pytorch3d==0.5.0
trimesh==3.9.34
numpy==1.21.2
opencv-python==4.5.4.58
scipy==1.7.1
h5py==3.2.1
open3d==0.15.2
pyrender>=0.1.43
pybullet==3.1.7
```

---

## Conclusion

The **lego_release** and **IKEA-Manuals-at-Work** repositories have complementary strengths:

1. **LEGO**: Sophisticated MEPNet architecture for assembly prediction with careful handling of discrete brick types and spatial relationships

2. **IKEA**: Rich real-world dataset with 6D poses, camera calibration, and diverse furniture categories

**Key Challenges for Integration**:
- Discrete (LEGO) to continuous (IKEA) pose representation
- Fixed brick types to variable part geometry
- Synthetic data to real-world complexity

**Next Steps**: 
1. Implement IKEA dataset loader (section 5.1)
2. Create adaptive voxelizer for arbitrary geometries
3. Modify MEPNet architecture for 6D pose regression
4. Implement training pipeline with IKEA data

