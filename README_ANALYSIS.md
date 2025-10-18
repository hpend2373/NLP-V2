# IKEA-MEPNet Adapter: Complete Repository Analysis

## Overview

This directory contains comprehensive analysis of the **lego_release** and **IKEA-Manuals-at-Work** repositories to guide the development of an adapter for applying MEPNet to IKEA assembly data.

## Analysis Documents

### 1. ANALYSIS_SUMMARY.md (Quick Reference)
**Size**: 185 lines | **Read Time**: 5-10 minutes

High-level overview containing:
- Executive summary of both repositories
- Key findings and strengths
- Critical integration points
- Implementation roadmap (Phase 1-3)
- Quick reference for dependencies
- File locations and next steps

**Start here** if you want a quick overview of what needs to be done.

### 2. REPOSITORY_ANALYSIS.md (Detailed Technical Reference)
**Size**: 761 lines | **Read Time**: 45-60 minutes

Comprehensive technical analysis containing:

**Part 1: LEGO Release Repository** (Sections 1.1-1.5)
- Complete directory structure
- MEPNet architecture deep dive
  - HourglassShapeCondModel (main model)
  - PyTorch3D integration
  - Voxel representation scheme
- Dataset structure (LegoKpsShapeCondDataset)
- Data format specifications
- Training pipeline details
- Hyperparameters and loss functions

**Part 2: IKEA-Manuals-at-Work Repository** (Sections 2.1-2.6)
- Complete directory structure
- Dataset structure and data.json format
- 3D model organization (OBJ files)
- Rendering utility (render_part.py)
- Camera parameter representation
- 6D pose specification
- Dataset statistics

**Part 3: Integration Challenges** (Section 3)
- Data format gaps (coordinate systems, brick vs parts, resolution)
- Voxel encoding differences
- Training data preprocessing requirements
- Loss function adaptations

**Part 4-6: Implementation Guide**
- File mapping summary
- Recommended integration architecture
- Dependencies alignment
- Conclusion and next steps

## Key Statistics

| Metric | Value |
|--------|-------|
| lego_release Lines | 15,038 |
| IKEA Lines | 22,122 |
| Total Code Analyzed | 37,160 |
| Analysis Document Lines | 761 |
| Summary Lines | 185 |
| Date | 2025-10-17 |

## Repository Locations

- **lego_release**: `/Users/minyeop/NLP/ikea_mepnet_adapter/lego_release/`
- **IKEA-Manuals-at-Work**: `/Users/minyeop/NLP/ikea_mepnet_adapter/IKEA-Manuals-at-Work/`
- **Analysis Docs**: `/Users/minyeop/NLP/ikea_mepnet_adapter/`

## Quick Navigation

### For Understanding MEPNet Architecture
- Read: REPOSITORY_ANALYSIS.md, Section 1.2 (MEPNet Architecture)
- Reference Files:
  - `/lego_release/models/hourglass_shape_cond_model.py`
  - `/lego_release/bricks/brick_info.py`

### For Understanding IKEA Dataset
- Read: REPOSITORY_ANALYSIS.md, Section 2 (IKEA Repository)
- Reference Files:
  - `/IKEA-Manuals-at-Work/data/data.json`
  - `/IKEA-Manuals-at-Work/render_part.py`

### For Implementation Strategy
- Read: ANALYSIS_SUMMARY.md, Section "Implementation Roadmap"
- Also: REPOSITORY_ANALYSIS.md, Section 5 (Recommended Integration Architecture)

### For Integration Challenges
- Read: REPOSITORY_ANALYSIS.md, Section 3 (Key Integration Points)
- Reference: REPOSITORY_ANALYSIS.md, Part 4 (File Mapping Summary)

## Critical Files Reference

### LEGO System (to adapt from)
| File | Lines | Purpose | Adaptation |
|------|-------|---------|-----------|
| `hourglass_shape_cond_model.py` | 823 | Main MEPNet | Modify heads for 6D pose |
| `legokps_shape_cond_dataset.py` | 833 | Data loading | Complete rewrite for IKEA |
| `brick_info.py` | 1752 | Brick system | Replace with part geometry |
| `train_kp.py` | 284 | Training loop | Minor modifications |

### IKEA Assets (to integrate)
| File | Size | Purpose | Integration |
|------|------|---------|-------------|
| `data.json` | 206MB | Annotations | Data source (Git-LFS) |
| `parts/*/` | ~ | 3D models | Shape conditioning input |
| `render_part.py` | 82 | Rendering | Projection & visualization |
| `datasheet.md` | - | Specs | Reference format |

## Implementation Phases

### Phase 1: Data Adapters (Foundation)
Create new modules to load and preprocess IKEA data:
1. `ikea_dataset.py` - Load data.json
2. `part_loader.py` - Load OBJ meshes
3. `voxelizer.py` - Voxelize arbitrary geometries
4. `preprocessing.py` - Image/annotation transforms

### Phase 2: Model Modifications (Architecture)
Adapt MEPNet for continuous 6D poses:
1. `ikea_model.py` - Adapted MEPNet
2. `part_encoder.py` - Geometry encoder
3. `pose_head.py` - 6D regression
4. Loss function adaptations

### Phase 3: Inference & Validation (Testing)
Complete the pipeline:
1. Inference implementation
2. Evaluation metrics
3. Visualization tools
4. Performance benchmarking

## Key Technical Insights

### MEPNet Design
- Shape-conditioned architecture (adapts to different part types)
- Multi-scale hourglass with residual connections
- Per-part-type heatmap generation
- Instance-aware assembly ordering
- Physics-based spatial constraints

### IKEA Data Richness
- 36 furniture models with 50-100 parts each
- 98 real assembly videos
- 34,441 annotated frames
- 6D pose + camera calibration per part
- Multi-part sequential assembly

### Integration Requirements
- Support arbitrary part geometry (not just discrete types)
- Continuous 6D pose regression (not discrete rotations)
- Dynamic voxel grids (not fixed 130^3)
- Real-world image complexity (not synthetic white backgrounds)
- Temporal assembly sequences (not static single steps)

## Dependencies Summary

### Core Libraries
- **PyTorch**: 1.8.0 - 1.10.0
- **PyTorch3D**: 0.5.0 (pinned)
- **OpenCV**: 4.5.4+
- **Open3D**: 0.15.2

### Supporting Libraries
- NumPy, SciPy, scikit-image
- PyRender, PyBullet (for 3D operations)
- PyYAML, h5py (for data I/O)

See REPOSITORY_ANALYSIS.md Section 6 for complete requirements.

## Next Steps

1. **Read the Analysis Documents**
   - Start with ANALYSIS_SUMMARY.md (10 min)
   - Then read REPOSITORY_ANALYSIS.md in detail (60 min)

2. **Explore the Repositories**
   - Study key files referenced in the analysis
   - Run data inspection scripts on IKEA dataset
   - Review MEPNet architecture in detail

3. **Begin Implementation**
   - Start with Phase 1 (data adapters)
   - Use LEGO dataset loader as template
   - Implement IKEA-specific components

4. **Reference and Validate**
   - Cross-reference with original papers
   - Validate data formats with actual files
   - Test adapters with sample data

## Document Structure

Both analysis documents follow this structure:

```
Analysis Document
├── Part 1: lego_release Details
│   ├── Directory structure
│   ├── Model architecture
│   ├── Dataset format
│   ├── Training pipeline
│   └── Dependencies
├── Part 2: IKEA Details
│   ├── Directory structure
│   ├── Dataset format
│   ├── 3D model organization
│   ├── Rendering utilities
│   └── Dependencies
├── Part 3: Integration
│   ├── Data format gaps
│   ├── Architecture differences
│   ├── Preprocessing needs
│   └── Loss function changes
├── Part 4-6: Implementation
│   ├── File mapping
│   ├── Recommended architecture
│   ├── Dependency resolution
│   └── Conclusion
```

## How to Use This Analysis

### For Architecture Understanding
Sections to read in order:
1. ANALYSIS_SUMMARY.md - "MEPNet Architecture Overview"
2. REPOSITORY_ANALYSIS.md - Section 1.2 (Core Model Architecture)
3. Reference: `lego_release/models/hourglass_shape_cond_model.py`

### For Data Format Understanding
Sections to read in order:
1. REPOSITORY_ANALYSIS.md - Section 1.3 (LegoKpsShapeCondDataset)
2. REPOSITORY_ANALYSIS.md - Section 2.2 (IKEA Dataset Structure)
3. REPOSITORY_ANALYSIS.md - Section 2.3 (Rendering Utility)

### For Implementation Planning
Sections to read in order:
1. ANALYSIS_SUMMARY.md - "Implementation Roadmap"
2. REPOSITORY_ANALYSIS.md - Section 3 (Integration Challenges)
3. REPOSITORY_ANALYSIS.md - Section 5 (Recommended Integration Architecture)

## Questions and References

For questions about specific components:
- **MEPNet Architecture**: REPOSITORY_ANALYSIS.md, Section 1.2
- **Data Loading**: REPOSITORY_ANALYSIS.md, Section 1.3
- **IKEA Format**: REPOSITORY_ANALYSIS.md, Section 2.2-2.4
- **Integration**: REPOSITORY_ANALYSIS.md, Section 3-5

For file references:
- **LEGO**: REPOSITORY_ANALYSIS.md, Part 4, Table 1
- **IKEA**: REPOSITORY_ANALYSIS.md, Part 4, Table 2
- **Locations**: REPOSITORY_ANALYSIS.md, Section 2.6

---

**Analysis Completed**: 2025-10-17
**Repositories Analyzed**: lego_release (15,038 lines) + IKEA-Manuals-at-Work (22,122 lines)
**Total Analysis**: 946 documentation lines

For detailed technical specifications, see REPOSITORY_ANALYSIS.md
For quick reference and roadmap, see ANALYSIS_SUMMARY.md
