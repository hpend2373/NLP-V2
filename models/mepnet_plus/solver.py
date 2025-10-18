"""
MEPNet+ Pose Solver
Combines MEPNet perception with constraint-based refinement and analysis-by-synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import trimesh
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation
import warnings

# PyTorch3D imports (for rendering)
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftSilhouetteShader,
        TexturesVertex
    )
    from pytorch3d.ops import chamfer_distance
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    warnings.warn("PyTorch3D not available. Rendering-based refinement disabled.")


class PoseSolver:
    """
    Main pose solver combining:
    1. Coarse pose from MEPNet
    2. Constraint-based candidate generation
    3. Analysis-by-synthesis refinement
    """

    def __init__(
        self,
        perception_model: nn.Module,
        constraint_engine: Any,
        assets_registry: Any,
        device: str = 'cuda',
        use_rendering: bool = True,
        refinement_iterations: int = 10
    ):
        """
        Args:
            perception_model: MEPNet perception model
            constraint_engine: Constraint engine instance
            assets_registry: Assets registry instance
            device: Device for computation
            use_rendering: Whether to use rendering-based refinement
            refinement_iterations: Number of refinement iterations
        """
        self.perception_model = perception_model
        self.constraint_engine = constraint_engine
        self.registry = assets_registry
        self.device = device
        self.use_rendering = use_rendering and PYTORCH3D_AVAILABLE
        self.refinement_iterations = refinement_iterations

        # Initialize renderer if available
        if self.use_rendering:
            self.renderer = DifferentiableRenderer(device=device)

        # Move model to device
        self.perception_model = self.perception_model.to(device)
        self.perception_model.eval()

    def solve(
        self,
        manual_image: torch.Tensor,
        part_shapes: Optional[List[Any]] = None,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        assembly_state: Optional[Any] = None,
        part_hints: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete pose solving pipeline

        Args:
            manual_image: Input manual image (B, 3, H, W)
            part_shapes: List of part shapes (meshes or point clouds)
            camera_params: Camera parameters
            assembly_state: Current assembly state
            part_hints: Hints about which parts to detect

        Returns:
            List of solved poses with metadata
        """
        B = manual_image.shape[0]
        results = []

        # Move inputs to device
        manual_image = manual_image.to(self.device)

        for b in range(B):
            batch_result = self._solve_single(
                manual_image[b:b+1],
                part_shapes[b] if part_shapes else None,
                {k: v[b:b+1] for k, v in camera_params.items()} if camera_params else None,
                assembly_state,
                part_hints
            )
            results.append(batch_result)

        return results

    def _solve_single(
        self,
        image: torch.Tensor,
        part_shapes: Optional[Any],
        camera: Optional[Dict[str, torch.Tensor]],
        assembly_state: Optional[Any],
        part_hints: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Solve for a single image"""

        # Step 1: Get coarse poses from perception model
        coarse_poses = self._get_coarse_poses(image, part_shapes, camera)

        # Step 2: Generate constraint-based candidates
        candidates = self._generate_candidates(coarse_poses, assembly_state, part_hints)

        # Step 3: Refine poses
        refined_poses = []
        for candidate in candidates:
            if self.use_rendering:
                refined = self._refine_with_rendering(
                    candidate,
                    image,
                    camera,
                    assembly_state
                )
            else:
                refined = self._refine_with_optimization(
                    candidate,
                    coarse_poses,
                    assembly_state
                )
            refined_poses.append(refined)

        # Step 4: Rank and filter
        final_poses = self._rank_and_filter(refined_poses, image, camera)

        return final_poses

    def _get_coarse_poses(
        self,
        image: torch.Tensor,
        part_shapes: Optional[Any],
        camera: Optional[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, Any]]:
        """Get initial poses from perception model"""

        with torch.no_grad():
            # Forward through perception model
            outputs = self.perception_model(image, part_shapes)

            # Decode predictions
            coarse_poses = self.perception_model.decode_predictions(
                outputs,
                camera,
                threshold=0.3
            )

        return coarse_poses[0] if coarse_poses else []

    def _generate_candidates(
        self,
        coarse_poses: List[Dict],
        assembly_state: Optional[Any],
        part_hints: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Generate pose candidates using constraints"""

        candidates = []

        for pose in coarse_poses:
            # Get part ID (map from detection to actual part)
            if part_hints and pose['part_id'] < len(part_hints):
                part_id = part_hints[pose['part_id']]
            else:
                part_id = f"part_{pose['part_id']}"

            # Use constraint engine to get valid poses
            constraint_poses = self.constraint_engine.get_valid_poses(
                part_id,
                assembly_state,
                max_candidates=5
            )

            # Merge coarse pose with constraint-based poses
            for R, t, score in constraint_poses:
                candidate = {
                    'part_id': part_id,
                    'R_coarse': pose['R'],
                    't_coarse': pose['t'],
                    'R': R,
                    't': t,
                    'score': score * pose['score'],
                    'keypoints_2d': pose.get('keypoints_2d'),
                    'mask': pose.get('mask')
                }
                candidates.append(candidate)

            # Also keep original coarse pose as candidate
            candidates.append({
                'part_id': part_id,
                'R_coarse': pose['R'],
                't_coarse': pose['t'],
                'R': pose['R'],
                't': pose['t'],
                'score': pose['score'],
                'keypoints_2d': pose.get('keypoints_2d'),
                'mask': pose.get('mask')
            })

        return candidates

    def _refine_with_rendering(
        self,
        candidate: Dict[str, Any],
        image: torch.Tensor,
        camera: Dict[str, torch.Tensor],
        assembly_state: Optional[Any]
    ) -> Dict[str, Any]:
        """Refine pose using differentiable rendering"""

        if not self.use_rendering:
            return candidate

        # Get part mesh
        part_info = self.registry.get_part(candidate['part_id'])
        if part_info is None:
            return candidate

        mesh = part_info.mesh

        # Convert to PyTorch3D mesh
        verts = torch.from_numpy(mesh.vertices).float().to(self.device)
        faces = torch.from_numpy(mesh.faces).long().to(self.device)

        # Initial pose
        R = torch.from_numpy(candidate['R']).float().to(self.device)
        t = torch.from_numpy(candidate['t']).float().to(self.device)

        # Make pose parameters learnable
        R_param = nn.Parameter(matrix_to_rotation_6d(R))
        t_param = nn.Parameter(t)

        # Optimizer
        optimizer = torch.optim.Adam([R_param, t_param], lr=0.01)

        # Refinement loop
        best_loss = float('inf')
        best_R, best_t = R.clone(), t.clone()

        for iter in range(self.refinement_iterations):
            optimizer.zero_grad()

            # Convert 6D rotation back to matrix
            R_current = rotation_6d_to_matrix(R_param)

            # Transform vertices
            verts_transformed = verts @ R_current.T + t_param

            # Render
            rendered = self.renderer.render(
                verts_transformed.unsqueeze(0),
                faces.unsqueeze(0),
                camera
            )

            # Compute loss
            loss = self._compute_rendering_loss(
                rendered,
                image,
                candidate.get('mask')
            )

            # Add regularization
            # Penalize large deviations from coarse pose
            R_coarse = torch.from_numpy(candidate['R_coarse']).float().to(self.device)
            t_coarse = torch.from_numpy(candidate['t_coarse']).float().to(self.device)

            reg_loss = 0.1 * torch.norm(R_current - R_coarse) + 0.1 * torch.norm(t_param - t_coarse)
            loss = loss + reg_loss

            # Backward
            loss.backward()
            optimizer.step()

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_R = R_current.clone()
                best_t = t_param.clone()

        # Update candidate with refined pose
        candidate['R'] = best_R.detach().cpu().numpy()
        candidate['t'] = best_t.detach().cpu().numpy()
        candidate['refinement_loss'] = best_loss

        return candidate

    def _refine_with_optimization(
        self,
        candidate: Dict[str, Any],
        coarse_poses: List[Dict],
        assembly_state: Optional[Any]
    ) -> Dict[str, Any]:
        """Refine pose using scipy optimization (no rendering)"""

        part_info = self.registry.get_part(candidate['part_id'])
        if part_info is None:
            return candidate

        # Define objective function
        def objective(params):
            # Unpack parameters (rotation as axis-angle, translation)
            rot_vec = params[:3]
            trans = params[3:6]

            # Convert to rotation matrix
            R = Rotation.from_rotvec(rot_vec).as_matrix()

            # Compute cost
            cost = 0.0

            # Distance from coarse pose
            R_coarse = candidate['R_coarse']
            t_coarse = candidate['t_coarse']

            rot_diff = np.linalg.norm(R - R_coarse, 'fro')
            trans_diff = np.linalg.norm(trans - t_coarse)

            cost += rot_diff + trans_diff

            # Collision penalty
            if assembly_state:
                collision = self._check_collision_soft(
                    part_info.mesh,
                    R,
                    trans,
                    assembly_state
                )
                cost += 10.0 * collision

            # Connection alignment bonus
            connection_score = self._score_connections(
                part_info,
                R,
                trans,
                assembly_state
            )
            cost -= connection_score

            return cost

        # Initial parameters
        R_init = candidate['R']
        t_init = candidate['t']
        rot_vec_init = Rotation.from_matrix(R_init).as_rotvec()
        params_init = np.concatenate([rot_vec_init, t_init])

        # Bounds
        bounds = [
            (-np.pi, np.pi),  # rotation x
            (-np.pi, np.pi),  # rotation y
            (-np.pi, np.pi),  # rotation z
            (t_init[0] - 0.1, t_init[0] + 0.1),  # translation x
            (t_init[1] - 0.1, t_init[1] + 0.1),  # translation y
            (t_init[2] - 0.1, t_init[2] + 0.1),  # translation z
        ]

        # Optimize
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}
        )

        # Update candidate
        if result.success:
            rot_vec = result.x[:3]
            trans = result.x[3:6]
            candidate['R'] = Rotation.from_rotvec(rot_vec).as_matrix()
            candidate['t'] = trans
            candidate['optimization_cost'] = result.fun

        return candidate

    def _compute_rendering_loss(
        self,
        rendered: Dict[str, torch.Tensor],
        target_image: torch.Tensor,
        target_mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute loss between rendered and target images"""

        loss = 0.0

        # Silhouette loss
        if 'silhouette' in rendered and target_mask is not None:
            target_mask_torch = torch.from_numpy(target_mask).float().to(self.device)
            silhouette_loss = F.binary_cross_entropy(
                rendered['silhouette'].squeeze(),
                target_mask_torch
            )
            loss += silhouette_loss

        # RGB loss (if available)
        if 'rgb' in rendered:
            # Simple L2 loss on non-background pixels
            rgb_loss = F.mse_loss(rendered['rgb'], target_image)
            loss += 0.1 * rgb_loss

        # Depth loss (if available)
        if 'depth' in rendered and 'target_depth' in rendered:
            depth_loss = F.l1_loss(rendered['depth'], rendered['target_depth'])
            loss += 0.1 * depth_loss

        return loss

    def _check_collision_soft(
        self,
        mesh: trimesh.Trimesh,
        R: np.ndarray,
        t: np.ndarray,
        assembly_state: Any
    ) -> float:
        """Soft collision check returning continuous penalty"""

        if not assembly_state or not assembly_state.assembled_parts:
            return 0.0

        # Transform mesh
        mesh_transformed = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        mesh_transformed.apply_transform(transform)

        total_penetration = 0.0

        # Check against each assembled part
        for part_data in assembly_state.assembled_parts.values():
            assembled_mesh = part_data['mesh']

            # Quick AABB check
            if not mesh_transformed.bounds_overlap(assembled_mesh.bounds):
                continue

            # Sample points and check distances
            sample_points = mesh_transformed.sample(100)
            distances = assembled_mesh.nearest.signed_distance(sample_points)

            # Penalize negative distances (penetration)
            penetration = np.maximum(-distances, 0)
            total_penetration += np.sum(penetration)

        return total_penetration / 100.0  # Normalize

    def _score_connections(
        self,
        part_info: Any,
        R: np.ndarray,
        t: np.ndarray,
        assembly_state: Any
    ) -> float:
        """Score based on connection alignment"""

        if not assembly_state or not part_info.connection_points:
            return 0.0

        score = 0.0

        # Transform connection points
        for conn in part_info.connection_points:
            conn_pos_world = R @ conn.position + t
            conn_normal_world = R @ conn.normal

            # Find nearest connection in assembly
            min_dist = float('inf')
            for assembled_id, assembled_data in assembly_state.assembled_parts.items():
                assembled_info = self.registry.get_part(assembled_id)
                if not assembled_info:
                    continue

                assembled_R = assembled_data['pose']['R']
                assembled_t = assembled_data['pose']['t']

                for assembled_conn in assembled_info.connection_points:
                    # Transform to world
                    assembled_pos = assembled_R @ assembled_conn.position + assembled_t
                    assembled_normal = assembled_R @ assembled_conn.normal

                    # Check alignment
                    dist = np.linalg.norm(conn_pos_world - assembled_pos)
                    normal_alignment = -np.dot(conn_normal_world, assembled_normal)

                    if dist < min_dist and normal_alignment > 0.8:
                        min_dist = dist
                        score += np.exp(-dist / 0.01) * normal_alignment

        return score

    def _rank_and_filter(
        self,
        poses: List[Dict[str, Any]],
        image: torch.Tensor,
        camera: Optional[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, Any]]:
        """Rank and filter final poses"""

        # Sort by combined score
        for pose in poses:
            # Combine different scores
            combined_score = pose['score']

            if 'refinement_loss' in pose:
                combined_score *= np.exp(-pose['refinement_loss'])

            if 'optimization_cost' in pose:
                combined_score *= np.exp(-pose['optimization_cost'])

            pose['final_score'] = combined_score

        # Sort by final score
        poses.sort(key=lambda x: x['final_score'], reverse=True)

        # Filter overlapping poses
        filtered = []
        for pose in poses:
            # Check if too similar to already selected poses
            is_duplicate = False
            for selected in filtered:
                if selected['part_id'] == pose['part_id']:
                    # Check pose similarity
                    R_diff = np.linalg.norm(selected['R'] - pose['R'], 'fro')
                    t_diff = np.linalg.norm(selected['t'] - pose['t'])

                    if R_diff < 0.1 and t_diff < 0.01:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered.append(pose)

        return filtered


class DifferentiableRenderer:
    """Differentiable renderer using PyTorch3D"""

    def __init__(self, device: str = 'cuda', image_size: int = 512):
        self.device = device
        self.image_size = image_size

        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D required for differentiable rendering")

        # Rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True
        )

    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Render mesh with given camera parameters

        Args:
            vertices: (B, V, 3) vertices
            faces: (B, F, 3) faces
            camera_params: Camera parameters dict

        Returns:
            Rendered outputs dict
        """
        B = vertices.shape[0]

        # Create meshes
        meshes = Meshes(verts=vertices, faces=faces)

        # Setup camera
        K = camera_params.get('K')  # (B, 3, 3)
        R = camera_params.get('R', torch.eye(3).expand(B, -1, -1).to(self.device))
        t = camera_params.get('t', torch.zeros(B, 3).to(self.device))

        # Extract camera parameters
        if K is not None:
            focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
            principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
        else:
            focal_length = torch.ones(B, 2).to(self.device) * self.image_size
            principal_point = torch.ones(B, 2).to(self.device) * self.image_size / 2

        cameras = PerspectiveCameras(
            R=R,
            T=t,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((self.image_size, self.image_size),) * B,
            device=self.device
        )

        # Create renderer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        # Render silhouette
        shader = SoftSilhouetteShader()
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        )

        # Add simple texture (white)
        verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(verts=vertices, faces=faces, textures=textures)

        # Render
        images = renderer(meshes)

        outputs = {
            'silhouette': images[..., 3],  # Alpha channel
            'rgb': images[..., :3],  # RGB channels
        }

        # Also get depth if needed
        fragments = rasterizer(meshes)
        outputs['depth'] = fragments.zbuf[..., 0]

        return outputs


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation"""
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to matrix"""
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)