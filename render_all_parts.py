import os
import sys
import glob
import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image
from tqdm import tqdm

def render_mesh(mesh_path, output_path, azimuth=0.0, width=512, height=512):
    """Render a single mesh to an image with specific azimuth."""
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Normalize mesh to unit sphere
        vertices = mesh.vertices
        center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2
        scale = 1.0 / np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        mesh.vertices = (mesh.vertices - center) * scale
        
        # Create pyrender mesh
        mesh_pr = pyrender.Mesh.from_trimesh(mesh)
        
        # Scene
        scene = pyrender.Scene(bg_color=[255, 255, 255])
        scene.add(mesh_pr)
        
        # Camera Pose (Orbit)
        radius = 1.5
        x = radius * np.sin(azimuth)
        z = radius * np.cos(azimuth)
        
        # LookAt Matrix
        # Forward vector (camera to origin) = (-x, 0, -z)
        f = -np.array([x, 0, z])
        f = f / np.linalg.norm(f)
        
        # Up vector
        u = np.array([0.0, 1.0, 0.0])
        
        # Right vector
        r = np.cross(f, u)
        r = r / np.linalg.norm(r)
        
        # Recompute Up
        u = np.cross(r, f)
        
        # Camera matrix (columns: r, u, -f, pos)
        # PyRender/OpenGL camera looks down -Z. So -Z should be f. Z should be -f.
        pose = np.eye(4)
        pose[:3, 0] = r
        pose[:3, 1] = u
        pose[:3, 2] = -f
        pose[:3, 3] = [x, 0, z]
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=pose)
        
        # Light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(light, pose=pose)
        
        # Render
        r = pyrender.OffscreenRenderer(width, height)
        color, depth = r.render(scene)
        
        # Save
        im = Image.fromarray(color)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        im.save(output_path)
        
        # Cleanup
        r.delete()
        
        return True
    except Exception as e:
        # print(f"Failed to render {mesh_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Render all IKEA parts to 2D images")
    parser.add_argument("--parts_dir", type=str, default="/Users/minyeop/NLP/NLP-V2/IKEA-Manuals-at-Work/data/parts", help="Input parts directory")
    parser.add_argument("--output_dir", type=str, default="/Users/minyeop/NLP/NLP-V2/IKEA-Manuals-at-Work/data/rendered_parts", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of parts to render")
    parser.add_argument("--filter", type=str, default=None, help="Filter parts by path (e.g. 'applaro')")
    args = parser.parse_args()

    # Find all .obj files
    obj_files = glob.glob(os.path.join(args.parts_dir, "**/*.obj"), recursive=True)
    
    if args.filter:
        obj_files = [f for f in obj_files if args.filter.lower() in f.lower()]
        
    print(f"Found {len(obj_files)} parts.")
    
    if args.limit:
        obj_files = obj_files[:args.limit]
        print(f"Limiting to {args.limit} parts.")

    success_count = 0
    for obj_path in tqdm(obj_files, desc="Rendering"):
        # Construct output path preserving structure
        rel_path = os.path.relpath(obj_path, args.parts_dir)
        base_output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0])
        
        # Render 4 views
        views = [0, 90, 180, 270]
        part_success = False
        
        for i, azimuth_deg in enumerate(views):
            output_path = f"{base_output_path}_v{i}.png"
            
            if os.path.exists(output_path):
                part_success = True
                continue
            
            azimuth = np.radians(azimuth_deg)
            if render_mesh(obj_path, output_path, azimuth=azimuth):
                part_success = True
            else:
                # Fallback to simple matplotlib projection if pyrender fails (only for v0)
                if i == 0:
                    try:
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        
                        mesh = trimesh.load(obj_path, force='mesh')
                        vertices = mesh.vertices
                        # Center and scale
                        center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2
                        scale = 1.0 / np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
                        vertices = (vertices - center) * scale
                        
                        fig = plt.figure(figsize=(5, 5))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.view_init(elev=30, azim=45)
                        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='gray')
                        ax.axis('off')
                        
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                        part_success = True
                    except Exception as e2:
                        pass

        if part_success:
            success_count += 1

    print(f"Rendered {success_count}/{len(obj_files)} parts.")

if __name__ == "__main__":
    main()
