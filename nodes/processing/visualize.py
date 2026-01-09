# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Visualization nodes for SAM 3D Body outputs.

Provides nodes for rendering and visualizing 3D mesh reconstructions.
"""
import os
import json
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
import folder_paths
from ..base import numpy_to_comfy_image
from server import PromptServer
from comfy_api.latest import IO, ComfyExtension, InputImpl, UI
# Add sam-3d-body to Python path if it exists
_SAM3D_BODY_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "sam-3d-body"
if _SAM3D_BODY_PATH.exists() and str(_SAM3D_BODY_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3D_BODY_PATH))




class SAM3DBodyVisualize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {}),
                "image": ("IMAGE", {}),
                "render_mode": (["overlay", "mesh_only", "side_by_side"], {"default": "overlay"}),
            },
            "optional": {
                "camera_info": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "visualize"
    CATEGORY = "SAM3DBody/visualization"
    
    def visualize(self, mesh_data, image, unique_id, render_mode="overlay", camera_info=""):
            # 1. Setup Canvas
            try:
                img_rgb = image[0].cpu().numpy()
                h, w = img_rgb.shape[:2]
                cx, cy = w / 2.0, h / 2.0 
    
                if render_mode == "mesh_only":
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    bg_image = None
                else:
                    canvas = (img_rgb * 255.0).astype(np.uint8).copy()
                    bg_image = canvas.copy()
            except Exception as e:
                print(f"[SAM3DBody] Setup failed: {e}")
                return (image,)
    
            # 2. Extract Data
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            if vertices is None or faces is None: return (image,)
            
            # Copy vertices to avoid graph mutation
            if hasattr(vertices, "cpu"): vertices = vertices.cpu().numpy().copy()
            else: vertices = vertices.copy()
                
            if hasattr(faces, "cpu"): faces = faces.cpu().numpy()
    
            # Extract Original Metadata
            camera_raw = mesh_data.get("camera", {})
            f_meta = mesh_data.get("focal_length", None)
            
            t_raw = np.array([0., 0., 0.])
            if isinstance(camera_raw, dict):
                t_raw = np.array(camera_raw.get("translation", [0., 0., 0.]))
            elif camera_raw is not None:
                t_raw = np.array(camera_raw)
            if hasattr(t_raw, "cpu"): t_raw = t_raw.cpu().numpy()
            
            # Focal Length default
            if f_meta is None and isinstance(camera_raw, dict):
                f_meta = camera_raw.get("focal_length", None)
            if hasattr(f_meta, "item"): f_val = f_meta.item()
            elif hasattr(f_meta, "cpu"): f_val = f_meta.cpu().numpy()
            else: f_val = float(f_meta) if f_meta else 5000.0
    
            # --- SYNC: Send Metadata to Viewport ---
            PromptServer.instance.send_sync("sam3d_mesh_update", {
                "node_id": unique_id,
                "vertices": vertices.flatten().tolist(),
                "faces": faces.flatten().tolist(),
                "cam_pos": t_raw.tolist(), 
                "focal_length": f_val,
                "width": w,
                "height": h
            })
    
            # =========================================================
            # COORDINATE PROCESSING
            # =========================================================
            use_manual = False
            manual_params = {}
            
            if camera_info and camera_info.strip():
                try:
                    manual_params = json.loads(camera_info)
                    if "position" in manual_params: use_manual = True
                except: pass
    
            if use_manual:
                # --- BRANCH A: MANUAL INTERACTIVE MODE ---
                p = manual_params.get("position", {})
                t = manual_params.get("target", {})
                fov_deg = manual_params.get("fov", 30.0)
                
                # 1. Replicate JS Mesh Rotation (Fixes Vertical Flip)
                verts_world = vertices.copy()
                verts_world[:, 1] *= -1.0
                verts_world[:, 2] *= -1.0
                
                # 2. Parse JS Camera WITH X INVERSION (Fixes Horizontal Flip)
                # We negate X to match the panning direction of the renderer
                cam_pos = np.array([-p.get("x", 0), p.get("y", 0), p.get("z", 2)])
                cam_target = np.array([-t.get("x", 0), t.get("y", 0), t.get("z", 0)])
    
                # 3. Build Rotation Matrix (LookAt)
                view_vec = cam_target - cam_pos
                dist = np.linalg.norm(view_vec)
                if dist > 1e-9: view_vec /= dist
                else: view_vec = np.array([0., 0., -1.])
                
                z_axis = -view_vec 
                
                up_ref = np.array([0., 1., 0.])
                x_axis = np.cross(up_ref, z_axis)
                x_axis /= (np.linalg.norm(x_axis) + 1e-9)
                
                y_axis = np.cross(z_axis, x_axis)
                
                R = np.vstack([x_axis, y_axis, z_axis])
                
                # 4. Transform Vertices
                v_centered = verts_world - cam_pos
                v_view_gl = v_centered @ R.T 
                
                # 5. Convert to CV Screen Space
                v_view = v_view_gl.copy()
                v_view[:, 1] *= -1.0 
                v_view[:, 2] *= -1.0 
    
                # 6. Calculate Focal Length
                f = h / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    
            else:
                # --- BRANCH B: LEGACY PIPELINE ---
                f = f_val
                if f < 10.0: f *= max(h, w)
                
                cam_world_pos = t_raw.copy()
                cam_world_pos[0] *= -1.0
                
                verts_world = vertices.copy()
                verts_world[:, 1] *= -1.0
                verts_world[:, 2] *= -1.0
                
                v_gl_view = verts_world - cam_world_pos
                
                v_view = v_gl_view.copy()
                v_view[:, 1] *= -1.0 
                v_view[:, 2] *= -1.0 
    
            # =========================================================
            # RENDER LOOP
            # =========================================================
            z_depth = v_view[:, 2]
            z_safe = np.maximum(z_depth, 0.1)
    
            screen_x = (f * v_view[:, 0] / z_safe) + cx
            screen_y = (f * v_view[:, 1] / z_safe) + cy 
            
            points_2d = np.stack([screen_x, screen_y], axis=1).astype(np.int32)
            
            face_verts = v_view[faces] 
            edge1 = face_verts[:, 1] - face_verts[:, 0]
            edge2 = face_verts[:, 2] - face_verts[:, 0]
            normals = np.cross(edge1, edge2)
            normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
            
            intensity = np.abs(normals[:, 2]) 
            intensity = np.clip(intensity, 0.3, 1.0)
            
            face_depths = np.mean(z_depth[faces], axis=1)
            sort_idx = np.argsort(face_depths)[::-1] 
            
            mesh_layer = np.zeros_like(canvas)
            base_color = np.array([200, 200, 200]) 
            faces_2d = points_2d[faces]
            
            img_h, img_w = canvas.shape[:2]
    
            for idx in sort_idx:
                if face_depths[idx] < 0.1: continue
                
                cnt = faces_2d[idx]
                if np.all(cnt[:,0] < 0) or np.all(cnt[:,0] > img_w) or \
                np.all(cnt[:,1] < 0) or np.all(cnt[:,1] > img_h):
                    continue
                
                color = (base_color * intensity[idx]).astype(np.uint8)
                cv2.fillPoly(mesh_layer, [cnt], color=color.tolist(), lineType=cv2.LINE_AA)
    
            if render_mode == "mesh_only":
                final_img = mesh_layer
            elif render_mode == "side_by_side":
                final_img = np.hstack([bg_image, mesh_layer])
            else: 
                mask = np.any(mesh_layer > 0, axis=-1)
                final_img = bg_image.copy()
                alpha = 0.7
                final_img[mask] = (mesh_layer[mask] * alpha + bg_image[mask] * (1-alpha)).astype(np.uint8)
                
            return (numpy_to_comfy_image(final_img),)
    

class SAM3DBodyExportMesh:
    """
    Exports SAM 3D Body mesh to STL format.

    Saves the reconstructed 3D mesh as ASCII STL for use in 3D viewers and editors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "filename": ("STRING", {
                    "default": "output_mesh.stl",
                    "tooltip": "Output filename (exports as ASCII STL)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export_mesh"
    CATEGORY = "SAM3DBody/io"

    def export_mesh(self, mesh_data, filename="output_mesh.stl"):
        """Export mesh to file."""

        print(f"[SAM3DBody] Exporting mesh to {filename}")

        try:
            import os

            # Use ComfyUI's output directory
            output_dir = os.path.join(folder_paths.get_input_directory(), "3d")
            full_path = os.path.join(output_dir, filename)

            # Extract mesh data
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)

            if vertices is None or faces is None:
                raise ValueError("No mesh data available to export")

            # Convert to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            # Export to STL format
            self._export_stl(vertices, faces, full_path)

            print(f"[SAM3DBody] [OK] Mesh exported to {full_path}")
            return (filename,)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _export_obj(self, vertices, faces, filepath):
        """Export mesh to OBJ format."""
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _export_ply(self, vertices, faces, filepath):
        """Export mesh to PLY format."""
        with open(filepath, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _export_stl(self, vertices, faces, filepath):
        """Export mesh to ASCII STL format."""
        import numpy as np

        # Apply 180° X-rotation to undo MHR coordinate transform (flip both Y and Z)
        # This matches what the renderer does for visualization
        vertices_flipped = vertices.copy()
        vertices_flipped[:, 1] = -vertices_flipped[:, 1]  # Flip Y
        vertices_flipped[:, 2] = -vertices_flipped[:, 2]  # Flip Z

        with open(filepath, 'w') as f:
            # Write STL header
            f.write("solid mesh\n")

            # Write each triangle face
            for face in faces:
                # Get the three vertices of the triangle
                v0 = vertices_flipped[int(face[0])]
                v1 = vertices_flipped[int(face[1])]
                v2 = vertices_flipped[int(face[2])]

                # Calculate face normal using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)

                # Normalize the normal vector
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0.0, 0.0, 1.0])  # Default normal if degenerate

                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            # Write STL footer
            f.write("endsolid mesh\n")


class SAM3DBodyGetVertices:
    """
    Extracts vertex data from SAM 3D Body output.

    Useful for custom processing or analysis of the reconstructed mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_vertices"
    CATEGORY = "SAM3DBody/utilities"

    def get_vertices(self, mesh_data):
        """Extract and display vertex information."""

        try:
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            joints = mesh_data.get("joints", None)

            info_lines = ["[SAM3DBody] Mesh Information:"]

            if vertices is not None:
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                info_lines.append(f"Vertices: {len(vertices)} points")
                info_lines.append(f"Vertex shape: {vertices.shape}")

            if faces is not None:
                if isinstance(faces, torch.Tensor):
                    faces = faces.cpu().numpy()
                info_lines.append(f"Faces: {len(faces)} triangles")

            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                info_lines.append(f"Joints: {len(joints)} keypoints")

            info = "\n".join(info_lines)
            print(info)

            return (info,)

        except Exception as e:
            error_msg = f"[SAM3DBody] [ERROR] Failed to get mesh info: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyVisualize": SAM3DBodyVisualize,
    "SAM3DBodyExportMesh": SAM3DBodyExportMesh,
    "SAM3DBodyGetVertices": SAM3DBodyGetVertices,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyVisualize": "SAM 3D Body: Visualize Mesh",
    "SAM3DBodyExportMesh": "SAM 3D Body: Export Mesh",
    "SAM3DBodyGetVertices": "SAM 3D Body: Get Mesh Info",
}
