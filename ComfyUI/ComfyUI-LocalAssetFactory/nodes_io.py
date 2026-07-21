"""
ComfyUI-LocalAssetFactory · I/O Nodes
Utility nodes for input/output operations.
"""

from __future__ import annotations

import logging
import os
import torch
import numpy as np
from PIL import Image

import folder_paths

from .utilities.image_conversion import pil_to_comfy_tensor

log = logging.getLogger(__name__)

_CATEGORY = "Asset Factory/IO"

class LoadMultiviewDirectoryNode:
    """
    Loads up to 4 views from a directory based on filename suffixes,
    OR allows uploading/selecting individual files from the ComfyUI input folder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        return {
            "required": {
                "use_directory": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "directory_path": ("STRING", {"default": r"C:\path\to\folder"}),
            },
            "optional": {
                "front_image": (["none"] + sorted(files), {"image_upload": True}),
                "left_image": (["none"] + sorted(files), {"image_upload": True}),
                "right_image": (["none"] + sorted(files), {"image_upload": True}),
                "back_image": (["none"] + sorted(files), {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("FRONT_IMAGE", "LEFT_IMAGE", "RIGHT_IMAGE", "BACK_IMAGE")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, use_directory: bool, directory_path: str, front_image="none", left_image="none", right_image="none", back_image="none"):
        front = None
        left = None
        right = None
        back = None
        
        def load_img(path):
            try:
                img = Image.open(path).convert("RGB")
                return pil_to_comfy_tensor(img).squeeze(0)
            except Exception as e:
                log.error(f"Error loading {path}: {e}")
                return None

        # Option A: Directory based loading
        if use_directory and os.path.isdir(directory_path):
            files = os.listdir(directory_path)
            for f in files:
                f_lower = f.lower()
                if not (f_lower.endswith('.png') or f_lower.endswith('.jpg') or f_lower.endswith('.jpeg')):
                    continue
                    
                full_path = os.path.join(directory_path, f)
                if "_front" in f_lower or "front" in f_lower:
                    if front is None: front = load_img(full_path)
                elif "_left" in f_lower or "left" in f_lower:
                    if left is None: left = load_img(full_path)
                elif "_right" in f_lower or "right" in f_lower:
                    if right is None: right = load_img(full_path)
                elif "_back" in f_lower or "back" in f_lower:
                    if back is None: back = load_img(full_path)
                    
        # Option B: Individual file uploads (will override directory if specified)
        input_dir = folder_paths.get_input_directory()
        if front_image != "none": front = load_img(os.path.join(input_dir, front_image))
        if left_image != "none": left = load_img(os.path.join(input_dir, left_image))
        if right_image != "none": right = load_img(os.path.join(input_dir, right_image))
        if back_image != "none": back = load_img(os.path.join(input_dir, back_image))

        empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        front = front.unsqueeze(0) if front is not None else empty_tensor
        left = left.unsqueeze(0) if left is not None else empty_tensor
        right = right.unsqueeze(0) if right is not None else empty_tensor
        back = back.unsqueeze(0) if back is not None else empty_tensor

        return (front, left, right, back)


class SaveMeshToGLBNode:
    """Saves a MESH object to disk as a GLB file and outputs the file path string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "filename_prefix": ("STRING", {"default": "hunyuan3d_character"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def execute(self, mesh, filename_prefix: str = "hunyuan3d_character"):
        try:
            from comfy_extras.nodes_save_3d import save_glb, get_mesh_batch_item
        except ImportError:
            log.error("Could not import nodes_save_3d from comfy_extras")
            return ("",)

        output_dir = folder_paths.get_output_directory()
        out_folder = os.path.join(output_dir, "local_asset_factory")
        os.makedirs(out_folder, exist_ok=True)
        filepath = os.path.join(out_folder, f"{filename_prefix}.glb")

        try:
            vertices_i, faces_i, v_colors, uvs_i = get_mesh_batch_item(mesh, 0)
            save_glb(vertices_i, faces_i, filepath, uvs=uvs_i, vertex_colors=v_colors)
            log.info("Saved GLB mesh to %s", filepath)
            return (filepath,)
        except Exception as e:
            log.error("Failed to save GLB: %s", e)
            return ("",)


# ── ComfyUI Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AssetFactory_LoadMultiviewDirectory": LoadMultiviewDirectoryNode,
    "AssetFactory_SaveMeshToGLB": SaveMeshToGLBNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetFactory_LoadMultiviewDirectory": "Asset Factory · Load Multiview Directory",
    "AssetFactory_SaveMeshToGLB": "Asset Factory · Save Mesh to GLB",
}

