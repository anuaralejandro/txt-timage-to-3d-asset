"""
ComfyUI-LocalAssetFactory · Image Processing Nodes
Utility nodes for pre-processing images (e.g. background removal).
"""

from __future__ import annotations

import logging
import torch
import numpy as np
from PIL import Image

try:
    import rembg
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

from local_asset_factory.preflight.alpha_detector import detect_alpha
from local_asset_factory.preflight.checkerboard_detector import detect_checkerboard
from .utilities.image_conversion import comfy_tensor_to_pil, pil_to_comfy_tensor

log = logging.getLogger(__name__)

_CATEGORY = "Asset Factory/Image"


class RemoveFakeBackgroundNode:
    """
    Analyzes an image to see if it has a fake background (solid color or checkerboard).
    If it does, it removes it using rembg (U2-Net).
    If the image already has true transparency, it leaves it alone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "force_remove": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "Auto"}),
                "alpha_matting": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, IMAGE, force_remove: bool, alpha_matting: bool = True):
        if not HAS_REMBG:
            log.error("rembg is not installed. Please wait for the setup to complete.")
            return (IMAGE, "Error: rembg not installed.")

        # IMAGE is typically shape (B, H, W, C)
        result_tensors = []
        status_msgs = []

        for b in range(IMAGE.shape[0]):
            img_tensor = IMAGE[b]
            pil_img = comfy_tensor_to_pil(img_tensor.unsqueeze(0))
            
            # Check if background is fake
            needs_removal = force_remove
            if not needs_removal:
                # Detect baked checkerboard
                checkerboard_res = detect_checkerboard(pil_img)
                if checkerboard_res.detected:
                    needs_removal = True
                else:
                    # Check alpha channel
                    alpha_res = detect_alpha(pil_img)
                    if not alpha_res.true_alpha:
                        # Fully opaque or fake alpha
                        needs_removal = True

            if needs_removal:
                log.info(f"Removing background from image {b+1}/{IMAGE.shape[0]}")
                try:
                    # Using rembg to remove background
                    # alpha_matting=True helps with hair and fine details
                    no_bg_pil = rembg.remove(pil_img, alpha_matting=alpha_matting)
                    out_tensor = pil_to_comfy_tensor(no_bg_pil)
                    result_tensors.append(out_tensor.squeeze(0))
                    status_msgs.append("Background removed.")
                except Exception as e:
                    log.error(f"Failed to remove background: {e}")
                    result_tensors.append(img_tensor)
                    status_msgs.append(f"Error removing bg: {e}")
            else:
                log.info(f"Image {b+1}/{IMAGE.shape[0]} already has true transparency. Skipping.")
                result_tensors.append(img_tensor)
                status_msgs.append("True alpha detected. Skipped.")

        out_batch = torch.stack(result_tensors, dim=0)
        return (out_batch, " | ".join(status_msgs))

class RemoveMultiviewFakeBackgroundNode:
    """
    Analyzes up to 4 views (Front, Left, Right, Back) in a single node
    and removes the fake background for all of them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_remove": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "Auto"}),
                "alpha_matting": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
            },
            "optional": {
                "FRONT_IMAGE": ("IMAGE",),
                "LEFT_IMAGE": ("IMAGE",),
                "RIGHT_IMAGE": ("IMAGE",),
                "BACK_IMAGE": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("FRONT_IMAGE", "LEFT_IMAGE", "RIGHT_IMAGE", "BACK_IMAGE", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, force_remove: bool, alpha_matting: bool = True, FRONT_IMAGE=None, LEFT_IMAGE=None, RIGHT_IMAGE=None, BACK_IMAGE=None):
        processor = RemoveFakeBackgroundNode()
        
        status_msgs = []
        
        def process_view(img, name):
            if img is None:
                return None
            # Return empty dummy if shape is weird (like the ones from directory loader)
            if img.shape[1] == 64 and img.shape[2] == 64:
                return img
                
            out, msg = processor.execute(img, force_remove, alpha_matting)
            status_msgs.append(f"{name}: {msg}")
            return out

        front = process_view(FRONT_IMAGE, "Front")
        left = process_view(LEFT_IMAGE, "Left")
        right = process_view(RIGHT_IMAGE, "Right")
        back = process_view(BACK_IMAGE, "Back")
        
        # If any is missing, supply dummy
        empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        return (
            front if front is not None else empty_tensor,
            left if left is not None else empty_tensor,
            right if right is not None else empty_tensor,
            back if back is not None else empty_tensor,
            " | ".join(status_msgs) if status_msgs else "No images provided"
        )


class ApplyManualMaskNode:
    """
    Applies a ComfyUI Mask to an image's Alpha channel.
    Useful for manually erasing background artifacts using the native MaskEditor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "MASK": ("MASK",),
                "action": (["erase_mask_area", "keep_only_mask_area"], {"default": "erase_mask_area"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, IMAGE, MASK, action="erase_mask_area"):
        result = []
        for i in range(IMAGE.shape[0]):
            img = IMAGE[i].clone()
            
            mask = MASK[i] if i < MASK.shape[0] else MASK[0]
            
            # Ensure mask is 2D/3D and matched dimensions
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)
            
            if img.shape[-1] == 3:
                alpha = torch.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype, device=img.device)
                img = torch.cat((img, alpha), dim=-1)
                
            if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                m = mask.unsqueeze(0).permute(0, 3, 1, 2)
                import torch.nn.functional as F
                m = F.interpolate(m, size=(img.shape[0], img.shape[1]), mode='bilinear', align_corners=False)
                mask = m.permute(0, 2, 3, 1).squeeze(0)
                
            if action == "erase_mask_area":
                img[..., 3] = img[..., 3] * (1.0 - mask[..., 0])
            else:
                img[..., 3] = img[..., 3] * mask[..., 0]
                
            result.append(img)
            
        return (torch.stack(result, dim=0),)


# ── ComfyUI Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AssetFactory_RemoveFakeBackground": RemoveFakeBackgroundNode,
    "AssetFactory_RemoveMultiviewFakeBackground": RemoveMultiviewFakeBackgroundNode,
    "AssetFactory_ApplyManualMask": ApplyManualMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetFactory_RemoveFakeBackground": "Asset Factory · Remove Fake Background",
    "AssetFactory_RemoveMultiviewFakeBackground": "Asset Factory · Remove Fake Background (Multi-view)",
    "AssetFactory_ApplyManualMask": "Asset Factory · Apply Manual Mask",
}
