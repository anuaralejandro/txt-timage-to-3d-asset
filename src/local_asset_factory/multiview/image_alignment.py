"""
local_asset_factory · multiview · image_alignment
Joint anatomical registration and validation for Multiview images.
"""
from typing import Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image

def clean_hidden_rgb(img_rgba: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Cleans RGB values where alpha is 0 to avoid checkerboard bleeding on resize."""
    if img_rgba.mode != "RGBA":
        return img_rgba
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    mask = alpha == 0
    arr[mask, 0] = bg_color[0]
    arr[mask, 1] = bg_color[1]
    arr[mask, 2] = bg_color[2]
    return Image.fromarray(arr, "RGBA")

def detect_extremities(alpha_channel: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
    ys, xs = np.where(alpha_channel > 10)
    if len(xs) == 0:
        return None
        
    top = (xs[np.argmin(ys)], ys.min())
    bottom = (xs[np.argmax(ys)], ys.max())
    left = (xs.min(), ys[np.argmin(xs)])
    right = (xs.max(), ys[np.argmax(xs)])
    
    return {
        "head_top": top,
        "feet_bottom": bottom,
        "arm_left": left,
        "arm_right": right
    }

def validate_t_pose(extremities: Dict[str, Tuple[int, int]], width: int, height: int) -> Tuple[bool, float]:
    """Validates if the character is in T-pose by checking arm span vs height."""
    arm_span = extremities["arm_right"][0] - extremities["arm_left"][0]
    body_height = extremities["feet_bottom"][1] - extremities["head_top"][1]
    
    if body_height == 0:
        return False, 0.0
        
    ratio = arm_span / body_height
    is_t_pose = ratio > 0.6
    return is_t_pose, ratio

def calculate_joint_registration(views: Dict[str, Image.Image], pad_fraction: float = 0.05) -> Dict[str, Any]:
    """
    Calculates a joint bounding box and scale to normalize all views consistently.
    Outputs the global center_y, scale factor, and crops for each image.
    """
    stats = {}
    max_height = 0
    max_width = 0
    
    for name, img in views.items():
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        ext = detect_extremities(alpha)
        
        if ext is None:
            stats[name] = {"valid": False}
            continue
            
        h = ext["feet_bottom"][1] - ext["head_top"][1]
        w = ext["arm_right"][0] - ext["arm_left"][0]
        center_y = (ext["feet_bottom"][1] + ext["head_top"][1]) / 2
        center_x = (ext["arm_right"][0] + ext["arm_left"][0]) / 2
        
        max_height = max(max_height, h)
        max_width = max(max_width, w)
        
        is_t_pose, ratio = validate_t_pose(ext, img.width, img.height)
        
        stats[name] = {
            "valid": True,
            "extremities": ext,
            "height": h,
            "width": w,
            "center_x": center_x,
            "center_y": center_y,
            "is_t_pose": is_t_pose,
            "span_ratio": ratio
        }
    
    registration = {}
    for name, stat in stats.items():
        if not stat["valid"]:
            registration[name] = None
            continue
            
        # Target bounding box based on global max_height and max_width
        # We need a square that fits the global max dimension with padding
        global_max_dim = max(max_height, max_width)
        padded_dim = int(global_max_dim * (1 + pad_fraction * 2))
        
        cx = stat["center_x"]
        cy = stat["center_y"]
        
        # Calculate cropping boundaries (could be out of image bounds)
        x0 = int(cx - padded_dim / 2)
        y0 = int(cy - padded_dim / 2)
        x1 = int(cx + padded_dim / 2)
        y1 = int(cy + padded_dim / 2)
        
        registration[name] = {
            "valid": True,
            "crop_box": (x0, y0, x1, y1),
            "t_pose_valid": stat["is_t_pose"],
            "span_ratio": stat["span_ratio"],
            "padded_dim": padded_dim
        }
        
    return registration

def apply_joint_registration(img: Image.Image, reg: Dict, target_size: Tuple[int, int]) -> Image.Image:
    """Applies the joint registration crop and resizes."""
    x0, y0, x1, y1 = reg["crop_box"]
    padded_dim = reg["padded_dim"]
    
    # Create a blank square canvas of padded_dim
    canvas = Image.new("RGBA", (padded_dim, padded_dim), (0, 0, 0, 0))
    
    # Calculate overlap
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(img.width, x1)
    src_y1 = min(img.height, y1)
    
    dest_x0 = src_x0 - x0
    dest_y0 = src_y0 - y0
    
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop = img.crop((src_x0, src_y0, src_x1, src_y1))
        canvas.paste(crop, (dest_x0, dest_y0))
        
    return canvas.resize(target_size, Image.LANCZOS)
