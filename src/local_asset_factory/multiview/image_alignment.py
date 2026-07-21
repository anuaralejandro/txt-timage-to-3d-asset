"""
local_asset_factory · multiview · image_alignment
Joint anatomical registration and validation for Multiview images.
"""

from typing import Dict, Tuple, Optional
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
    # A human T-pose typically has arm span ~ body height. Anime characters might have larger heads.
    # We expect ratio > 0.6 for a valid T-pose (A-pose or T-pose).
    is_t_pose = ratio > 0.6
    return is_t_pose, ratio

def calculate_joint_registration(views: Dict[str, Image.Image]) -> Dict[str, Dict]:
    """
    Calculates a joint bounding box and scale to normalize all views consistently.
    """
    stats = {}
    max_height = 0
    max_center_y = 0
    
    # Extract extremities for each view
    for name, img in views.items():
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        ext = detect_extremities(alpha)
        
        if ext is None:
            stats[name] = {"valid": False}
            continue
            
        height = ext["feet_bottom"][1] - ext["head_top"][1]
        center_y = (ext["feet_bottom"][1] + ext["head_top"][1]) / 2
        
        if height > max_height:
            max_height = height
        
        is_t_pose, ratio = validate_t_pose(ext, img.width, img.height)
        
        stats[name] = {
            "valid": True,
            "extremities": ext,
            "height": height,
            "center_y": center_y,
            "is_t_pose": is_t_pose,
            "span_ratio": ratio
        }
    
    # Calculate crop boxes to align centers and scale
    registration = {}
    for name, stat in stats.items():
        if not stat["valid"]:
            registration[name] = None
            continue
            
        # Target bounding box based on global max_height
        # to ensure all views have the exact same pixel-to-meter scale
        registration[name] = {
            "target_height": max_height,
            "t_pose_valid": stat["is_t_pose"]
        }
        
    return registration
