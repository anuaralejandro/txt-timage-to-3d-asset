"""
local_asset_factory · segmentation · labels
Defines anatomical semantic labels, anatomical L/R conventions, and visualization palettes.
"""

from enum import IntEnum
from typing import Dict, Tuple

class SemanticLabel(IntEnum):
    UNASSIGNED = 0
    TORSO = 1
    NECK = 2
    HEAD = 3
    HAIR = 4
    ARM_UPPER_L = 5
    ARM_LOWER_L = 6
    HAND_L = 7
    ARM_UPPER_R = 8
    ARM_LOWER_R = 9
    HAND_R = 10
    LEG_UPPER_L = 11
    LEG_LOWER_L = 12
    FOOT_L = 13
    LEG_UPPER_R = 14
    LEG_LOWER_R = 15
    FOOT_R = 16
    OTHER = 17

# Human-readable names map
LABEL_NAMES: Dict[int, str] = {
    SemanticLabel.UNASSIGNED: "other",
    SemanticLabel.TORSO: "torso",
    SemanticLabel.NECK: "neck",
    SemanticLabel.HEAD: "head",
    SemanticLabel.HAIR: "hair",
    SemanticLabel.ARM_UPPER_L: "upper_arm_L",
    SemanticLabel.ARM_LOWER_L: "lower_arm_L",
    SemanticLabel.HAND_L: "hand_L",
    SemanticLabel.ARM_UPPER_R: "upper_arm_R",
    SemanticLabel.ARM_LOWER_R: "lower_arm_R",
    SemanticLabel.HAND_R: "hand_R",
    SemanticLabel.LEG_UPPER_L: "thigh_L",
    SemanticLabel.LEG_LOWER_L: "lower_leg_L",
    SemanticLabel.FOOT_L: "foot_L",
    SemanticLabel.LEG_UPPER_R: "thigh_R",
    SemanticLabel.LEG_LOWER_R: "lower_leg_R",
    SemanticLabel.FOOT_R: "foot_R",
    SemanticLabel.OTHER: "other",
}

NAME_TO_LABEL: Dict[str, SemanticLabel] = {
    v: SemanticLabel(k) for k, v in LABEL_NAMES.items()
}

# RGB Color palette for debug GLB rendering and preview overlays (normalized 0.0 - 1.0)
LABEL_COLORS_RGB: Dict[int, Tuple[float, float, float]] = {
    SemanticLabel.UNASSIGNED: (0.4, 0.4, 0.4),  # Gray
    SemanticLabel.TORSO: (0.55, 0.48, 0.45),    # Brownish Gray
    SemanticLabel.NECK: (0.7, 0.6, 0.5),       # Neck Skin
    SemanticLabel.HEAD: (0.95, 0.85, 0.2),      # Yellow (Head)
    SemanticLabel.HAIR: (0.9, 0.7, 0.1),       # Gold (Hair)
    SemanticLabel.ARM_UPPER_L: (0.4, 0.1, 0.5), # Dark Purple
    SemanticLabel.ARM_LOWER_L: (0.85, 0.35, 0.35), # Pastel Red
    SemanticLabel.HAND_L: (0.95, 0.5, 0.5),    # Light Red/Pink
    SemanticLabel.ARM_UPPER_R: (0.5, 0.15, 0.6), # Dark Violet
    SemanticLabel.ARM_LOWER_R: (0.9, 0.4, 0.4),   # Coral Red
    SemanticLabel.HAND_R: (0.95, 0.6, 0.6),    # Soft Pink
    SemanticLabel.LEG_UPPER_L: (0.2, 0.5, 0.25),   # Dark Green
    SemanticLabel.LEG_LOWER_L: (0.6, 0.4, 0.8),    # Light Purple
    SemanticLabel.FOOT_L: (0.75, 0.5, 0.9),     # Magenta
    SemanticLabel.LEG_UPPER_R: (0.25, 0.55, 0.3),  # Emerald Green
    SemanticLabel.LEG_LOWER_R: (0.65, 0.45, 0.85), # Violet
    SemanticLabel.FOOT_R: (0.8, 0.55, 0.95),    # Soft Purple
    SemanticLabel.OTHER: (0.3, 0.3, 0.3),      # Dark Gray
}

# RGB Color palette in 0-255 uint8 format for 2D image overlays
LABEL_COLORS_UINT8: Dict[int, Tuple[int, int, int]] = {
    k: (int(r * 255), int(g * 255), int(b * 255))
    for k, (r, g, b) in LABEL_COLORS_RGB.items()
}

def resolve_anatomical_side(is_front_view: bool, screen_left: bool) -> SemanticLabel:
    """
    Resolves anatomical Left vs Right based on view direction and screen position.
    In a front view:
    - Screen Left is character's Right (ARM_R / LEG_R).
    - Screen Right is character's Left (ARM_L / LEG_L).
    In a back view:
    - Screen Left is character's Left (ARM_L / LEG_L).
    - Screen Right is character's Right (ARM_R / LEG_R).
    """
    if is_front_view:
        # Front view: anatomical right is on screen left
        return SemanticLabel.ARM_UPPER_R if screen_left else SemanticLabel.ARM_UPPER_L
    else:
        # Back view: anatomical left is on screen left
        return SemanticLabel.ARM_UPPER_L if screen_left else SemanticLabel.ARM_UPPER_R
