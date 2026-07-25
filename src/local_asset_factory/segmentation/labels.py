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
    ARM_UPPER_R = 7
    ARM_LOWER_R = 8
    LEG_UPPER_L = 9
    LEG_LOWER_L = 10
    LEG_UPPER_R = 11
    LEG_LOWER_R = 12

# Human-readable names map
LABEL_NAMES: Dict[int, str] = {
    SemanticLabel.UNASSIGNED: "unassigned",
    SemanticLabel.TORSO: "torso",
    SemanticLabel.NECK: "neck",
    SemanticLabel.HEAD: "head",
    SemanticLabel.HAIR: "hair",
    SemanticLabel.ARM_UPPER_L: "arm_upper_L",
    SemanticLabel.ARM_LOWER_L: "arm_lower_L",
    SemanticLabel.ARM_UPPER_R: "arm_upper_R",
    SemanticLabel.ARM_LOWER_R: "arm_lower_R",
    SemanticLabel.LEG_UPPER_L: "leg_upper_L",
    SemanticLabel.LEG_LOWER_L: "leg_lower_L",
    SemanticLabel.LEG_UPPER_R: "leg_upper_R",
    SemanticLabel.LEG_LOWER_R: "leg_lower_R",
}

NAME_TO_LABEL: Dict[str, SemanticLabel] = {
    v: SemanticLabel(k) for k, v in LABEL_NAMES.items()
}

# RGB Color palette for debug GLB rendering and preview overlays (normalized 0.0 - 1.0)
LABEL_COLORS_RGB: Dict[int, Tuple[float, float, float]] = {
    SemanticLabel.UNASSIGNED: (0.4, 0.4, 0.4),  # Gray
    SemanticLabel.TORSO: (0.55, 0.48, 0.45),    # Brownish Gray (Dorso)
    SemanticLabel.NECK: (0.55, 0.48, 0.45),     # Match Torso
    SemanticLabel.HEAD: (0.95, 0.85, 0.2),      # Yellow (Cabeza/Cabello)
    SemanticLabel.HAIR: (0.95, 0.85, 0.2),      # Yellow
    SemanticLabel.ARM_UPPER_L: (0.4, 0.1, 0.5), # Dark Purple (Brazos)
    SemanticLabel.ARM_UPPER_R: (0.4, 0.1, 0.5), # Dark Purple
    SemanticLabel.ARM_LOWER_L: (0.85, 0.35, 0.35), # Pastel Red (Antebrazos)
    SemanticLabel.ARM_LOWER_R: (0.85, 0.35, 0.35), # Pastel Red
    SemanticLabel.LEG_UPPER_L: (0.2, 0.5, 0.25),   # Dark Green (Muslos)
    SemanticLabel.LEG_UPPER_R: (0.2, 0.5, 0.25),   # Dark Green
    SemanticLabel.LEG_LOWER_L: (0.6, 0.4, 0.8),    # Light Purple (Piernas)
    SemanticLabel.LEG_LOWER_R: (0.6, 0.4, 0.8),    # Light Purple
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
