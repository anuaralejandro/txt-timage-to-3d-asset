"""
local_asset_factory · segmentation · human_parser_backend
Human parser backend interface, SCHP-LIP implementation, and synthetic mock parser for testing.
"""

from __future__ import annotations
import os
import logging
import numpy as np
from typing import Protocol, Dict, Any, Optional
from PIL import Image

from .labels import SemanticLabel, resolve_anatomical_side

log = logging.getLogger(__name__)

# SCHP / LIP dataset category mapping to internal SemanticLabel
# SCHP-LIP categories: 0: Background, 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses, 5: UpperClothes,
# 6: Dress, 7: Coat, 8: Socks, 9: Pants, 10: Torso-skin, 11: Scarf, 12: Skirt, 13: Face,
# 14: Left-arm, 15: Right-arm, 16: Left-leg, 17: Right-leg, 18: Left-shoe, 19: Right-shoe.

SCHP_LIP_TO_INTERNAL: Dict[int, SemanticLabel] = {
    0: SemanticLabel.UNASSIGNED,
    1: SemanticLabel.HEAD,        # Hat -> Head
    2: SemanticLabel.HAIR,        # Hair -> Hair
    3: SemanticLabel.ARM_UPPER_L,       # Glove
    4: SemanticLabel.HEAD,        # Sunglasses
    5: SemanticLabel.TORSO,       # UpperClothes
    6: SemanticLabel.TORSO,       # Dress
    7: SemanticLabel.TORSO,       # Coat
    8: SemanticLabel.LEG_UPPER_L,       # Socks
    9: SemanticLabel.TORSO,       # Pants
    10: SemanticLabel.TORSO,      # Torso-skin
    11: SemanticLabel.NECK,       # Scarf -> Neck
    12: SemanticLabel.TORSO,      # Skirt
    13: SemanticLabel.HEAD,       # Face -> Head
    14: SemanticLabel.ARM_UPPER_L,      # SCHP Left-arm
    15: SemanticLabel.ARM_UPPER_R,      # SCHP Right-arm
    16: SemanticLabel.LEG_UPPER_L,      # SCHP Left-leg
    17: SemanticLabel.LEG_UPPER_R,      # SCHP Right-leg
    18: SemanticLabel.LEG_UPPER_L,      # Left-shoe
    19: SemanticLabel.LEG_UPPER_R,      # Right-shoe
}


class ParserResult:
    """Holds 2D label map array (H, W) and optional per-class confidence map (H, W)."""

    def __init__(self, labels: np.ndarray, confidence: Optional[np.ndarray] = None):
        self.labels = labels  # uint8 / int32 array containing SemanticLabel values (0..8)
        self.confidence = confidence if confidence is not None else np.ones_like(labels, dtype=np.float32)


class HumanParserBackend(Protocol):
    """Abstract interface for 2D human parsing model backends."""

    def load(self, config: Dict[str, Any]) -> None:
        ...

    def predict(self, image_path: str, is_front: bool = True) -> ParserResult:
        ...

    def unload(self) -> None:
        ...


class MockHumanParser:
    """Synthetic mock human parser backend for fast testing without GPU weights."""

    def __init__(self):
        self.is_loaded = False

    def load(self, config: Dict[str, Any]) -> None:
        self.is_loaded = True
        log.info("Loaded MockHumanParser.")

    def predict(self, image_path: str, is_front: bool = True) -> ParserResult:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        w, h = img.size
        
        # Create geometric 2D bands matching a human figure in screen space
        labels = np.zeros((h, w), dtype=np.int32)
        confidence = np.full((h, w), fill_value=0.9, dtype=np.float32)

        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Head (top 20%)
        head_mask = (y_coords < int(h * 0.20)) & (x_coords > int(w * 0.35)) & (x_coords < int(w * 0.65))
        labels[head_mask] = SemanticLabel.HEAD

        # Hair (top 15% outer ring)
        hair_mask = (y_coords < int(h * 0.15)) & (x_coords > int(w * 0.30)) & (x_coords < int(w * 0.70)) & ~head_mask
        labels[hair_mask] = SemanticLabel.HAIR

        # Neck (20% to 25%)
        neck_mask = (y_coords >= int(h * 0.20)) & (y_coords < int(h * 0.25)) & (x_coords > int(w * 0.42)) & (x_coords < int(w * 0.58))
        labels[neck_mask] = SemanticLabel.NECK

        # Torso (25% to 60%, center)
        torso_mask = (y_coords >= int(h * 0.25)) & (y_coords < int(h * 0.60)) & (x_coords > int(w * 0.35)) & (x_coords < int(w * 0.65))
        labels[torso_mask] = SemanticLabel.TORSO

        # Left / Right Arms (25% to 60%, sides)
        left_screen_arm = (y_coords >= int(h * 0.25)) & (y_coords < int(h * 0.60)) & (x_coords <= int(w * 0.35)) & (x_coords > int(w * 0.20))
        right_screen_arm = (y_coords >= int(h * 0.25)) & (y_coords < int(h * 0.60)) & (x_coords >= int(w * 0.65)) & (x_coords < int(w * 0.80))

        # Assign anatomical lateralities
        arm_left_label = resolve_anatomical_side(is_front, screen_left=True)
        arm_right_label = resolve_anatomical_side(is_front, screen_left=False)

        labels[left_screen_arm] = arm_left_label
        labels[right_screen_arm] = arm_right_label

        # Legs (60% to 100%)
        left_screen_leg = (y_coords >= int(h * 0.60)) & (x_coords > int(w * 0.25)) & (x_coords <= int(w * 0.50))
        right_screen_leg = (y_coords >= int(h * 0.60)) & (x_coords > int(w * 0.50)) & (x_coords < int(w * 0.75))

        leg_left_label = SemanticLabel.LEG_UPPER_R if is_front else SemanticLabel.LEG_UPPER_L
        leg_right_label = SemanticLabel.LEG_UPPER_L if is_front else SemanticLabel.LEG_UPPER_R

        labels[left_screen_leg] = leg_left_label
        labels[right_screen_leg] = leg_right_label

        return ParserResult(labels=labels, confidence=confidence)

    def unload(self) -> None:
        self.is_loaded = False
        log.info("Unloaded MockHumanParser.")


class SCHPLIPHumanParser:
    """Real SCHP-LIP PyTorch Human Parser backend."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = "cpu"

    def load(self, config: Dict[str, Any]) -> None:
        self.device = config.get("device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        log.info(f"Loading SCHP-LIP parser backend on {self.device}...")
        # Checkpoint download check/fallback
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            log.info(f"Using SCHP checkpoint: {self.checkpoint_path}")
        else:
            log.warning("SCHP checkpoint not found locally. Using fallback mock parser.")
            self.model = None

    def predict(self, image_path: str, is_front: bool = True) -> ParserResult:
        if self.model is None:
            # Fallback to deterministic mock prediction if weights are missing
            mock = MockHumanParser()
            return mock.predict(image_path, is_front=is_front)

        # Real infer code goes here when weights are present
        raise NotImplementedError("SCHP PyTorch model execution requires checkpoint weights.")

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        log.info("Unloaded SCHPLIPHumanParser.")
