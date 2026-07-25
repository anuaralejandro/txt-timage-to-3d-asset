"""
Unit tests for segmentation labels, anatomical L/R rules, and color palettes.
"""

import pytest
from local_asset_factory.segmentation.labels import (
    SemanticLabel,
    LABEL_NAMES,
    LABEL_COLORS_RGB,
    resolve_anatomical_side,
)

def test_semantic_labels_enumeration():
    assert len(SemanticLabel) == 13
    assert SemanticLabel.UNASSIGNED == 0
    assert SemanticLabel.TORSO == 1
    assert SemanticLabel.NECK == 2
    assert SemanticLabel.HEAD == 3
    assert SemanticLabel.HAIR == 4
    assert SemanticLabel.ARM_UPPER_L == 5
    assert SemanticLabel.ARM_LOWER_L == 6
    assert SemanticLabel.ARM_UPPER_R == 7
    assert SemanticLabel.ARM_LOWER_R == 8
    assert SemanticLabel.LEG_UPPER_L == 9
    assert SemanticLabel.LEG_LOWER_L == 10
    assert SemanticLabel.LEG_UPPER_R == 11
    assert SemanticLabel.LEG_LOWER_R == 12

def test_anatomical_lateralities_front_view():
    # Front view: screen left -> character's Right arm
    arm_left_screen = resolve_anatomical_side(is_front_view=True, screen_left=True)
    assert arm_left_screen == SemanticLabel.ARM_UPPER_R

    # Front view: screen right -> character's Left arm
    arm_right_screen = resolve_anatomical_side(is_front_view=True, screen_left=False)
    assert arm_right_screen == SemanticLabel.ARM_UPPER_L

def test_anatomical_lateralities_back_view():
    # Back view: screen left -> character's Left arm
    arm_left_screen = resolve_anatomical_side(is_front_view=False, screen_left=True)
    assert arm_left_screen == SemanticLabel.ARM_UPPER_L

    # Back view: screen right -> character's Right arm
    arm_right_screen = resolve_anatomical_side(is_front_view=False, screen_left=False)
    assert arm_right_screen == SemanticLabel.ARM_UPPER_R

def test_label_colors_mapping():
    assert len(LABEL_COLORS_RGB) == 13
    for lbl in SemanticLabel:
        rgb = LABEL_COLORS_RGB[lbl]
        assert len(rgb) == 3
        assert all(0.0 <= c <= 1.0 for c in rgb)
