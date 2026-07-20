"""
local_asset_factory · parts3d · part_classifier
Classifies raw 3D mesh parts into retopology pipelines.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..domain.enums import PartClass

log = logging.getLogger(__name__)

# Keyword map for semantic name -> PartClass
KEYWORD_RULES: Dict[str, PartClass] = {
    "body": PartClass.ORGANIC_DEFORMING,
    "head": PartClass.ORGANIC_DEFORMING,
    "face": PartClass.ORGANIC_DEFORMING,
    "arm": PartClass.ORGANIC_DEFORMING,
    "leg": PartClass.ORGANIC_DEFORMING,
    "hand": PartClass.ORGANIC_DEFORMING,
    "skin": PartClass.ORGANIC_DEFORMING,

    "hair": PartClass.HAIR_RIGID,
    "ponytail": PartClass.HAIR_SECONDARY_MOTION,
    "bangs": PartClass.HAIR_SECONDARY_MOTION,

    "top": PartClass.CLOTH_DEFORMING,
    "shirt": PartClass.CLOTH_DEFORMING,
    "pants": PartClass.CLOTH_DEFORMING,
    "shorts": PartClass.CLOTH_DEFORMING,
    "skirt": PartClass.CLOTH_DEFORMING,
    "jacket": PartClass.CLOTH_DEFORMING,
    "cape": PartClass.CLOTH_DEFORMING,

    "boot": PartClass.HARD_SURFACE,
    "shoe": PartClass.HARD_SURFACE,
    "armor": PartClass.HARD_SURFACE,
    "helmet": PartClass.HARD_SURFACE,
    "belt": PartClass.HARD_SURFACE,
    "buckle": PartClass.HARD_SURFACE,
    "glove": PartClass.HARD_SURFACE,
    "button": PartClass.HARD_SURFACE,
    "accessory": PartClass.ACCESSORY,
    "accessories": PartClass.ACCESSORY,
    "ribbon": PartClass.ACCESSORY,
    "weapon": PartClass.ACCESSORY,
}


def classify_part_by_name(semantic_name: str) -> PartClass:
    """
    Classify a part name into a PartClass based on keyword matching.
    Default fallback is ORGANIC_DEFORMING.
    """
    s_lower = semantic_name.lower()
    for kw, p_cls in KEYWORD_RULES.items():
        if kw in s_lower:
            return p_cls
    return PartClass.ORGANIC_DEFORMING
