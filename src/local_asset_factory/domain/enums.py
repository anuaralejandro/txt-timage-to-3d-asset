"""
local_asset_factory · domain · enums
Typed enumerations for the character pipeline.
"""
from __future__ import annotations

from enum import Enum


class ViewOrientation(str, Enum):
    FRONT = "front"
    LEFT = "left"
    BACK = "back"
    RIGHT = "right"

    @classmethod
    def canonical_order(cls) -> list["ViewOrientation"]:
        return [cls.FRONT, cls.LEFT, cls.BACK, cls.RIGHT]

    @classmethod
    def checkpoint_views(cls) -> list["ViewOrientation"]:
        """Views that Hunyuan3D-2mv actually accepts (per official docs)."""
        return [cls.FRONT, cls.LEFT, cls.BACK]

    @classmethod
    def qa_views(cls) -> list["ViewOrientation"]:
        """Views used for scoring/QA (right reserved for QA only)."""
        return [cls.FRONT, cls.LEFT, cls.BACK, cls.RIGHT]


class PartClass(str, Enum):
    """Semantic classification of a mesh part for retopology strategy."""
    ORGANIC_DEFORMING = "organic_deforming"
    ORGANIC_RIGID = "organic_rigid"
    CLOTH_DEFORMING = "cloth_deforming"
    CLOTH_RIGID = "cloth_rigid"
    HAIR_RIGID = "hair_rigid"
    HAIR_SECONDARY_MOTION = "hair_secondary_motion"
    HARD_SURFACE = "hard_surface"
    ACCESSORY = "accessory"

    def retopo_strategy(self) -> str:
        """Return the retopology strategy name for this part class."""
        return {
            self.ORGANIC_DEFORMING: "organic",
            self.ORGANIC_RIGID: "organic",
            self.CLOTH_DEFORMING: "organic",
            self.CLOTH_RIGID: "organic",
            self.HAIR_RIGID: "hair",
            self.HAIR_SECONDARY_MOTION: "hair",
            self.HARD_SURFACE: "hard_surface",
            self.ACCESSORY: "hard_surface",
        }[self]


class PipelineStage(str, Enum):
    """Ordered pipeline stages."""
    PREFLIGHT = "preflight"
    CANONICAL_VIEWS = "canonical_views"
    SAM_SEGMENTATION = "sam_segmentation"
    MULTIVIEW_CONSISTENCY = "multiview_consistency"
    GEOMETRY_GENERATION = "geometry_generation"
    SCORING = "scoring"
    PART_SEGMENTATION = "part_segmentation"
    RETOPOLOGY = "retopology"
    UV_BAKE = "uv_bake"
    PAINT = "paint"
    RIGGING = "rigging"
    ANIMATION_QA = "animation_qa"
    OPTIMIZATION = "optimization"
    EXPORT = "export"

    @classmethod
    def ordered(cls) -> list["PipelineStage"]:
        return [
            cls.PREFLIGHT, cls.CANONICAL_VIEWS, cls.SAM_SEGMENTATION,
            cls.MULTIVIEW_CONSISTENCY, cls.GEOMETRY_GENERATION, cls.SCORING,
            cls.PART_SEGMENTATION, cls.RETOPOLOGY, cls.UV_BAKE, cls.PAINT,
            cls.RIGGING, cls.ANIMATION_QA, cls.OPTIMIZATION, cls.EXPORT,
        ]


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class PaintMode(str, Enum):
    TOON_MOBILE = "toon_mobile"
    PBR_MOBILE = "pbr_mobile"


class MobileProfile(str, Enum):
    ANDROID_LOW = "android_low"
    ANDROID_MID = "android_mid"
    ANDROID_HIGH = "android_high"
    IOS_MID = "ios_mid"
    IOS_HIGH = "ios_high"


class CameraType(str, Enum):
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"


class OmniControlType(str, Enum):
    """Supported control types for Hunyuan3D-Omni."""
    POSE = "pose"
    BBOX = "bbox"
    VOXEL = "voxel"
    POINT = "point"


class RigMethod(str, Enum):
    RIGANYTHING = "riganything"
    RIGIFY = "rigify"


class PreflightFailure(str, Enum):
    NO_TRUE_ALPHA = "no_true_alpha"
    BAKED_CHECKERBOARD_BACKGROUND = "baked_checkerboard_background"
    DOMINANT_BACKGROUND = "dominant_background"
    SUBJECT_CROPPED = "subject_cropped"
    MISSING_LIMB = "missing_limb"
    INVALID_POSE = "invalid_pose"
    INCONSISTENT_VIEWS = "inconsistent_views"
    DUPLICATE_VIEWS = "duplicate_views"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    CLOTHING_INCONSISTENT = "clothing_inconsistent"


class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"
