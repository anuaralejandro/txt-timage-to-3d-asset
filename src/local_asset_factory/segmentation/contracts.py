"""
local_asset_factory · segmentation · contracts
Dataclasses and JSON contract models for semantic segmentation and texture manifest.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os

@dataclass
class SemanticPartsManifest:
    """Schema version 1.0 manifest for semantic_parts.json."""
    schema_version: str = "1.0"
    source_glb: str = ""
    segmented_glb: str = ""
    coordinate_system: str = "Y_UP_RIGHT_HANDED"
    labels: Dict[str, str] = field(default_factory=dict)
    face_count: int = 0
    face_labels_path: str = ""
    proxy_used: bool = False
    proxy_face_count: Optional[int] = None
    per_label_face_counts: Dict[str, int] = field(default_factory=dict)
    confidence_summary: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    models: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_glb": self.source_glb,
            "segmented_glb": self.segmented_glb,
            "coordinate_system": self.coordinate_system,
            "labels": self.labels,
            "face_count": self.face_count,
            "face_labels_path": self.face_labels_path,
            "proxy_used": self.proxy_used,
            "proxy_face_count": self.proxy_face_count,
            "per_label_face_counts": self.per_label_face_counts,
            "confidence_summary": self.confidence_summary,
            "warnings": self.warnings,
            "models": self.models,
            "settings": self.settings,
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SemanticPartsManifest":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ViewRenderInfo:
    """Metadata for a rendered 2D orthographic camera view."""
    view_name: str  # front, left, right, back, etc.
    rgb_path: str
    depth_path: str
    normals_path: str
    face_id_path: str
    camera_json_path: str
    view_angle_deg: float = 0.0
    is_front: bool = False
    is_back: bool = False


@dataclass
class Keypoint2D:
    x: float
    y: float
    confidence: float


@dataclass
class PoseDetectionResult:
    view_name: str
    keypoints: Dict[str, Keypoint2D] = field(default_factory=dict)
    has_head: bool = False
    has_shoulders: bool = False
    has_limbs: bool = False


@dataclass
class VRAMLogEntry:
    phase: str
    gpu_name: str
    peak_allocated_mb: float
    peak_reserved_mb: float
    elapsed_sec: float
