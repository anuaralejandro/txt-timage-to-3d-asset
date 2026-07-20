"""
local_asset_factory · domain · models
All Pydantic v2 contracts for the character pipeline.

Every artifact includes: id, relative_path, sha256, producer,
checkpoint, revision, seed, created_at, metadata.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import (
    CameraType,
    MobileProfile,
    OmniControlType,
    PaintMode,
    PartClass,
    PipelineStage,
    PreflightFailure,
    RigMethod,
    SeverityLevel,
    StageStatus,
    ViewOrientation,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file. Returns empty string if file not found."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return ""


# ---------------------------------------------------------------------------
# Base artifact meta (all artifacts share these fields)
# ---------------------------------------------------------------------------

class ArtifactMeta(BaseModel):
    """
    Base contract for every pipeline artifact.
    All paths are RELATIVE to the job root (artifacts/<job_id>/).
    """
    id: str = Field(default_factory=_new_id)
    relative_path: str = ""
    sha256: str = ""
    producer: str = ""           # e.g. "hunyuan3d_2mv", "sam3_1", "blender"
    checkpoint: str = ""         # model checkpoint identifier
    revision: str = ""           # git/HF revision hash
    seed: int = 0
    created_at: str = Field(default_factory=_utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def compute_sha256(self, job_root: str | Path) -> "ArtifactMeta":
        """Compute and set sha256 from the artifact file."""
        if self.relative_path:
            full = Path(job_root) / self.relative_path
            self.sha256 = sha256_file(full)
        return self


# ---------------------------------------------------------------------------
# Input contracts
# ---------------------------------------------------------------------------

class InputImage(ArtifactMeta):
    """A raw input image provided by the user."""
    orientation: ViewOrientation = ViewOrientation.FRONT
    width: int = 0
    height: int = 0
    has_alpha: bool = False
    true_alpha: bool = False       # alpha is real transparency (not all opaque)
    format: str = "PNG"


class AssetRequest(BaseModel):
    """
    User-provided asset request — top-level input to the pipeline.

    Views contract:
      front: required
      left:  required
      back:  required
      right: required for QA (used in scoring, not fed to checkpoint directly)
    """
    id: str = Field(default_factory=_new_id)
    created_at: str = Field(default_factory=_utcnow)

    # Identity
    asset_name: str
    asset_type: str = "humanoid_character"
    style: str = "anime_low_poly"

    # Target
    target_platform: MobileProfile = MobileProfile.ANDROID_MID
    pose: str = "strict_t_pose"
    paint_mode: PaintMode = PaintMode.TOON_MOBILE

    # Budgets
    triangle_budget_lod0: int = Field(default=28000, gt=0)
    texture_resolution: int = 2048

    # Input views (relative paths or absolute paths)
    views: Dict[str, str] = Field(default_factory=dict)

    # Optional overrides
    seeds: List[int] = Field(default_factory=lambda: [11, 29, 47, 83])
    inference_steps: List[int] = Field(default_factory=lambda: [30, 40])
    enable_omni_pose: bool = True
    enable_sam3: bool = True
    enable_hunyuan_part: bool = True
    enable_riganything: bool = False    # non-commercial
    enable_text_to_asset: bool = False  # phase 2+

    @field_validator("views")
    @classmethod
    def _require_front(cls, v: dict) -> dict:
        if "front" not in v:
            raise ValueError("views.front is required")
        return v


# ---------------------------------------------------------------------------
# Canonical views
# ---------------------------------------------------------------------------

class Keypoint2D(BaseModel):
    name: str
    x: float
    y: float
    confidence: float = 1.0
    visible: bool = True


class BoundingBox2D(BaseModel):
    x: float
    y: float
    w: float
    h: float


class CanonicalView(ArtifactMeta):
    """
    A normalized input view ready for the pipeline.
    Relative to the canonical_views used by Hunyuan3D-2mv.
    """
    orientation: ViewOrientation
    width: int = 0
    height: int = 0
    subject_bbox: Optional[BoundingBox2D] = None
    camera_type: CameraType = CameraType.ORTHOGRAPHIC
    camera_yaw_deg: float = 0.0
    camera_pitch_deg: float = 0.0
    keypoints_2d: List[Keypoint2D] = Field(default_factory=list)
    # Path to per-part semantic mask file (PNG composite)
    semantic_mask_path: str = ""
    identity_embedding: List[float] = Field(default_factory=list)
    # Is this view used as checkpoint input or QA-only?
    for_checkpoint: bool = True
    for_qa: bool = True


class CanonicalViewSet(BaseModel):
    """
    The full set of canonical views for a pipeline job.
    front/left/back: checkpoint input + QA
    right: QA only (Hunyuan3D-2mv does not accept it per official docs)
    """
    front: CanonicalView
    left: CanonicalView
    back: CanonicalView
    right: Optional[CanonicalView] = None    # QA-only view

    def checkpoint_views(self) -> Dict[str, str]:
        """Return {orientation: path} dict for Hunyuan3D-2mv checkpoint."""
        return {
            "front": self.front.relative_path,
            "left": self.left.relative_path,
            "back": self.back.relative_path,
        }

    def all_views(self) -> List[CanonicalView]:
        views = [self.front, self.left, self.back]
        if self.right:
            views.append(self.right)
        return views


# ---------------------------------------------------------------------------
# SAM 3.1 segmentation
# ---------------------------------------------------------------------------

class SemanticMask(ArtifactMeta):
    """A single semantic mask for one part in one view."""
    view_orientation: ViewOrientation
    part_name: str                  # e.g. "body", "hair", "boot_left"
    confidence: float = 0.0
    area_px: int = 0
    rle_path: str = ""              # path to RLE JSON
    # Runtime
    inference_ms: float = 0.0


class SemanticMaskSet(BaseModel):
    """All semantic masks for all views."""
    job_id: str
    masks: List[SemanticMask] = Field(default_factory=list)
    taxonomy: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow)

    def by_part(self, part_name: str) -> List[SemanticMask]:
        return [m for m in self.masks if m.part_name == part_name]

    def by_view(self, orientation: ViewOrientation) -> List[SemanticMask]:
        return [m for m in self.masks if m.view_orientation == orientation]


# ---------------------------------------------------------------------------
# Geometry generation
# ---------------------------------------------------------------------------

class GeometryRequest(BaseModel):
    """Request sent to a geometry backend."""
    job_id: str
    views: Dict[str, str]            # {orientation: path} — only checkpoint views
    seeds: List[int] = Field(default_factory=lambda: [11, 29, 47, 83])
    inference_steps: int = 30
    variant: str = "normal"          # "normal" | "turbo"
    omni_control: Optional[OmniControlType] = None
    pose_path: Optional[str] = None  # for Omni pose control
    timeout_seconds: int = 600


class CandidateMetrics(BaseModel):
    """Geometric and perceptual metrics for a geometry candidate."""
    silhouette_iou: float = 0.0
    semantic_part_iou: float = 0.0
    keypoint_error: float = 0.0
    pose_error: float = 0.0
    head_body_ratio_error: float = 0.0
    arm_separation: float = 0.0
    leg_separation: float = 0.0
    symmetry: float = 0.0
    non_manifold_edges: int = 0
    degenerate_faces: int = 0
    connected_components: int = 1
    normal_consistency: float = 0.0
    surface_noise: float = 0.0
    # Weighted composite score (higher = better)
    composite_score: float = 0.0

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or {
            "semantic_part_iou": 0.25,
            "silhouette_iou": 0.20,
            "keypoint_error": 0.15,
            "pose_error": 0.15,
            "mesh_health": 0.10,
            "part_separability": 0.10,
            "surface_noise": 0.05,
        }
        mesh_health = 1.0 - min(
            (self.non_manifold_edges + self.degenerate_faces) / max(1, 1000), 1.0
        )
        score = (
            self.semantic_part_iou * w.get("semantic_part_iou", 0)
            + self.silhouette_iou * w.get("silhouette_iou", 0)
            + max(0, 1.0 - self.keypoint_error) * w.get("keypoint_error", 0)
            + max(0, 1.0 - self.pose_error) * w.get("pose_error", 0)
            + mesh_health * w.get("mesh_health", 0)
            + self.arm_separation * w.get("part_separability", 0)
            + max(0, 1.0 - self.surface_noise) * w.get("surface_noise", 0)
        )
        self.composite_score = round(score, 4)
        return self.composite_score


class GeometryCandidate(ArtifactMeta):
    """
    A raw geometry candidate produced by a backend.
    Never modified after generation — preservation is mandatory.
    """
    backend: str = ""                      # "hunyuan3d_2mv" | "hunyuan3d_omni"
    variant: str = "normal"
    omni_control: Optional[OmniControlType] = None
    input_views: List[str] = Field(default_factory=list)
    render_paths: Dict[str, str] = Field(default_factory=dict)   # {view: path}
    metrics: Optional[CandidateMetrics] = None
    warnings: List[str] = Field(default_factory=list)
    runtime_seconds: float = 0.0
    peak_vram_mb: int = 0
    passed_gates: bool = False
    rank: int = -1                         # -1 = unranked


class SelectedGeometry(ArtifactMeta):
    """The single geometry candidate selected by the scorer."""
    candidate_id: str
    backend: str = ""
    rank: int = 0
    score: float = 0.0
    metrics: Optional[CandidateMetrics] = None
    selection_reason: str = ""
    rejected_candidates: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3D part segmentation
# ---------------------------------------------------------------------------

class BoundingBox3D(BaseModel):
    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0
    max_z: float = 0.0


class Part3D(ArtifactMeta):
    """
    A single semantically-labeled mesh part from Hunyuan3D-Part.
    Each part drives its own retopology strategy.
    """
    semantic_name: str                     # e.g. "body", "boot_left", "ponytail"
    confidence: float = 0.0
    source_face_ids: List[int] = Field(default_factory=list)
    bbox: Optional[BoundingBox3D] = None
    symmetry_partner: Optional[str] = None # e.g. "boot_right" for "boot_left"
    part_class: PartClass = PartClass.ORGANIC_DEFORMING
    rig_policy: str = "deform"             # "deform" | "rigid" | "secondary"
    material_policy: str = "shared"        # "shared" | "separate"
    topology_policy: str = "organic"       # driven by part_class.retopo_strategy()


class PartSet(BaseModel):
    """All 3D parts from Hunyuan3D-Part segmentation."""
    job_id: str
    selected_geometry_id: str
    parts: List[Part3D] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow)
    fallback_used: bool = False            # True if SAM mask projection was used
    warnings: List[str] = Field(default_factory=list)

    def by_class(self, cls: PartClass) -> List[Part3D]:
        return [p for p in self.parts if p.part_class == cls]

    def by_name(self, name: str) -> Optional[Part3D]:
        return next((p for p in self.parts if p.semantic_name == name), None)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class TopologyResult(ArtifactMeta):
    """Result of per-part retopology."""
    part_name: str
    part_class: PartClass = PartClass.ORGANIC_DEFORMING
    strategy_used: str = ""
    triangle_count: int = 0
    quad_ratio: float = 0.0             # % of faces that are quads (pre-triangulation)
    has_joint_loops: bool = False
    manifold: bool = False
    deformation_qa_passed: bool = False
    warnings: List[str] = Field(default_factory=list)


class RecomposedMesh(ArtifactMeta):
    """Final recomposed low-poly character after all parts are retopologized."""
    part_topology_ids: List[str] = Field(default_factory=list)
    total_triangles: int = 0
    total_parts: int = 0
    manifold: bool = False
    profile: str = "normal_anime"       # "normal_anime" | "chibi_4_heads" | "chibi_5_heads"


# ---------------------------------------------------------------------------
# UV and bake
# ---------------------------------------------------------------------------

class UVResult(ArtifactMeta):
    """UV unwrap and atlas result."""
    atlas_count: int = 1                # <= 2 for mobile
    texel_density_avg: float = 0.0
    seam_count: int = 0
    has_overlaps: bool = False
    padding_px: int = 4
    atlas_paths: List[str] = Field(default_factory=list)


class BakeResult(ArtifactMeta):
    """Result of high-to-low baking."""
    base_color_path: str = ""
    normal_path: str = ""
    ao_path: str = ""
    roughness_path: str = ""
    metallic_path: str = ""
    emissive_path: str = ""
    opacity_path: str = ""


# ---------------------------------------------------------------------------
# Texturing
# ---------------------------------------------------------------------------

class TextureResult(ArtifactMeta):
    """Final texture result from Hunyuan3D-Paint."""
    paint_mode: PaintMode = PaintMode.TOON_MOBILE
    base_color_path: str = ""
    normal_path: str = ""
    roughness_path: str = ""
    metallic_path: str = ""
    outline_mask_path: str = ""
    material_count: int = 1
    max_materials: int = 2
    resolution: int = 2048


# ---------------------------------------------------------------------------
# Rigging
# ---------------------------------------------------------------------------

class BoneInfo(BaseModel):
    name: str
    parent: Optional[str] = None
    head: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    tail: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    is_deform: bool = True


class RigResult(ArtifactMeta):
    """Rig result — skeleton + weights."""
    method: RigMethod = RigMethod.RIGIFY
    bone_count: int = 0
    deform_bone_count: int = 0
    max_influences: int = 4
    has_unweighted_vertices: bool = False
    weight_sum_valid: bool = True
    has_ponytail_bones: bool = False
    has_skirt_bones: bool = False
    rigged_glb_path: str = ""
    skeleton_json_path: str = ""
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Animation QA
# ---------------------------------------------------------------------------

class AnimationQAResult(ArtifactMeta):
    """Result of a single animation QA pass."""
    animation_name: str
    unweighted_vertices: int = 0
    max_influences_found: int = 0
    weight_sum_error: float = 0.0
    self_intersections: int = 0
    part_penetrations: int = 0
    volume_loss_pct: float = 0.0
    uv_stretch_avg: float = 0.0
    normal_flips: int = 0
    detached_parts: int = 0
    passed: bool = True
    severity: SeverityLevel = SeverityLevel.INFO
    blocking: bool = False              # True = blocks export


class AnimationQASuite(BaseModel):
    """Full animation QA suite results."""
    job_id: str
    results: List[AnimationQAResult] = Field(default_factory=list)
    all_passed: bool = True
    export_blocked: bool = False
    created_at: str = Field(default_factory=_utcnow)

    def blocking_failures(self) -> List[AnimationQAResult]:
        return [r for r in self.results if r.blocking and not r.passed]


# ---------------------------------------------------------------------------
# Mobile LOD
# ---------------------------------------------------------------------------

LOD_BUDGETS: Dict[str, Dict[str, int]] = {
    "android_low": {"lod0": 16000, "lod1": 8000, "lod2": 3500, "lod3": 1200},
    "android_mid": {"lod0": 28000, "lod1": 14000, "lod2": 6500, "lod3": 2200},
    "android_high": {"lod0": 40000, "lod1": 20000, "lod2": 9000, "lod3": 3000},
    "ios_mid": {"lod0": 28000, "lod1": 14000, "lod2": 6500, "lod3": 2200},
    "ios_high": {"lod0": 40000, "lod1": 20000, "lod2": 9000, "lod3": 3000},
}


class LODResult(ArtifactMeta):
    """A single LOD level output."""
    lod_level: int = 0                  # 0 = highest detail
    triangle_count: int = 0
    budget_triangles: int = 0
    within_budget: bool = True
    skinning_preserved: bool = True
    texture_resolution: int = 2048
    ktx2_compressed: bool = False
    glb_path: str = ""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportResult(ArtifactMeta):
    """Final export result including all LODs."""
    lods: List[LODResult] = Field(default_factory=list)
    manifest_path: str = ""
    qc_report_path: str = ""
    turntable_path: str = ""
    preview_paths: List[str] = Field(default_factory=list)
    gltf_valid: bool = False
    blender_reimport_passed: bool = False
    all_validations_passed: bool = False


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

class PreflightCheck(BaseModel):
    """Result of a single preflight check."""
    check: PreflightFailure
    passed: bool
    message: str = ""
    severity: SeverityLevel = SeverityLevel.ERROR


class PreflightResult(BaseModel):
    """Full preflight validation result."""
    job_id: str
    view: Optional[str] = None
    checks: List[PreflightCheck] = Field(default_factory=list)
    passed: bool = True
    created_at: str = Field(default_factory=_utcnow)

    def failures(self) -> List[PreflightCheck]:
        return [c for c in self.checks if not c.passed]

    def fatal_failures(self) -> List[PreflightCheck]:
        return [c for c in self.checks if not c.passed and c.severity == SeverityLevel.FATAL]


# ---------------------------------------------------------------------------
# Pipeline Manifest
# ---------------------------------------------------------------------------

class CheckpointInfo(BaseModel):
    """Records a specific model checkpoint used in the pipeline."""
    name: str
    checkpoint: str
    revision: str = ""
    license: str = ""
    noncommercial: bool = False


class StageRecord(BaseModel):
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    artifact_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PipelineManifest(BaseModel):
    """
    Complete, reproducible manifest for a pipeline run.
    Records every model, seed, hash, and license.
    """
    schema_version: str = "2.0"
    job_id: str = Field(default_factory=_new_id)
    created_at: str = Field(default_factory=_utcnow)
    completed_at: Optional[str] = None
    status: StageStatus = StageStatus.PENDING

    # Request
    request: Optional[AssetRequest] = None

    # Pipeline stages
    stages: List[StageRecord] = Field(
        default_factory=lambda: [
            StageRecord(stage=s) for s in PipelineStage.ordered()
        ]
    )

    # Checkpoints used
    checkpoints: List[CheckpointInfo] = Field(default_factory=list)

    # Top-level artifacts
    canonical_views: Optional[str] = None      # relative path
    selected_geometry_id: Optional[str] = None
    export_result: Optional[ExportResult] = None

    # Stats
    total_candidates_generated: int = 0
    total_runtime_seconds: float = 0.0
    peak_vram_mb: int = 0

    # License declaration
    license_mode: str = "private_noncommercial_research"
    license_accepted: bool = False

    def stage(self, s: PipelineStage) -> Optional[StageRecord]:
        return next((r for r in self.stages if r.stage == s), None)

    def mark_stage_running(self, s: PipelineStage) -> None:
        rec = self.stage(s)
        if rec:
            rec.status = StageStatus.RUNNING
            rec.started_at = _utcnow()

    def mark_stage_done(self, s: PipelineStage, artifact_ids: List[str] = None) -> None:
        rec = self.stage(s)
        if rec:
            rec.status = StageStatus.COMPLETED
            rec.completed_at = _utcnow()
            if artifact_ids:
                rec.artifact_ids.extend(artifact_ids)

    def mark_stage_failed(self, s: PipelineStage, error: str) -> None:
        rec = self.stage(s)
        if rec:
            rec.status = StageStatus.FAILED
            rec.completed_at = _utcnow()
            rec.errors.append(error)
