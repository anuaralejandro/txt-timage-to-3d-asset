"""
Unit tests: Pydantic v2 domain contracts.
Tests validation, serialization, defaults, and business logic.
No GPU required.
"""
import pytest
from datetime import datetime

from local_asset_factory.domain.models import (
    AssetRequest,
    ArtifactMeta,
    CanonicalView,
    CanonicalViewSet,
    CandidateMetrics,
    GeometryCandidate,
    Part3D,
    PartSet,
    PipelineManifest,
    PreflightResult,
    PreflightCheck,
    sha256_file,
)
from local_asset_factory.domain.enums import (
    MobileProfile,
    PaintMode,
    PartClass,
    PipelineStage,
    StageStatus,
    ViewOrientation,
    PreflightFailure,
    SeverityLevel,
)


# ---------------------------------------------------------------------------
# ViewOrientation
# ---------------------------------------------------------------------------

class TestViewOrientation:
    def test_canonical_order(self):
        order = ViewOrientation.canonical_order()
        assert order[0] == ViewOrientation.FRONT
        assert order[-1] == ViewOrientation.RIGHT

    def test_checkpoint_views_excludes_right(self):
        """RIGHT must NOT be in checkpoint views per Hunyuan3D-2mv official docs."""
        cv = ViewOrientation.checkpoint_views()
        assert ViewOrientation.RIGHT not in cv
        assert ViewOrientation.FRONT in cv
        assert ViewOrientation.LEFT in cv
        assert ViewOrientation.BACK in cv

    def test_qa_views_includes_right(self):
        """RIGHT must be available for QA/scoring."""
        qv = ViewOrientation.qa_views()
        assert ViewOrientation.RIGHT in qv


# ---------------------------------------------------------------------------
# PartClass
# ---------------------------------------------------------------------------

class TestPartClass:
    def test_retopo_strategy_organic(self):
        assert PartClass.ORGANIC_DEFORMING.retopo_strategy() == "organic"
        assert PartClass.CLOTH_DEFORMING.retopo_strategy() == "organic"

    def test_retopo_strategy_hard_surface(self):
        assert PartClass.HARD_SURFACE.retopo_strategy() == "hard_surface"
        assert PartClass.ACCESSORY.retopo_strategy() == "hard_surface"

    def test_retopo_strategy_hair(self):
        assert PartClass.HAIR_RIGID.retopo_strategy() == "hair"
        assert PartClass.HAIR_SECONDARY_MOTION.retopo_strategy() == "hair"


# ---------------------------------------------------------------------------
# ArtifactMeta
# ---------------------------------------------------------------------------

class TestArtifactMeta:
    def test_default_id_generated(self):
        a = ArtifactMeta()
        assert len(a.id) == 36  # UUID4

    def test_created_at_is_utc_isoformat(self):
        a = ArtifactMeta()
        # Should be parseable
        dt = datetime.fromisoformat(a.created_at.replace("Z", "+00:00"))
        assert dt is not None

    def test_metadata_defaults_empty(self):
        a = ArtifactMeta()
        assert a.metadata == {}

    def test_serialization_roundtrip(self):
        a = ArtifactMeta(
            relative_path="candidates/hunyuan2mv/raw_11_normal.glb",
            producer="hunyuan3d_2mv",
            checkpoint="tencent/Hunyuan3D-2mv",
            seed=11,
        )
        data = a.model_dump()
        b = ArtifactMeta.model_validate(data)
        assert b.relative_path == a.relative_path
        assert b.seed == 11


# ---------------------------------------------------------------------------
# AssetRequest
# ---------------------------------------------------------------------------

class TestAssetRequest:
    def test_front_view_required(self):
        with pytest.raises(Exception):
            AssetRequest(
                asset_name="jace",
                views={"left": "left.png", "back": "back.png"},  # no front
            )

    def test_front_view_accepted(self):
        r = AssetRequest(
            asset_name="jace",
            views={
                "front": "front.png",
                "left": "left.png",
                "back": "back.png",
                "right": "right.png",
            },
        )
        assert r.asset_name == "jace"
        assert "front" in r.views

    def test_default_platform(self):
        r = AssetRequest(asset_name="test", views={"front": "f.png"})
        assert r.target_platform == MobileProfile.ANDROID_MID

    def test_default_paint_mode(self):
        r = AssetRequest(asset_name="test", views={"front": "f.png"})
        assert r.paint_mode == PaintMode.TOON_MOBILE

    def test_default_seeds(self):
        r = AssetRequest(asset_name="test", views={"front": "f.png"})
        assert 11 in r.seeds
        assert len(r.seeds) >= 4

    def test_omni_enabled_by_default(self):
        r = AssetRequest(asset_name="test", views={"front": "f.png"})
        assert r.enable_omni_pose is True

    def test_riganything_disabled_by_default(self):
        """RigAnything is non-commercial — disabled by default."""
        r = AssetRequest(asset_name="test", views={"front": "f.png"})
        assert r.enable_riganything is False


# ---------------------------------------------------------------------------
# CanonicalViewSet
# ---------------------------------------------------------------------------

class TestCanonicalViewSet:
    def _make_view(self, orientation: ViewOrientation, path: str) -> CanonicalView:
        return CanonicalView(
            orientation=orientation,
            relative_path=path,
            producer="normalizer",
        )

    def test_checkpoint_views_excludes_right(self):
        vs = CanonicalViewSet(
            front=self._make_view(ViewOrientation.FRONT, "views/front.png"),
            left=self._make_view(ViewOrientation.LEFT, "views/left.png"),
            back=self._make_view(ViewOrientation.BACK, "views/back.png"),
            right=self._make_view(ViewOrientation.RIGHT, "views/right.png"),
        )
        cv = vs.checkpoint_views()
        assert "right" not in cv
        assert "front" in cv
        assert "left" in cv
        assert "back" in cv

    def test_right_optional(self):
        vs = CanonicalViewSet(
            front=self._make_view(ViewOrientation.FRONT, "views/front.png"),
            left=self._make_view(ViewOrientation.LEFT, "views/left.png"),
            back=self._make_view(ViewOrientation.BACK, "views/back.png"),
        )
        assert vs.right is None
        assert len(vs.all_views()) == 3


# ---------------------------------------------------------------------------
# CandidateMetrics
# ---------------------------------------------------------------------------

class TestCandidateMetrics:
    def test_composite_score_calculated(self):
        m = CandidateMetrics(
            silhouette_iou=0.8,
            semantic_part_iou=0.75,
            keypoint_error=0.1,
            pose_error=0.05,
            arm_separation=0.9,
            surface_noise=0.1,
        )
        score = m.compute_composite()
        assert 0.0 <= score <= 1.0
        assert m.composite_score == score

    def test_zero_score_for_bad_mesh(self):
        m = CandidateMetrics(
            silhouette_iou=0.0,
            semantic_part_iou=0.0,
            keypoint_error=1.0,
            pose_error=1.0,
            non_manifold_edges=10000,
            degenerate_faces=5000,
        )
        score = m.compute_composite()
        assert score < 0.3  # bad mesh should rank low


# ---------------------------------------------------------------------------
# PartSet
# ---------------------------------------------------------------------------

class TestPartSet:
    def test_by_class(self):
        ps = PartSet(
            job_id="test",
            selected_geometry_id="geom_001",
            parts=[
                Part3D(semantic_name="body", part_class=PartClass.ORGANIC_DEFORMING),
                Part3D(semantic_name="boot_left", part_class=PartClass.HARD_SURFACE),
                Part3D(semantic_name="boot_right", part_class=PartClass.HARD_SURFACE),
            ]
        )
        hard = ps.by_class(PartClass.HARD_SURFACE)
        assert len(hard) == 2
        organic = ps.by_class(PartClass.ORGANIC_DEFORMING)
        assert len(organic) == 1

    def test_by_name(self):
        ps = PartSet(
            job_id="test",
            selected_geometry_id="geom_001",
            parts=[
                Part3D(semantic_name="ponytail", part_class=PartClass.HAIR_SECONDARY_MOTION),
            ]
        )
        pt = ps.by_name("ponytail")
        assert pt is not None
        assert pt.semantic_name == "ponytail"

        missing = ps.by_name("nonexistent")
        assert missing is None


# ---------------------------------------------------------------------------
# PipelineManifest
# ---------------------------------------------------------------------------

class TestPipelineManifest:
    def test_stages_initialized(self):
        m = PipelineManifest()
        assert len(m.stages) == len(PipelineStage.ordered())

    def test_stage_transitions(self):
        m = PipelineManifest()
        m.mark_stage_running(PipelineStage.PREFLIGHT)
        rec = m.stage(PipelineStage.PREFLIGHT)
        assert rec.status == StageStatus.RUNNING
        assert rec.started_at is not None

        m.mark_stage_done(PipelineStage.PREFLIGHT, artifact_ids=["abc-123"])
        assert rec.status == StageStatus.COMPLETED
        assert "abc-123" in rec.artifact_ids

    def test_stage_failure(self):
        m = PipelineManifest()
        m.mark_stage_failed(PipelineStage.SAM_SEGMENTATION, "CUDA OOM")
        rec = m.stage(PipelineStage.SAM_SEGMENTATION)
        assert rec.status == StageStatus.FAILED
        assert "CUDA OOM" in rec.errors[0]

    def test_serialization_roundtrip(self):
        m = PipelineManifest()
        m.mark_stage_running(PipelineStage.PREFLIGHT)
        m.mark_stage_done(PipelineStage.PREFLIGHT)
        data = m.model_dump_json()
        m2 = PipelineManifest.model_validate_json(data)
        assert m2.job_id == m.job_id

    def test_license_mode_default(self):
        m = PipelineManifest()
        assert m.license_mode == "private_noncommercial_research"
        assert m.license_accepted is False


# ---------------------------------------------------------------------------
# PreflightResult
# ---------------------------------------------------------------------------

class TestPreflightResult:
    def test_failures_filter(self):
        r = PreflightResult(job_id="test")
        r.checks = [
            PreflightCheck(
                check=PreflightFailure.BAKED_CHECKERBOARD_BACKGROUND,
                passed=False,
                severity=SeverityLevel.FATAL,
            ),
            PreflightCheck(
                check=PreflightFailure.NO_TRUE_ALPHA,
                passed=True,
                severity=SeverityLevel.INFO,
            ),
        ]
        assert len(r.failures()) == 1
        assert r.failures()[0].check == PreflightFailure.BAKED_CHECKERBOARD_BACKGROUND
