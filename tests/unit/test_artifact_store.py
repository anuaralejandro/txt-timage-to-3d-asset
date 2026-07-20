"""
Unit tests: Artifact store.
Tests atomic writes, SHA-256, resume markers, cache, and failed candidate preservation.
No GPU required.
"""
import json
import os
import pytest
from pathlib import Path

from local_asset_factory.observability.artifact_store import ArtifactStore
from local_asset_factory.domain.models import ArtifactMeta, PipelineManifest
from local_asset_factory.domain.enums import PipelineStage


class TestArtifactStore:
    @pytest.fixture
    def store(self, tmp_path) -> ArtifactStore:
        s = ArtifactStore(base_dir=str(tmp_path), job_id="test-job-001")
        s.initialize()
        return s

    def test_directory_structure_created(self, store, tmp_path):
        job_root = tmp_path / "artifacts" / "test-job-001"
        assert job_root.exists()
        assert (job_root / "input").exists()
        assert (job_root / "candidates" / "hunyuan2mv").exists()
        assert (job_root / "candidates" / "omni").exists()
        assert (job_root / "exports").exists()
        assert (job_root / "logs").exists()

    def test_atomic_write_bytes(self, store):
        dest = store.atomic_write("input/test_file.bin", b"hello bytes")
        assert dest.exists()
        assert dest.read_bytes() == b"hello bytes"

    def test_atomic_write_str(self, store):
        dest = store.atomic_write("logs/test.log", "hello string")
        assert dest.read_text(encoding="utf-8") == "hello string"

    def test_atomic_copy(self, store, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"\x00\x01\x02\x03")
        dest = store.atomic_copy(src, "candidates/hunyuan2mv/raw_11_normal.glb")
        assert dest.exists()
        assert dest.read_bytes() == b"\x00\x01\x02\x03"

    def test_sha256_registration(self, store):
        store.atomic_write("input/front.png", b"test image content")
        artifact = ArtifactMeta(relative_path="input/front.png", producer="test")
        registered = store.register(artifact)
        assert len(registered.sha256) == 64  # sha256 hex
        assert registered.sha256 != ""

    def test_resume_markers(self, store):
        assert not store.has_stage("preflight")
        store.mark_stage_done("preflight")
        assert store.has_stage("preflight")
        store.clear_stage("preflight")
        assert not store.has_stage("preflight")

    def test_cancellation(self, store):
        assert not store.is_cancelled()
        store.cancel()
        assert store.is_cancelled()

    def test_save_and_load_manifest(self, store):
        m = PipelineManifest()
        m.mark_stage_running(PipelineStage.PREFLIGHT)
        store.save_manifest(m)

        m2 = store.load_manifest()
        assert m2 is not None
        assert m2.job_id == m.job_id

    def test_load_manifest_returns_none_when_missing(self, store):
        result = store.load_manifest()
        assert result is None

    def test_save_and_load_json(self, store):
        data = {"score": 0.85, "candidate": "abc123"}
        store.save_json("metrics.json", data)
        loaded = store.load_json("metrics.json")
        assert loaded["score"] == 0.85

    def test_cache_key_operations(self, store):
        key = "abc123def456"
        assert not store.cache_key_exists(key)
        store.record_cache_hit(key, "candidates/raw_11.glb")
        assert store.cache_key_exists(key)
        cached = store.get_cached_path(key)
        assert cached == "candidates/raw_11.glb"

    def test_preserve_failed_candidate(self, store):
        store.preserve_failed_candidate(
            candidate_id="00000000-1234-5678-abcd-ef0000000001",
            reason="low silhouette IoU: 0.12",
            mesh_path="candidates/hunyuan2mv/raw_29_normal.glb"
        )
        # File should exist in logs/
        logs_dir = store.job_root / "logs"
        rejected_files = list(logs_dir.glob("rejected_*.json"))
        assert len(rejected_files) == 1
        data = json.loads(rejected_files[0].read_text())
        assert "low silhouette IoU" in data["reason"]

    def test_path_helpers(self, store):
        abs_path = store.path("input/front.png")
        assert abs_path.is_absolute()
        assert str(store.job_root) in str(abs_path)

    def test_exists(self, store):
        assert not store.exists("input/front.png")
        store.atomic_write("input/front.png", b"data")
        assert store.exists("input/front.png")
