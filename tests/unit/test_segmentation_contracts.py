"""
Unit tests for segmentation dataclasses, manifests, and JSON serialization.
"""

import os
import json
import tempfile
import pytest
from local_asset_factory.segmentation.contracts import SemanticPartsManifest, ViewRenderInfo

def test_manifest_serialization_roundtrip():
    manifest = SemanticPartsManifest(
        schema_version="1.0",
        source_glb="/path/to/mesh.glb",
        segmented_glb="/path/to/segmented.glb",
        face_count=1000,
        per_label_face_counts={"torso": 500, "head": 500},
        proxy_used=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "semantic_parts.json")
        manifest.save(json_path)
        assert os.path.isfile(json_path)

        loaded = SemanticPartsManifest.load(json_path)
        assert loaded.schema_version == "1.0"
        assert loaded.face_count == 1000
        assert loaded.proxy_used is True
        assert loaded.per_label_face_counts["torso"] == 500
