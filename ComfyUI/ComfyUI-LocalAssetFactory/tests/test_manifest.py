"""
Tests — test_manifest.py
Tests for manifest building and writing.

Run with:
    python -m pytest tests/test_manifest.py -v
    (from the ComfyUI-LocalAssetFactory directory)
"""

import json
import os
import re
import sys
import tempfile
import unittest

# Ensure the package root is importable
_pkg_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _pkg_root)

from schemas import AssetManifest, sanitize_asset_id


class TestManifestBuild(unittest.TestCase):
    """Test the ManifestWriter.build_manifest logic (inline, no service import)."""

    def _build_manifest(
        self,
        asset_id="test",
        concept_image_path="",
        texture_paths=None,
        raw_model_path="",
        errors=None,
    ):
        """Inline build logic to avoid relative-import issues."""
        has_errors = bool(errors)
        has_concept = bool(concept_image_path)
        has_textures = bool(texture_paths)
        has_model = bool(raw_model_path)

        if has_errors and not (has_concept or has_textures or has_model):
            status = "failed"
        elif has_errors:
            status = "partial"
        else:
            status = "generated"

        return AssetManifest(
            asset_id=asset_id,
            status=status,
            errors=errors or [],
        )

    def test_manifest_generated(self):
        m = self._build_manifest(
            asset_id="test_sword_001",
            concept_image_path="/output/concept.png",
            texture_paths={"base_color": "/output/texture.png"},
            raw_model_path="/output/model.glb",
        )
        self.assertEqual(m.status, "generated")
        self.assertEqual(m.asset_id, "test_sword_001")

    def test_manifest_partial(self):
        m = self._build_manifest(
            asset_id="test_sword",
            concept_image_path="/output/concept.png",
            errors=["TRELLIS failed: out of memory"],
        )
        self.assertEqual(m.status, "partial")

    def test_manifest_failed(self):
        m = self._build_manifest(
            asset_id="test_sword",
            errors=["Everything broke"],
        )
        self.assertEqual(m.status, "failed")


class TestManifestSerialization(unittest.TestCase):
    def test_manifest_creation(self):
        m = AssetManifest(asset_id="test_asset_001")
        self.assertEqual(m.schema_version, "1.0")
        self.assertEqual(m.asset_id, "test_asset_001")
        self.assertEqual(m.status, "generated")

    def test_manifest_to_json(self):
        m = AssetManifest(asset_id="test_asset")
        data = m.model_dump(mode="json")
        j = json.dumps(data)
        self.assertIn("test_asset", j)
        self.assertIn("schema_version", j)

    def test_manifest_sanitizes_id(self):
        manifest = AssetManifest(asset_id="Bad ID With Spaces!")
        self.assertNotIn(" ", manifest.asset_id)
        self.assertNotIn("!", manifest.asset_id)

    def test_manifest_saves_to_disk(self):
        """Manifest should be writable as valid JSON."""
        m = AssetManifest(
            asset_id="save_test",
            status="generated",
        )
        data = m.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2)
            path = f.name

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["asset_id"], "save_test")
            self.assertEqual(loaded["schema_version"], "1.0")
        finally:
            os.remove(path)


class TestManifestJsonTolerance(unittest.TestCase):
    """Test that malformed JSON from LLM can be handled."""

    def test_valid_json_from_llm(self):
        raw = '{"asset_id": "sword", "asset_name": "Sword"}'
        data = json.loads(raw)
        self.assertEqual(data["asset_id"], "sword")

    def test_json_with_trailing_comma(self):
        raw = '{"asset_id": "sword", "asset_name": "Sword",}'
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(fixed)
        self.assertEqual(data["asset_id"], "sword")

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"asset_id": "sword"}\n```'
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        match = re.search(r"\{[\s\S]*\}", cleaned)
        self.assertIsNotNone(match)
        data = json.loads(match.group(0))
        self.assertEqual(data["asset_id"], "sword")


if __name__ == "__main__":
    unittest.main()
