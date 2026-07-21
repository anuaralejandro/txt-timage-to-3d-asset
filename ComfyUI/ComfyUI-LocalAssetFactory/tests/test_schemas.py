"""
Tests — test_schemas.py
Tests for Pydantic schema validation, sanitisation, and error handling.
"""

import json
import os
import sys
import unittest

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schemas import (
    AssetRequest,
    AssetSpecification,
    AssetManifest,
    BlenderReport,
    MobileRequirements,
    ModelingRequirements,
    TexturePromptSet,
    sanitize_asset_id,
)


class TestSanitizeAssetId(unittest.TestCase):
    """Test the sanitize_asset_id function."""

    def test_normal_name(self):
        self.assertEqual(sanitize_asset_id("Royal Training Sword"), "royal_training_sword")

    def test_special_chars(self):
        self.assertEqual(sanitize_asset_id("Sword/of:Death!"), "sword_of_death")

    def test_multiple_underscores(self):
        self.assertEqual(sanitize_asset_id("a___b___c"), "a_b_c")

    def test_empty(self):
        self.assertEqual(sanitize_asset_id(""), "unnamed_asset")

    def test_spaces_only(self):
        self.assertEqual(sanitize_asset_id("   "), "unnamed_asset")

    def test_no_spaces(self):
        result = sanitize_asset_id("my_sword_001")
        self.assertNotIn(" ", result)

    def test_already_valid(self):
        self.assertEqual(sanitize_asset_id("valid_id_123"), "valid_id_123")


class TestAssetRequest(unittest.TestCase):
    """Test AssetRequest validation."""

    def _make_request(self, **overrides):
        defaults = {
            "asset_name": "Test Sword",
            "asset_type": "weapon",
            "description_es": "Una espada de prueba",
            "visual_style": "low_poly_anime",
            "gameplay_role": "hero",
            "target_platform": "android_mid",
            "triangle_budget": 2500,
            "texture_resolution": 512,
            "seed": 0,
        }
        defaults.update(overrides)
        return AssetRequest(**defaults)

    def test_valid_request(self):
        req = self._make_request()
        self.assertEqual(req.asset_name, "Test Sword")
        self.assertEqual(req.triangle_budget, 2500)

    def test_invalid_asset_type(self):
        with self.assertRaises(ValueError):
            self._make_request(asset_type="invalid_type")

    def test_invalid_visual_style(self):
        with self.assertRaises(ValueError):
            self._make_request(visual_style="cyberpunk")

    def test_invalid_gameplay_role(self):
        with self.assertRaises(ValueError):
            self._make_request(gameplay_role="boss_enemy")

    def test_invalid_platform(self):
        with self.assertRaises(ValueError):
            self._make_request(target_platform="ps5")

    def test_invalid_texture_resolution(self):
        with self.assertRaises(ValueError):
            self._make_request(texture_resolution=300)

    def test_triangle_budget_zero(self):
        with self.assertRaises(ValueError):
            self._make_request(triangle_budget=0)

    def test_triangle_budget_negative(self):
        with self.assertRaises(ValueError):
            self._make_request(triangle_budget=-100)


class TestAssetSpecification(unittest.TestCase):
    """Test AssetSpecification validation and sanitisation."""

    def test_asset_id_sanitized(self):
        spec = AssetSpecification(asset_id="My Cool Sword!", asset_name="My Cool Sword")
        self.assertEqual(spec.asset_id, "my_cool_sword")

    def test_asset_id_already_valid(self):
        spec = AssetSpecification(asset_id="valid_sword_001", asset_name="Valid Sword")
        self.assertEqual(spec.asset_id, "valid_sword_001")

    def test_serialization_roundtrip(self):
        spec = AssetSpecification(
            asset_id="test_item",
            asset_name="Test Item",
            concept_prompt_en="A test prompt",
            negative_prompt_en="bad stuff",
        )
        data = spec.model_dump(mode="json")
        spec2 = AssetSpecification(**data)
        self.assertEqual(spec.asset_id, spec2.asset_id)
        self.assertEqual(spec.concept_prompt_en, spec2.concept_prompt_en)


class TestMobileRequirements(unittest.TestCase):
    def test_valid(self):
        m = MobileRequirements(triangle_budget=3000, texture_resolution=512)
        self.assertEqual(m.triangle_budget, 3000)

    def test_invalid_resolution(self):
        with self.assertRaises(ValueError):
            MobileRequirements(texture_resolution=333)

    def test_materials_range(self):
        with self.assertRaises(ValueError):
            MobileRequirements(maximum_materials=0)
        with self.assertRaises(ValueError):
            MobileRequirements(maximum_materials=5)


class TestTexturePromptSet(unittest.TestCase):
    def test_all_prompts_filters_empty(self):
        t = TexturePromptSet(base_color="stone texture", ornament_variant="", material_style="")
        prompts = t.all_prompts()
        self.assertEqual(len(prompts), 1)
        self.assertIn("base_color", prompts)

    def test_all_prompts_includes_extra(self):
        t = TexturePromptSet(
            base_color="wood",
            extra={"glow_map": "glowing runes", "empty": ""},
        )
        prompts = t.all_prompts()
        self.assertEqual(len(prompts), 2)
        self.assertIn("glow_map", prompts)


class TestAssetManifest(unittest.TestCase):
    def test_manifest_creation(self):
        m = AssetManifest(asset_id="test_asset_001")
        self.assertEqual(m.schema_version, "1.0")
        self.assertEqual(m.asset_id, "test_asset_001")
        self.assertEqual(m.status, "generated")

    def test_manifest_serialization(self):
        m = AssetManifest(asset_id="test_asset")
        data = m.model_dump(mode="json")
        j = json.dumps(data)
        self.assertIn("test_asset", j)


class TestBlenderReport(unittest.TestCase):
    def test_default_report(self):
        r = BlenderReport()
        self.assertFalse(r.success)
        self.assertEqual(r.preview_renders, [])
        self.assertEqual(r.errors, [])


if __name__ == "__main__":
    unittest.main()
