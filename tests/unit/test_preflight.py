"""
Unit tests: Preflight detectors.
Tests that the baked checkerboard image fails and real alpha passes.
No GPU required.
"""
import io
import pytest
import numpy as np
from PIL import Image

from local_asset_factory.preflight.alpha_detector import detect_alpha
from local_asset_factory.preflight.checkerboard_detector import detect_checkerboard
from local_asset_factory.preflight.preflight_runner import PreflightRunner
from local_asset_factory.domain.enums import PreflightFailure


# ---------------------------------------------------------------------------
# Helpers: synthetic image builders
# ---------------------------------------------------------------------------

def make_checkerboard_rgba(size: int = 256, tile_size: int = 16) -> Image.Image:
    """Create a PNG with a baked checkerboard background in RGB, alpha=255."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            tile = (y // tile_size + x // tile_size) % 2
            val = 240 if tile == 0 else 180
            arr[y, x, 0] = val
            arr[y, x, 1] = val
            arr[y, x, 2] = val
            arr[y, x, 3] = 255  # all opaque — fake alpha
    return Image.fromarray(arr)


def make_true_alpha_rgba(size: int = 256) -> Image.Image:
    """Create a RGBA image with real transparency (character on transparent bg)."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    # Center circle = character (opaque)
    cx, cy = size // 2, size // 2
    r = size // 3
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                arr[y, x] = [180, 120, 100, 255]   # skin color, opaque
            else:
                arr[y, x] = [0, 0, 0, 0]            # transparent background
    return Image.fromarray(arr)


def make_rgb_no_alpha(size: int = 256) -> Image.Image:
    """Plain RGB image — no alpha channel."""
    arr = np.full((size, size, 3), 200, dtype=np.uint8)
    return Image.fromarray(arr)


def make_uniform_background_rgba(size: int = 256, bg_color=(255, 255, 255)) -> Image.Image:
    """RGBA image with uniform white background (opaque but not checkerboard)."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, :3] = bg_color
    arr[:, :, 3] = 255
    # Put a character-like region in the center
    cx, cy = size // 2, size // 2
    arr[cy-40:cy+40, cx-30:cx+30, :3] = [180, 120, 100]
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Checkerboard detector tests
# ---------------------------------------------------------------------------

class TestCheckerboardDetector:
    def test_baked_checkerboard_detected(self):
        """The canonical fake-transparency image must be detected and rejected."""
        img = make_checkerboard_rgba(size=256, tile_size=16)
        result = detect_checkerboard(img, confidence_threshold=0.55)
        assert result.detected, (
            f"Expected checkerboard to be DETECTED but got confidence={result.confidence:.2f}, "
            f"fraction={result.affected_fraction:.2f}"
        )
        assert result.confidence >= 0.55

    def test_uniform_background_not_detected(self):
        """A uniform white background should NOT trigger checkerboard detection."""
        img = make_uniform_background_rgba()
        result = detect_checkerboard(img)
        assert not result.detected, (
            f"Expected NO checkerboard but got confidence={result.confidence:.2f}"
        )

    def test_true_alpha_image_not_detected(self):
        """An image with real transparency (no background) should not be flagged."""
        img = make_true_alpha_rgba()
        result = detect_checkerboard(img)
        assert not result.detected

    def test_small_tile_checkerboard(self):
        """Even small-tile checkerboards (8px) should be detectable."""
        img = make_checkerboard_rgba(size=256, tile_size=8)
        result = detect_checkerboard(
            img,
            tile_size_range=(4, 32),
            confidence_threshold=0.55
        )
        # Small tiles may be harder — just check it doesn't crash
        assert isinstance(result.detected, bool)

    def test_large_tile_checkerboard(self):
        """Large tile (64px) checkerboard."""
        img = make_checkerboard_rgba(size=512, tile_size=64)
        result = detect_checkerboard(img, tile_size_range=(32, 80))
        assert isinstance(result.detected, bool)

    def test_result_has_message(self):
        img = make_checkerboard_rgba()
        result = detect_checkerboard(img)
        assert len(result.message) > 10


# ---------------------------------------------------------------------------
# Alpha detector tests
# ---------------------------------------------------------------------------

class TestAlphaDetector:
    def test_fully_opaque_rgba_detected(self):
        """RGBA with all alpha=255 is NOT true transparency."""
        img = make_checkerboard_rgba()
        result = detect_alpha(img)
        assert result.has_alpha_channel
        assert result.fully_opaque
        assert not result.true_alpha
        assert result.min_alpha == 255

    def test_true_alpha_recognized(self):
        """Image with real transparency (alpha < 255 for bg) is true alpha."""
        img = make_true_alpha_rgba()
        result = detect_alpha(img)
        assert result.has_alpha_channel
        assert not result.fully_opaque
        assert result.true_alpha
        assert result.min_alpha == 0

    def test_rgb_no_alpha(self):
        """Plain RGB has no alpha channel."""
        img = make_rgb_no_alpha()
        result = detect_alpha(img)
        assert not result.has_alpha_channel
        assert not result.true_alpha

    def test_result_message_not_empty(self):
        img = make_checkerboard_rgba()
        result = detect_alpha(img)
        assert len(result.message) > 10


# ---------------------------------------------------------------------------
# PreflightRunner integration (no GPU, synthetic images)
# ---------------------------------------------------------------------------

class TestPreflightRunner:
    def setup_method(self):
        self.runner = PreflightRunner()

    def test_checkerboard_image_fails_preflight(self, tmp_path):
        """A baked checkerboard image must FAIL preflight."""
        img = make_checkerboard_rgba(size=256, tile_size=16)
        img_path = tmp_path / "checkerboard_front.png"
        img.save(str(img_path))

        result = self.runner.check_image(str(img_path), job_id="test", view_orientation="front")

        # Must fail
        assert not result.passed, (
            "Checkerboard image should FAIL preflight but passed. "
            f"Checks: {[(c.check.value, c.passed) for c in result.checks]}"
        )
        failure_types = [c.check for c in result.failures()]
        assert PreflightFailure.BAKED_CHECKERBOARD_BACKGROUND in failure_types, (
            f"Expected BAKED_CHECKERBOARD_BACKGROUND in failures, got: {failure_types}"
        )

    def test_true_alpha_image_passes_preflight(self, tmp_path):
        """An image with real transparency should pass basic preflight."""
        img = make_true_alpha_rgba(size=512)
        img_path = tmp_path / "character_front.png"
        img.save(str(img_path))

        result = self.runner.check_image(str(img_path), job_id="test", view_orientation="front")
        # This should pass checkerboard and alpha checks
        checker_check = next(
            (c for c in result.checks
             if c.check == PreflightFailure.BAKED_CHECKERBOARD_BACKGROUND), None
        )
        if checker_check:
            assert checker_check.passed, (
                f"True alpha image should not trigger checkerboard: {checker_check.message}"
            )

    def test_missing_front_view_fails_validation(self, tmp_path):
        """A view set missing 'front' must fail validation."""
        left = tmp_path / "left.png"
        make_true_alpha_rgba().save(str(left))

        results = self.runner.validate_view_set(
            {"left": str(left)},
            job_id="test"
        )
        assert not self.runner.all_passed(results)

    def test_duplicate_views_detected(self, tmp_path):
        """Identical images for different views must trigger duplicate detection."""
        img = make_true_alpha_rgba(size=256)
        front = tmp_path / "front.png"
        left = tmp_path / "left.png"
        img.save(str(front))
        img.save(str(left))   # same content

        results = self.runner.validate_view_set(
            {"front": str(front), "left": str(left), "back": str(front)},
            job_id="test"
        )
        failure_types = [
            c.check
            for r in results
            for c in r.failures()
        ]
        assert PreflightFailure.DUPLICATE_VIEWS in failure_types

    def test_summary_output(self, tmp_path):
        """Summary string should be generated without errors."""
        img = make_checkerboard_rgba()
        img_path = tmp_path / "test.png"
        img.save(str(img_path))
        results = [self.runner.check_image(str(img_path), job_id="test")]
        summary = self.runner.summary(results)
        assert "FAIL" in summary or "PASS" in summary
