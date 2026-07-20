"""
local_asset_factory · preflight · preflight_runner
Orchestrates all preflight checks for a view or view set.

Gates:
  alpha:
    require_true_alpha_or_uniform_background: true
  checkerboard:
    reject_baked_pattern: true
  pose:
    max_shoulder_angle_error_deg: 5
    max_elbow_flexion_deg: 7
  multiview:
    require_front: true
    require_left: true
    require_back: true
    right_view_policy: validation

CLI usage:
    asset-factory preflight input/front.png
    asset-factory validate-views views.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from ..domain.enums import PreflightFailure, SeverityLevel
from ..domain.models import PreflightCheck, PreflightResult
from .alpha_detector import detect_alpha
from .checkerboard_detector import detect_checkerboard

log = logging.getLogger(__name__)


class PreflightConfig:
    """Preflight gate configuration."""
    require_true_alpha_or_uniform_background: bool = True
    reject_baked_checkerboard: bool = True
    require_front: bool = True
    require_left: bool = True
    require_back: bool = True
    right_view_policy: str = "validation"    # "validation" | "optional" | "required"
    max_shoulder_angle_error_deg: float = 5.0
    max_elbow_flexion_deg: float = 7.0
    min_resolution: int = 512
    max_background_fraction: float = 0.50
    checkerboard_confidence_threshold: float = 0.55


DEFAULT_CONFIG = PreflightConfig()


class PreflightRunner:
    """
    Runs all preflight checks on a single image or a set of views.
    Returns structured results with severity levels.
    Failures with severity FATAL block the pipeline.
    """

    def __init__(self, config: Optional[PreflightConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def check_image(
        self,
        image_path: str | Path,
        job_id: str = "unknown",
        view_orientation: str = "front",
    ) -> PreflightResult:
        """
        Run all applicable checks on a single image file.
        Returns PreflightResult with all check outcomes.
        """
        path = Path(image_path)
        result = PreflightResult(job_id=job_id, view=view_orientation)
        checks: List[PreflightCheck] = []

        if not path.exists():
            checks.append(PreflightCheck(
                check=PreflightFailure.SUBJECT_CROPPED,
                passed=False,
                message=f"Image file not found: {path}",
                severity=SeverityLevel.FATAL,
            ))
            result.checks = checks
            result.passed = False
            return result

        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            checks.append(PreflightCheck(
                check=PreflightFailure.SUBJECT_CROPPED,
                passed=False,
                message=f"Cannot open image: {e}",
                severity=SeverityLevel.FATAL,
            ))
            result.checks = checks
            result.passed = False
            return result

        # --- Check 1: Checkerboard background ---
        checker_result = detect_checkerboard(
            img,
            confidence_threshold=self.config.checkerboard_confidence_threshold,
        )
        if self.config.reject_baked_checkerboard:
            checks.append(PreflightCheck(
                check=PreflightFailure.BAKED_CHECKERBOARD_BACKGROUND,
                passed=not checker_result.detected,
                message=checker_result.message,
                severity=SeverityLevel.FATAL if checker_result.detected else SeverityLevel.INFO,
            ))

        # --- Check 2: Alpha channel quality ---
        alpha_result = detect_alpha(img)
        if self.config.require_true_alpha_or_uniform_background:
            # Fail if: has RGBA but no true alpha (baked checkerboard scenario)
            if alpha_result.has_alpha_channel and not alpha_result.true_alpha:
                checks.append(PreflightCheck(
                    check=PreflightFailure.NO_TRUE_ALPHA,
                    passed=False,
                    message=alpha_result.message,
                    severity=SeverityLevel.ERROR,
                ))
            else:
                checks.append(PreflightCheck(
                    check=PreflightFailure.NO_TRUE_ALPHA,
                    passed=True,
                    message=alpha_result.message,
                    severity=SeverityLevel.INFO,
                ))

        # --- Check 3: Resolution ---
        w, h = img.size
        if min(w, h) < self.config.min_resolution:
            checks.append(PreflightCheck(
                check=PreflightFailure.INSUFFICIENT_RESOLUTION,
                passed=False,
                message=f"Image {w}x{h} is below minimum {self.config.min_resolution}px",
                severity=SeverityLevel.ERROR,
            ))

        result.checks = checks
        result.passed = all(c.passed for c in checks if c.severity in (
            SeverityLevel.ERROR, SeverityLevel.FATAL
        ))
        return result

    def validate_view_set(
        self,
        views: Dict[str, str],
        job_id: str = "unknown",
    ) -> List[PreflightResult]:
        """
        Validate a complete view set.
        Returns one PreflightResult per view.
        """
        results: List[PreflightResult] = []

        # Check required views present
        missing = []
        if self.config.require_front and "front" not in views:
            missing.append("front")
        if self.config.require_left and "left" not in views:
            missing.append("left")
        if self.config.require_back and "back" not in views:
            missing.append("back")
        if self.config.right_view_policy == "required" and "right" not in views:
            missing.append("right")

        if missing:
            dummy = PreflightResult(job_id=job_id, view="view_set")
            dummy.checks = [
                PreflightCheck(
                    check=PreflightFailure.INCONSISTENT_VIEWS,
                    passed=False,
                    message=f"Missing required views: {missing}",
                    severity=SeverityLevel.FATAL,
                )
            ]
            dummy.passed = False
            results.append(dummy)
            return results

        # Check for duplicate views (by hash)
        seen_hashes: Dict[str, str] = {}
        for orientation, path in views.items():
            p = Path(path)
            if p.exists():
                import hashlib
                h = hashlib.md5(p.read_bytes()).hexdigest()[:12]
                if h in seen_hashes:
                    dup_result = PreflightResult(job_id=job_id, view=orientation)
                    dup_result.checks = [
                        PreflightCheck(
                            check=PreflightFailure.DUPLICATE_VIEWS,
                            passed=False,
                            message=(
                                f"View '{orientation}' appears to be identical to "
                                f"view '{seen_hashes[h]}' (same image hash)."
                            ),
                            severity=SeverityLevel.ERROR,
                        )
                    ]
                    dup_result.passed = False
                    results.append(dup_result)
                else:
                    seen_hashes[h] = orientation

        # Check each view individually
        for orientation, path in views.items():
            r = self.check_image(path, job_id=job_id, view_orientation=orientation)
            results.append(r)

        return results

    def all_passed(self, results: List[PreflightResult]) -> bool:
        return all(r.passed for r in results)

    def summary(self, results: List[PreflightResult]) -> str:
        lines = []
        for r in results:
            view = r.view or "unknown"
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {view}")
            for c in r.failures():
                lines.append(f"         {c.severity.value.upper()}: {c.check.value} — {c.message}")
        return "\n".join(lines)
