"""
Image Quality Gate — Pre-OCR Document Image Validation
=======================================================

Validates uploaded document images for sufficient visual quality before they
reach downstream processing. Catches blurry, over/under-exposed, low-
contrast, low-resolution, and blank documents early and returns user-friendly
guidance for recapture.

Metrics (industry-standard):
  • Blur         — Laplacian variance (Pech-Pacheco filter)
  • Contrast     — Normalised histogram standard deviation
  • Brightness   — Mean pixel intensity (over/under-exposure)
  • Resolution   — Shortest-side pixel count (too-small = illegible)
  • Blank page   — Non-zero pixel ratio (empty scans)

All metrics run locally with zero LLM calls — fast (<50 ms per page) and cheap.

Source: Ported from TantorGenAIBackend/Apps/CREDITDECISIONING/ImageQualityGate.py
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Optional PDF support
try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.debug("PyMuPDF (fitz) not available — PDF quality assessment disabled.")


# ============================================================================
# METRIC NAMES
# ============================================================================


class QualityMetric(str, Enum):
    BLUR = "blur"
    CONTRAST = "contrast"
    BRIGHTNESS = "brightness"
    RESOLUTION = "resolution"
    BLANK_PAGE = "blank_page"


# ============================================================================
# DEFAULT THRESHOLDS  (tuned for 144 DPI financial documents)
# ============================================================================

_DEFAULT_BLUR_THRESHOLD = 100.0  # Laplacian variance — < 100 = blurry
_DEFAULT_MIN_CONTRAST = 0.10  # Normalised std dev — < 0.10 = washed out
_DEFAULT_MAX_CONTRAST = 0.45  # Normalised std dev — > 0.45 = too harsh/binary
_DEFAULT_MIN_BRIGHTNESS = 40.0  # Mean pixel — < 40 = underexposed
_DEFAULT_MAX_BRIGHTNESS = 215.0  # Mean pixel — > 215 = overexposed
_DEFAULT_MIN_SHORTEST_SIDE = 600  # Pixels — < 600px = too small for OCR
_DEFAULT_MIN_NON_BLANK_PCT = 1.0  # Percent — < 1% non-zero pixels = blank/empty


# ============================================================================
# USER-FRIENDLY MESSAGES  (per metric, shown to end-user for recapture)
# ============================================================================

_USER_MESSAGES: Dict[str, str] = {
    QualityMetric.BLUR: (
        "This document image is too blurry. Please recapture with:\n"
        "  • Steady hands or a flatbed scanner\n"
        "  • Good lighting — avoid dim environments\n"
        "  • Camera held parallel to the document"
    ),
    QualityMetric.CONTRAST: (
        "This document image has poor contrast (washed out or faded). Please:\n"
        "  • Ensure the document is well-lit and evenly illuminated\n"
        "  • Avoid shadows falling across the page\n"
        "  • Use a dark background if capturing on a phone"
    ),
    QualityMetric.BRIGHTNESS: (
        "This document image is too dark or too bright. Please:\n"
        "  • Capture in even, ambient lighting (not direct sunlight)\n"
        "  • Turn off flash if using a phone camera\n"
        "  • Avoid heavy back-lighting"
    ),
    QualityMetric.RESOLUTION: (
        "This document image resolution is too low for accurate OCR. Please:\n"
        "  • Hold the camera closer to the document\n"
        "  • Use a higher-resolution scanner setting (minimum 144 DPI)\n"
        "  • Ensure the document fills most of the frame"
    ),
    QualityMetric.BLANK_PAGE: (
        "This page appears to be blank or mostly empty. Please:\n"
        "  • Verify all pages of the document are included\n"
        "  • Check that the scanner/camera captured the content\n"
        "  • Ensure pages aren't stuck together in the scanner"
    ),
}


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class MetricResult:
    """Outcome of a single quality metric check."""

    metric: str
    value: float
    threshold: float
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value
            if isinstance(self.metric, Enum)
            else str(self.metric),
            "value": round(float(self.value), 4),
            "threshold": float(self.threshold),
            "passed": bool(self.passed),
        }


@dataclass
class QualityResult:
    """Aggregated quality assessment for a single page/image."""

    passed: bool
    overall_score: float  # 0.0 — 1.0 composite over all metrics
    metrics: List[MetricResult]  # per-metric details
    failure_reasons: List[str]  # user-facing failure messages
    user_message: str  # single combined message (newline-separated)
    page_number: int = 1
    filename: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_score": round(self.overall_score, 4),
            "metrics": [m.as_dict() for m in self.metrics],
            "failure_reasons": self.failure_reasons,
            "user_message": self.user_message,
            "page_number": self.page_number,
            "filename": self.filename,
        }


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ImageQualityConfig:
    """Tunable thresholds for the image quality gate.

    All thresholds have sensible defaults tuned for financial documents at
    ~144 DPI.

    Environment variable overrides (applied after dict config):
        IQG_ENABLED             → enabled
        IQG_BLUR_THRESHOLD       → blur_threshold
        IQG_MIN_CONTRAST         → min_contrast
        IQG_MIN_BRIGHTNESS       → min_brightness
        IQG_MAX_BRIGHTNESS       → max_brightness
        IQG_MIN_SHORTEST_SIDE    → min_shortest_side_px
        IQG_MIN_NON_BLANK_PCT    → min_non_blank_pct
    """

    enabled: bool = True
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD
    min_contrast: float = _DEFAULT_MIN_CONTRAST
    max_contrast: float = _DEFAULT_MAX_CONTRAST
    min_brightness: float = _DEFAULT_MIN_BRIGHTNESS
    max_brightness: float = _DEFAULT_MAX_BRIGHTNESS
    min_shortest_side_px: int = _DEFAULT_MIN_SHORTEST_SIDE
    min_non_blank_pct: float = _DEFAULT_MIN_NON_BLANK_PCT

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ImageQualityConfig":
        if not d:
            return cls()
        return cls(
            enabled=d.get("enabled", True),
            blur_threshold=d.get("blur_threshold", _DEFAULT_BLUR_THRESHOLD),
            min_contrast=d.get("min_contrast", _DEFAULT_MIN_CONTRAST),
            max_contrast=d.get("max_contrast", _DEFAULT_MAX_CONTRAST),
            min_brightness=d.get("min_brightness", _DEFAULT_MIN_BRIGHTNESS),
            max_brightness=d.get("max_brightness", _DEFAULT_MAX_BRIGHTNESS),
            min_shortest_side_px=d.get("min_shortest_side", _DEFAULT_MIN_SHORTEST_SIDE),
            min_non_blank_pct=d.get("min_non_blank_pct", _DEFAULT_MIN_NON_BLANK_PCT),
        )

    @classmethod
    def from_env(cls) -> "ImageQualityConfig":
        return cls(
            enabled=os.getenv("IQG_ENABLED", "true").lower() == "true",
            blur_threshold=float(
                os.getenv("IQG_BLUR_THRESHOLD", str(_DEFAULT_BLUR_THRESHOLD))
            ),
            min_contrast=float(
                os.getenv("IQG_MIN_CONTRAST", str(_DEFAULT_MIN_CONTRAST))
            ),
            max_contrast=float(
                os.getenv("IQG_MAX_CONTRAST", str(_DEFAULT_MAX_CONTRAST))
            ),
            min_brightness=float(
                os.getenv("IQG_MIN_BRIGHTNESS", str(_DEFAULT_MIN_BRIGHTNESS))
            ),
            max_brightness=float(
                os.getenv("IQG_MAX_BRIGHTNESS", str(_DEFAULT_MAX_BRIGHTNESS))
            ),
            min_shortest_side_px=int(
                os.getenv("IQG_MIN_SHORTEST_SIDE", str(_DEFAULT_MIN_SHORTEST_SIDE))
            ),
            min_non_blank_pct=float(
                os.getenv("IQG_MIN_NON_BLANK_PCT", str(_DEFAULT_MIN_NON_BLANK_PCT))
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "blur_threshold": self.blur_threshold,
            "min_contrast": self.min_contrast,
            "max_contrast": self.max_contrast,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "min_shortest_side_px": self.min_shortest_side_px,
            "min_non_blank_pct": self.min_non_blank_pct,
        }


# ============================================================================
# IMAGE QUALITY GATE
# ============================================================================


class ImageQualityGate:
    """Pre-OCR image quality assessor.

    Runs 5 fast computer-vision checks on each page before any LLM call.
    Configurable thresholds; disabled by default so existing workflows are
    unaffected until a template opts in.

    Thread-safe: all metric methods are @staticmethod or operate on local state.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        base = ImageQualityConfig.from_dict(config)
        env_overrides = ImageQualityConfig.from_env()

        self.config = ImageQualityConfig(
            enabled=base.enabled
            if config and "enabled" in config
            else env_overrides.enabled,
            blur_threshold=base.blur_threshold
            if config and "blur_threshold" in config
            else env_overrides.blur_threshold,
            min_contrast=base.min_contrast
            if config and "min_contrast" in config
            else env_overrides.min_contrast,
            max_contrast=base.max_contrast
            if config and "max_contrast" in config
            else env_overrides.max_contrast,
            min_brightness=base.min_brightness
            if config and "min_brightness" in config
            else env_overrides.min_brightness,
            max_brightness=base.max_brightness
            if config and "max_brightness" in config
            else env_overrides.max_brightness,
            min_shortest_side_px=base.min_shortest_side_px
            if config and "min_shortest_side" in config
            else env_overrides.min_shortest_side_px,
            min_non_blank_pct=base.min_non_blank_pct
            if config and "min_non_blank_pct" in config
            else env_overrides.min_non_blank_pct,
        )

        logger.info(
            "[QualityGate] Initialized: enabled=%s, blur_thresh=%.1f, "
            "contrast=%.2f, brightness=[%.0f..%.0f], min_side=%dpx, blank=%.1f%%",
            self.config.enabled,
            self.config.blur_threshold,
            self.config.min_contrast,
            self.config.min_brightness,
            self.config.max_brightness,
            self.config.min_shortest_side_px,
            self.config.min_non_blank_pct,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def assess_single_image(
        self,
        image: Image.Image,
        page_number: int = 1,
        filename: str = "",
    ) -> QualityResult:
        """Run all quality metrics on a single PIL image."""
        if not self.config.enabled:
            return QualityResult(
                passed=True,
                overall_score=1.0,
                metrics=[],
                failure_reasons=[],
                user_message="",
                page_number=page_number,
                filename=filename,
            )

        metric_results: List[MetricResult] = []
        failure_reasons: List[str] = []

        metric_results.append(self._check_blur(image))
        metric_results.append(self._check_contrast(image))
        metric_results.append(self._check_brightness(image))
        metric_results.append(self._check_resolution(image))
        metric_results.append(self._check_blank_page(image))

        for mr in metric_results:
            if not mr.passed:
                failure_reasons.append(
                    _USER_MESSAGES.get(mr.metric, f"Quality check failed: {mr.metric}")
                )

        passed_count = sum(1 for m in metric_results if m.passed)
        overall_score = passed_count / len(metric_results) if metric_results else 1.0

        user_message = "\n\n".join(failure_reasons) if failure_reasons else ""

        return QualityResult(
            passed=len(failure_reasons) == 0,
            overall_score=overall_score,
            metrics=metric_results,
            failure_reasons=failure_reasons,
            user_message=user_message,
            page_number=page_number,
            filename=filename,
        )

    def assess_file(
        self,
        file_path: str,
        filename: str = "",
    ) -> List[QualityResult]:
        """Assess quality of all pages in a file (PDF or single image)."""
        ext = Path(file_path).suffix.lower()
        results: List[QualityResult] = []

        if ext == ".pdf":
            if not HAS_FITZ:
                logger.warning(
                    "[QualityGate] PyMuPDF not installed — cannot assess PDF '%s'",
                    filename,
                )
                results.append(
                    QualityResult(
                        passed=True,
                        overall_score=1.0,
                        metrics=[],
                        failure_reasons=[],
                        user_message="",
                        page_number=1,
                        filename=filename,
                    )
                )
                return results
            try:
                pdf_doc = fitz.open(file_path)
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    result = self.assess_single_image(img, page_num + 1, filename)
                    results.append(result)
                pdf_doc.close()
            except Exception as e:
                logger.error("[QualityGate] Failed to assess PDF '%s': %s", filename, e)
                results.append(
                    QualityResult(
                        passed=False,
                        overall_score=0.0,
                        metrics=[],
                        failure_reasons=[f"Could not read file: {e}"],
                        user_message=f"Could not read file: {e}",
                        page_number=1,
                        filename=filename,
                    )
                )
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            try:
                img = Image.open(file_path)
                result = self.assess_single_image(img, 1, filename)
                results.append(result)
            except Exception as e:
                logger.error(
                    "[QualityGate] Failed to assess image '%s': %s", filename, e
                )
                results.append(
                    QualityResult(
                        passed=False,
                        overall_score=0.0,
                        metrics=[],
                        failure_reasons=[f"Could not read image: {e}"],
                        user_message=f"Could not read image: {e}",
                        page_number=1,
                        filename=filename,
                    )
                )
        else:
            logger.warning(
                "[QualityGate] Unsupported file type '%s' for '%s' — skipping",
                ext,
                filename,
            )
            results.append(
                QualityResult(
                    passed=True,
                    overall_score=1.0,
                    metrics=[],
                    failure_reasons=[],
                    user_message="",
                    page_number=1,
                    filename=filename,
                )
            )

        return results

    # ── Per-Metric Checks ───────────────────────────────────────────────────

    def _check_blur(self, image: Image.Image) -> MetricResult:
        img_gray = self._to_grayscale_cv(image)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return MetricResult(
            metric=QualityMetric.BLUR,
            value=round(laplacian_var, 2),
            threshold=self.config.blur_threshold,
            passed=laplacian_var >= self.config.blur_threshold,
        )

    def _check_contrast(self, image: Image.Image) -> MetricResult:
        img_gray = self._to_grayscale_cv(image)
        std_dev = img_gray.std() / 255.0
        passed = self.config.min_contrast <= std_dev <= self.config.max_contrast
        relevant_threshold = (
            self.config.min_contrast
            if std_dev < self.config.min_contrast
            else self.config.max_contrast
        )
        return MetricResult(
            metric=QualityMetric.CONTRAST,
            value=round(std_dev, 4),
            threshold=relevant_threshold,
            passed=passed,
        )

    def _check_brightness(self, image: Image.Image) -> MetricResult:
        img_gray = self._to_grayscale_cv(image)
        mean = float(img_gray.mean())
        if self.config.min_brightness <= mean <= self.config.max_brightness:
            passed = True
            relevant_threshold = self.config.max_brightness
        else:
            passed = False
            relevant_threshold = (
                self.config.min_brightness
                if mean < self.config.min_brightness
                else self.config.max_brightness
            )
        return MetricResult(
            metric=QualityMetric.BRIGHTNESS,
            value=round(mean, 2),
            threshold=relevant_threshold,
            passed=passed,
        )

    def _check_resolution(self, image: Image.Image) -> MetricResult:
        w, h = image.size
        shortest = min(w, h)
        return MetricResult(
            metric=QualityMetric.RESOLUTION,
            value=float(shortest),
            threshold=float(self.config.min_shortest_side_px),
            passed=shortest >= self.config.min_shortest_side_px,
        )

    def _check_blank_page(self, image: Image.Image) -> MetricResult:
        img_gray = self._to_grayscale_cv(image)
        total_pixels = img_gray.size
        non_zero = cv2.countNonZero(img_gray)
        non_zero_pct = (non_zero / total_pixels) * 100.0
        return MetricResult(
            metric=QualityMetric.BLANK_PAGE,
            value=round(non_zero_pct, 2),
            threshold=self.config.min_non_blank_pct,
            passed=non_zero_pct >= self.config.min_non_blank_pct,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_grayscale_cv(image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("L"))
        return arr.astype(np.uint8)
