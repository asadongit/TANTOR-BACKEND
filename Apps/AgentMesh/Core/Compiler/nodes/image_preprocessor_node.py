"""
Image Preprocessor Node
Cleans raw signature and stamp crops extracted from documents.

Strategy (Approach 2 - Document-Adaptive):
- Signature: CLAHE contrast enhancement → Gaussian adaptive binarization → black ink on white.
  Works on any ink color, paper color, or lighting condition.
- Stamp: Light denoise + normalize only (no binarization). Preserves color/texture features
  that SIFT and the Siamese model rely on for matching.
"""

import base64
import io
import logging
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _b64_to_cv2(b64_str: str) -> np.ndarray:
    """Decode a base64 image string (with or without data-URI prefix) to a BGR cv2 array."""
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _cv2_to_b64(img: np.ndarray, fmt: str = ".png") -> str:
    """Encode a cv2 BGR array to a base64 PNG string."""
    success, buffer = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image to base64")
    return base64.b64encode(buffer).decode("utf-8")


def _ensure_white_background(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert image to RGBA, make near-white pixels fully transparent,
    then composite onto a pure white canvas. Ensures no grey artifacts.
    """
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    # Mark very light pixels (background) as transparent
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    rgba[:, :, 3] = cv2.bitwise_not(mask)
    # Composite onto white
    white = np.ones_like(img_bgr, dtype=np.uint8) * 255
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    result = (img_bgr.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# Core Processing Functions
# ─────────────────────────────────────────────

def preprocess_signature(img_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    Adaptive signature preprocessing — works on any document type.

    Strategy:
      1. Convert to greyscale.
      2. Apply CLAHE to boost local contrast (handles poor lighting/faded ink).
      3. Gaussian adaptive threshold → binary image (inverted: ink=black, bg=white).
      4. Light morphological closing to reconnect broken strokes.

    This is document-agnostic: it makes no assumptions about ink color or paper color.
    The mode param is kept for API compatibility but is unused.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: boost local contrast so faint ink becomes distinguishable
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive Gaussian threshold — block size 21, C=10 works well for documents
    binary = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )
    # binary is already: ink=black (0), bg=white (255) — correct orientation

    # Light closing to reconnect thin strokes broken by thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Convert back to 3-channel for consistency with rest of pipeline
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return output


def preprocess_stamp(img_bgr: np.ndarray, stamp_color: str = "auto") -> np.ndarray:
    """
    Light stamp preprocessing — preserves color and texture for SIFT/Siamese matching.

    Strategy:
      - Denoise with a fast bilateral filter (preserves edges, removes grain).
      - Normalize brightness with CLAHE on the L channel (handles variable lighting).
      - No binarization — SIFT and the Siamese model need real texture/color information.

    The stamp_color param is kept for API compatibility but is unused.
    """
    # Bilateral filter: smooths noise while keeping stamp edges sharp
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Normalize brightness in LAB color space via CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a, b])
    output = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return output


def _get_b64_list(data) -> list:
    """Normalize form field data (single dict, list of dicts, or raw string) to a list of b64 strings."""
    if not data:
        return []
    if isinstance(data, list):
        return [d.get("data") if isinstance(d, dict) else d for d in data if d]
    if isinstance(data, dict):
        return [data.get("data")] if data.get("data") else []
    return [data]


# ─────────────────────────────────────────────
# Node Builder
# ─────────────────────────────────────────────

def build_image_preprocessor_node(node_config: NodeConfig) -> Callable:
    """
    Builds an Image Preprocessor node.

    Reads raw signature and stamp base64 images from context,
    applies document-adaptive preprocessing, and writes cleaned images back
    as 'clean_signature_b64' and 'clean_stamp_b64'.

    Config options (node_config extra fields):
        signature_mode: kept for compatibility, unused (adaptive threshold is always used)
        stamp_color:    kept for compatibility, unused (bilateral+CLAHE always used)
    """
    signature_mode = getattr(node_config, "signature_mode", "auto")
    stamp_color = getattr(node_config, "stamp_color", "auto")

    async def execute_preprocessing(state: AgentState):
        ctx = state.get("context", {})
        node_id = node_config.id
        results = {}
        errors = []

        # Pull raw images from context (set by Signature/Stamp Extractor node)
        raw_sig_data = ctx.get("payload", {}).get("signature_b64") or ctx.get("signature_b64")
        raw_stamp_data = ctx.get("payload", {}).get("stamp_b64") or ctx.get("stamp_b64")

        sig_b64_list = _get_b64_list(raw_sig_data)
        stamp_b64_list = _get_b64_list(raw_stamp_data)

        if not sig_b64_list and not stamp_b64_list:
            msg = f"[WARN] Preprocessor {node_id}: No signature_b64 or stamp_b64 found in context. Passing through unchanged."
            logger.warning(msg)
            return {
                "messages": [AIMessage(content=msg)],
                "active_agent": node_id,
                "context": ctx
            }

        # ── Process Signatures ──
        clean_signatures = []
        for i, b64 in enumerate(sig_b64_list):
            if not b64:
                continue
            try:
                img_bgr = _b64_to_cv2(b64)
                clean_img = preprocess_signature(img_bgr, mode=signature_mode)
                clean_b64 = _cv2_to_b64(clean_img)
                clean_signatures.append(clean_b64)
                logger.info(f"Preprocessor {node_id}: Signature {i+1}/{len(sig_b64_list)} cleaned OK")
            except Exception as e:
                errors.append(f"Signature {i+1} error: {e}")
                logger.error(f"Preprocessor {node_id}: Signature {i+1} failed: {e}")

        # ── Process Stamps ──
        clean_stamps = []
        for i, b64 in enumerate(stamp_b64_list):
            if not b64:
                continue
            try:
                img_bgr = _b64_to_cv2(b64)
                clean_img = preprocess_stamp(img_bgr, stamp_color=stamp_color)
                clean_b64 = _cv2_to_b64(clean_img)
                clean_stamps.append(clean_b64)
                logger.info(f"Preprocessor {node_id}: Stamp {i+1}/{len(stamp_b64_list)} cleaned OK")
            except Exception as e:
                errors.append(f"Stamp {i+1} error: {e}")
                logger.error(f"Preprocessor {node_id}: Stamp {i+1} failed: {e}")

        # ── Update context ──
        updated_ctx = {
            **ctx,
            "clean_signature_b64": clean_signatures[0] if len(clean_signatures) == 1 else clean_signatures,
            "clean_stamp_b64": clean_stamps[0] if len(clean_stamps) == 1 else clean_stamps,
            # Keep originals available downstream
            "raw_signature_b64": sig_b64_list,
            "raw_stamp_b64": stamp_b64_list,
        }

        summary = (
            f"✅ Preprocessor {node_id}: Cleaned {len(clean_signatures)} signature(s) "
            f"and {len(clean_stamps)} stamp(s)."
        )
        if errors:
            summary += f" ⚠️ Errors: {'; '.join(errors)}"

        logger.info(summary)
        return {
            "messages": [AIMessage(content=summary, name=f"Preprocessor_{node_id}")],
            "active_agent": node_id,
            "context": updated_ctx
        }

    return RunnableLambda(execute_preprocessing)
