import base64
import json
import logging
import os
import re
import uuid
from typing import Callable, Any, List, Dict, Optional
import cv2
import numpy as np

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_config import get_vision_llm
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color-based ink candidate detection
# ---------------------------------------------------------------------------

def get_ink_candidates(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Uses color-specific HSV thresholding to find all colored ink blobs
    (red stamps, blue/dark signatures).  Avoids detecting black printed text.
    """
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    candidates = []

    # ── 1. Red ink (stamps / seals) ──────────────────────────────────────────
    # High saturation (>=95) to skip pink card backgrounds; catches saturated red stamp ink.
    lower_red1 = np.array([0,   95,  50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165,  95,  50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    kernel_stamp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_stamp)
    contours_red, _ = cv2.findContours(closed_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # Minimum area of 1500px² filters small noise; real stamps are substantially sized
        if area > 1500:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect-ratio guard: stamps are roughly square (not flat strips)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if bw < w * 0.8 and bh < h * 0.8 and aspect < 3.5:
                # Must be at least 40×40 pixels to be a real stamp impression
                if bw >= 40 and bh >= 40:
                    pad = 15
                    candidates.append({
                        "type": "stamp",
                        "color": "red",
                        "box": [max(0, y - pad), max(0, x - pad),
                                min(h, y + bh + pad), min(w, x + bw + pad)],
                        "area": int(area)
                    })

    # ── 2. Blue / purple ink (stamps / signatures) ────────────────────────────
    # Range H=80-145 covers cyan-blue through blue to purple/violet.
    # Official stamps often use royal blue (H≈100-120) or dark blue (H≈85-100).
    lower_blue = np.array([80,  25,  30])
    upper_blue = np.array([145, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove red regions so stamp ink doesn't bleed into signature candidates
    lower_red_sub1 = np.array([0,  30, 30])
    upper_red_sub1 = np.array([25, 255, 255])
    lower_red_sub2 = np.array([155, 30, 30])
    upper_red_sub2 = np.array([180, 255, 255])
    mask_red_sub = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_sub1, upper_red_sub1),
        cv2.inRange(hsv, lower_red_sub2, upper_red_sub2)
    )
    mask_blue = cv2.subtract(mask_blue, mask_red_sub)

    contours_raw, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_strokes_mask = np.zeros_like(mask_blue)

    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if area > 800 or (bw > 45 and bh > 45):
            if not (bh < 22 and bw > 80):
                pad = 15
                if bw < w * 0.8 and bh < h * 0.65:
                    candidates.append({
                        "type": "signature",
                        "color": "blue",
                        "is_large_blue_contour": True,
                        "box": [max(0, y - pad), max(0, x - pad),
                                min(h, y + bh + pad), min(w, x + bw + pad)],
                        "area": int(area)
                    })
        else:
            if area > 15 and max(bw, bh) > 5:
                cv2.drawContours(small_strokes_mask, [cnt], -1, 255, -1)

    # Erase large-blue regions from the small-strokes mask to prevent merging
    for cand in candidates:
        if cand.get("is_large_blue_contour"):
            y1, x1, y2, x2 = cand["box"]
            cv2.rectangle(small_strokes_mask, (x1, y1), (x2, y2), 0, -1)

    kernel_sig = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
    closed_sig = cv2.morphologyEx(small_strokes_mask, cv2.MORPH_CLOSE, kernel_sig)
    contours_sig, _ = cv2.findContours(closed_sig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_sig:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < w * 0.8 and bh < h * 0.65 and not (bh < 22 and bw > 80):
                pad = 15
                candidates.append({
                    "type": "signature",
                    "box": [max(0, y - pad), max(0, x - pad),
                            min(h, y + bh + pad), min(w, x + bw + pad)],
                    "area": int(area)
                })

    # ── Iterative union-merge (same type only) ────────────────────────────────
    changed = True
    while changed:
        changed = False
        merged_list: List[Dict] = []
        for cand in candidates:
            cy1, cx1, cy2, cx2 = cand["box"]
            did_merge = False
            for uc in merged_list:
                if cand["type"] != uc["type"]:
                    continue
                uy1, ux1, uy2, ux2 = uc["box"]
                iy1, ix1 = max(cy1, uy1), max(cx1, ux1)
                iy2, ix2 = min(cy2, uy2), min(cx2, ux2)
                if iy2 > iy1 and ix2 > ix1:
                    inter = (iy2 - iy1) * (ix2 - ix1)
                    c_area = max(1, (cy2 - cy1) * (cx2 - cx1))
                    u_area = max(1, (uy2 - uy1) * (ux2 - ux1))
                    if inter / min(c_area, u_area) > 0.15:
                        uc["box"] = [min(uy1, cy1), min(ux1, cx1),
                                     max(uy2, cy2), max(ux2, cx2)]
                        did_merge = True
                        changed = True
                        break
            if not did_merge:
                merged_list.append({**cand})
        candidates = merged_list

    return candidates


# ---------------------------------------------------------------------------
# Precise crop trimming
# ---------------------------------------------------------------------------

def trim_to_ink(img: np.ndarray, box: List[int], cand_type: str = "general",
                margin: int = 12) -> List[int]:
    """
    Tightly trims a bounding box to the actual ink pixels using color-aware masks.

    For STAMPS  → uses the dominant ink color (red OR blue/purple) so we snap
                  exactly to the stamp ring, not to adjacent printed text.
    For SIGNATURES → uses non-red colored ink to isolate pen strokes from
                  any overlapping stamp border.
    Falls back to Otsu grayscale if no color mask yields enough pixels.
    """
    y1, x1, y2, x2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return box

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv_crop[:, :, 0]
    s_ch = hsv_crop[:, :, 1]
    v_ch = hsv_crop[:, :, 2]
    mask: Optional[np.ndarray] = None

    if cand_type == "stamp":
        # ── Try red ink first ──
        red_mask = (
            ((h_ch < 20) | (h_ch > 155)) & (s_ch >= 50) & (v_ch >= 40)
        ).astype(np.uint8) * 255

        # ── Then blue/purple ink ──
        blue_mask = (
            (h_ch >= 80) & (h_ch <= 145) & (s_ch >= 35) & (v_ch >= 40)
        ).astype(np.uint8) * 255

        n_red  = int(cv2.countNonZero(red_mask))
        n_blue = int(cv2.countNonZero(blue_mask))

        if n_red >= n_blue and n_red >= 80:
            logger.debug(f"trim_to_ink: stamp → RED mask ({n_red}px)")
            mask = red_mask
            margin = 25
        elif n_blue >= 80:
            logger.debug(f"trim_to_ink: stamp → BLUE/PURPLE mask ({n_blue}px)")
            mask = blue_mask
            margin = 25
        else:
            # No dominant colored ink — return padded box as-is.
            # NEVER use Otsu for stamps: Otsu catches printed text (names/titles) near the stamp.
            logger.debug("trim_to_ink: stamp → no colored ink found, returning padded box")
            return box

    elif cand_type == "signature":
        # Non-red colored ink (blue pen, black ink that has slight hue, etc.)
        sig_mask = cv2.inRange(hsv_crop,
                               np.array([15, 12, 15]),
                               np.array([165, 255, 255]))
        # Subtract red to remove overlapping stamp borders
        red_sub = cv2.bitwise_or(
            cv2.inRange(hsv_crop, np.array([0,  15, 15]), np.array([25, 255, 255])),
            cv2.inRange(hsv_crop, np.array([155, 15, 15]), np.array([180, 255, 255]))
        )
        sig_mask = cv2.subtract(sig_mask, red_sub)

        if cv2.countNonZero(sig_mask) >= 20:
            mask = sig_mask
        else:
            # Otsu fallback with red subtraction
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            red_sub2 = cv2.bitwise_or(
                cv2.inRange(hsv_crop, np.array([0,  15, 15]), np.array([25, 255, 255])),
                cv2.inRange(hsv_crop, np.array([155, 15, 15]), np.array([180, 255, 255]))
            )
            mask = cv2.subtract(mask, red_sub2)

    # Global Otsu fallback — only for non-stamp types
    # Stamps must never use Otsu: it catches all dark pixels including printed text
    if mask is None or cv2.countNonZero(mask) < 20:
        if cand_type == "stamp":
            return box  # Return padded box, never Otsu for stamps
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate slightly to bridge hairline gaps in thin strokes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) == 0:
        return box

    nx1 = int(coords[:, 0, 0].min())
    ny1 = int(coords[:, 0, 1].min())
    nx2 = int(coords[:, 0, 0].max())
    ny2 = int(coords[:, 0, 1].max())

    result_y1 = max(0,            y1 + ny1 - margin)
    result_x1 = max(0,            x1 + nx1 - margin)
    result_y2 = min(img.shape[0], y1 + ny2 + margin)
    result_x2 = min(img.shape[1], x1 + nx2 + margin)

    # For signatures: clamp the top so the crop cannot expand ABOVE the original
    # (padded) LLM box top — prevents stamp rings or printed text above the
    # signature from pulling the crop boundary upward.
    if cand_type == "signature":
        result_y1 = max(result_y1, y1)

    return [result_y1, result_x1, result_y2, result_x2]


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_json_robust(text: str) -> Dict[str, Any]:
    """Parses JSON from an LLM response, tolerating common formatting errors
    including truncated responses missing closing braces/brackets."""
    # Fix trailing double-quote after a number: : 835" → : 835
    text = re.sub(r'(:\s*\d+(?:\.\d+)?)"', r'\1', text)

    # First try: find a well-formed {..} block
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        raw_json = json_match.group(0)
        try:
            return json.loads(raw_json)
        except Exception:
            try:
                import ast
                return ast.literal_eval(raw_json)
            except Exception:
                pass

    # Second try: find a partial { block and auto-complete it
    partial_match = re.search(r"\{.*", text, re.DOTALL)
    if partial_match:
        partial = partial_match.group(0)
        # Try adding common missing suffixes to close truncated JSON
        for suffix in ["}", "]}", "}]}", "}}]", "}}"]:
            try:
                return json.loads(partial + suffix)
            except Exception:
                continue

    return {}


# ---------------------------------------------------------------------------
# LLM detection prompts
# ---------------------------------------------------------------------------

async def detect_signatures_only(llm: Any, img: np.ndarray) -> List[Dict[str, Any]]:
    """
    LLM call that detects ONLY handwritten signatures in the full document image.
    Returns a list of bounding boxes in 0-1000 normalized coordinates.
    """
    height, width, _ = img.shape
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    prompt = f"""You are a document analysis expert.
Image size: {width}×{height} pixels.

YOUR TASK: Locate every HANDWRITTEN SIGNATURE in this document image.

═══════════════════════════════════════════
WHAT IS A HANDWRITTEN SIGNATURE?
═══════════════════════════════════════════
A handwritten signature is a person's unique mark written by hand — it may be:
• A flowing cursive name (fully or partially legible)
• A stylized scribble, monogram, or personal mark
• Initials written in a practiced, personal style
• Any hand-drawn mark that clearly serves as a personal authorization

Signatures are almost always found:
• Near a "Signature:" label, a signature line, or at the bottom of a document
• In designated signature boxes or fields
• Alongside a printed name, title, or date

═══════════════════════════════════════════
STRICT EXCLUSIONS — DO NOT RETURN:
═══════════════════════════════════════════
• Printed text, letterheads, titles, or labels (even if near a signature)
• Printed logos, crests, emblems, or decorative elements
• Official rubber stamps or ink seals (circular/rectangular bordered marks)
• Handwritten DATES only (e.g., "12/03/2024", "July 2024") — NOT a signature
• Handwritten numbers, codes, or annotations that are clearly NOT a name
• Checkbox marks, tick marks, or correction marks
• Form field labels such as "Signature:", "Date:", "Name:"
• Any element that is PART of the pre-printed document design

═══════════════════════════════════════════
COORDINATE INSTRUCTIONS (CRITICAL):
═══════════════════════════════════════════
• Use a 0-1000 normalized scale where:
    x=0 is left edge,  x=1000 is right edge
    y=0 is top edge,   y=1000 is bottom edge
• Draw each bounding box TIGHTLY around the ACTUAL STROKE PIXELS only.
  Do NOT include large empty white margins around the signature.
• If multiple separate signatures exist, return ALL of them as separate entries.

Return ONLY valid JSON — no markdown, no explanation:
{{"signatures":[{{"x1":int,"y1":int,"x2":int,"y2":int}}]}}

If no handwritten signatures are found, return: {{"signatures":[]}}
"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]

    for attempt in range(3):
        try:
            response = await llm.ainvoke([HumanMessage(content=content)])
            text = response.content.strip()
            logger.info(f"Signature detection raw response: {text[:400]}")
            result = parse_json_robust(text)
            sigs = result.get("signatures", [])
            if isinstance(sigs, list):
                return sigs
        except Exception as e:
            logger.warning(f"Signature detection attempt {attempt + 1} failed: {e}")
    return []


async def detect_stamps_only(llm: Any, img: np.ndarray) -> List[Dict[str, Any]]:
    """
    LLM call that detects ONLY hand-applied ink stamps/seals in the full document image.
    Returns a list of bounding boxes in 0-1000 normalized coordinates.
    """
    height, width, _ = img.shape
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    prompt = f"""You are a document analysis expert specializing in detecting official stamps.
Image size: {width}×{height} pixels.

YOUR TASK: Locate every HAND-APPLIED INK STAMP (rubber stamp / official seal) in this document image.
Return ALL stamps — they may be RED, BLUE, PURPLE, or BLACK ink.

═══════════════════════════════════════════
WHAT IS A HAND-APPLIED INK STAMP?
═══════════════════════════════════════════
A real ink stamp has ALL of these characteristics:
1. A VISIBLE BORDER — circular ring, oval ring, or rectangular frame that outlines the stamp
2. INK IMPRESSION — the stamp was physically pressed onto the paper (ink may be red, blue, purple, or black)
3. IMPERFECT APPLICATION — the ink may be uneven, patchy, smudged, slightly rotated, or partially faint
4. OVERLAPS PAGE CONTENT — the stamp impression sits ON TOP of the printed document

VERY COMMON STAMP TYPES (detect all of these):
• Blue circular seal with Khmer/Arabic/Latin text around a ring — VERY COMMON on official documents
• Blue or purple oval entry/exit stamp with date inside
• Red circular stamp with a central image (temple, emblem) and text ring
• Rectangular "APPROVED" or "RECEIVED" ink stamp in any color
• Blue or purple notary/ministry seal pressed onto the document

═══════════════════════════════════════════
STRICT EXCLUSIONS — DO NOT RETURN:
═══════════════════════════════════════════
⚠️  LOGOS & PRINTED EMBLEMS:
• National coat-of-arms, government emblems, or institution logos printed as part of the document
• Colorful crests or medallion artwork with perfect, uniform colors and crisp edges
• Ministry logos integrated into the letterhead

⚠️  DOCUMENT BACKGROUND / PAPER ELEMENTS:
• The document’s paper background, spine, edge, or binding strip — even if dark-colored
• Decorative page borders or frame lines that are PART of the document design
• Dark vertical or horizontal strips at the document edges (these are the paper/folder, NOT stamps)
• Any region that is just document paper without a visible stamp impression on top

⚠️  OTHER EXCLUSIONS:
• Handwritten signatures (no border, just pen strokes)
• Printed barcodes, QR codes, or watermarks

═══════════════════════════════════════════
COORDINATE INSTRUCTIONS (CRITICAL):
═══════════════════════════════════════════
• Use a 0-1000 normalized scale where:
    x=0 is LEFT edge,  x=1000 is RIGHT edge
    y=0 is TOP edge,   y=1000 is BOTTOM edge
• Draw each bounding box TIGHTLY around the stamp impression only.
  Include the full outer border ring but exclude empty white space around it.
• If multiple separate stamps exist, return ALL of them as separate entries.

Return ONLY valid JSON — no markdown, no explanation:
{{"stamps":[{{"x1":int,"y1":int,"x2":int,"y2":int}}]}}

If no ink stamps are found, return: {{"stamps":[]}}
"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]

    for attempt in range(3):
        try:
            response = await llm.ainvoke([HumanMessage(content=content)])
            text = response.content.strip()
            logger.info(f"Stamp detection raw response: {text[:400]}")
            result = parse_json_robust(text)
            stmps = result.get("stamps", [])
            if isinstance(stmps, list):
                return stmps
        except Exception as e:
            logger.warning(f"Stamp detection attempt {attempt + 1} failed: {e}")
    return []


# ---------------------------------------------------------------------------
# Per-crop classifiers (used for fallback color candidates)
# ---------------------------------------------------------------------------

async def classify_stamp(llm: Any, crop_img: np.ndarray) -> Dict[str, Any]:
    """
    Classifies a cropped region as either a hand-applied ink stamp or a printed logo/crest.
    Returns {"is_stamp": bool, "label": str}.
    """
    _, buffer = cv2.imencode(".png", crop_img)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    prompt = """Analyze this closely cropped image from a document.
Determine whether it shows a HAND-APPLIED INK STAMP or something else (logo, background, edge, etc.).

HAND-APPLIED INK STAMP (is_stamp = true) MUST have ALL THREE of these:
1. A CLEARLY VISIBLE circular, oval, or rectangular BORDER RING enclosing the content
2. Text, symbols, or an emblem INSIDE that border
3. Ink that looks physically pressed — may be slightly uneven, smudged, or rotated

The ink color can be red, blue, purple, or black.
Blue circular seals are extremely common on official government documents.

NOT a stamp (is_stamp = false) — return false for:
• Dark or black background strips — these are document edges, bindings, or paper spine NOT stamps
• A region that is mostly one dark solid color with no visible stamp border or text
• A narrow vertical or horizontal strip without a circular/rectangular stamp impression
• Printed logos, coat-of-arms, crests with perfect uniform colors (no ink smudging)
• Document paper background, even if it has some texture or color variation
• Any region without a CLEARLY VISIBLE bordered stamp impression on a lighter background

❌ CRITICAL RULE: If the crop is mostly dark/black background, or if you cannot clearly see a stamp
   border ring plus interior content, the answer is is_stamp=false. Dark strips = paper edge, NOT stamp.

Response must be JSON only:
{"is_stamp": true | false, "label": "Short English description"}
"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]

    for attempt in range(3):
        try:
            response = await llm.ainvoke([HumanMessage(content=content)])
            text = response.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if "is_stamp" in result:
                    return result
        except Exception as e:
            logger.warning(f"Stamp classification attempt {attempt + 1} failed: {e}")
    return {"is_stamp": False, "label": ""}


async def classify_signature(llm: Any, crop_img: np.ndarray) -> Dict[str, Any]:
    """
    Classifies a cropped region as either a handwritten signature or not.
    Returns {"is_signature": bool, "label": str}.
    """
    _, buffer = cv2.imencode(".png", crop_img)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    prompt = """Analyze this closely cropped image from a document.
Determine whether it shows a HANDWRITTEN SIGNATURE.

A HANDWRITTEN SIGNATURE (is_signature = true):
• A person's name or personal mark written by hand
• May be neat cursive, a quick scribble, initials, or a stylized monogram
• Exhibits flowing strokes with natural variation in pressure and direction
• Typically found at the end of a document, near a "Signature:" line

NOT a signature (is_signature = false):
• Printed text, logos, crests, or document elements
• A rubber stamp or official seal impression
• A date, number, or annotation written by hand (e.g., "12/03/2024")
• Clearly readable block-letter text or notes
• Barcodes, QR codes, or form fields

Response must be JSON only:
{"is_signature": true | false, "label": "Short English description"}
"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]

    for attempt in range(3):
        try:
            response = await llm.ainvoke([HumanMessage(content=content)])
            text = response.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if "is_signature" in result:
                    return result
        except Exception as e:
            logger.warning(f"Signature classification attempt {attempt + 1} failed: {e}")
    return {"is_signature": False, "label": ""}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_extracted_elements(elements: List[Dict[str, Any]],
                                   prefer_type: str) -> List[Dict[str, Any]]:
    """Deduplicates by IoU overlap, keeping the largest / preferred-type element."""
    elements.sort(key=lambda x: (0 if x.get("cand_type") == prefer_type else 1,
                                  -x.get("area", 0)))
    unique: List[Dict] = []
    for el in elements:
        cy1, cx1, cy2, cx2 = el["pixel_coords"]
        dup = False
        for ue in unique:
            uy1, ux1, uy2, ux2 = ue["pixel_coords"]
            iy1, ix1 = max(cy1, uy1), max(cx1, ux1)
            iy2, ix2 = min(cy2, uy2), min(cx2, ux2)
            if iy2 > iy1 and ix2 > ix1:
                inter = (iy2 - iy1) * (ix2 - ix1)
                c_area = max(1, (cy2 - cy1) * (cx2 - cx1))
                u_area = max(1, (uy2 - uy1) * (ux2 - ux1))
                if inter / min(c_area, u_area) > 0.15:
                    dup = True
                    break
        if not dup:
            clean = el.copy()
            clean.pop("cand_type", None)
            clean.pop("area", None)
            unique.append(clean)
    return unique


def _boxes_overlap(a: List[int], b: List[int], threshold: float = 0.40) -> bool:
    """Returns True if the smaller box overlaps the larger by at least `threshold`."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    if iy2 <= iy1 or ix2 <= ix1:
        return False
    inter = (iy2 - iy1) * (ix2 - ix1)
    a_area = max(1, (ay2 - ay1) * (ax2 - ax1))
    b_area = max(1, (by2 - by1) * (bx2 - bx1))
    return inter / min(a_area, b_area) > threshold


# ---------------------------------------------------------------------------
# Main per-frame processing
# ---------------------------------------------------------------------------

async def process_image_frame(llm: Any, img: np.ndarray,
                               page_num: int,
                               thread_id: str = "default") -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts signatures and stamps from a single image frame.

    Pipeline:
    1. Two focused LLM calls (signatures only, stamps only) with strict prompts.
    2. trim_to_ink refines each LLM box to the actual ink pixels.
    3. classify_stamp is called on every LLM-detected stamp to reject printed logos.
    4. Color-based fallback runs ALWAYS for elements that the LLM missed.
    5. Cross-type deduplication removes any stamp that is 85%+ covered by a signature.
    """
    height, width, _ = img.shape

    def build_element(y1: int, x1: int, y2: int, x2: int,
                      cand_type: str, label: str, source: str) -> Dict[str, Any]:
        filename = f"crop_{page_num}_{cand_type}_{uuid.uuid4().hex[:12]}.png"
        safe_thread = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id)) if thread_id else "default"
        temp_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, safe_thread, "extracted"))
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        crop_img = img[y1:y2, x1:x2]
        cv2.imwrite(file_path, crop_img)
        return {
            "base64": "",
            "url": f"/temp/{safe_thread}/extracted/{filename}",
            "box_2d": [int(y1 * 1000 / height), int(x1 * 1000 / width),
                       int(y2 * 1000 / height), int(x2 * 1000 / width)],
            "pixel_coords": [int(y1), int(x1), int(y2), int(x2)],
            "page": page_num,
            "cand_type": cand_type,
            "area": int((y2 - y1) * (x2 - x1)),
            "label": label,
            "source": source,
        }

    raw_signatures: List[Dict] = []
    raw_stamps: List[Dict] = []

    # ── Step 1a: LLM signature detection ─────────────────────────────────────
    sig_regions = await detect_signatures_only(llm, img)
    logger.info(f"Page {page_num}: LLM returned {len(sig_regions)} signature region(s)")

    for sig in sig_regions:
        y1_n = max(0, min(1000, sig.get("y1", 0)))
        x1_n = max(0, min(1000, sig.get("x1", 0)))
        y2_n = max(0, min(1000, sig.get("y2", 0)))
        x2_n = max(0, min(1000, sig.get("x2", 0)))

        y1 = int(y1_n * height / 1000)
        x1 = int(x1_n * width  / 1000)
        y2 = int(y2_n * height / 1000)
        x2 = int(x2_n * width  / 1000)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)

        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            logger.info(f"  Skipping tiny sig region ({bw}×{bh})")
            continue
        if bw > width * 0.70 or bh > height * 0.70:
            logger.info(f"  Skipping oversized sig region ({bw}×{bh})")
            continue

        # Pad generously then trim to actual ink strokes.
        # A 35px pad ensures we capture signature strokes the LLM may have under-boxed.
        pad_sig = 35
        padded = [max(0, y1 - pad_sig), max(0, x1 - pad_sig),
                  min(height, y2 + pad_sig), min(width, x2 + pad_sig)]
        refined = trim_to_ink(img, padded, "signature")
        rbw, rbh = refined[3] - refined[1], refined[2] - refined[0]
        # If trim_to_ink collapsed OR shrank the result to less than 60% of the LLM box,
        # fall back to the padded LLM box — this prevents blue stamp ring pixels
        # from masking out parts of a signature that overlaps a stamp.
        llm_area = max(1, bw * bh)
        ref_area = rbw * rbh
        if rbw < 5 or rbh < 5 or (ref_area / llm_area) < 0.35:
            logger.info(f"  trim_to_ink over-shrank sig box, using padded LLM box instead")
            refined = padded

        raw_signatures.append(build_element(
            refined[0], refined[1], refined[2], refined[3],
            "signature", "Handwritten Signature", "llm_sig"
        ))

    # ── Step 1b: LLM stamp detection + logo rejection ────────────────────────
    stamp_regions = await detect_stamps_only(llm, img)
    logger.info(f"Page {page_num}: LLM returned {len(stamp_regions)} stamp region(s)")

    for stmp in stamp_regions:
        y1_n = max(0, min(1000, stmp.get("y1", 0)))
        x1_n = max(0, min(1000, stmp.get("x1", 0)))
        y2_n = max(0, min(1000, stmp.get("y2", 0)))
        x2_n = max(0, min(1000, stmp.get("x2", 0)))

        y1 = int(y1_n * height / 1000)
        x1 = int(x1_n * width  / 1000)
        y2 = int(y2_n * height / 1000)
        x2 = int(x2_n * width  / 1000)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)

        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            logger.info(f"  Skipping tiny stamp region ({bw}×{bh})")
            continue
        if bw > width * 0.70 or bh > height * 0.70:
            logger.info(f"  Skipping oversized stamp region ({bw}×{bh})")
            continue

        # Expand generously then trim to ink — large pad ensures the full circular
        # stamp outline is captured even when the LLM returned a tight/partial box.
        pad_stmp = 50
        padded = [max(0, y1 - pad_stmp), max(0, x1 - pad_stmp),
                  min(height, y2 + pad_stmp), min(width, x2 + pad_stmp)]
        refined = trim_to_ink(img, padded, "stamp")
        rbw, rbh = refined[3] - refined[1], refined[2] - refined[0]
        # Use padded box if trim collapsed or collapsed below 50% of the LLM box
        llm_area_s = max(1, bw * bh)
        ref_area_s = rbw * rbh
        if rbw < 5 or rbh < 5 or (ref_area_s / llm_area_s) < 0.30:
            logger.info(f"  trim_to_ink over-shrank stamp box, using padded LLM box instead")
            refined = padded

        # ── Logo-rejection gate: classify the refined crop ─────────────────
        crop = img[refined[0]:refined[2], refined[1]:refined[3]]
        if crop.size > 0:
            verdict = await classify_stamp(llm, crop)
            explicitly_rejected = (
                not verdict.get("is_stamp", True)  # LLM said NOT a stamp
                and verdict.get("label", "") != ""  # AND it gave a real reason (not a failure)
            )
            if explicitly_rejected:
                logger.info(f"  LLM stamp at ({refined}) rejected as logo/crest: {verdict['label']}")
                continue
            # classify_stamp failed (empty label) → trust the main detection; it already filtered logos
            label = verdict.get("label", "") or "Official Stamp"
        else:
            label = "Official Stamp"

        raw_stamps.append(build_element(
            refined[0], refined[1], refined[2], refined[3],
            "stamp", label, "llm_stamp"
        ))

    # ── Step 2: Color-based fallback (always runs to catch missed elements) ──
    color_candidates = get_ink_candidates(img)
    logger.info(f"Page {page_num}: Color fallback checking {len(color_candidates)} candidate(s)")

    for cc in color_candidates:
        cy1, cx1, cy2, cx2 = cc["box"]

        # ── Fallback stamps ───────────────────────────────────────────────────
        if cc["type"] == "stamp":
            if any(_boxes_overlap([cy1, cx1, cy2, cx2], s["pixel_coords"], 0.40)
                   for s in raw_stamps):
                continue  # already detected

            # Size & shape sanity: real stamps are roughly square and at least 50×50px
            cbw, cbh = cx2 - cx1, cy2 - cy1
            if cbw < 50 or cbh < 50:
                logger.debug(f"  Color stamp candidate too small ({cbw}×{cbh}), skipping")
                continue
            c_aspect = max(cbw, cbh) / max(1, min(cbw, cbh))
            if c_aspect > 3.0:
                logger.debug(f"  Color stamp candidate wrong aspect ({c_aspect:.1f}), skipping")
                continue

            # Brightness guard: real stamp ink sits on a lighter paper background.
            # A crop whose average HSV brightness is very low is a dark paper edge/binding, not a stamp.
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_v = float(hsv_crop[:, :, 2].mean())
            if mean_v < 45:
                logger.info(f"  Color stamp candidate too dark (mean_V={mean_v:.0f}), likely paper edge — skipping")
                continue
            # Colored-ink check: at least some pixels must have meaningful saturation
            # (pure dark gray/black background has near-zero saturation)
            mean_s = float(hsv_crop[:, :, 1].mean())
            if mean_s < 12:
                logger.info(f"  Color stamp candidate has no colored ink (mean_S={mean_s:.0f}) — skipping")
                continue

            verdict = await classify_stamp(llm, crop)
            if verdict.get("is_stamp", False):
                refined = trim_to_ink(img, [cy1, cx1, cy2, cx2], "stamp")
                raw_stamps.append(build_element(
                    refined[0], refined[1], refined[2], refined[3],
                    "stamp", verdict.get("label", "Official Stamp") or "Official Stamp", "color"
                ))

        # ── Fallback signatures ───────────────────────────────────────────────
        elif cc["type"] == "signature":
            if any(_boxes_overlap([cy1, cx1, cy2, cx2], s["pixel_coords"], 0.40)
                   for s in raw_signatures):
                continue  # already detected
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            verdict = await classify_signature(llm, crop)
            if verdict.get("is_signature", False):
                refined = trim_to_ink(img, [cy1, cx1, cy2, cx2], "signature")
                raw_signatures.append(build_element(
                    refined[0], refined[1], refined[2], refined[3],
                    "signature",
                    verdict.get("label", "Handwritten Signature") or "Handwritten Signature",
                    "color"
                ))

    # ── Step 3: Within-type deduplication ────────────────────────────────────
    stamps     = deduplicate_extracted_elements(raw_stamps,     "stamp")
    signatures = deduplicate_extracted_elements(raw_signatures, "signature")

    # ── Step 4: Cross-type dedup — stamp fully covered by a signature → drop ─
    clean_stamps = []
    for s in stamps:
        sy1, sx1, sy2, sx2 = s["pixel_coords"]
        if any(_boxes_overlap([sy1, sx1, sy2, sx2], sig["pixel_coords"], 0.85)
               for sig in signatures):
            continue
        clean_stamps.append(s)

    return {"signatures": signatures, "stamps": clean_stamps}


# ---------------------------------------------------------------------------
# LangGraph node builder
# ---------------------------------------------------------------------------

def build_signature_stamp_extractor_node(node_config: NodeConfig) -> Callable:
    """
    Builds a LangGraph-compatible Signature & Stamp Extractor node.
    Extracts all handwritten signatures and official ink stamps from an
    uploaded document image or PDF, saves cropped images, and returns metadata.
    """
    llm = get_vision_llm(
        provider=node_config.llm.provider    if node_config.llm else None,
        model=node_config.llm.model          if node_config.llm else None,
        temperature=node_config.llm.temperature if node_config.llm else 0.0
    )

    async def execute_extraction(state: AgentState):
        context = state.get("context", {})
        thread_id = context.get("thread_id", "default")

        input_key  = getattr(node_config, "extractor_input_key",  "latest_file_path") or "latest_file_path"
        output_key = getattr(node_config, "extractor_output_key", "extracted_elements") or "extracted_elements"

        file_path = context.get(input_key) or context.get("latest_file_path")
        if not file_path and context.get("payload", {}).get("uploaded_files"):
            file_path = context["payload"]["uploaded_files"][0].get("path")

        if not file_path or not os.path.exists(file_path):
            error_msg = f"Signature Extractor Error: No file found. Searched context keys: '{input_key}', 'latest_file_path', payload sub-keys. Got: '{file_path}'."
            return {
                "messages": [AIMessage(content=error_msg,
                                       response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": "FileNotFound"},
                "context": {output_key: None},
            }

        ext = os.path.splitext(file_path)[1].lower()
        signatures: List[Dict] = []
        stamps: List[Dict] = []

        try:
            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Could not load image at {file_path}")
                res = await process_image_frame(llm, img, page_num=1, thread_id=thread_id)
                signatures.extend(res["signatures"])
                stamps.extend(res["stamps"])

            elif ext == ".pdf":
                try:
                    import fitz
                except ImportError:
                    raise ImportError("PDF extraction requires 'pymupdf'. Please install it.")

                doc = fitz.open(file_path)
                max_pages = min(len(doc), 10)
                for i in range(max_pages):
                    page = doc.load_page(i)
                    pix = page.get_pixmap()
                    nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        res = await process_image_frame(llm, img, page_num=i + 1, thread_id=thread_id)
                        signatures.extend(res["signatures"])
                        stamps.extend(res["stamps"])
                doc.close()

            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Build markdown summary
            parts = [f"### Extracted Elements ({len(signatures)} signatures, {len(stamps)} stamps)"]
            if stamps:
                parts.append("\n**Stamps:**")
                for idx, s in enumerate(stamps):
                    parts.append(f"- **Stamp {idx + 1}** ({s['label']}):")
                    parts.append(f"  ![Stamp {idx + 1}]({s['url']})")
            if signatures:
                parts.append("\n**Signatures:**")
                for idx, s in enumerate(signatures):
                    parts.append(f"- **Signature {idx + 1}** ({s['label']}):")
                    parts.append(f"  ![Signature {idx + 1}]({s['url']})")

            return {
                "messages": [AIMessage(content="\n".join(parts),
                                       response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "context": {
                    output_key: {
                        "signatures": signatures,
                        "stamps": stamps,
                        "total_signatures": len(signatures),
                        "total_stamps": len(stamps),
                    }
                },
            }

        except Exception as e:
            error_msg = f"Signature Extractor Node failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "messages": [AIMessage(content=error_msg,
                                       response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": type(e).__name__},
                "context": {output_key: None},
            }

    return RunnableLambda(execute_extraction)
