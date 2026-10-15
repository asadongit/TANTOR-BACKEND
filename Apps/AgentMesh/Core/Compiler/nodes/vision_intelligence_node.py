import os
import io
import uuid
import base64
import logging
import threading
import tempfile
import atexit
import shutil
import urllib.request
import urllib.error
import anyio
import cv2
import numpy as np
from typing import Callable, Dict, Any, Optional
from langchain_core.messages import AIMessage
from PIL import Image as PILImage

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.Compiler.nodes.utils import get_production_data, get_nested_val, resolve_temp_path, make_session_dir
from Apps.AgentMesh.Core.Compiler.nodes.signature_stamp_extractor import (
    get_ink_candidates,
    trim_to_ink,
)
from Apps.AgentMesh.Core.Compiler.nodes.image_quality_gate import ImageQualityGate

logger = logging.getLogger(__name__)

# ── Model Singleton Cache (thread-safe) ────────────────────────────────────
_VISION_LOCK = threading.Lock()
_INSIGHTFACE_APP = None
_INSIGHTFACE_LOADED = False
_VISION_TEMP_DIRS: set = set()


def _cleanup_temp_dirs():
    for d in list(_VISION_TEMP_DIRS):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    _VISION_TEMP_DIRS.clear()


atexit.register(_cleanup_temp_dirs)


def _get_insightface_app():
    """Lazy-loads InsightFace model suite (buffalo_l) once per process (thread-safe)."""
    global _INSIGHTFACE_APP, _INSIGHTFACE_LOADED
    if _INSIGHTFACE_LOADED:
        return _INSIGHTFACE_APP

    with _VISION_LOCK:
        if _INSIGHTFACE_LOADED:
            return _INSIGHTFACE_APP
        _INSIGHTFACE_LOADED = True
        try:
            import insightface

            model_dir = os.environ.get(
                "INSIGHTFACE_MODEL_DIR",
                os.path.join(os.path.expanduser("~"), ".insightface", "models"),
            )
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            app = insightface.app.FaceAnalysis(
                name="buffalo_l", root=model_dir, providers=["CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _INSIGHTFACE_APP = app
            logger.info(
                f"InsightFace (buffalo_l) initialized successfully. "
                f"Model root: {model_dir}"
            )
        except Exception as e:
            logger.warning(
                f"InsightFace engine not initialized: {e}. "
                f"Falling back to OpenCV heuristic face matcher."
            )
            _INSIGHTFACE_APP = None
    return _INSIGHTFACE_APP


def _compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _resolve_image_data(scope: Dict[str, Any], key_name: str) -> Optional[np.ndarray]:
    if not key_name:
        return None
    val = get_nested_val(scope, key_name)
    if not val:
        val = scope.get(key_name)
        
    if isinstance(val, dict):
        val = val.get("vfs_path") or val.get("file_path") or val.get("path") or val.get("url")
        
    if not val or not isinstance(val, str):
        return None
        
    if val and isinstance(val, str) and not os.path.exists(val) and not val.startswith(("http://", "https://", "data:")):
        resolved = resolve_temp_path(val)
        if resolved:
            val = resolved

    if os.path.exists(val):
        try:
            return cv2.imread(val)
        except Exception:
            logger.exception(f"Failed to read image file {val}")
            return None

    # Check if it's a URL
    if val.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(
                val, headers={"User-Agent": "AgentMesh-VisionIntelligence/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                img_bytes = resp.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to decode image from URL: {val}")
            return img
        except Exception as e:
            logger.warning(f"Failed to fetch image from URL {val}: {e}")
            return None

    # Check if it's a base64 string
    try:
        b64_str = val
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ── OpenCV fallback: ORB feature matching ─────────────────────────────────
def _orb_face_match(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0.0
        matches = sorted(matches, key=lambda x: x.distance)
        good = [m for m in matches[:50] if m.distance < 50]
        score = len(good) / max(len(kp1), len(kp2))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: Biometric Face Verification (InsightFace buffalo_l)
# ─────────────────────────────────────────────────────────────────────────────
def _run_face_verification(
    scope: Dict[str, Any], node_config: NodeConfig
) -> Dict[str, Any]:
    id_key = node_config.vision_id_image_key or "latest_file_path"
    selfie_key = node_config.vision_selfie_image_key or "vcip_capture_path"
    threshold = node_config.vision_face_match_threshold or 0.65

    img_id = _resolve_image_data(scope, id_key)
    img_selfie = _resolve_image_data(scope, selfie_key)

    if img_id is None or img_selfie is None:
        missing = []
        if img_id is None:
            missing.append(f"ID image ('{id_key}')")
        if img_selfie is None:
            missing.append(f"Selfie image ('{selfie_key}')")
        logger.warning(
            f"Vision Intelligence ({node_config.id}): Missing image sources: {', '.join(missing)}"
        )
        return {
            "vision_mode": "face_verification",
            "status": "SKIPPED_MISSING_INPUTS",
            "missing_inputs": missing,
            "face_match_score": 0.0,
            "face_match_decision": "REQUIRES_INPUT",
            "message": f"Biometric Face Match waiting for images: missing {', '.join(missing)}.",
        }

    app = _get_insightface_app()

    if app:
        try:
            faces_id = app.get(img_id)
            faces_selfie = app.get(img_selfie)

            if not faces_id or not faces_selfie:
                return {
                    "vision_mode": "face_verification",
                    "status": "FACE_NOT_DETECTED",
                    "face_detected_in_id": len(faces_id) > 0,
                    "face_detected_in_selfie": len(faces_selfie) > 0,
                    "face_match_score": 0.0,
                    "face_match_decision": "REJECT",
                    "message": f"Face detection failed: ID face found = {len(faces_id) > 0}, Selfie face found = {len(faces_selfie) > 0}.",
                }

            emb_id = faces_id[0].embedding
            emb_selfie = faces_selfie[0].embedding

            score = _compute_cosine_similarity(emb_id, emb_selfie)
            score_normalized = float(np.clip(score, 0.0, 1.0))
            decision = "PASS" if score_normalized >= threshold else "REJECT"

            return {
                "vision_mode": "face_verification",
                "status": "SUCCESS",
                "engine": "InsightFace (buffalo_l)",
                "face_match_score": round(score_normalized, 4),
                "threshold": threshold,
                "face_match_decision": decision,
                "bbox_id": faces_id[0].bbox.tolist()
                if hasattr(faces_id[0], "bbox")
                else [],
                "bbox_selfie": faces_selfie[0].bbox.tolist()
                if hasattr(faces_selfie[0], "bbox")
                else [],
                "message": f"Biometric Face Match completed: Score = {round(score_normalized * 100, 1)}% ({decision}).",
            }
        except Exception as e:
            logger.exception(f"InsightFace processing failed: {e}")

    # OpenCV Fallback: ORB feature matching (more meaningful than Laplacian heuristic)
    logger.warning("InsightFace unavailable — using OpenCV ORB fallback face matcher.")
    sim_score = _orb_face_match(img_id, img_selfie)
    decision = "PASS" if sim_score >= threshold else "REJECT"

    return {
        "vision_mode": "face_verification",
        "status": "SUCCESS",
        "engine": "OpenCV ORB Fallback",
        "face_match_score": round(sim_score, 4),
        "threshold": threshold,
        "face_match_decision": decision,
        "message": f"Biometric Face Match (ORB Fallback): Score = {round(sim_score * 100, 1)}% ({decision}).",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: Signature & Stamp Detection (CV Fast Mode or Vision LLM Deep Analysis)
# ─────────────────────────────────────────────────────────────────────────────
async def _run_signature_stamp_detection(
    scope: Dict[str, Any], node_config: NodeConfig, state: AgentState
) -> Dict[str, Any]:
    id_key = (
        getattr(node_config, "vision_id_image_key", None)
        or getattr(node_config, "extractor_input_key", None)
        or "latest_file_path"
    )
    
    file_path_raw = get_nested_val(scope, id_key) or scope.get(id_key)
    if isinstance(file_path_raw, dict):
        file_path_raw = (
            file_path_raw.get("vfs_path")
            or file_path_raw.get("file_path")
            or file_path_raw.get("path")
            or file_path_raw.get("url")
        )
    
    if not file_path_raw or not isinstance(file_path_raw, str):
        if scope.get("payload", {}).get("uploaded_files"):
            file_path_raw = scope["payload"]["uploaded_files"][0].get("path")

    enable_llm = getattr(node_config, "vision_enable_llm", False) or getattr(node_config, "enable_llm_analysis", False)

    resolved_path = (
        resolve_temp_path(file_path_raw)
        if (
            isinstance(file_path_raw, str)
            and not os.path.exists(file_path_raw)
            and not file_path_raw.startswith(("http://", "https://", "data:"))
        )
        else file_path_raw
    )

    is_pdf = (
        isinstance(resolved_path, str)
        and resolved_path.lower().endswith(".pdf")
        and os.path.exists(resolved_path)
    )

    thread_id = scope.get("thread_id", "default")

    # ── Option A: Vision LLM Deep Analysis ─────────────────────────────────────
    if enable_llm and getattr(node_config, "llm", None):
        try:
            from Apps.AgentMesh.Utilities.LLM.llm_config import get_vision_llm
            from Apps.AgentMesh.Core.Compiler.nodes.signature_stamp_extractor import process_image_frame

            llm = get_vision_llm(
                provider=node_config.llm.provider if node_config.llm else None,
                model=node_config.llm.model if node_config.llm else None,
                temperature=node_config.llm.temperature if node_config.llm else 0.0,
            )

            signatures = []
            stamps = []

            if is_pdf:
                import fitz
                doc = fitz.open(resolved_path)
                max_pages = min(len(doc), 10)
                for i in range(max_pages):
                    page = doc.load_page(i)
                    pix = page.get_pixmap()
                    nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        res = await process_image_frame(llm, img, page_num=i + 1, thread_id=thread_id)
                        signatures.extend(res.get("signatures", []))
                        stamps.extend(res.get("stamps", []))
                doc.close()
            else:
                img = _resolve_image_data(scope, id_key)
                if img is not None:
                    res = await process_image_frame(llm, img, page_num=1, thread_id=thread_id)
                    signatures.extend(res.get("signatures", []))
                    stamps.extend(res.get("stamps", []))
                else:
                    return {
                        "vision_mode": "signature_stamp_detection",
                        "status": "SKIPPED_MISSING_INPUT",
                        "message": f"Signature/Stamp detection waiting for valid image/document at key '{id_key}'.",
                    }

            parts = [f"### Extracted Elements ({len(signatures)} signatures, {len(stamps)} stamps)"]
            if stamps:
                parts.append("\n**Stamps:**")
                for idx, s in enumerate(stamps):
                    parts.append(f"- **Stamp {idx + 1}** ({s.get('label', 'Stamp')}):")
                    if s.get("url"):
                        parts.append(f"  ![Stamp {idx + 1}]({s['url']})")
            if signatures:
                parts.append("\n**Signatures:**")
                for idx, s in enumerate(signatures):
                    parts.append(f"- **Signature {idx + 1}** ({s.get('label', 'Signature')}):")
                    if s.get("url"):
                        parts.append(f"  ![Signature {idx + 1}]({s['url']})")

            formatted_msg = "\n".join(parts)

            return {
                "vision_mode": "signature_stamp_detection",
                "status": "SUCCESS",
                "llm_enabled": True,
                "stamps_detected_count": len(stamps),
                "signatures_detected_count": len(signatures),
                "stamps": stamps,
                "signatures": signatures,
                "message": f"Vision LLM extraction complete: Found {len(signatures)} signatures and {len(stamps)} stamps.",
                "formatted_summary": formatted_msg,
                "extracted_elements": {
                    "signatures": signatures,
                    "stamps": stamps,
                    "total_signatures": len(signatures),
                    "total_stamps": len(stamps),
                },
            }
        except Exception as e:
            logger.exception(f"Vision LLM extraction failed in Vision Intelligence: {e}")

    # ── Option B: Fast CV-Only Ink Blob Candidate Detection ───────────────────
    images_to_process = []
    if is_pdf:
        try:
            import fitz
            doc = fitz.open(resolved_path)
            max_pages = min(len(doc), 10)
            for i in range(max_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap()
                nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    images_to_process.append((img, i + 1))
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to decode PDF for CV processing: {e}")
    else:
        img = _resolve_image_data(scope, id_key)
        if img is not None:
            images_to_process.append((img, 1))

    if not images_to_process:
        return {
            "vision_mode": "signature_stamp_detection",
            "status": "SKIPPED_MISSING_INPUT",
            "message": f"Signature/Stamp detection waiting for document image key '{id_key}'.",
        }

    detected_stamps = []
    detected_signatures = []
    temp_dir = make_session_dir(thread_id, "vision_crops")

    import re
    safe_thread = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id))

    for img, page_num in images_to_process:
        raw_candidates = get_ink_candidates(img)
        for cand in raw_candidates:
            box = cand["box"]
            trimmed = trim_to_ink(img, box, cand_type=cand["type"])
            y1, x1, y2, x2 = trimmed
            bw = x2 - x1
            bh = y2 - y1

            entry = {
                "type": cand["type"].upper(),
                "bbox": [x1, y1, bw, bh],
                "page": page_num,
                "confidence": round(min(0.99, cand["area"] / 5000.0), 2),
                "area_px": cand["area"],
            }

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crop_filename = f"{cand['type']}_p{page_num}_{len(detected_stamps + detected_signatures)}_{uuid.uuid4().hex[:8]}.png"
                crop_path = os.path.join(temp_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                entry["crop_path"] = crop_path
                entry["url"] = f"/temp/{safe_thread}/vision_crops/{crop_filename}"

            if cand["type"] == "stamp":
                detected_stamps.append(entry)
            else:
                detected_signatures.append(entry)

    return {
        "vision_mode": "signature_stamp_detection",
        "status": "SUCCESS",
        "llm_enabled": False,
        "stamps_detected_count": len(detected_stamps),
        "signatures_detected_count": len(detected_signatures),
        "stamps": detected_stamps,
        "signatures": detected_signatures,
        "temp_dir": temp_dir,
        "message": f"Detection complete: Found {len(detected_signatures)} signatures and {len(detected_stamps)} stamps.",
        "extracted_elements": {
            "signatures": detected_signatures,
            "stamps": detected_stamps,
            "total_signatures": len(detected_signatures),
            "total_stamps": len(detected_stamps),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: Document Quality Gate (ImageQualityGate engine)
# ─────────────────────────────────────────────────────────────────────────────
def _run_document_quality_gate(
    scope: Dict[str, Any], node_config: NodeConfig
) -> Dict[str, Any]:
    id_key = node_config.vision_id_image_key or "latest_file_path"
    img = _resolve_image_data(scope, id_key)

    if img is None:
        return {
            "vision_mode": "document_quality_gate",
            "status": "SKIPPED_MISSING_INPUT",
            "message": f"Quality gate waiting for image source '{id_key}'.",
        }

    # Convert OpenCV BGR → PIL RGB for ImageQualityGate
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    # Build config from node_config fields (user overrides)
    qc = {}
    if getattr(node_config, "vision_quality_blur_threshold", None) is not None:
        qc["blur_threshold"] = node_config.vision_quality_blur_threshold
    if getattr(node_config, "vision_quality_min_contrast", None) is not None:
        qc["min_contrast"] = node_config.vision_quality_min_contrast
    if getattr(node_config, "vision_quality_max_contrast", None) is not None:
        qc["max_contrast"] = node_config.vision_quality_max_contrast
    if getattr(node_config, "vision_quality_min_brightness", None) is not None:
        qc["min_brightness"] = node_config.vision_quality_min_brightness
    if getattr(node_config, "vision_quality_max_brightness", None) is not None:
        qc["max_brightness"] = node_config.vision_quality_max_brightness
    if getattr(node_config, "vision_quality_min_shortest_side", None) is not None:
        qc["min_shortest_side"] = node_config.vision_quality_min_shortest_side

    gate = ImageQualityGate(config=qc if qc else None)
    result = gate.assess_single_image(pil_img)

    quality_status = "PASS" if result.passed else "FAIL"

    return {
        "vision_mode": "document_quality_gate",
        "status": "SUCCESS",
        "quality_score": round(result.overall_score, 4),
        "quality_status": quality_status,
        "metrics": [m.as_dict() for m in result.metrics],
        "failure_reasons": result.failure_reasons,
        "user_message": result.user_message,
        "message": (
            f"Quality Gate ({quality_status}): Score = {round(result.overall_score * 100, 1)}% "
            f"({result.failure_reasons[0] if result.failure_reasons else 'All checks passed'})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4: Generic Object Detection / Future Extension Stub
# ─────────────────────────────────────────────────────────────────────────────
def _run_object_detection(
    scope: Dict[str, Any], node_config: NodeConfig
) -> Dict[str, Any]:
    id_key = node_config.vision_id_image_key or "latest_file_path"
    img = _resolve_image_data(scope, id_key)

    if img is None:
        return {
            "vision_mode": "object_detection",
            "status": "SKIPPED_MISSING_INPUT",
            "message": f"Object detection waiting for image source '{id_key}'.",
        }

    h, w, _ = img.shape
    return {
        "vision_mode": "object_detection",
        "status": "SUCCESS",
        "detected_objects": [
            {"label": "document", "confidence": 0.98, "bbox": [0, 0, w, h]}
        ],
        "message": f"Object Detection complete: Analyzed image bounds ({w}x{h}).",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Builder Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def build_vision_intelligence_node(node_config: NodeConfig) -> Callable:
    async def execute_vision_intelligence(state: AgentState):
        await anyio.sleep(0)
        scope = get_production_data(state)
        mode = (
            getattr(node_config, "vision_mode", "face_verification")
            or "face_verification"
        )
        output_key = (
            getattr(node_config, "vision_output_key", "vision_result")
            or "vision_result"
        )

        logger.info(
            f"Vision Intelligence ({node_config.id}): Executing mode '{mode}'..."
        )

        if mode == "ocr":
            from Apps.AgentMesh.Core.Compiler.nodes.ocr_node import build_ocr_node
            node_config.destination_key = output_key
            ocr_runnable = build_ocr_node(node_config)
            res = await ocr_runnable.ainvoke(state)
            if isinstance(res, dict):
                ctx = res.get("context", {})
                extracted_text = (
                    ctx.get(output_key)
                    or ctx.get("latest_ocr_output")
                    or ctx.get(f"ocr_output_{node_config.id}")
                    or res.get(f"ocr_output_{node_config.id}")
                    or res.get("latest_ocr_output")
                    or ""
                )
                new_ctx = {
                    **ctx,
                    output_key: extracted_text,
                    "latest_ocr_output": extracted_text
                }
                res["context"] = new_ctx
                res[output_key] = extracted_text
                res["latest_ocr_output"] = extracted_text
            return res

        if mode == "signature_stamp_detection":
            result = await _run_signature_stamp_detection(scope, node_config, state)
        elif mode == "document_quality_gate":
            result = _run_document_quality_gate(scope, node_config)
        elif mode == "object_detection":
            result = _run_object_detection(scope, node_config)
        else:
            result = _run_face_verification(scope, node_config)

        msg_content = result.get("formatted_summary") or result.get(
            "message", f"Vision Intelligence mode '{mode}' completed successfully."
        )

        ai_msg = AIMessage(
            content=msg_content,
            name=node_config.id,
            response_metadata={"active_agent": node_config.id},
        )

        context_updates = {
            output_key: result,
            **result,
        }

        # Also mirror 'extracted_elements' to context if present for backwards compatibility
        if "extracted_elements" in result:
            context_updates["extracted_elements"] = result["extracted_elements"]

        return {
            "active_agent": node_config.id,
            "messages": [ai_msg],
            "context": context_updates,
        }

    return execute_vision_intelligence
