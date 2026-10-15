import logging
import base64
import os
from typing import Callable, Optional
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageStat
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def _get_base64_from_context(ctx: dict, input_key: Optional[str] = None) -> Optional[str]:
    """Helper to extract a base64 string from the context based on common patterns."""
    
    def extract_b64(val):
        if not val: return None
        # If it's a list from the frontend, grab the first element
        if isinstance(val, list) and len(val) > 0:
            val = val[0]
        
        # The frontend form output might be a dict with a 'data' key containing the base64 string
        if isinstance(val, dict) and "data" in val:
            val = val["data"]
            
        if isinstance(val, str) and (val.startswith("data:image") or len(val) > 100):
            return val
        return None

    if input_key:
        val = ctx.get(input_key) or ctx.get("payload", {}).get(input_key)
        res = extract_b64(val)
        if res: return res

    # Fallback to common keys
    for key in ["signature_b64", "stamp_b64", "base64", "image_b64", "file_b64", "latest_file_path"]:
        val = ctx.get(key) or ctx.get("payload", {}).get(key)
        res = extract_b64(val)
        if res: return res
    
    return None

def _decode_to_cv2(b64_str: str) -> np.ndarray:
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def build_image_validator_node(node_config: NodeConfig) -> Callable:
    """
    Builds an Image Quality Validator node.
    It reads an image from context, calculates Standard Deviation and Edge Score,
    and raises an error if the image is blank or lacks sharp ink strokes.
    """
    
    # Configuration from UI
    min_std_dev = float(getattr(node_config, "min_std_dev", 8.0))
    min_edge_score = float(getattr(node_config, "min_edge_score", 100.0))
    input_key = getattr(node_config, "input_key", None)
    
    async def execute_validation(state: AgentState):
        ctx = state.get("context", {})
        node_id = node_config.id
        
        b64_str = _get_base64_from_context(ctx, input_key)
        if not b64_str:
            # Check if it's a file path instead of base64
            file_path = ctx.get("latest_file_path") or ctx.get("payload", {}).get("latest_file_path")
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode('utf-8')
            else:
                logger.warning(f"[Validator {node_id}] No image found in context to validate. Passing through.")
                return {
                    "messages": [AIMessage(content="No image found to validate. Passing through.")],
                    "active_agent": node_id,
                    "context": ctx
                }
            
        try:
            cv_img = _decode_to_cv2(b64_str)
            if cv_img is None:
                raise ValueError("Could not decode base64 image data.")
                
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Check 1: Minimum Standard Deviation (Variance)
            std_dev = np.std(gray)
            
            # Check 2: Edge Density (Laplacian variance)
            edge_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            logger.info(f"[Validator {node_id}] StdDev: {std_dev:.2f} | EdgeScore: {edge_score:.2f}")
            
            # The user explicitly requested it only fail for blank images.
            # A blank image will have very low variance (< 5.0) and edge score (< 15.0).
            # The previous default of 100.0 was failing legitimate document scans.
            is_blank = std_dev < 5.0 or edge_score < 15.0
            
            logger.info(f"[Validator {node_id}] StdDev: {std_dev:.2f} | EdgeScore: {edge_score:.2f} | IsBlank: {is_blank}")
            
            if is_blank:
                reason = "appears completely blank"
                error_msg = (
                    f"Invalid Image: The uploaded image {reason} "
                    f"(StdDev: {std_dev:.1f}, EdgeScore: {edge_score:.1f}). "
                    f"Required Minimums: StdDev=5.0, EdgeScore=15.0."
                )
                logger.error(f"[Validator {node_id}] FAILED: {error_msg}")
                # Hard-stop the pipeline by raising an exception
                raise ValueError(error_msg)

            # Passed!
            success_msg = f"✅ Image Validation Passed (StdDev: {std_dev:.2f}, EdgeScore: {edge_score:.2f})"
            logger.info(f"[Validator {node_id}] {success_msg}")
            
            return {
                "messages": [AIMessage(content=success_msg, name=node_id)],
                "active_agent": node_id,
                "context": ctx
            }

        except Exception as e:
            error_msg = f"Image validation error: {str(e)}"
            logger.error(f"[Validator {node_id}] {error_msg}")
            # We MUST re-raise the exception to halt the LangGraph pipeline.
            # Returning a dict just passes the FAILED state to the next node.
            raise ValueError(error_msg) from e
            
    return RunnableLambda(execute_validation)
