import cv2
import numpy as np
import base64
import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def build_stamp_matcher_node(node_config: NodeConfig) -> Callable:
    """
    Stamp Matcher Node using SIFT keypoints.
    Takes input stamp and reference stamp from context.
    """
    match_threshold = getattr(node_config, "match_threshold", 15)
    lowe_ratio = getattr(node_config, "lowe_ratio", 0.75)
    enable_ocr = getattr(node_config, "enable_ocr", False)
    ocr_prompt = getattr(node_config, "ocr_prompt", "Extract the company registration number from this stamp image.")
    target_context_key = getattr(node_config, "target_context_key", "stamp_ref_id")
    enable_fallback_search = getattr(node_config, "enable_fallback_search", True)  # default ON
    vector_collection = getattr(node_config, "vector_collection", "ReferenceStamps")

    async def execute_match(state: AgentState):
        ctx = state.get("context", {})
        node_id = node_config.id
        
        b64_stamp_input = ctx.get("payload", {}).get("stamp_b64") or ctx.get("stamp_b64")
        b64_stamp_ref = ctx.get("payload", {}).get("reference_stamp_b64") or ctx.get("reference_stamp_b64")
        
        # Attempt to get the targeted key, fallback to common 'reference_id' if not found
        reference_id = ctx.get("payload", {}).get(target_context_key) or ctx.get(target_context_key)
        if not reference_id:
            reference_id = ctx.get("payload", {}).get("reference_id") or ctx.get("reference_id")
        
        # If no reference stamp is directly provided, attempt to fetch from Qdrant
        if not b64_stamp_ref:
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            client = qdrant_manager._get_client()
            batch_vectors = ctx.get("batch_vectors", [])
            stamp_vectors = [v for v in batch_vectors if v.get("type") == "Stamp"]
            
            if stamp_vectors:
                query_vector = stamp_vectors[0].get("vector")
                best_match = None
                
                # 1:1 Fetch Attempt
                if reference_id:
                    try:
                        filter_condition = Filter(must=[FieldCondition(key="reference_id", match=MatchValue(value=reference_id))])
                        search_result = client.query_points(
                            collection_name=vector_collection,
                            query=query_vector,
                            query_filter=filter_condition,
                            limit=1
                        ).points
                        if search_result: best_match = search_result[0]
                    except Exception as e:
                        logger.warning(f"Stamp Matcher: 1:1 Qdrant fetch failed: {e}")
                        
                # 1:N Fallback
                if not best_match and enable_fallback_search:
                    try:
                        search_result = client.query_points(
                            collection_name=vector_collection,
                            query=query_vector,
                            limit=1
                        ).points
                        if search_result: best_match = search_result[0]
                    except Exception as e:
                        logger.warning(f"Stamp Matcher: 1:N Qdrant fallback failed: {e}")
                        
                if best_match and best_match.payload:
                    b64_stamp_ref = best_match.payload.get("stamp_b64") or best_match.payload.get("reference_stamp_b64")
        
        if not b64_stamp_input or not b64_stamp_ref:
            msg = f"Stamp Matcher {node_id}: Missing stamp_b64 or could not fetch reference_stamp_b64 from Qdrant."
            return {"messages": [AIMessage(content=f"[WARNING] {msg}")], "active_agent": node_id}

        def decode_img(b64_str):
            if isinstance(b64_str, list): b64_str = b64_str[0]
            if isinstance(b64_str, dict): b64_str = b64_str.get("data", "")
            if "base64," in b64_str:
                b64_str = b64_str.split("base64,")[1]
            nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return img

        try:
            img_input = decode_img(b64_stamp_input)
            img_ref = decode_img(b64_stamp_ref)
            
            # SIFT matching
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_input, None)
            kp2, des2 = sift.detectAndCompute(img_ref, None)
            
            if des1 is None or des2 is None:
                raise ValueError("SIFT could not compute descriptors for one or both images.")
                
            # FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
                    
            # Geometric Verification (RANSAC)
            num_good_matches = 0
            M = None
            if len(good_matches) >= 4:
                # Extract the X,Y coordinates of the matches
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find Homography using RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    # Count only the inliers that fit the geometric shape
                    num_good_matches = int(np.sum(mask))
                else:
                    num_good_matches = 0
            else:
                # If there are less than 4 matches, a homography cannot be computed.
                num_good_matches = 0
                    
            is_match = num_good_matches >= match_threshold
            
            # Secondary Verification (OCR)
            ocr_text = None
            if is_match and enable_ocr and M is not None:
                try:
                    h, w = img_ref.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    x, y, w_bb, h_bb = cv2.boundingRect(dst)
                    
                    # Ensure within bounds
                    x = max(0, x)
                    y = max(0, y)
                    w_bb = min(img_input.shape[1] - x, w_bb)
                    h_bb = min(img_input.shape[0] - y, h_bb)
                    
                    if w_bb > 0 and h_bb > 0:
                        cropped_stamp = img_input[y:y+h_bb, x:x+w_bb]
                        _, buffer = cv2.imencode('.jpg', cropped_stamp)
                        cropped_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        from Apps.AgentMesh.Utilities.LLM.llm_config import get_vision_llm
                        from langchain_core.messages import HumanMessage
                        
                        llm = get_vision_llm()
                        content = [
                            {"type": "text", "text": ocr_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_b64}"}}
                        ]
                        response = await llm.ainvoke([HumanMessage(content=content)])
                        ocr_text = response.content
                except Exception as e:
                    logger.error(f"Stamp Matcher OCR failed: {e}")
                    ocr_text = f"[OCR ERROR] {str(e)}"
            
            result_data = {
                "good_matches": num_good_matches,
                "raw_sift_matches": len(good_matches),
                "is_match": is_match,
                "threshold_used": match_threshold
            }
            if ocr_text:
                result_data["ocr_extracted_text"] = ocr_text
                
            match_str = "MATCH" if is_match else "NO MATCH"
            summary = f"✅ Stamp Matcher completed: {match_str} ({num_good_matches} geometric matches)"
            if ocr_text:
                summary += f"\n📄 OCR Result: {ocr_text}"
            logger.info(summary)
            
            return {
                "messages": [AIMessage(content=summary, name=f"StampMatch_{node_id}")],
                "active_agent": node_id,
                "context": {
                    **ctx,
                    "stamp_match_result": result_data
                }
            }
            
        except Exception as e:
            error_msg = f"Stamp Matcher Node {node_id} failed: {str(e)}"
            logger.exception(error_msg)
            return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}
            
    return RunnableLambda(execute_match)
