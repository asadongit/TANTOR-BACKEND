import logging
import os
import base64
import time
from typing import Callable, Dict, Any
import anyio
from langchain_core.messages import AIMessage
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

# Dummy proxy IP address used for Mule/Fraud validation routing in the mesh
PROXY_IP_FALLBACK = "104.28.12.50"

def _read_file_sync(path: str) -> bytes:
    """Helper for reading file content synchronously in thread pool."""
    with open(path, "rb") as f:
        return f.read()

def _write_file_sync(path: str, data: bytes):
    """Helper for writing file content synchronously in thread pool."""
    with open(path, "wb") as f:
        f.write(data)

def _compress_image_data(img_data: bytes) -> bytes:
    """Helper function to perform in-memory image compression/resizing."""
    from io import BytesIO
    from PIL import Image
    
    img = Image.open(BytesIO(img_data))
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    
    out_io = BytesIO()
    img.save(out_io, format="JPEG", quality=75)
    return out_io.getvalue()

async def _process_pre_saved_path(pre_saved_path: str, user_data: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Handles logic when the frontend has already compressed and uploaded the file."""
    filepath = os.path.abspath(pre_saved_path)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Uploaded temp file not found at path: {filepath}")
    
    # Check file size for safety (Face++ 2MB limit)
    file_size = os.path.getsize(filepath)
    logger.info(f"Camera node {node_id}: Found pre-saved temp image at {filepath} ({file_size} bytes).")
    
    # Compress on backend if file exceeds 1MB for safety
    if file_size > 1024 * 1024:
        try:
            img_data = await anyio.to_thread.run_sync(_read_file_sync, filepath)
            compressed_data = _compress_image_data(img_data)
            await anyio.to_thread.run_sync(_write_file_sync, filepath, compressed_data)
            logger.info(f"Camera node {node_id}: Compressed pre-saved file from {file_size} to {os.path.getsize(filepath)} bytes.")
        except Exception:
            logger.exception(f"Camera node {node_id}: Failed to run backend compression.")

    context_updates = {
        "vcip_capture_path": filepath,
        "latest_file_path": filepath
    }
    
    try:
        img_bytes = await anyio.to_thread.run_sync(_read_file_sync, filepath)
        context_updates["vcip_capture_base64"] = base64.b64encode(img_bytes).decode('utf-8')
    except Exception:
        logger.exception(f"Camera node {node_id}: Failed to generate base64 update.")
        
    return context_updates

async def _process_base64_image(image_base64: str, node_id: str, thread_id: str) -> Dict[str, Any]:
    """Handles fallback logic when receiving raw base64 data."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
        
    image_data = base64.b64decode(image_base64)
    
    # Compress if exceeds 1MB
    if len(image_data) > 1024 * 1024:
        try:
            image_data = _compress_image_data(image_data)
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Camera node {node_id}: Compressed base64 image to {len(image_data)} bytes.")
        except Exception:
            logger.exception(f"Camera node {node_id}: Failed to compress base64 image.")
 
    # Ensure uploads dir exists
    upload_dir = os.path.join(settings.TEMP_MOUNT, thread_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    filename = f"vcip_capture_{int(time.time())}.jpg"
    filepath = os.path.abspath(os.path.join(upload_dir, filename))
    
    await anyio.to_thread.run_sync(_write_file_sync, filepath, image_data)
    
    return {
        "vcip_capture_path": filepath,
        "vcip_capture_base64": image_base64,
        "latest_file_path": filepath,
        "filename": filename
    }

def build_camera_node(node_config: NodeConfig) -> Callable:
    """
    Build a Camera capture node.
    Supports direct vcip_capture_path from the hybrid frontend upload,
    and falls back to vcip_image_base64 decoding.
    """
    async def camera_capture(state: AgentState):
        logger.debug(f"Camera node {node_config.id} active: Checking for capture data.")
        
        user_data = state.get("user_input") if isinstance(state, dict) else getattr(state, "user_input", {})
        if not isinstance(user_data, dict):
            user_data = {}
            
        pre_saved_path = user_data.get("vcip_capture_path")
        image_base64 = user_data.get("vcip_image_base64")
        
        logger.info(f"Camera Node user_data keys: {list(user_data.keys())}")
        logger.info(f"Camera Node user_ip value: {user_data.get('user_ip')}")
        
        context_updates = {}
        status_msg = "Waiting for Camera capture..."

        if pre_saved_path:
            try:
                context_updates = await _process_pre_saved_path(pre_saved_path, user_data, node_config.id)
                status_msg = f"Camera capture processed successfully from pre-saved path: {os.path.basename(pre_saved_path)}."
                logger.info(f"Camera Node {node_config.id}: {status_msg}")
            except Exception as e:
                logger.exception(f"Camera Node {node_config.id} Error processing pre-saved path: {e}")
                status_msg = f"Error processing Camera capture: {e}"

        elif image_base64:
            try:
                thread_id = state.get("thread_id") or state.get("context", {}).get("thread_id") or "default_thread"
                res = await _process_base64_image(image_base64, node_config.id, thread_id)
                filename = res.pop("filename")
                context_updates = res
                status_msg = f"Camera capture saved successfully as {filename}."
                logger.info(f"Camera Node {node_config.id}: {status_msg}")
            except Exception as e:
                logger.exception(f"Camera Node {node_config.id} Error decoding base64 image: {e}")
                status_msg = f"Error processing Camera capture: {e}"
        else:
            logger.debug(f"Camera node {node_config.id}: No capture data found in user_input.")

        # Inject common telemetry variables (with proxy IP fallback if needed)
        if context_updates:
            client_ip = user_data.get("user_ip")
            context_updates["vcip_capture_user_ip"] = client_ip if client_ip not in [None, "Unknown", ""] else PROXY_IP_FALLBACK
            context_updates["vcip_capture_device_fingerprint"] = user_data.get("device_fingerprint", "Unknown Device")

        # Yield execution time control back to event loop (asynchronous feature compliance)
        await anyio.sleep(0)

        return {
            "active_agent": node_config.id,
            "messages": [AIMessage(content=status_msg, response_metadata={"active_agent": node_config.id})],
            "context": context_updates
        }
    
    return camera_capture
