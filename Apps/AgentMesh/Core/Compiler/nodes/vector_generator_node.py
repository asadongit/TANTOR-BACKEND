import base64
import importlib
import io
import logging
import os
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Deep-learning dependencies are loaded LAZILY, never at import time.
#
# `nodes/__init__.py` imports every node module eagerly, so anything imported at
# the top of this file is imported by every workflow compile in the system. That
# had two consequences worth avoiding:
#
#   1. A missing optional package (torchvision) raised ModuleNotFoundError during
#      `import nodes`, which failed the compile of EVERY workflow — including
#      workflows containing no vector_generator node at all.
#   2. Importing torch costs seconds and hundreds of MB of RSS, paid on every
#      startup by every deployment, whether or not this node is ever used.
#
# Both are fixed by importing inside the functions that need them, and by
# reporting a missing dependency as an actionable node-level error instead of a
# system-wide import crash.
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_DEPS = ("torch", "torchvision", "PIL")


class VectorDependencyError(RuntimeError):
    """Raised when this node runs but its optional ML dependencies are absent."""


def missing_vector_deps() -> list:
    """Return the names of any vector dependencies that cannot be imported.

    Cheap: uses find_spec, so it does not actually load torch.
    """
    missing = []
    for name in VECTOR_DEPS:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def _require_vector_deps() -> None:
    """Fail with an actionable message if the ML stack is not installed."""
    missing = missing_vector_deps()
    if missing:
        raise VectorDependencyError(
            f"The Vector Generator node requires {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. Install the "
            f"matched CPU build with:\n"
            f"    uv pip install \"torch==2.12.0\" torchvision "
            f"--index-url https://download.pytorch.org/whl/cpu\n"
            f"Pin torch explicitly — installing torchvision alone can pull a "
            f"different torch build and break other nodes."
        )


def _build_siamese_class():
    """Define the model class at call time, since it subclasses torch.nn.Module.

    A `class X(nn.Module)` at module scope would execute at import and therefore
    require torch at import — exactly what this module avoids.
    """
    import torch.nn as nn

    class SiameseResNet50(nn.Module):
        def __init__(self, vector_dimensions=256):
            super(SiameseResNet50, self).__init__()
            from torchvision.models import resnet50
            self.backbone = resnet50(weights=None)
            # Replace the original fc layer with Identity
            self.backbone.fc = nn.Identity()

            # Add the head that matches the saved state_dict
            self.head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, int(vector_dimensions))
            )

        def forward(self, x):
            # x goes through backbone, which now outputs (B, 2048)
            x = self.backbone(x)
            # x goes through head
            return self.head(x)

    return SiameseResNet50


# Singleton to cache the model in memory so we don't load it on every document
_model_instance = None

def get_siamese_model(model_path: str, vector_dimensions: int = 256):
    global _model_instance
    if _model_instance is None:
        _require_vector_deps()
        import torch

        logger.info(f"Loading Siamese ResNet50 model from {model_path} with {vector_dimensions} dimensions...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _build_siamese_class()(vector_dimensions=vector_dimensions)

        try:
            # Try loading as state_dict (Recommended)
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            logger.warning(f"Failed to load as state_dict ({e}). Trying full model load...")
            # Fallback if the user saved the entire model object
            model = torch.load(model_path, map_location=device)

        model.to(device)
        model.eval()
        _model_instance = (model, device)

    return _model_instance

def build_vector_generator_node(node_config: NodeConfig) -> Callable:
    """
    Builds a Vector Generator node.
    Takes a base64 image from the context and outputs an N-d float array (vector) to the context.
    """
    vector_dimensions = getattr(node_config, "vector_dimensions", 256)
    model_name = getattr(node_config, "model_name", "resnet50")

    # Surface a missing ML stack at COMPILE time as a warning, so it is visible
    # immediately rather than only when the node first runs. We warn instead of
    # raising so an incomplete environment cannot block editing or compiling a
    # canvas; execution still fails loudly via _require_vector_deps().
    _missing = missing_vector_deps()
    if _missing:
        logger.warning(
            f"Vector Generator {getattr(node_config, 'id', '?')} compiled, but "
            f"{', '.join(_missing)} {'is' if len(_missing) == 1 else 'are'} not "
            f"installed. This node will fail if executed."
        )

    # Standard ResNet preprocessing, built once on first execution and cached.
    # Deferred rather than built here so that compiling a workflow never imports
    # torchvision — compile stays fast and dependency-free.
    _preprocess_cache = {}

    def _get_preprocess():
        if "p" not in _preprocess_cache:
            from torchvision import transforms
            _preprocess_cache["p"] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        return _preprocess_cache["p"]

    async def execute_generation(state: AgentState):
        ctx = state.get("context", {})
        node_id = node_config.id
        
        # Read keys from UI config, default to raw inputs
        sig_key = getattr(node_config, "input_signature_key", "signature_b64") or "signature_b64"
        stamp_key = getattr(node_config, "input_stamp_key", "stamp_b64") or "stamp_b64"
        
        b64_sig = ctx.get("payload", {}).get(sig_key) or ctx.get(sig_key)
        b64_stamp = ctx.get("payload", {}).get(stamp_key) or ctx.get(stamp_key)
        
        def get_images(data):
            if not data: return []
            if isinstance(data, list):
                # Handle both list of dicts (from frontend form) and list of strings
                return [d.get("data") if isinstance(d, dict) else d for d in data if d]
            elif isinstance(data, dict):
                return [data.get("data")]
            return [data]

        sig_images = get_images(b64_sig)
        stamp_images = get_images(b64_stamp)
            
        if not sig_images and not stamp_images:
            msg = f"Vector Generator {node_id}: No signature_b64 or stamp_b64 found in payload."
            return {"messages": [AIMessage(content=f"[ERROR] {msg}")], "active_agent": node_id}

        try:
            # Fail with an actionable message before touching the model or files.
            _require_vector_deps()
            import torch
            from PIL import Image

            preprocess = _get_preprocess()

            # Map frontend model_name to the actual path
            if model_name == "resnet50":
                model_path = r"C:\Tantor\AgentMesh2.0\Backend\models\wingbankpoc\siamese_resnet50_signature1.pth"
            else:
                model_path = r"C:\Tantor\AgentMesh2.0\Backend\models\wingbankpoc\siamese_resnet50_signature1.pth" # fallback
                
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            model, device = get_siamese_model(model_path, vector_dimensions=vector_dimensions)

            batch_vectors = []
            
            # Extract reference_id from context to pass along
            ref_id = ctx.get("payload", {}).get("reference_id") or ctx.get("reference_id") or "unknown"
            
            def process_image(b64_img, img_type):
                original_b64 = b64_img
                if "base64," in b64_img:
                    b64_img = b64_img.split("base64,")[1]
                img_data = base64.b64decode(b64_img)
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    vector_tensor = model(input_tensor)
                vector_list = vector_tensor.cpu().numpy().flatten().tolist()
                
                payload_key = "stamp_b64" if img_type == "Stamp" else "signature_b64"
                return {"type": img_type, "vector": vector_list, "payload": {payload_key: original_b64, "reference_id": ref_id, "type": img_type}}

            for img in sig_images:
                if img: batch_vectors.append(process_image(img, "Signature"))
            for img in stamp_images:
                if img: batch_vectors.append(process_image(img, "Stamp"))
                
            summary = f"✅ Vector Generator: Successfully generated {len(batch_vectors)} embedding vectors ({len(sig_images)} Signatures, {len(stamp_images)} Stamps)."
            logger.info(summary)
            
            return {
                "messages": [AIMessage(content=summary, name=f"VectorGen_{node_id}")],
                "active_agent": node_id,
                "context": {
                    **ctx,
                    "batch_vectors": batch_vectors
                }
            }
            
        except Exception as e:
            error_msg = f"Vector Generator Node {node_id} failed: {str(e)}"
            logger.exception(error_msg)
            return {"messages": [AIMessage(content=f"[ERROR] {error_msg}")], "active_agent": node_id}
            
    return RunnableLambda(execute_generation)
