from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import os
import shutil
import tempfile
import logging
import re
from pathvalidate import sanitize_filename

from Apps.AgentMesh.Schema.RunSchema import IngestTextRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG & Ingestion"])

# --- Security Constants ---
MAX_DOC_SIZE = 10 * 1024 * 1024  # 10MB
MAX_MEDIA_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_DOCS = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".flac"}
ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}

def validate_params(collection_name: str, chunk_size: int, chunk_overlap: int):
    if not re.match(r"^[a-zA-Z0-9_\-]+$", collection_name):
        raise HTTPException(status_code=400, detail="Invalid collection_name. Use only alphanumeric, _, or -")
    if not (100 <= chunk_size <= 10000):
        raise HTTPException(status_code=400, detail="chunk_size must be between 100 and 10,000")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")

@router.get("/collections")
async def list_collections(vector_store: str = "chroma"):
    """List all collections from specified vector store (chroma or qdrant)."""
    try:
        if vector_store.lower() == "qdrant":
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
            cols = qdrant_manager.list_collections()
        else:
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import chroma_manager
            cols = chroma_manager.list_collections()
        return {"vector_store": vector_store, "collections": cols}
    except Exception as e:
        logger.error(f"Error fetching collections for {vector_store}: {e}")
        return {"vector_store": vector_store, "collections": ["default"], "error": str(e)}

@router.get("/collections/details")
async def get_collection_details(
    collection_name: str = "default", 
    vector_store: str = "chroma",
    limit: int = 50,
    offset: int = 0
):
    """Get metadata metrics (vector count, dimension, metric, file breakdown) and paginated chunks for a collection."""
    try:
        if vector_store.lower() == "qdrant":
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
            details = qdrant_manager.get_collection_details(collection_name, limit=limit, offset=offset)
        else:
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import chroma_manager
            details = chroma_manager.get_collection_details(collection_name, limit=limit, offset=offset)
        return {"vector_store": vector_store, "details": details}
    except Exception as e:
        logger.error(f"Error fetching details for collection '{collection_name}': {e}")
        return {
            "vector_store": vector_store,
            "details": {
                "name": collection_name,
                "exists": False,
                "total_chunks": 0,
                "vector_size": 384,
                "distance": "COSINE",
                "files": []
            }
        }

@router.delete("/collections")
async def delete_collection(collection_name: str, vector_store: str = "chroma"):
    """Delete a collection from Chroma or Qdrant."""
    try:
        if vector_store.lower() == "qdrant":
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Qdrant_instance import qdrant_manager
            client = qdrant_manager._get_client()
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
        else:
            from Apps.AgentMesh.Utilities.Tools.RagTools.VectorDatabaseModules.Chroma_instance import chroma_manager
            client = chroma_manager._get_client()
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
        return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_upload_file_safely(file: UploadFile, allowed_exts: set, max_size: int) -> str:
    # 1. Validate Extension
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"File extension {ext} not allowed. Supported: {allowed_exts}")
    
    # 2. Sanitize Filename for temp storage
    safe_name = sanitize_filename(file.filename or "upload")
    
    # 3. Read and check size
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        total_size = 0
        while True:
            chunk = await file.read(8192) # 8KB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail=f"File too large. Max size allowed is {max_size / (1024*1024)}MB")
            tmp.write(chunk)
        return tmp.name

@router.post("/ingest")
async def ingest_data(
    file: UploadFile = File(None),
    text: str = Form(None),
    collection_name: str = Form("default"),
    vector_store: str = Form("chroma"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    chunking_strategy: str = Form("recursive"),
    model_name: str = Form("base"),
    target_type: str = Form("vector"),
    llm_provider: str = Form(None),
    llm_model: str = Form(None),
    allowed_nodes: str = Form(None),
    allowed_relationships: str = Form(None),
    graph_database: str = Form(None),
    context_hint: str = Form(None)
):
    from Apps.AgentMesh.Core.Services.IngestionService import IngestionService
    from Apps.AgentMesh.Core.Services.GraphIngestionService import GraphIngestionService
    from Apps.AgentMesh.Utilities.Audio.audio_ingestion import AudioIngestionService
    
    # Parse allowed nodes/relationships lists
    nodes_list = [n.strip() for n in allowed_nodes.split(",") if n.strip()] if allowed_nodes else None
    rels_list = [r.strip() for r in allowed_relationships.split(",") if r.strip()] if allowed_relationships else None

    if target_type == "vector" and chunking_strategy != "markdown_heading":
        validate_params(collection_name, chunk_size, chunk_overlap)
    
    # 1. Text Ingestion
    if text and not file:
        try:
            if target_type == "graph":
                count = await GraphIngestionService.ingest_text(
                    text=text,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    allowed_nodes=nodes_list,
                    allowed_relationships=rels_list,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy,
                    metadata={},
                    graph_database=graph_database,
                    context_hint=context_hint
                )
            else:
                count = await IngestionService.ingest_text(
                    text=text,
                    collection_name=collection_name,
                    vector_store_type=vector_store,
                    metadata={},
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy
                )
            return {"status": "success", "chunks_ingested": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    # 2. File Ingestion
    if file:
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        
        # Audio
        if ext in ALLOWED_AUDIO:
            if target_type == "graph":
                raise HTTPException(status_code=400, detail="Graph DB ingestion currently only supports text-based documents.")
            tmp_path = await save_upload_file_safely(file, ALLOWED_AUDIO, MAX_MEDIA_SIZE)
            try:
                count = await AudioIngestionService.ingest_audio(
                    file_path=tmp_path,
                    collection_name=collection_name,
                    vector_store_type=vector_store,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy,
                    model_name=model_name
                )
                return {"status": "success", "filename": file.filename, "chunks_ingested": count}
            except Exception as e:
                logger.error(f"Audio ingestion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        # Video
        elif ext in ALLOWED_VIDEO:
            if target_type == "graph":
                raise HTTPException(status_code=400, detail="Graph DB ingestion currently only supports text-based documents.")
            tmp_path = await save_upload_file_safely(file, ALLOWED_VIDEO, MAX_MEDIA_SIZE)
            try:
                count = await AudioIngestionService.ingest_video(
                    file_path=tmp_path,
                    collection_name=collection_name,
                    vector_store_type=vector_store,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy,
                    model_name=model_name
                )
                return {"status": "success", "filename": file.filename, "chunks_ingested": count}
            except Exception as e:
                logger.error(f"Video ingestion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        # Documents / Images
        elif ext in ALLOWED_DOCS:
            is_media = ext in {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
            
            # Special media bypass for OCR (Only relevant for vector store currently)
            if is_media and target_type == "vector":
                from Apps.AgentMesh.Config import settings
                upload_dir = os.path.join(settings.UPLOAD_MOUNT if hasattr(settings, "UPLOAD_MOUNT") else "./mnt/uploads")
                os.makedirs(upload_dir, exist_ok=True)
                
                safe_name = sanitize_filename(file.filename or "file")
                import time
                final_filename = f"{int(time.time())}_{safe_name}"
                final_path = os.path.join(upload_dir, final_filename)
                
                with open(final_path, "wb") as f:
                    await file.seek(0)
                    shutil.copyfileobj(file.file, f)
                    
                return {
                    "status": "success", 
                    "filename": file.filename, 
                    "file_path": final_path,
                    "chunks_ingested": 0,
                    "message": f"File uploaded to {final_path}. Skipped vector ingestion for processing."
                }
                
            # Regular text-based docs or PDF for graph
            tmp_path = await save_upload_file_safely(file, ALLOWED_DOCS, MAX_DOC_SIZE)
            try:
                if target_type == "graph":
                    count = await GraphIngestionService.ingest_file(
                        file_path=tmp_path,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        allowed_nodes=nodes_list,
                        allowed_relationships=rels_list,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunking_strategy=chunking_strategy,
                        original_filename=file.filename,
                        graph_database=graph_database,
                        context_hint=context_hint
                    )
                else:
                    count = await IngestionService.ingest_file(
                        file_path=tmp_path,
                        collection_name=collection_name,
                        vector_store_type=vector_store,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunking_strategy=chunking_strategy,
                        original_filename=file.filename
                    )
                return {
                    "status": "success", 
                    "filename": file.filename, 
                    "file_path": tmp_path,
                    "chunks_ingested": count
                }
            except Exception as e:
                logger.error(f"File ingestion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
            
    raise HTTPException(status_code=400, detail="Must provide either text or a file.")
