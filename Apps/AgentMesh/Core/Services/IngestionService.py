import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Apps.AgentMesh.Utilities.Tools.RagTools.vector_store_factory import store_in_vector_store
from Apps.AgentMesh.Utilities.Tools.RagTools.embedding_factory import init_embedding_model
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service responsible for bulk ingestion of data into vector stores.
    Handles loading, chunking, and storage.
    """

    @classmethod
    async def ingest_text(
        cls, 
        text: str, 
        collection_name: str, 
        metadata: Optional[dict] = None,
        vector_store_type: str = "chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive"
    ):
        """Processes a raw string of text."""
        # 1. Select Strategy
        docs = []
        filename = (metadata or {}).get("filename", "raw_text")
        source = (metadata or {}).get("source", "manual_input")

        if chunking_strategy == "none" or chunk_size <= 0:
            docs = [Document(page_content=text, metadata=metadata or {})]
        elif chunking_strategy == "markdown_heading":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.MarkdownHeadingSplitter import chunk_markdown_by_headings
            docs = chunk_markdown_by_headings(text, filename, source, max_size=2000)
        elif chunking_strategy == "recursive":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.RecursiveCharacterTextSplitter import chunk_document
            docs = chunk_document(text, filename, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunking_strategy == "token":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.TokenTextSplitter import chunk_document
            docs = chunk_document(text, filename, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunking_strategy == "json":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.JsonChunking import chunk_document
            docs = chunk_document(text, filename, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # Default fallback
            docs = [Document(page_content=text, metadata=metadata or {})]
        
        # 3. Store
        embeddings = init_embedding_model()
        store_in_vector_store(docs, vector_store_type, embeddings, collection_name=collection_name)
        
        return len(docs)

    @classmethod
    async def ingest_file(
        cls, 
        file_path: str, 
        collection_name: str, 
        vector_store_type: str = "chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        original_filename: Optional[str] = None
    ):
        """Loads and processes a file based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = original_filename or os.path.basename(file_path)
        loader = None
        
        # Select appropriate loader
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        elif ext in [".txt", ".md"]:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(file_path)
        elif ext == ".json":
            if chunking_strategy == "json":
                # Special handling for JSON strategy - we read raw and let JsonChunking handle it
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.JsonChunking import chunk_document
                docs = chunk_document(content, filename, file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                embeddings = init_embedding_model()
                store_in_vector_store(docs, vector_store_type, embeddings, collection_name=collection_name)
                return len(docs)
            else:
                from langchain_community.document_loaders import JSONLoader
                try:
                    loader = JSONLoader(file_path, jq_schema=".", text_content=False)
                except Exception:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path, encoding="utf-8")
        
        if not loader:
            raise ValueError(f"Unsupported file type: {ext}")

        # 1. Load context
        raw_docs = loader.load()
        full_text = "\n".join([d.page_content for d in raw_docs])

        # 2. Chunk based on strategy
        if chunking_strategy == "none" or chunk_size <= 0:
            docs = raw_docs
        elif chunking_strategy == "markdown_heading":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.MarkdownHeadingSplitter import chunk_markdown_by_headings
            docs = chunk_markdown_by_headings(full_text, filename, file_path, max_size=2000)
        elif chunking_strategy == "recursive":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.RecursiveCharacterTextSplitter import chunk_document
            docs = chunk_document(full_text, filename, file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunking_strategy == "token":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.TokenTextSplitter import chunk_document
            docs = chunk_document(full_text, filename, file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunking_strategy == "json":
             # Fallback if someone uses JSON strategy on non-JSON file (though handled above for JSON files)
             from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.JsonChunking import chunk_document
             docs = chunk_document(full_text, filename, file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            docs = raw_docs
        
        # 3. Store
        embeddings = init_embedding_model()
        store_in_vector_store(docs, vector_store_type, embeddings, collection_name=collection_name)
        
        return len(docs)
