from typing import List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool, Tool, StructuredTool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Utilities.Tools.RagTools.vector_store_factory import query_vector_store, store_in_vector_store
from Apps.AgentMesh.Utilities.Tools.RagTools.embedding_factory import init_embedding_model
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from langchain_core.documents import Document

# --- Inputs Schema ---

class MemoryStoreInput(BaseModel):
    content: str = Field(description="The factual information or detail to remember for this specific conversation thread")

class MemoryRetrieveInput(BaseModel):
    query: str = Field(description="The question or topic to search for in this conversation's memory")

class RagSearchInput(BaseModel):
    query: str = Field(description="The search query to look up in the vector knowledge base")
    collection_name: str = Field(..., description="The specific vector collection to search (e.g., 'technical_docs', 'proposals')")
    k: int = Field(default=5, description="Number of document snippets to retrieve")
    threshold: float = Field(default=0.5, description="Similarity threshold (0.0 to 1.0). Higher means more relevant results.")
    vector_store_type: str = Field(default="chroma", description="The type of vector store: 'chroma' or 'qdrant'")
    search_type: str = Field(default="similarity", description="Search strategy: 'similarity', 'hybrid', or 'keyword'")
    show_citations: bool = Field(default=False, description="Whether to include citation (filenames) in results")

# --- 1. Thread-Based Memory Tools ---

@tool("store_memory", args_schema=MemoryStoreInput)
async def store_memory(content: str, config: RunnableConfig) -> str:
    """
    Store information into the agent's long-term memory for THIS specific thread.
    The agent will automatically use the current thread ID to isolate this memory.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    collection_name = f"thread_{thread_id}"
    
    print(f"\n💾 [MEMORY] Storing in thread-collection '{collection_name}': '{content[:50]}...'")
    embeddings = init_embedding_model()
    doc = Document(page_content=content, metadata={"source": "agent_memory", "thread_id": thread_id})
    
    store_in_vector_store([doc], "chroma", embeddings, collection_name=collection_name)
    return f"Successfully saved to memory for thread: {thread_id}"

@tool("retrieve_memory", args_schema=MemoryRetrieveInput)
async def retrieve_memory(query: str, config: RunnableConfig) -> str:
    """
    Retrieve facts previously stored in memory for THIS specific thread.
    Use this to recall past details provided by the user or learned from experience.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    collection_name = f"thread_{thread_id}"
    
    print(f"\n🔍 [MEMORY] Querying thread-collection '{collection_name}' for: '{query}'...")
    embeddings = init_embedding_model()
    results = query_vector_store(query, "chroma", embeddings, collection_name=collection_name, k=5)
    
    if not results:
        return f"No relevant thread-memory found for '{query}'."
    
    return "\n\n".join([f"Memory Snippet:\n{doc.page_content}" for doc in results])

# --- 2. Highly Configurable RAG Tool ---

@tool("search_rag", args_schema=RagSearchInput)
async def search_rag(
    query: str, 
    collection_name: str, 
    k: int = 5, 
    threshold: float = 0.5, 
    vector_store_type: str = "chroma",
    search_type: str = "similarity",
    show_citations: bool = False
) -> str:
    """
    Advanced RAG search tool. Allows searching specific collections with customizable 
    top-K results, similarity thresholds, vector store types, and search strategies (hybrid/keyword).
    """
    import os
    print(f"\n🔍 [RAG] Searching '{collection_name}' ({search_type}, k={k}) for: '{query}'...")
    embeddings = init_embedding_model()
    
    results = query_vector_store(
        query, 
        vector_store_type, 
        embeddings, 
        collection_name=collection_name, 
        k=k,
        threshold=threshold,
        search_type=search_type
    )
    
    if not results:
        return f"No relevant information found in the '{collection_name}' collection."
    
    formatted_results = []
    should_show_citations = str(show_citations).lower() in ("true", "1", "yes") if isinstance(show_citations, str) else bool(show_citations)
    for doc in results:
        if should_show_citations:
            filename = doc.metadata.get("filename") or (os.path.basename(doc.metadata.get("source", "")) if doc.metadata.get("source") else "RAG")
            formatted_results.append(f"Source: {filename}\n{doc.page_content}")
        else:
            formatted_results.append(doc.page_content)
            
    return "\n\n".join(formatted_results)

# --- Registry Integration ---

def register_rag_tools():
    # Register the tools with metadata for the UI
    
    registry.register(
        store_memory,
        ToolMeta(
            name="store_memory", 
            display_name="Memory Storage (Thread)", 
            description="Save facts for this specific conversation.", 
            category="Memory",
            package="built-in",
            version="1.1.0",
            ai_visible_fields=["content"], # LLM only fills the content
            configs=[
                {
                    "name": "vector_store_type", 
                    "type": "string", 
                    "description": "The vector database provider", 
                    "required": False,
                    "options": ["chroma", "qdrant"],
                    "default": "chroma"
                }
            ]
        )
    )
    registry.register(
        retrieve_memory,
        ToolMeta(
            name="retrieve_memory", 
            display_name="Memory Retrieval (Thread)", 
            description="Recall facts for this specific conversation.", 
            category="Memory",
            package="built-in",
            version="1.1.0",
            ai_visible_fields=["query"], # LLM only fills the query
            configs=[
                {
                    "name": "vector_store_type", 
                    "type": "string", 
                    "description": "Which vector store to query", 
                    "required": False,
                    "options": ["chroma", "qdrant"],
                    "default": "chroma"
                }
            ]
        )
    )
    registry.register(
        search_rag,
        ToolMeta(
            name="search_rag", 
            display_name="Advanced RAG Search", 
            description="Search specific collections with custom config.", 
            category="RAG",
            package="built-in",
            version="1.1.0",
            ai_visible_fields=["query"], # LLM only fills the query
            configs=[
                {"name": "collection_name", "type": "string", "description": "Vector collection name", "required": True, "default": "default"},
                {"name": "k", "type": "number", "description": "Number of results to return", "required": False, "default": 5},
                {"name": "threshold", "type": "number", "description": "Similarity threshold (0.0 to 1.0)", "required": True, "default": 0.5},
                {
                    "name": "search_type", 
                    "type": "string", 
                    "description": "Search strategy: similarity, hybrid, keyword", 
                    "required": False,
                    "options": ["similarity", "hybrid", "keyword"],
                    "default": "similarity"
                },
                {
                    "name": "show_citations",
                    "type": "boolean",
                    "description": "Include source file names in retrieval results",
                    "required": False,
                    "options": ["true", "false"],
                    "default": "false"
                },
                {
                    "name": "vector_store_type", 
                    "type": "string", 
                    "description": "The vector database provider", 
                    "required": False,
                    "options": ["chroma", "qdrant"],
                    "default": "chroma"
                }
            ]
        )
    )

# Automatically register on import
register_rag_tools()
