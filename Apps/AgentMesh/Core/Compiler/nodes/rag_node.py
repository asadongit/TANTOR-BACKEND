"""
RAG Node Builder
Handles retrieval-augmented generation with vector stores.
"""

import logging
from typing import Callable

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from Apps.AgentMesh.Utilities.Tools.RagTools.embedding_factory import init_embedding_model
from Apps.AgentMesh.Utilities.Tools.RagTools.vector_store_factory import query_vector_store

logger = logging.getLogger(__name__)


def build_rag_node(node_config: NodeConfig) -> Callable:
    """
    Build a RAG node that retrieves context from vector store
    and answers questions based on that context.
    """
    llm = init_chat_model(node_config.llm)
    embeddings = init_embedding_model(node_config.embedding_model)
    
    async def execute_rag(state: AgentState):
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""
        
        logger.debug(f"RAG node {node_config.id}: Processing query: {query[:50]}...")
        
        # Perform similarity search
        docs = query_vector_store(
            query=query,
            vector_store_type=node_config.vector_store or "chroma",
            collection_name=node_config.vector_collection or "default",
            embedding_model=embeddings,
            k=node_config.k_results or 5,
            threshold=node_config.threshold or 0.5,
            search_type=node_config.search_type or "similarity"
        )
        
        import os
        logger.debug(f"RAG node {node_config.id}: Retrieved {len(docs)} documents")
        
        context_chunks = []
        for d in docs:
            if getattr(node_config, "show_citations", False):
                filename = d.metadata.get("filename") or (os.path.basename(d.metadata.get("source", "")) if d.metadata.get("source") else "Unknown")
                context_chunks.append(f"Source: {filename}\nContent: {d.page_content}")
            else:
                context_chunks.append(d.page_content)
        context = "\n\n".join(context_chunks)
        
        # Build chat history from previous messages
        chat_history = []
        for m in messages[:-1]:  # Exclude the latest query
            msg_type = getattr(m, 'type', None) or m.__class__.__name__.replace('Message', '').lower()
            chat_history.append(f"{msg_type}: {m.content}")
        
        history_str = "\n".join(chat_history[-5:])  # Keep last 5 for context limit
        
        # Base RAG prompt (Internal/Predefined)
        system_content = (
            "You are a helpful assistant answering based on the provided context.\n"
            "Use the following context to answer the question. If you cannot find the answer, state that you don't know.\n\n"
            f"Recent Conversation History:\n{history_str}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )
        
        # Add Custom/Extra Instructions if provided
        if node_config.task:
            system_content += f"\nAdditional Instructions: {node_config.task}\n"
            
        system_content += "\nFinal Answer:"
        
        messages_to_send = [
            SystemMessage(content=system_content),
        ]
        
        response = await llm.ainvoke(messages_to_send)
        
        logger.debug(f"RAG node {node_config.id}: Generated response")
        
        return {
            "messages": [response],
            "active_agent": node_config.id,
            "context": {
                **state.get("context", {}),
                "retrieved_context": context
            }
        }
    
    return RunnableLambda(execute_rag)
