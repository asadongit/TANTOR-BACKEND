import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from Apps.AgentMesh.Utilities.Tools.GraphTool.graph_tools import Neo4jConnectionManager
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

logger = logging.getLogger(__name__)

from Apps.AgentMesh.Core.Services.GraphExtractor import GraphExtractor


class GraphIngestionService:
    """
    Service responsible for loading documents, chunking them,
    extracting entity/relationship graphs via LLM, and ingesting them to Neo4j.
    """

    @staticmethod
    def _sanitize_properties(properties: dict) -> dict:
        import json
        sanitized = {}
        for k, v in properties.items():
            if isinstance(v, (dict, list)):
                if isinstance(v, list):
                    if all(isinstance(x, (str, int, float, bool)) for x in v):
                        sanitized[k] = v
                    else:
                        sanitized[k] = json.dumps(v)
                else:
                    sanitized[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    @classmethod
    async def ingest_text(
        cls,
        text: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        metadata: Optional[dict] = None,
        graph_database: Optional[str] = None,
        context_hint: Optional[str] = None
    ) -> int:
        """Processes a raw string of text, extracts graph triples, and saves to Neo4j."""
        # 1. Chunking
        filename = (metadata or {}).get("filename", "raw_text")
        source = (metadata or {}).get("source", "manual_input")

        if chunking_strategy == "none" or (chunking_strategy == "recursive" and chunk_size <= 0):
            docs = [Document(page_content=text, metadata=metadata or {})]
        elif chunking_strategy == "markdown_heading":
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.MarkdownHeadingSplitter import chunk_markdown_by_headings
            docs = chunk_markdown_by_headings(text, filename, source, max_size=2000)
        elif chunking_strategy == "markdown_context" or (chunking_strategy == "recursive" and (filename.endswith(".md") or text.strip().startswith("#"))):
            from Apps.AgentMesh.Utilities.Tools.RagTools.ChunkingModules.MarkdownContextSplitter import chunk_markdown_with_context
            docs = chunk_markdown_with_context(text, filename, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
            docs = [Document(page_content=text, metadata=metadata or {})]

        # 2. Extract Triples using custom GraphExtractor
        llm = get_llm_provider(provider=llm_provider, model=llm_model)
        extractor = GraphExtractor(
            llm=llm,
            allowed_nodes=allowed_nodes or [],
            allowed_relationships=allowed_relationships or [],
            context_hint=context_hint
        )

        graph_docs = extractor.convert_to_graph_documents(docs)

        # 3. Store in Neo4j
        uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("Neo4j credentials are not configured")

        driver = await Neo4jConnectionManager.get_driver(uri, user, password)

        import hashlib

        # Group nodes by dynamic multiple label key and relationships by type
        nodes_by_label = {}
        relationships_by_type = {}

        node_count = 0
        rel_count = 0

        for idx, graph_doc in enumerate(graph_docs):
            # Calculate a unique chunk ID using md5 hash of page content
            page_content = graph_doc.source.page_content if hasattr(graph_doc, "source") and graph_doc.source else ""
            chunk_hash = hashlib.md5(page_content.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{filename}_ch_{idx}_{chunk_hash}"

            for node in graph_doc.nodes:
                types = node.properties.get("types") or [node.type or "Entity"]
                clean_types = []
                for t in types:
                    clean_t = t.strip().replace(" ", "_")
                    if clean_t.replace("_", "").isalnum():
                        clean_types.append(clean_t)
                if not clean_types:
                    clean_types = ["Entity"]
                
                label_key = ":".join(f"`{lt}`" for lt in clean_types)
                
                if label_key not in nodes_by_label:
                    nodes_by_label[label_key] = []
                
                props = dict(node.properties or {})
                description = props.pop("description", "")
                summary = props.pop("summary", "")
                tags = props.pop("tags", [])
                
                # Make sure ID/name is stored in properties too
                props["name"] = node.id
                
                # Sanitize properties to prevent Neo4j TypeErrors (no maps allowed in properties)
                props = cls._sanitize_properties(props)
                
                if not any(n["id"] == node.id for n in nodes_by_label[label_key]):
                    nodes_by_label[label_key].append({
                        "id": node.id,
                        "description": description,
                        "summary": summary,
                        "tags": tags,
                        "chunk_id": chunk_id,
                        "properties": props
                    })
                    node_count += 1

            for rel in graph_doc.relationships:
                rel_type = (rel.type or "RELATED_TO").strip().replace(" ", "_")
                if not rel_type.replace("_", "").isalnum():
                    rel_type = "RELATED_TO"

                # Resolve dynamic source label keys
                source_types = rel.source.properties.get("types") or [rel.source.type or "Entity"]
                clean_source_types = []
                for t in source_types:
                    clean_t = t.strip().replace(" ", "_")
                    if clean_t.replace("_", "").isalnum():
                        clean_source_types.append(clean_t)
                if not clean_source_types:
                    clean_source_types = ["Entity"]
                source_label_key = ":".join(f"`{st}`" for st in clean_source_types)

                # Resolve dynamic target label keys
                target_types = rel.target.properties.get("types") or [rel.target.type or "Entity"]
                clean_target_types = []
                for t in target_types:
                    clean_t = t.strip().replace(" ", "_")
                    if clean_t.replace("_", "").isalnum():
                        clean_target_types.append(clean_t)
                if not clean_target_types:
                    clean_target_types = ["Entity"]
                target_label_key = ":".join(f"`{tt}`" for tt in clean_target_types)

                group_key = (source_label_key, rel_type, target_label_key)

                if group_key not in relationships_by_type:
                    relationships_by_type[group_key] = []

                props = dict(rel.properties or {})
                description = props.pop("description", "")
                
                # Sanitize properties to prevent Neo4j TypeErrors (no maps allowed in properties)
                props = cls._sanitize_properties(props)

                relationships_by_type[group_key].append({
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "description": description,
                    "chunk_id": chunk_id,
                    "properties": props
                })
                rel_count += 1

        session_params = {}
        if graph_database:
            session_params["database"] = graph_database

        async with driver.session(**session_params) as session:
            # Batch merge nodes with weight counts, descriptions, summaries, tags, and source chunks
            for label, batch in nodes_by_label.items():
                query = f"""
                UNWIND $batch AS row
                MERGE (n {{id: row.id}})
                ON CREATE SET 
                    n:{label},
                    n += row.properties,
                    n.description = row.description,
                    n.summary = row.summary,
                    n.tags = row.tags,
                    n.weight = 1,
                    n.source_chunks = [row.chunk_id]
                ON MATCH SET 
                    n:{label},
                    n += row.properties,
                    n.description = CASE WHEN n.description IS NULL OR n.description = "" THEN row.description ELSE n.description END,
                    n.summary = CASE WHEN n.summary IS NULL OR n.summary = "" THEN row.summary ELSE n.summary END,
                    n.weight = n.weight + 1,
                    n.source_chunks = CASE WHEN row.chunk_id IN n.source_chunks THEN n.source_chunks ELSE n.source_chunks + row.chunk_id END
                """
                await session.run(query, {"batch": batch})

            # Batch merge relationships using dynamic label MERGE to connect nodes robustly
            for (source_label, rel_type, target_label), batch in relationships_by_type.items():
                query = f"""
                UNWIND $batch AS row
                MERGE (source {{id: row.source_id}})
                SET source:{source_label}
                MERGE (target {{id: row.target_id}})
                SET target:{target_label}
                MERGE (source)-[r:`{rel_type}`]->(target)
                ON CREATE SET 
                    r += row.properties,
                    r.description = row.description,
                    r.weight = 1,
                    r.source_chunks = [row.chunk_id]
                ON MATCH SET 
                    r += row.properties,
                    r.description = CASE WHEN r.description IS NULL OR r.description = "" THEN row.description ELSE r.description END,
                    r.weight = r.weight + 1,
                    r.source_chunks = CASE WHEN row.chunk_id IN r.source_chunks THEN r.source_chunks ELSE r.source_chunks + row.chunk_id END
                """
                await session.run(query, {"batch": batch})

        logger.info(f"Ingested {node_count} nodes and {rel_count} relationships into Neo4j graph database: {graph_database or 'default'}.")
        return node_count

    @classmethod
    async def ingest_file(
        cls,
        file_path: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        original_filename: Optional[str] = None,
        graph_database: Optional[str] = None,
        context_hint: Optional[str] = None
    ) -> int:
        """Loads and processes a file, extracts graph triples, and saves to Neo4j."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = original_filename or os.path.basename(file_path)
        loader = None

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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return await cls.ingest_text(
                text=content,
                llm_provider=llm_provider,
                llm_model=llm_model,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                metadata={"filename": filename, "source": file_path},
                graph_database=graph_database,
                context_hint=context_hint
            )

        if not loader:
            raise ValueError(f"Unsupported file type: {ext}")

        raw_docs = loader.load()
        full_text = "\n".join([d.page_content for d in raw_docs])

        return await cls.ingest_text(
            text=full_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            metadata={"filename": filename, "source": file_path},
            graph_database=graph_database,
            context_hint=context_hint
        )
