import json
import logging
import re
from typing import List, Optional, Tuple, Sequence, Dict, Any
from pydantic import BaseModel, Field, AliasChoices, field_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument

logger = logging.getLogger(__name__)

# --- JSON Repair Utility ---

def repair_json(json_str: str) -> str:
    """Basic attempt to repair truncated JSON by adding missing closing characters."""
    json_str = json_str.strip()
    if not json_str:
        return "{}"
    
    stack = []
    in_string = False
    escaped = False
    
    for char in json_str:
        if char == '"' and not escaped:
            in_string = not in_string
        elif char == '\\' and not escaped:
            escaped = True
            continue
        elif not in_string:
            if char == '{': stack.append('}')
            elif char == '[': stack.append(']')
            elif char in ('}', ']'):
                if stack and stack[-1] == char:
                    stack.pop()
        escaped = False
        
    if in_string:
        json_str += '"'
    
    while stack:
        json_str += stack.pop()
        
    return json_str


# --- Pydantic Schema for Structured Output ---

class ExtractedNode(BaseModel):
    id: str = Field(description="Name or human-readable unique identifier of the node/entity. IMPORTANT: For financial metrics, periodic values, or events, include the fiscal year or period in the ID to ensure uniqueness across different years (e.g., 'Net Automotive Debt FY2023-24' instead of just 'Net Automotive Debt'). Do not use parentheses.")
    types: List[str] = Field(default_factory=list, description="The labels or types of the node. The first label should be a standard category (e.g., PERSON, CORPORATION, PRODUCT), and subsequent labels should be more specific (e.g., SUV, CEO).")
    description: Optional[str] = Field(None, description="Detailed context or description of what/who this entity is.")
    summary: Optional[str] = Field(None, description="A brief summary of the entity's context snippet or main description in the text.")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of search aliases, keywords, or topics associated with this entity.")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key-value pairs of additional attributes or properties.")

    @field_validator('id', mode='before')
    @classmethod
    def coerce_id(cls, v: Any) -> str:
        if isinstance(v, dict):
            for key in ['id', 'name', 'value']:
                if key in v and v[key]:
                    return str(v[key])
            if v:
                return str(next(iter(v.values())))
        return str(v) if v is not None else ""

class ExtractedRelationship(BaseModel):
    source_node_id: str = Field(description="Name or human-readable unique identifier of the source node.", validation_alias=AliasChoices('source_node_id', 'source'))
    source_node_types: List[str] = Field(default_factory=list, description="The labels or types of the source node.")
    target_node_id: str = Field(description="Name or human-readable unique identifier of the target node.", validation_alias=AliasChoices('target_node_id', 'target'))
    target_node_types: List[str] = Field(default_factory=list, description="The labels or types of the target node.")
    type: Optional[str] = Field(default="RELATED_TO", description="The type of the relationship connecting the source and target.", validation_alias=AliasChoices('type', 'relation'))
    description: Optional[str] = Field(None, description="Detailed explanation of how/why these entities are connected in the context of the source text.")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key-value pairs of additional attributes or properties.")

    @field_validator('source_node_id', 'target_node_id', mode='before')
    @classmethod
    def coerce_id(cls, v: Any) -> str:
        if isinstance(v, dict):
            for key in ['id', 'name', 'value']:
                if key in v and v[key]:
                    return str(v[key])
            if v:
                return str(next(iter(v.values())))
        return str(v) if v is not None else ""

class KnowledgeGraphExtraction(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list, description="List of all extracted nodes/entities.")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="List of all extracted relationships.")


# --- Custom Graph Extractor Class ---

class GraphExtractor:
    """
    Custom extraction transformer to convert documents into graphs.
    Replaces deprecated langchain-experimental dependency and supports strict schema,
    date standardization, and metadata injection.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = None,
        allowed_relationships: List[str] = None,
        strict_mode: bool = True,
        context_hint: Optional[str] = None
    ) -> None:
        self.llm = llm
        self.allowed_nodes = allowed_nodes or []
        self.allowed_relationships = allowed_relationships or []
        self.strict_mode = strict_mode
        self.context_hint = context_hint

        # Define prompts dynamically based on constraints
        node_rules = (
            f"The standard categories should prioritize or match: {self.allowed_nodes}"
            if self.allowed_nodes else "Standard categories (use these if matching, or dynamically create descriptive labels based on content): PERSON, CORPORATION, BUSINESS_SEGMENT, FINANCIAL_METRIC, PRODUCT, LOCATION, DATE, EVENT, STRATEGIC_STATEMENT, CONTACT_INFO."
        )
        relationship_rules = (
            f"The output relation types should prioritize or match: {self.allowed_relationships}"
            if self.allowed_relationships else "Relations represent connections between concepts or entities (e.g. HAS_VISION, HAS_METRIC, WON_AWARD, PRODUCED, RELATED_TO)."
        )

        system_prompt = (
            "You are a hyperscale Information Extraction System. Your mission is to build a high-density, open-schema Knowledge Graph.\n\n"
            "RULES:\n"
            "1. OUTPUT ONLY VALID JSON MATCHING THE SCHEMA.\n"
            "2. ZERO SUMMARIZATION & QUOTA: Never summarize lists or skip sections. Extract all entities, properties, facts, and relationships from the text.\n"
            f"- **Allowed Labels**: {node_rules}\n"
            f"- **Allowed Relations**: {relationship_rules}\n"
        )

        if self.context_hint:
            system_prompt += (
                f"3. CONTEXTUAL ENRICHMENT: Every single entity extracted from the chunk MUST have a direct relationship to the context hint entity '{self.context_hint}'.\n"
                f"   - NO ORPHAN NODES: Every entity you extract MUST have a relationship to the primary context '{self.context_hint}'.\n"
            )
        else:
            system_prompt += (
                "3. CONTEXTUAL ENRICHMENT: Identify the primary topic/subject of the document chunk and ensure that other extracted entities are connected to it. Avoid orphan nodes.\n"
            )

        system_prompt += (
            "4. DUAL-LABELING (HIERARCHICAL): Every entity must have a LIST of at least 2 types in 'types':\n"
            "   - Label 1 (Standard Category): e.g. ORGANIZATION, PERSON, LOCATION, METRIC, STATEMENT.\n"
            "   - Label 2+ (Specific Domain Category): e.g. 'Software_Company', 'City', 'Revenue', 'Mission'. Dynamically derive these from the content.\n"
            "5. MANDATORY PROPERTY: Include a 'summary' property for EVERY node detailing its context in the text.\n"
            "6. OPEN METADATA & PROPERTY EXTRACTION: Extract all facts, strategic attributes (such as objectives, vision, statements, or values), and properties inside the 'properties' dictionary of the parent entity, in addition to creating specific nodes/relationships if they are complex.\n"
            "7. STRICT TEMPORAL ANCHOR & STANDARDIZATION:\n"
            "   - Date Standardization: Format all extracted dates or temporal attributes in properties to standard ISO format (YYYY-MM-DD) or normalized ranges (e.g., '2023-2024' or 'Q1_2024').\n"
            "8. ENTITY UNIQUENESS (PERIODIC VALUES): For entities that vary by time, phase, or period, always include the period/year in the entity name/ID (e.g. 'Sales Q1 2024' or 'Revenue 2022-2023') to prevent overwriting. You MUST NOT group multiple period values into properties of a single consolidated node; extract a separate, distinct node for each individual period to allow direct node-level queries.\n"
            "9. MULTI-COLUMN TABLE EXTRACTION (NO TRUNCATION): When extracting data from tables or grid structures, you MUST extract separate nodes and relationships for EVERY single data cell or column. You are strictly forbidden from truncating or skipping columns on the right edge of a table even if they represent older or less prominent data. Extract all columns systematically to ensure a lossless mapping of the table grid.\n"
            " - APPROXIMATE VALUES: You must extract and preserve numerical values even if they contain approximate indicators, prefixes, ranges, or suffixes (e.g., '~5,000', '10,000+', or '1,000-2,000'). Store these as string properties in their original format rather than dropping the row.\n"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Use the given format to extract information from the following input: {input}")
        ])

        # Bind structured output to LLM
        try:
            self.chain = self.prompt | llm.with_structured_output(KnowledgeGraphExtraction)
        except Exception:
            # Fallback if with_structured_output is not supported
            self.chain = self.prompt | llm

    def process_response(self, document: Document, raw_schema: KnowledgeGraphExtraction) -> GraphDocument:
        """Processes extracted raw schema, handles strict filters and maps to GraphDocument."""
        nodes = []
        relationships = []

        # Map nodes
        for en in raw_schema.nodes:
            props = dict(en.properties or {})
            # Standardize core properties
            props["description"] = en.description or ""
            props["summary"] = en.summary or ""
            props["tags"] = en.tags or []
            
            # Map hierarchical types list
            types = en.types or []
            if not types:
                types = ["Entity"]
            props["types"] = types

            # Inject document metadata onto nodes
            if document.metadata:
                props.update(document.metadata)
            
            # Map type (first type in types list) to safe string for langchain Node compatibility
            node_type = types[0]
            nodes.append(Node(id=en.id, type=node_type, properties=props))

        # Map relationships
        for er in raw_schema.relationships:
            source_types = er.source_node_types or ["Entity"]
            target_types = er.target_node_types or ["Entity"]
            rel_type = er.type or "RELATED_TO"

            source_node = Node(id=er.source_node_id, type=source_types[0], properties={"types": source_types})
            target_node = Node(id=er.target_node_id, type=target_types[0], properties={"types": target_types})

            props = dict(er.properties or {})
            props["description"] = er.description or ""
            # Inject document metadata onto relationships
            if document.metadata:
                props.update(document.metadata)

            relationships.append(Relationship(
                source=source_node,
                target=target_node,
                type=rel_type,
                properties=props
            ))

        # Recover missing nodes from relationships to prevent data loss
        existing_node_ids = {n.id for n in nodes}
        for r in relationships:
            if r.source.id not in existing_node_ids:
                nodes.append(r.source)
                existing_node_ids.add(r.source.id)
            if r.target.id not in existing_node_ids:
                nodes.append(r.target)
                existing_node_ids.add(r.target.id)

        # Filter strictly if strict_mode is True
        if self.strict_mode:
            if self.allowed_nodes:
                lower_allowed_nodes = {n.lower() for n in self.allowed_nodes}
                nodes = [n for n in nodes if any(t.lower() in lower_allowed_nodes for t in n.properties.get("types", []))]
                relationships = [
                    r for r in relationships
                    if any(t.lower() in lower_allowed_nodes for t in r.source.properties.get("types", []))
                    and any(t.lower() in lower_allowed_nodes for t in r.target.properties.get("types", []))
                ]
            if self.allowed_relationships:
                lower_allowed_rels = {r.lower() for r in self.allowed_relationships}
                relationships = [r for r in relationships if r.type.lower() in lower_allowed_rels]

        # Ensure no orphan relationships referencing non-existent nodes
        node_ids = {n.id for n in nodes}
        relationships = [
            r for r in relationships
            if r.source.id in node_ids and r.target.id in node_ids
        ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(self, documents: Sequence[Document]) -> List[GraphDocument]:
        """Convert a sequence of documents into GraphDocuments."""
        results = []
        for doc in documents:
            raw_schema = None
            try:
                # Try structured extraction
                raw_schema = self.chain.invoke({"input": doc.page_content})
                if not isinstance(raw_schema, KnowledgeGraphExtraction):
                    raise ValueError("Chain did not return KnowledgeGraphExtraction object.")
            except Exception as e:
                logger.warning(f"Structured extraction failed or returned invalid format ({e}). Attempting fallback with JSON repair...")
                try:
                    # Fallback raw call to get JSON text
                    raw_llm = self.llm
                    formatted_prompt = self.prompt.format(input=doc.page_content)
                    raw_output = raw_llm.invoke(formatted_prompt).content.strip()
                    
                    # Strip markdown fences if present
                    if raw_output.startswith("```"):
                        lines = raw_output.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        raw_output = "\n".join(lines).strip()

                    try:
                        data = json.loads(raw_output)
                    except json.JSONDecodeError:
                        repaired = repair_json(raw_output)
                        data = json.loads(repaired)
                        logger.info("Successfully repaired truncated LLM JSON response.")

                    raw_schema = KnowledgeGraphExtraction(**data)
                except Exception as fallback_err:
                    logger.error(f"Fallback extraction and repair also failed for this chunk (skipping chunk): {fallback_err}")
                    raw_schema = KnowledgeGraphExtraction(nodes=[], relationships=[])

            if raw_schema:
                results.append(self.process_response(doc, raw_schema))

        return results
