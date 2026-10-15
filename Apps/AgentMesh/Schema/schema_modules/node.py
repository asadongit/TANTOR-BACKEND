from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

from .base import LLMConfig
from .form import FormField, FormDecision
from .weightage import WeightageCategory, RiskBand, CrossCategoryRule
from .api import SubApiConfig


class IterationPolicy(BaseModel):
    """Configuration for automated iterative loops (Self-Correction)"""

    enabled: bool = Field(False, description="Enable automated iterative refinement")
    max_iterations: int = Field(3, description="Maximum number of refinement cycles")
    evaluator_task: Optional[str] = Field(
        None, description="System prompt for the evaluator LLM"
    )
    evaluator_llm: Optional[LLMConfig] = None
    target_score: float = Field(
        0.8, description="Target quality score (0.0 to 1.0) to stop iterating"
    )
    inject_feedback_as: str = Field(
        "human", description="How to inject feedback: 'human' message or 'system' nudge"
    )


class NodeConfig(BaseModel):
    """Defines a step/node in the workflow"""
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique ID for the node within the workflow")
    type: str = Field(
        ..., description="Node type: llm_agent, tool_node, router, subgraph, etc."
    )

    # Frontend display metadata
    label: Optional[str] = Field(
        None, description="Human-readable display label for the node in the canvas"
    )
    position: Optional[Dict[str, float]] = Field(
        None, description="Canvas position {x, y} for visual layout"
    )
    name: Optional[str] = Field(
        None, description="Editable name for group or other nodes"
    )
    description: Optional[str] = Field(
        None, description="Editable description for group or other nodes"
    )
    internal_nodes: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="For group nodes: list of internal nodes"
    )
    internal_edges: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="For group nodes: list of internal edges"
    )
    original_edges: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="For group nodes: list of original edges before grouping",
    )

    # Agent Specific Configs
    task: Optional[str] = Field(
        None, description="System prompt or task description for agents"
    )
    llm: Optional[LLMConfig] = None
    memory: Optional[Dict[str, Any]] = Field(
        None, description="Memory configuration for the agent"
    )
    context_mode: str = Field(
        "smart",
        description="Context truncation strategy: smart (auto-budget), strict (aggressive truncation), passthrough (send full context)",
    )
    max_context_chars: Optional[int] = Field(
        None,
        description="Optional override for max context characters; if unset, backend auto-calculates based on model",
    )
    agent_output_key: Optional[str] = Field(
        None,
        description="Where to store the agent's answer when it replies with structured data. Set this whenever a later step needs that answer; without it the answer is only stored if it happens to contain certain field names.",
    )
    agent_context_exclude: Optional[List[str]] = Field(
        None,
        description="Context keys to keep out of this agent's prompt, in addition to the built-in list of bulky keys. Use for large values a later step needs but this agent does not.",
    )
    agent_context_include: Optional[List[str]] = Field(
        None,
        description="Context keys to send to this agent even if the built-in list would normally exclude them.",
    )

    # Answering again when the answer is obviously short
    # A model occasionally returns a fraction of what it should - the same
    # document that yields 244 lines returns 8, with no error and valid JSON.
    # Silence is the danger: a short answer looks like a successful one and
    # flows downstream. This states what a plausible answer looks like and asks
    # again when it is not met. No second model judges the first; the test is
    # arithmetic on the answer itself, so it costs nothing and cannot disagree
    # with itself.
    agent_retry_when: Optional[str] = Field(
        None,
        description="An expression describing an answer worth rejecting, e.g. len(answer.lines) < 20. When it holds, the agent is asked again. Leave empty for no retrying.",
    )
    agent_retry_max: int = Field(
        2,
        description="How many extra attempts to allow when the answer is rejected. Only used when a rejection test is set.",
    )

    # Batched extraction
    # Asking for a few hundred fields in one answer costs accuracy: the model
    # drops entries, or abandons the shape part way through. Splitting the list
    # and asking repeatedly keeps each answer small. Off unless switched on, and
    # when off the agent behaves exactly as it did before.
    agent_batch_enabled: bool = Field(
        False,
        description="Ask for a long list of fields in several smaller calls instead of one. Leave off unless the agent is being asked for more fields than it answers reliably.",
    )
    agent_batch_items: Optional[str] = Field(
        None,
        description="Where the list to split lives, e.g. {{ balance_sheet_fields }} or a plain context key. Must resolve to a list. Keep the list in workflow variables rather than in the prompt.",
    )
    agent_batch_size: int = Field(
        50,
        description="How many items to ask for per call. Smaller is usually more accurate and costs more calls.",
    )
    agent_batch_var: str = Field(
        "batch_items",
        description="The name this batch's items take in the prompt. A prompt saying {{ batch_items }} receives one batch at a time.",
    )
    agent_batch_merge: str = Field(
        "deep_merge",
        description="How the answers are combined: deep_merge (nested objects joined), shallow_merge (top level only), list_concat (answers kept as a list).",
    )
    agent_batch_merge_key: Optional[str] = Field(
        None,
        description="When answers contain lists of objects, the field that says which entries are the same one, e.g. 'slot'. Without it such lists are joined end to end instead of matched up.",
    )
    agent_batch_max_parallel: int = Field(
        1,
        description="How many calls may run at once. One at a time by default.",
    )
    agent_batch_on_error: str = Field(
        "continue",
        description="What a failed batch does: continue (its fields are left empty and the failure is reported) or fail (the node raises).",
    )

    # Iteration Policy
    iteration_policy: Optional[IterationPolicy] = Field(
        None, description="Policy for automated self-correction loops"
    )

    # Tool/Agent Options
    tools: Optional[List[str]] = Field(
        default_factory=list,
        description="List of tool names for the agent or tool_node",
    )
    tool_configs: Optional[Dict[str, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Tool-specific configurations like return_direct",
        repr=False,
    )
    return_direct: bool = Field(
        False,
        description="If true, the output of this node is returned immediately as the final result.",
    )

    # Payload Builder Specific Configs (Post-Processor)
    payload_template: Optional[Dict[str, Any]] = Field(
        None, description="A JSON skeleton with {{context.var}} placeholders"
    )
    payload_destination_key: str = Field(
        "webhook_payload", description="Context key to store the finalized JSON"
    )

    # Logic & Mapping Specific Configs
    branches: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of branches for switch node: [{name, condition}]",
    )
    condition: Optional[str] = Field(None, description="Expression for condition node")
    message: Optional[str] = Field(None, description="Static text for set_data node")
    wait_seconds: float = Field(0.0, description="Duration in seconds for wait node")

    # Output Parser Specific Configs (Auto-fixing JSON/data parser)
    # Auto-reads from latest message (no input_field config needed)
    # Follows AgentMesh pattern like rag_node, api_node, tool_node
    parse_mode: str = Field(
        "auto",
        description="Parse mode: 'auto' (standard then repair), 'json' (force JSON), 'strict' (fail on error)",
    )
    fallback_to_raw: bool = Field(
        True, description="Return original string if all parsing fails"
    )
    output_context_key: Optional[str] = Field(
        "parsed_output",
        description="Context key to store parsed result (default: parsed_output)",
    )

    # Audio Specific Configs
    tts_library: str = Field(
        "piper", description="TTS Library to use: piper or kittentts"
    )
    audio_voice: Optional[str] = Field(
        "en-US-GuyNeural", description="Voice ID for textToAudio node"
    )
    kittentts_model: str = Field(
        "KittenML/kitten-tts-micro-0.8", description="KittenTTS Model ID"
    )
    audio_speed: float = Field(
        1.0, description="Speech speed multiplier for KittenTTS synthesis"
    )
    mute_audio: bool = Field(
        False, description="Whether to skip audio generation for this node"
    )
    audio_input_text: Optional[str] = Field(
        None,
        description="Optional custom text or Jinja2 expression to convert to audio instead of last message",
    )

    # Error Handling Configs
    on_error: Optional[str] = Field(
        None, description="Optional ID of a node to jump to if this node fails."
    )
    continue_on_fail: bool = Field(
        False, description="Whether to continue execution if the node/API call fails"
    )

    # Direct Tool Specific Configs
    direct_tool_name: Optional[str] = Field(
        None, description="The name of the tool to execute directly"
    )
    tool_arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed directly to the tool"
    )
    destination_key: Optional[str] = Field(
        None, description="Context key where the tool execution result is saved"
    )

    # Router Specific Configs
    routing_key: Optional[str] = Field(
        None, description="State key to determine target node in conditional edges"
    )
    custom_python_rules: Optional[str] = Field(
        None, description="Custom python script for rule engine dynamic execution"
    )
    python_rules: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Frontend rule editor: array of {id, name, code, enabled} rule objects",
    )

    # Subgraph/Supervisor/Swarm Specific Configs
    workflow_id: Optional[str] = Field(
        None, description="Target Workflow ID to invoke as a subgraph"
    )
    member_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="List of node IDs that are members of this supervisor or swarm",
    )
    allowed_transitions: Optional[List[str]] = Field(
        None,
        description="Restrict which agents this member can hand off to. If None, can hand off to all members.",
    )

    # Human-in-the-Loop (HITL) Configs
    interrupt_before: bool = Field(
        False, description="Pause execution before this node starts."
    )
    interrupt_after: bool = Field(
        False, description="Pause execution after this node completes."
    )
    hitl_actions: Optional[List[str]] = Field(
        default_factory=lambda: ["approve", "rework", "reject"],
        description="Which HITL actions are available: approve, rework, reject",
    )
    mute_chat_bubble: bool = Field(
        False,
        description="Mute / hide execution bubble for this node in the chat interface.",
    )

    # Button Trigger Configs
    button_label: Optional[str] = Field(
        "Start Workflow", description="Display label for the button trigger"
    )
    button_input: Optional[str] = Field(
        "Start", description="Initial input sent by the button trigger"
    )
    button_category: Optional[str] = Field(
        "action", description="Visual theme category for the button trigger"
    )

    # STT (Speech-to-Text) Trigger Configs
    auto_stop: bool = Field(
        True,
        description="If true, recording stops automatically on silence. If false, user must click the button to stop.",
    )
    auto_start: bool = Field(
        False,
        description="If true, recording starts automatically when voice activity is detected (VAD). If false, user must click the mic button.",
    )
    silence_delay: int = Field(
        1500,
        description="Milliseconds to wait after silence before auto-submitting the transcript. Increase if you pause between words.",
    )
    stt_lang: str = Field(
        "en-US",
        description="BCP-47 language code for browser speech recognition (e.g. 'hi-IN' for Hindi, 'en-US' for English).",
    )

    # DeepAgent Specific Configs
    subagents: Optional[List[str]] = Field(
        default_factory=list,
        description="List of specialized subagent names to make available to the DeepAgent.",
    )
    deep_agent_backend: str = Field(
        "sqlite",
        description="Persistence backend for DeepAgent: local, sqlite, sandbox, ephemeral, contexthub",
    )
    deep_agent_root_dir: str = Field(
        "./mnt/deep_agents", description="Root directory for local/sqlite storage."
    )

    # Vision Intelligence Node Configs
    vision_mode: str = Field(
        "face_verification",
        description="Mode for Vision Intelligence node: face_verification, signature_stamp_detection, document_quality_gate, object_detection",
    )
    vision_face_match_threshold: float = Field(
        0.65,
        description="Cosine similarity threshold for face verification (0.0 to 1.0)",
    )
    vision_quality_min_score: float = Field(
        0.70, description="Minimum acceptable image quality score (0.0 to 1.0)"
    )
    vision_quality_blur_threshold: float = Field(
        100.0,
        description="Laplacian variance threshold — lower values = more sensitive to blur",
    )
    vision_quality_min_contrast: float = Field(
        0.10, description="Minimum normalised contrast std-dev (below = washed out)"
    )
    vision_quality_max_contrast: float = Field(
        0.45, description="Maximum normalised contrast std-dev (above = too harsh)"
    )
    vision_id_image_key: str = Field(
        "latest_file_path", description="Context key for primary ID document image path"
    )
    vision_selfie_image_key: str = Field(
        "vcip_capture_path", description="Context key for live selfie image path"
    )
    vision_output_key: str = Field(
        "vision_result", description="Context key to store vision engine result"
    )
    harness_profile: Optional[str] = Field(
        None,
        description="Named harness profile to tune agent behavior (e.g. 'advanced-reasoning', 'low-cost').",
    )
    code_interpreter: bool = Field(
        False,
        description="Enable the native lightweight code interpreter tool for data transformation and math.",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="List of paths to directories containing DeepAgent SKILL.md folders.",
    )
    max_iterations: int = Field(10, description="Max tool calls before agent stops")
    timeout_seconds: int = Field(300, description="Max execution time in seconds")
    context_schema: Optional[Dict[str, Any]] = Field(
        None, description="Schema for context data validation"
    )

    # RAG & Ingestion Specific Configs
    vector_store: Optional[str] = Field(
        None, description="Name of vector store type (e.g., chroma, qdrant)"
    )
    vector_collection: Optional[str] = Field(
        "default", description="Collection name within the vector store"
    )
    embedding_model: Optional[str] = Field(
        None, description="Embedding model for RAG node"
    )
    chunk_size: int = Field(1000, description="Size of each text chunk (characters)")
    chunk_overlap: int = Field(
        100, description="Overlap between chunks to maintain context"
    )
    chunking_strategy: Optional[str] = Field(
        "recursive", description="Type of chunking: recursive, token"
    )
    k_results: int = Field(5, description="Number of results to retrieve")
    threshold: float = Field(0.5, description="Similarity threshold for RAG retrieval")
    show_citations: bool = Field(
        False, description="Whether to show citations (filenames) in RAG results"
    )
    search_type: str = Field(
        "similarity", description="Search type: similarity, hybrid, keyword"
    )
    ingestion_source_key: Optional[str] = Field(
        None, description="Context key to ingest from (e.g., latest_ocr_output)"
    )
    data_type: str = Field(
        "text_rag", description="Data type to ingest: text_rag, precomputed_vector"
    )
    vector_context_key: str = Field(
        "signature_vector", description="Context key for precomputed vector data"
    )
    payload_context_key: str = Field(
        "reference_id", description="Context key for precomputed payload metadata"
    )

    # Preprocessor & Matcher Specific Configs
    sig_mode: str = Field(
        "auto", description="Signature ink color mode for Image Preprocessor"
    )
    stamp_color: str = Field(
        "auto", description="Stamp ink color mode for Image Preprocessor"
    )
    match_threshold: float = Field(
        0.75, description="Match threshold for Signature and Stamp matchers"
    )
    min_edge_score: float = Field(
        100.0, description="Minimum laplacian variance edge score for Image Validator"
    )
    min_std_dev: float = Field(
        8.0, description="Minimum standard deviation for Image Validator"
    )
    distance_metric: str = Field(
        "cosine", description="Distance metric for vector search"
    )
    lowe_ratio: float = Field(0.75, description="Lowe's ratio for SIFT matching")

    # Ingestion Adapter Specific Configs (Generic)
    source_format: Optional[str] = Field(
        "Auto-Detect", description="For ingestion_adapter: Expected source format"
    )
    validation_mode: Optional[str] = Field(
        "Strict", description="For ingestion_adapter: Strict or Lax validation"
    )
    target_serialization: Optional[str] = Field(
        "Canonical JSON", description="For ingestion_adapter: Output format"
    )
    flatten_depth: str = Field(
        "one_level",
        description="How deeply to flatten nested objects: none, one_level, recursive",
    )
    flatten_separator: str = Field(
        ".",
        description="Separator for flattened key paths (e.g., '.' gives 'address.city')",
    )
    field_aliases: Dict[str, str] = Field(
        default_factory=dict,
        description="User-configured field renaming: {'old_name': 'new_name', ...}",
    )
    default_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-configured default values for missing fields: {'field': 'default_val', ...}",
    )

    # Pre-Enrichment Configs
    external_integrations: Optional[List[str]] = Field(
        default_factory=list, description="External APIs to mock fetch"
    )
    internal_integrations: Optional[List[str]] = Field(
        default_factory=list, description="Internal APIs to mock fetch"
    )
    source_key: Optional[str] = Field(
        "canonical_transaction", description="Context key to read transaction from"
    )

    # Splitter & Aggregator Specific Configs
    split_field: Optional[str] = Field(
        None,
        description="Key in context/payload to split. Supports comma-separated fallback keys (e.g., 'latest_ocr_output, ocr_like_results') — tries each in order.",
    )
    batch_size: int = Field(
        1, description="Number of items to process in each iteration"
    )
    destination_field: Optional[str] = Field(
        None,
        description="Output key for the item (Splitter) or collection (Aggregator)",
    )
    discovery_keys: Optional[str] = Field(
        None, description="Comma-separated fallback keys for smart discovery"
    )

    # Aggregator Actions
    action: Optional[str] = Field(
        "append", description="Aggregation strategy: append, merge, concatenate, group_by"
    )
    field_to_aggregate: Optional[str] = Field(
        "item", description="Key in context to collect from"
    )
    separator: Optional[str] = Field(
        "\\n", description="Separator for concatenation strategy"
    )
    group_by: Optional[str] = Field(
        None, description="Key field to group by (e.g. 'class', 'document_type'). Used when action='group_by'."
    )

    # Classification Specific Configs
    classification_classes: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of classes and descriptions for classification",
    )
    classification_summary: bool = Field(
        False,
        description="If true, summarizes the input data in addition to classifying it",
    )
    classification_extract_identifiers: bool = Field(
        False,
        description="If true, extracts unique identifiers (NER) like IDs, names, or reference numbers into an array.",
    )
    classification_per_page: bool = Field(
        False,
        description=(
            "If true, classifies each page of the input separately and reports "
            "which pages hold which class, instead of giving the whole file one "
            "class. Costs one model call per page."
        ),
    )
    classification_input_field: Optional[str] = Field(
        None,
        description="Optional path or variable name containing input data to classify",
    )
    classification_engine: str = Field(
        "llm", description="Classification engine: 'llm' or 'regex'"
    )
    classification_input_key: Optional[str] = Field(
        "latest_ocr_output", description="Context key to read text from"
    )
    classification_max_input_chars: Optional[int] = Field(
        None,
        description=(
            "Cap on how much text is sent to the LLM prompt. 0 or None means no "
            "limit. Deciding a document's class needs a sample, not the whole "
            "document, and reasoning models given very long input can spend "
            "their entire output budget thinking and never emit an answer. Only "
            "the prompt is capped - the full text is still stored on the result "
            "and passed downstream."
        ),
    )
    classification_output_key: Optional[str] = Field(
        None, description="Context key where the result dict is saved"
    )
    engine_mode: Optional[str] = Field(
        "both",
        description="Engine mode for sensitivity classifier: both, presidio, regex",
    )
    custom_entities: Optional[str] = Field(
        "", description="Custom entities for sensitivity classifier"
    )
    output_format: Optional[str] = Field(
        "markdown", description="Output format for sensitivity classifier"
    )

    # API Specific Configs
    api_description: Optional[str] = Field(
        None, description="Optional description of the primary API call"
    )
    api_enabled: bool = Field(
        True, description="Whether the primary API call is enabled"
    )
    api_execution_mode: str = Field(
        "sequential", description="Execution mode: sequential or parallel"
    )
    additional_apis: List[SubApiConfig] = Field(
        default_factory=list,
        description="Optional additional APIs for hybrid multi-API mode",
        repr=False,
    )
    connection_id: Optional[str] = Field(
        None, description="ID of the saved connection profile in the registry"
    )
    api_protocol: str = Field("REST", description="Protocol type: REST, SOAP, GraphQL")
    api_method: str = "GET"
    api_url: Optional[str] = None
    api_headers: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_params: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_body: Optional[str] = Field(None, repr=False)
    graphql_query: Optional[str] = Field(
        None, description="GraphQL specific query string"
    )
    api_auth_type: str = "None"
    api_auth_config: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_response_format: str = "json"
    use_curl: bool = Field(
        False,
        description="Whether to parse and execute a raw cURL command instead of the standard fields",
    )
    curl_command: Optional[str] = Field(
        None, description="The raw cURL command string (supports Jinja interpolation)", repr=False
    )
    api_max_retries: int = Field(
        1, description="Max retry attempts for the primary API"
    )
    api_retry_delay: float = Field(
        2.0, description="Delay between retries in seconds for the primary API"
    )

    # File Store Specific Configs
    file_format: str = Field(
        "csv",
        description=(
            "Format to save: csv, excel, txt, md. Markdown keeps the text "
            "exactly as it was produced, so a laid-out table stays laid out."
        ),
    )
    file_name: Optional[str] = Field(
        "output", description="Name of the file to save (supports interpolation)"
    )
    file_data_key: Optional[str] = Field(
        None, description="Key in context/state to save as file data"
    )
    file_overwrite: bool = Field(
        True, description="Whether to overwrite if file exists (CSV only for append)"
    )

    # Report Generation Specific Configs
    template_id: Optional[str] = Field(
        None,
        description="ID of the DocumentTemplateModel to use for generating the report",
    )
    input_data_key: Optional[str] = Field(
        "report_data",
        description="Context key containing the JSON/Dict data to map into the template",
    )
    output_file_name: str = Field(
        "generated_report",
        description="Base name for the output file (without extension)",
    )
    # ── Derived Fields ────────────────────────────────────────────────────────
    df_fields: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description=(
            "Values to calculate, one per row: "
            "{name, expression, group, description, enabled}. The expression is "
            "plain arithmetic over field names already in the data, for example "
            "'revenue - cost' or 'total / count * 100'. Rows are evaluated in "
            "order and each result is available to the rows below it, so one "
            "value can build on another."
        ),
    )
    df_input_key: Optional[str] = Field(
        None,
        description=(
            "Where to read the source values from. Leave blank to use everything "
            "available to the node."
        ),
    )
    df_output_key: Optional[str] = Field(
        "derived_fields",
        description="Where the calculated values are stored for later steps to use.",
    )
    df_on_error: Optional[str] = Field(
        "null",
        description=(
            "What to do when one row cannot be calculated. 'null' records the "
            "value as empty and carries on, 'skip' leaves it out entirely, "
            "'fail' stops the node. Rows are always independent, so one bad "
            "expression never silently affects another."
        ),
    )
    df_confidence_key: Optional[str] = Field(
        None,
        description=(
            "Optional. Where to read a map saying which incoming values are "
            "trustworthy — for example the '_confidence' value an OCR node "
            "writes beside its text. When set, every calculated row that reads "
            "a doubtful value is itself marked doubtful, and the verdicts are "
            "stored beside the results under the output key with '_confidence' "
            "appended. A value the map says nothing about counts as sound, so a "
            "partial map never floods the report with doubt. Leave blank and "
            "nothing is marked."
        ),
    )

    report_format: Optional[str] = Field(
        None,
        description="Output format for the report node: docx, pdf, xlsx, csv, json, md. Overrides the shared file_format field.",
    )
    report_mode: Optional[str] = Field(
        None,
        description=(
            "How a template is filled. 'template_map' scans every sheet for "
            "{{ path }} markers and replaces them deterministically, preserving "
            "merged cells, number formats and the template's own formulas - use "
            "this for exacting formats such as regulatory returns. 'ai_semantic' "
            "is the existing model-driven fill for narrative documents. Blank "
            "keeps the previous behaviour."
        ),
    )
    report_on_missing_target: Optional[str] = Field(
        None,
        description=(
            "template_map only: what to do when a {{ marker }} has no matching "
            "data. 'fail' (default) refuses to produce the file - a report that "
            "looks complete with a figure missing is more dangerous than none. "
            "'warn' leaves the marker visible in the cell. 'skip' blanks it."
        ),
    )

    # Debug Console Specific Configs
    debug_scope: str = Field(
        "context", description="What to print: context, messages, full_state, selective"
    )
    debug_key: Optional[str] = Field(
        None, description="If scope is selective, which key to print"
    )

    # Template Engine Specific Configs
    template_content: Optional[str] = Field(
        None, description="The raw Jinja2/Markdown template string."
    )
    template_file_path: Optional[str] = Field(
        None, description="Optional path to load a template from disk."
    )
    template_output_key: str = Field(
        "rendered_template", description="Context key to save the final output."
    )
    template_data_key: Optional[str] = Field(
        None,
        description="Specific context key to use as data. If None, uses full context.",
    )

    # Form Submission Specific Configs
    form_title: Optional[str] = Field(
        "Information Requested", description="Title shown at the top of the form"
    )
    form_description: Optional[str] = Field(
        None, description="Subtext or instructions for the form"
    )
    form_fields: List[FormField] = Field(
        default_factory=list, description="List of fields to render in the form"
    )
    form_decisions: List[FormDecision] = Field(
        default_factory=list,
        description="Optional list of custom decision buttons at the bottom of the form",
    )
    form_submit_label: str = Field(
        "Submit", description="Text for the submission button"
    )
    form_use_context_schema: bool = Field(
        False,
        description="If true, generates form fields dynamically from the context_schema",
    )

    # Notification Specific Configs
    notification_title: Optional[str] = Field(
        None, description="Title for the notification"
    )
    notification_message: Optional[str] = Field(
        None, description="Message body (supports template interpolation)"
    )
    notification_type: str = Field(
        "info", description="Type: info, success, warning, error"
    )
    mute_notification_message: bool = Field(
        True, description="Whether to skip returning a chat message for this node"
    )
    show_chat_button: bool = Field(
        False,
        description="Whether to show an 'Open Chat' button in the notification center",
    )

    # OCR Specific Configs
    ocr_auto_orient: bool = Field(
        False,
        description=(
            "Check which way up each page is before reading it, and turn it if "
            "needed. A scan fed in the wrong way round carries nothing to say "
            "so - the file's own rotation flag stays at zero - and a model will "
            "read such a page plausibly while losing much of it. The check runs "
            "on a thumbnail, so it costs a fraction of reading the page. Leave "
            "off for documents known to be upright."
        ),
    )
    ocr_passes: int = Field(
        1,
        description=(
            "How many times to read each page. One is the default and behaves "
            "exactly as before. Two or more reads every page again and reports "
            "which pages the readings disagreed about, in a companion value "
            "named after the destination key with '_confidence' appended. The "
            "first reading is always the one returned - a second reading is "
            "used to raise a hand, never to pick a different answer, because "
            "where a figure is hard to read the wrong readings tend to agree "
            "with each other. Costs one extra model call per page per pass."
        ),
    )
    ocr_stability_threshold: float = Field(
        1.0,
        description=(
            "Share of figures every reading must agree on before a page counts "
            "as stable, between 0 and 1. Only meaningful when ocr_passes is 2 "
            "or more. The default of 1.0 demands exact agreement, which is what "
            "pages that read well actually produce; lower it if clean pages are "
            "being reported unstable."
        ),
    )
    ocr_engine: str = Field(
        "vision",
        description="OCR Engine type: hybrid (text layer, then fallback), vision (LLM), standard (Local), or external (API Service)",
    )
    ocr_standard_model: str = Field(
        "tesseract", description="Local OCR model: tesseract, chandra, surya"
    )

    # Hybrid engine: read a PDF's embedded text layer, and only call an OCR
    # engine for pages that have none. Ignored unless ocr_engine == "hybrid",
    # so existing workflows are unaffected by these defaults.
    ocr_fallback_engine: str = Field(
        "vision",
        description="Hybrid only: engine used for pages with no readable text layer (vision | standard)",
    )
    ocr_text_layer_min_chars: int = Field(
        100,
        description="Hybrid only: characters of embedded text required to treat a PDF as digital. A scan yields almost none; a real page yields thousands.",
    )
    ocr_max_pages: int = Field(
        20,
        description="Maximum pages to read from one document. 0 = no limit. Applies to the hybrid AND vision engines. Raise it for long documents: a limit discards everything past it, so a 29-page pack whose accounts sit in the last pages yields nothing at all.",
    )
    ocr_on_max_pages: str = Field(
        "truncate",
        description="Behaviour when ocr_max_pages is exceeded — truncate (read the first N and warn) or fail (stop the run rather than return a partial document)",
    )
    ocr_input_key: str = Field(
        "latest_file_path",
        description="Context key containing the file path to OCR (default: latest_file_path)",
    )
    ocr_prompt: str = Field(
        "Please perform OCR on this document and return the full text accurately.",
        description="Instruction for the LLM to perform OCR.",
    )
    # External OCR Service Configs (used when ocr_engine == 'external')
    ocr_service_url: Optional[str] = Field(
        None,
        description="URL of the external OCR service (e.g., http://localhost:8003/api/ocr/upload)",
    )
    ocr_service_params: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra form fields to send with the file upload (e.g., provider, model, mask_pii)",
    )
    ocr_service_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers for the external service (e.g., Authorization)",
    )
    ocr_service_response_type: str = Field(
        "sse",
        description="Response type from external service: 'sse' (Server-Sent Events) or 'json' (standard JSON)",
    )
    ocr_service_file_field: str = Field(
        "files",
        description="Name of the multipart form field for the file upload (default: 'files')",
    )
    ocr_service_text_path: str = Field(
        "data.text",
        description="Dot-notation path to extract text from each SSE event or JSON response (default: 'data.text')",
    )
    ocr_multi_file_key: Optional[str] = Field(
        None,
        description="If set, reads a list of file info dicts from this context key (e.g., 'payload.uploaded_files') and processes all files",
    )

    # Multi-File Ingestion Configs
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["pdf", "jpg", "jpeg", "png", "webp"],
        description="List of allowed file extensions (e.g. ['pdf', 'jpg', 'png'])",
    )
    max_file_size_mb: float = Field(
        25.0, description="Maximum allowed file size per file in megabytes"
    )
    max_files_count: int = Field(
        10, description="Maximum number of files allowed in a single upload batch"
    )
    strategy_rules: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Dynamic list of strategy rules specifying parent file_type and child strategy (e.g. [{'id': 'rule_1', 'file_type': 'pdf', 'strategy': 'whole_file'}, {'id': 'rule_2', 'file_type': 'images', 'strategy': 'black_white'}])",
    )

    # PII Redaction Specific Configs
    pii_engine: str = Field(
        "local",
        description="PII masking engine: local (DataFog) or docbot (External API)",
    )
    datafog_engine: str = Field(
        "regex", description="Detection algorithm for local engine: regex or gliner"
    )
    pii_target: str = Field(
        "latest_message",
        description="What to redact: latest_message, context_variable, targeted_replacement",
    )
    context_variable_key: Optional[str] = Field(
        None, description="Specific context key to mask for context_variable target"
    )
    pii_entities: List[str] = Field(
        default_factory=lambda: ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"],
        description="Specific entities to look for (if supported by engine)",
    )
    pii_custom_regex: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Custom regex patterns: [{'label': 'Phone', 'pattern': '...'}]",
    )
    docbot_url: Optional[str] = Field(
        "http://localhost:8000", description="DocBot API base URL"
    )
    docbot_method: str = Field(
        "local", description="DocBot detection method: local or hybrid"
    )

    # Data Cleaning Specific Configs
    data_cleaning_rules: List[str] = Field(
        default_factory=lambda: ["GEN_TRIM_WHITESPACE"],
        description="Rules to run for cleaning: GEN_TRIM_WHITESPACE, GEN_PROPER_CASE, etc.",
    )
    data_cleaning_target: str = Field(
        "latest_message", description="Target source: latest_message, context_data"
    )
    data_cleaning_input_key: Optional[str] = Field(
        "",
        description="Context key containing the JSON input (only if target is context_data)",
    )
    data_cleaning_output_key: str = Field(
        "cleaned_data",
        description="Context key where cleaned data and metrics are saved",
    )

    # Webhook Response Specific Configs
    response_status: int = Field(
        200, description="HTTP status code for the webhook response"
    )
    response_headers: Dict[str, str] = Field(
        default_factory=dict, description="Custom HTTP headers for the response"
    )
    response_body_key: Optional[str] = Field(
        None,
        description="Context key to use as response body. If None, uses full context.",
    )

    # DB Node Shared Configs
    db_engine: Optional[str] = Field(
        "mongodb",
        description="The engine to connect to: mongodb, sql, postgresql, mysql, etc.",
    )
    db_connection_string: Optional[str] = Field(
        None, description="MongoDB connection URI. Falls back to MONGODB_URI env var."
    )
    db_database: Optional[str] = Field(None, description="MongoDB database name")
    db_collection: Optional[str] = Field(None, description="MongoDB collection name")
    db_operation: str = Field(
        "insert_one",
        description="Operation: insert_one, insert_many, find_one, find_many, update_one, delete_one",
    )
    db_document: Optional[Any] = Field(
        None,
        description="Document template for insert/update (supports {{context.xxx}} interpolation)",
    )
    db_query: Optional[Any] = Field(
        None, description="Query filter for find/update/delete (supports interpolation)"
    )
    db_output_key: str = Field(
        "db_result", description="Context key to store query results"
    )
    db_limit: int = Field(100, description="Max documents to return for find_many")

    # Ingestion Router & Event Emitter Specific Configs
    scan_mode: str = Field(
        "auto", description="Processing mode: auto, files, directory"
    )
    file_filters: List[str] = Field(
        default_factory=lambda: [".pdf", ".jpg", ".png", ".jpeg"],
        description="List of allowed file extensions for directory scanning",
    )
    concurrency_limit: int = Field(
        5,
        description="Maximum number of parallel child workflows to spawn (Semaphore limit)",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="URL to ping when emitting events (e.g. for SSE progress sync)",
    )
    emit_status: Optional[str] = Field(
        None, description="Status to emit (e.g., awaiting_classification_review)"
    )

    @field_validator("payload_template", mode="before")
    @classmethod
    def parse_json_strings(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Must be a valid JSON dictionary")
        return v

    # Network Graph Specific Configs
    graph_target_key: str = Field(
        "latest_message",
        description="Context key containing transaction details to ingest",
    )
    graph_neo4j_uri: Optional[str] = Field(
        None,
        description="Neo4j URI, defaults to NEO4J_URI environment variable or bolt://localhost:7687",
    )
    graph_neo4j_user: Optional[str] = Field(
        None,
        description="Neo4j User, defaults to NEO4J_USER environment variable or neo4j",
    )
    graph_neo4j_password: Optional[str] = Field(
        None,
        description="Neo4j password; defaults only to the securely injected NEO4J_PASSWORD environment variable",
        repr=False,
    )
    graph_collection: Optional[str] = Field(
        "FraudFlow",
        description="Dynamic label/collection name in Neo4j (defaults to FraudFlow)",
    )
    graph_ingestion_mode: str = Field(
        "visual", description="Ingestion mode: visual, or cypher"
    )
    graph_cypher_query: Optional[str] = Field(
        None, description="Raw Cypher query string for custom Neo4j execution"
    )
    graph_nodes_schema: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual dynamic nodes mapping"
    )
    graph_relations_schema: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual dynamic relationships mapping"
    )

    # Advanced JSON Transform (jt_*) Specific Configs
    jt_input_key: Optional[str] = Field(
        None,
        description="Context key to read input data from. If left empty, defaults to full root context.",
    )
    jt_output_key: Optional[str] = Field(
        "transformed_data", description="Context key to write the transform result to"
    )
    jt_mode: str = Field(
        "jmespath",
        description="Transform mode: 'jmespath' (query language), 'operations' (visual no-code), 'python_expr' (sandboxed Python)",
    )
    jt_query: Optional[str] = Field(
        None,
        description="JMESPath query string. e.g. 'job_status.tasks[*].task_id' or 'tasks[?status==`SUCCESS`].algorithm'",
    )
    jt_operations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of visual transform operations [{type, source_path, field, destination, ...}]",
    )
    jt_expression: Optional[str] = Field(
        None,
        description="Sandboxed Python expression. 'data' is the input variable. e.g. '[t[\"task_id\"] for t in data[\"tasks\"]]'",
    )
    jt_pass_through: bool = Field(
        False,
        description="If true and both input/output are dicts, merge result into original data",
    )
    extract_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of keyword & regex extraction patterns for pattern extract operations",
    )

    # Weightage Specific Configs
    weightage_categories: Optional[List[WeightageCategory]] = Field(
        default_factory=list, description="List of categories for weightage calculation"
    )
    weightage_output_key: Optional[str] = Field(
        "weightage_result", description="Context key to store the weightage result"
    )
    weightage_approve_threshold: float = Field(
        80.0, description="Minimum normalized score required to automatically approve"
    )
    weightage_review_threshold: float = Field(
        50.0, description="Minimum normalized score required to route for manual review"
    )
    weightage_risk_bands: Optional[List[RiskBand]] = Field(
        default_factory=list, description="Optional custom risk bands and grades"
    )
    weightage_output_flags: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of flag names to auto-resolve in context",
    )
    weightage_cross_rules: Optional[List[CrossCategoryRule]] = Field(
        default_factory=list,
        description="Optional list of rules evaluated across category scores",
    )

    # Signature & Stamp Extractor Specific Configs
    extractor_input_key: str = Field(
        "latest_file_path",
        description="Context key containing the file path(s) to extract signatures/stamps from (default: latest_file_path)",
    )
    extractor_output_key: str = Field(
        "extracted_elements",
        description="Context key to store the list of extracted signatures and stamps (default: extracted_elements)",
    )

    # Code Execution Node (ce_*) Specific Configs
    ce_language: str = Field(
        "python",
        description="Programming language: 'python' (default). 'javascript' reserved for future use.",
    )
    ce_code: Optional[str] = Field(
        None,
        description="User-defined Python script. Available variables: input_data, input_items, context. Set 'result' variable to produce output.",
    )
    ce_input_key: Optional[str] = Field(
        "latest_api_result",
        description="Context key to read input data from. Available in code as 'input_data'.",
    )
    ce_output_key: Optional[str] = Field(
        "code_result",
        description="Context key to write the execution result to.",
    )
    ce_mode: str = Field(
        "run_once",
        description="Execution mode: 'run_once' (script runs once with all input), 'run_for_each' (script runs per item in input list). '_index' variable available in per-item mode.",
    )
    ce_block_mode: str = Field(
        "single",
        description="Code block mode: 'single' (one script in ce_code) or 'multi' (sequential blocks in ce_code_blocks)",
    )
    ce_code_blocks: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Multi-block mode: array of {id, name, code, enabled} code blocks executed sequentially. Enabled blocks only.",
    )

    # File Extractor Specific Configs
    extract_file_path: Optional[str] = Field(
        None, description="Path to the file to extract (or jinja template)"
    )
    extract_format: str = Field(
        "text", description="Format to extract (text, json, base64)"
    )
    extract_output_key: Optional[str] = Field(
        "extracted_file_content", description="Context key to store the extracted content"
    )
    extract_skip_errors: bool = Field(
        False, description="If true, bypass errors and continue workflow"
    )

    # SMTP Email Node Specific Configs
    email_provider: str = Field(
        "Custom",
        description="SMTP email provider preset (Gmail, Outlook, SendGrid, Mailgun, Custom)",
    )
    email_to: Optional[str] = Field(None, description="Recipient email address")
    email_cc: Optional[str] = Field(
        None, description="CC email addresses (comma-separated)"
    )
    email_bcc: Optional[str] = Field(
        None, description="BCC email addresses (comma-separated)"
    )
    email_subject: Optional[str] = Field(None, description="Email subject")
    email_body: Optional[str] = Field(
        None, description="Email body (supports HTML and Jinja interpolation)"
    )
    email_html: bool = Field(False, description="Whether to send the email as HTML")
    email_smtp_server: str = Field("smtp.gmail.com", description="SMTP server hostname")
    email_smtp_port: int = Field(587, description="SMTP server port")
    email_user: Optional[str] = Field(
        None, description="SMTP authenticating username/email"
    )
    email_password: Optional[str] = Field(
        None, description="SMTP password or App password", repr=False
    )
    email_attachments: Optional[str] = Field(
        None, description="Attachments file paths, comma-separated"
    )
    email_headers: Optional[Dict[str, str]] = Field(
        None, description="Custom headers for the email"
    )
    email_smtp_timeout: int = Field(
        15, description="SMTP connection timeout in seconds"
    )
    email_auth_method: str = Field(
        "smtp", description="Authentication method for Email Node: smtp or oauth"
    )
    email_tenant_id: Optional[str] = Field(None, description="OAuth/Azure Tenant ID")
    email_client_id: Optional[str] = Field(None, description="OAuth/Azure Client ID")
    email_client_secret: Optional[str] = Field(
        None, description="OAuth/Azure Client Secret", repr=False
    )
    email_google_service_account_json: Optional[str] = Field(
        None, description="Google Service Account JSON key for Gmail OAuth", repr=False
    )
    email_google_delegated_user: Optional[str] = Field(
        None, description="Workspace email to impersonate for Gmail OAuth"
    )
