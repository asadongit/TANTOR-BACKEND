import logging
from typing import Dict, Any, Callable, List

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Core.Compiler import nodes as node_builders
from Apps.AgentMesh.Core.Compiler.nodes.utils import wrap_with_error_handling

logger = logging.getLogger(__name__)

_NODE_BUILDER_MAP: Dict[str, Callable] = {
    "llm_agent": node_builders.build_llm_agent,
    "deep_agent": node_builders.build_llm_deep_agent,
    "tool_node": node_builders.build_tool_node,
    "direct_tool": node_builders.build_direct_tool_node,
    "router": node_builders.build_router_node,
    "condition": node_builders.build_condition_node,
    "switch": node_builders.build_switch_node,
    "wait": node_builders.build_wait_node,
    "set_data": node_builders.build_set_data_node,
    "join": node_builders.build_join_node,
    "splitter": node_builders.build_splitter_node,
    "aggregator": node_builders.build_aggregator_node,
    "subgraph": node_builders.build_subgraph_node,
    "supervisor": node_builders.build_supervisor_node,
    "swarm": node_builders.build_swarm_node,
    "rag_node": node_builders.build_rag_node,
    "api_node": node_builders.build_api_node,
    "db_node": node_builders.build_db_node,
    "ingestion_node": node_builders.build_ingestion_node,
    "ingestion_adapter": node_builders.build_ingestion_adapter_node,
    "banking_enrichment": node_builders.build_banking_enrichment_node,
    "beneficiary_risk": node_builders.build_beneficiary_risk_node,
    "text_to_audio": node_builders.build_text_to_audio_node,
    "trigger": node_builders.build_trigger_node,
    "stt_trigger": node_builders.build_stt_trigger,
    "webhook_response": node_builders.build_webhook_response_node,
    "email_node": node_builders.build_email_node,
    "circuit_breaker": node_builders.build_circuit_breaker_node,
    "topic_publish": node_builders.build_topic_publish_node,
    "topic_consume": node_builders.build_topic_consume_node,
    "file_store": node_builders.build_file_store_node,
    "notification": node_builders.build_notification_node,
    "error_router": node_builders.build_error_router,
    "form_node": node_builders.build_form_node,
    "multi_file_ingestion": node_builders.build_multi_file_ingestion_node,
    "camera": node_builders.build_camera_node,
    "human_interaction": node_builders.build_human_interaction_node,
    "output_parser": node_builders.build_output_parser,
    "pii_node": node_builders.build_pii_node,
    "ocr_node": node_builders.build_ocr_node,
    "debug_node": node_builders.build_debug_node,
    "classification": node_builders.build_classification_node,
    "sensitivity_classifier": node_builders.build_sensitivity_classifier_node,
    "template_node": node_builders.build_template_node,
    "report_generation": node_builders.build_report_generation_node,
    "file_extractor": node_builders.build_file_extractor_node,
    "json_transform": node_builders.build_json_transform_node,
    "ingestion_router": node_builders.build_ingestion_router_node,
    "event_emitter": node_builders.build_event_emitter_node,
    "payload_builder": node_builders.build_payload_builder_node,
    "data_cleaning": node_builders.build_data_cleaning_node,
    "network_graph": node_builders.build_network_graph_node,
    "weightage": node_builders.build_weightage_node,
    "signature_stamp_extractor": node_builders.build_signature_stamp_extractor_node,
    "code_execution": node_builders.build_code_execution_node,
    "derived_fields": node_builders.build_derived_fields_node,
    "vector_generator": node_builders.build_vector_generator_node,
    "image_preprocessor": node_builders.build_image_preprocessor_node,
    "signature_matcher": node_builders.build_signature_matcher_node,
    "stamp_matcher": node_builders.build_stamp_matcher_node,
    "image_validator": node_builders.build_image_validator_node,
    "vision_intelligence": node_builders.build_vision_intelligence_node,
    "excel_to_json": node_builders.build_excel_to_json_node,
    "excel_to_json_node": node_builders.build_excel_to_json_node,
    "jev": node_builders.build_jev_node,
    "jev_node": node_builders.build_jev_node,
    "typesafe_ai": node_builders.build_jev_node,
    "system_one": node_builders.build_jev_node,
    "laya": node_builders.build_laya_node,
    "laya_node": node_builders.build_laya_node,
    "laya_decision": node_builders.build_laya_node,
    "local_decision": node_builders.build_laya_node,
}


class NodeFactory:

    # Only true functional synonyms where multiple visual triggers map to 1 backend builder
    _NODE_TYPE_ALIASES = {
        "button_trigger": "trigger",
        "auto_trigger": "trigger",
        # Entry-point triggers. The frontend exports these both into
        # `workflow.triggers` AND as canvas nodes (store/index.js), and
        # GraphCompiler routes graph entry from them by type, so they must be
        # buildable. build_trigger_node already handles all three explicitly —
        # they were simply missing from this map, which made any workflow
        # containing a webhook/cron/polling node fail to compile.
        "webhook": "trigger",
        "cron": "trigger",
        "polling": "trigger",
        "stt_node": "stt_trigger",
        "sttTrigger": "stt_trigger",
        "agentNode": "llm_agent",
        "textToAudio": "text_to_audio",
        # The palette creates nodes of type 'report_generator'; the builder is
        # registered as 'report_generation'. Neither name appeared in this map,
        # so ANY workflow containing a Report Generator node failed to compile
        # with "Unknown node type 'report_generator'".
        #
        # Aliasing rather than renaming either side: saved workflows already
        # carry 'report_generator' in their stored config, and renaming would
        # break every one of them.
        "report_generator": "report_generation",
        "json_transform_va": "json_transform",
        "pattern_extract": "json_transform",
        "rule_engine": "json_transform",
        "jevNode": "jev_node",
        "systemOne": "jev_node",
        "systemone": "jev_node",
        "typesafe_system_one": "jev_node",
        "layaNode": "laya_node",
        "layaDecision": "laya_node",
        "localDecision": "laya_node",
    }

    @classmethod
    def build(cls, node_config: NodeConfig, all_nodes: List[NodeConfig] = None, workflow_id: str = None) -> Callable:
        node_type = node_config.type

        # 1. Check explicit alias map
        canonical_type = cls._NODE_TYPE_ALIASES.get(node_type)

        # 2. Auto-normalize camelCase to snake_case if not explicitly aliased
        # e.g. textToAudio -> text_to_audio, visionIntelligence -> vision_intelligence
        if not canonical_type:
            import re
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', node_type).lower()
            canonical_type = cls._NODE_TYPE_ALIASES.get(snake, snake)

        builder = _NODE_BUILDER_MAP.get(canonical_type)
        if builder is None:
            supported = sorted(_NODE_BUILDER_MAP.keys())
            raise ValueError(
                f"Unknown node type '{node_type}'. "
                f"Supported types: {', '.join(supported)}"
            )

        # Some builders require additional arguments
        extra_kwargs = {}
        if canonical_type in ("deep_agent", "supervisor", "swarm"):
            extra_kwargs["all_nodes"] = all_nodes or []
        if canonical_type == "deep_agent":
            extra_kwargs["resolver"] = None
        if canonical_type == "rule_engine" and workflow_id:
            extra_kwargs["workflow_id"] = workflow_id

        logger.debug(
            f"NodeFactory building node '{node_config.id}' (type={node_type}) "
            f"with builder={builder.__name__}"
        )

        return wrap_with_error_handling(
            node_config.id,
            builder(node_config, **extra_kwargs),
            node_config.label or node_config.name,
            node_config.type,
        )
