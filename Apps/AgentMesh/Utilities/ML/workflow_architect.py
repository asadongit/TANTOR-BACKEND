import logging
import os
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

logger = logging.getLogger(__name__)

class WorkflowArchitect:
    def _format_node(self, node: Dict[str, Any]) -> str:
        nid = node.get("id")
        ndata = node.get("data", {})
        ntype = ndata.get("type", node.get("type", "unknown"))
        nlabel = ndata.get("label", nid)
        
        node_desc = f"- **Node: {nlabel}** (ID: `{nid}`, Type: `{ntype}`)"
        
        details = []
        if ndata.get("task"):
            details.append(f"Instructions/Task: {ndata.get('task')}")
        if ndata.get("description"):
            details.append(f"Description: {ndata.get('description')}")
        if ndata.get("tools"):
            details.append(f"Associated Tools: {', '.join(ndata.get('tools'))}")
        if ndata.get("poller_tool"):
            details.append(f"Poller Tool: {ndata.get('poller_tool')}")
        if ndata.get("cron_expression"):
            details.append(f"Cron Expression: {ndata.get('cron_expression')}")
        if ndata.get("webhook_path"):
            details.append(f"Webhook Path: {ndata.get('webhook_path')}")
            
        if details:
            node_desc += "\n  " + "\n  ".join(details)
        return node_desc

    def _format_edge(self, edge: Dict[str, Any]) -> str:
        source = edge.get("source")
        target = edge.get("target")
        condition = edge.get("data", {}).get("condition")
        cond_str = ""
        if condition and condition.get("key"):
            cond_str = f" [Condition: {condition.get('key')} {condition.get('operator', '==')} {condition.get('value')}]"
        return f"- `{source}` &rarr; `{target}`{cond_str}"

    async def explain(self, workflow_data: Dict[str, Any], llm_config: Dict[str, Any]) -> str:
        """
        Generates an architectural summary of a workflow using the configured LLM.
        """
        nodes: List[Dict[str, Any]] = workflow_data.get("nodes", [])
        edges: List[Dict[str, Any]] = workflow_data.get("edges", [])
        name: str = workflow_data.get("name", "Unnamed Workflow")
        description: str = workflow_data.get("description", "")

        # Format nodes and connections for the LLM
        nodes_summary = [self._format_node(n) for n in nodes]
        edges_summary = [self._format_edge(e) for e in edges]

        nodes_text = "\n".join(nodes_summary) if nodes_summary else "No nodes defined."
        edges_text = "\n".join(edges_summary) if edges_summary else "No connections defined."

        system_instruction = (
            "You are a Senior System Architect and AI Workflow Engineer.\n"
            "Your task is to analyze the provided graph configuration of an AI agentic workflow/pipeline "
            "and generate a clear, professional, and detailed architectural summary explaining what this workflow does.\n\n"
            "Format your output in clean, beautiful Markdown. Include the following sections:\n"
            "1. **Executive Summary**: A high-level overview of the workflow's main purpose.\n"
            "2. **Workflow Architecture & Execution Flow**: A step-by-step description of how data flows through the nodes and connections (triggers, agent steps, decision nodes, and outputs).\n"
            "3. **Component Roles & Configuration**: A summary table or list detailing the purpose and settings of each key node.\n"
            "4. **Recommendations/Insights** (Optional): Any architectural suggestions for improvement based on the layout."
        )

        prompt = (
            f"Workflow Name: {name}\n"
            f"Workflow Description: {description}\n\n"
            f"### Nodes List:\n{nodes_text}\n\n"
            f"### Edges / Connections:\n{edges_text}\n"
        )

        try:
            llm = get_llm_provider(
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                temperature=llm_config.get("temperature", 0.1),
                api_key=llm_config.get("api_key_override")
            )
            
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=prompt)
            ]
            
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.exception("Workflow architect generation failed")
            raise e

    async def repair_text(self, text: str, llm_config: Dict[str, Any]) -> str:
        """
        Repairs spelling, grammar, punctuation, and wording clarity of a text
        using the configured LLM (Architect path). Replaces the local Qwen
        grammar-fixer, so it uses the same cloud LLM as the rest of the Architect.
        """
        if not text or not text.strip():
            return text

        system_instruction = (
            "You are a senior technical writer and systems architect.\n"
            "Fix spelling, grammar, and punctuation errors and improve clarity and phrasing.\n"
            "Preserve the meaning, intent, and any technical terms exactly.\n"
            "Return ONLY the corrected text. Do not add explanations, quotes, or markdown formatting."
        )

        try:
            llm = get_llm_provider(
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                temperature=llm_config.get("temperature", 0.1),
                api_key=llm_config.get("api_key_override")
            )
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=text),
            ]
            response = await llm.ainvoke(messages)
            fixed = (response.content or text).strip()
            return fixed if fixed else text
        except Exception as e:
            logger.exception("Workflow architect text repair failed")
            raise e

    async def generate_rule(self, text: str, schema: str, existing_code: str, llm_config: Dict[str, Any]) -> str:
        """
        Converts a natural-language business rule into executable Python code
        using the configured LLM (Architect path).
        """
        system_instruction = (
            "You are an expert technical writer, Python developer, and systems architect.\n"
            "Your task is to convert the user's natural language business rule into executable Python code for a transaction fraud engine.\n"
            "The Python code will be executed dynamically inside a sandbox environment with access to ONLY a global dictionary variable `context`.\n"
            "You MUST start your script by safely extracting the transaction or variables from `context`. Usually this is:\n"
            "txn = context.get('canonical_transaction', {})\n"
            "The workflow schema defines the following available custom variables inside the transaction/context:\n"
            f"{schema or 'No specific variables provided. Rely on standard context.'}\n\n"
            "To allow a transaction, set `intent = 'allow'`.\n"
            "To block a transaction, set `intent = 'block'`.\n\n"
            "Respond with ONLY valid Python code. Do not include markdown code block backticks (like ```python or ```) or explanations. Just return the executable Python script content."
        )

        try:
            llm = get_llm_provider(
                provider=llm_config.get("provider") or os.getenv('LLM_PROVIDER_FAST', 'groq'),
                model=llm_config.get("model") or os.getenv('LLM_MODEL_FAST', 'llama-3.1-8b-instant'),
                temperature=llm_config.get("temperature", 0.1),
                api_key=llm_config.get("api_key_override")
            )
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=f"Convert this prompt to Python code:\n{text}"),
            ]
            response = await llm.ainvoke(messages)
            code = (response.content or "").strip()
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                code = "\n".join(lines).strip()
            return code
        except Exception as e:
            logger.exception("Workflow architect rule generation failed")
            raise e

    async def chat(self, workflow_data: Dict[str, Any], context: Dict[str, Any], chat_history: List[Dict[str, str]], llm_config: Dict[str, Any]) -> str:
        """
        Interactive chat session with the workflow configuration and runtime variables.
        """
        import json
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        nodes: List[Dict[str, Any]] = workflow_data.get("nodes", []) if workflow_data else []
        edges: List[Dict[str, Any]] = workflow_data.get("edges", []) if workflow_data else []
        name: str = workflow_data.get("name", "Unnamed Workflow") if workflow_data else "Unnamed Workflow"
        description: str = workflow_data.get("description", "") if workflow_data else ""

        nodes_summary = [self._format_node(n) for n in nodes]
        edges_summary = [self._format_edge(e) for e in edges]

        nodes_text = "\n".join(nodes_summary) if nodes_summary else "No nodes defined."
        edges_text = "\n".join(edges_summary) if edges_summary else "No connections defined."

        system_instruction = (
            "You are a Senior System Architect and AI Workflow Engineer.\n"
            "You are assisting a developer in inspecting, debugging, and understanding their AI workflow pipeline.\n"
            "You have access to the active workflow configuration, and the live runtime context / variables (if any run has occurred).\n\n"
        )
        if nodes or edges:
            system_instruction += (
                f"### Current Workflow Configuration:\n"
                f"Name: {name}\n"
                f"Description: {description}\n"
                f"Nodes:\n{nodes_text}\n"
                f"Connections:\n{edges_text}\n\n"
            )
        if context:
            # Filter out heavy/internal keys to save tokens
            filtered_context = {k: v for k, v in context.items() if not str(k).startswith("__") and k != "messages"}
            system_instruction += (
                f"### Active Data Explorer / Runtime Variables:\n"
                f"This represents the live variable values currently in the workflow state:\n"
                f"```json\n{json.dumps(filtered_context, indent=2, default=str)}\n```\n\n"
            )
        system_instruction += (
            "Answer the developer's questions clearly and professionally based on the workflow config and variables provided.\n"
            "If they ask for template syntax to access a variable, provide it as {{variable_name}} or {{nodes.node_id.key}} as appropriate."
        )

        try:
            llm = get_llm_provider(
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                temperature=llm_config.get("temperature", 0.2),
                api_key=llm_config.get("api_key_override")
            )
            
            messages = [SystemMessage(content=system_instruction)]
            
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
                    
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.exception("Workflow architect chat failed")
            raise e

workflow_architect = WorkflowArchitect()
