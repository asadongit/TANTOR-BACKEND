"""
Error Router Node Builder
Handles system-level errors by evaluating the error context and deciding on recovery strategies.
"""

import logging
import json
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from Apps.AgentMesh.Utilities.Observer import mark_langfuse_trace_error

logger = logging.getLogger(__name__)

class ErrorAnalysis(BaseModel):
    """Schema for error evaluation response"""
    category: str = Field(..., description="Category of error (transient, authentication, logic, validation, unknown)")
    logic: str = Field(..., description="Explanation of why the error occurred and what to do next")
    decision: str = Field(..., description="Action to take: 'retry', 'abort', 'human_help', 'continue'")
    suggestion: Optional[str] = Field(None, description="Detailed instructions for the retry or human helper")

def build_error_router(node_config: NodeConfig) -> Callable:
    """
    Build an error router node that uses an LLM (or static logic) to decide how to handle a failure.
    """
    llm = init_chat_model(node_config.llm)
    
    # Setup robust JSON parsing instead of brittle tool-calling
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=ErrorAnalysis)
    
    system_prompt = node_config.task or (
        "You are a System Error Handler. Analyze the error and decide on a recovery strategy."
    )
    
    full_system_prompt = (
        f"{system_prompt}\n\n"
        "Available Actions:\n"
        "- retry: Re-run the node (for transient/fixable errors).\n"
        "- abort: Stop the workflow (for fatal errors).\n"
        "- human_help: Request human intervention.\n\n"
        "CRITICAL: You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{format_instructions}\n"
        "Example output: {{\n  \"category\": \"transient\",\n  \"logic\": \"API timeout detected.\",\n  \"decision\": \"retry\",\n  \"suggestion\": null\n}}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        ("human", "ERROR DETAILS:\nSource Node: {error_source}\nError Message: {error_msg}\nError Type: {error_type}\n\nLast Context: {last_msg}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    error_chain = prompt | llm | parser

    async def execute_router(state: AgentState):
        error = state.get("error") or {}
        error_source = state.get("error_source", "Unknown")
        messages = state.get("messages", [])

        # Resolve thread_id for Langfuse trace-error marking on abort
        ctx = state.get("context") or {}
        thread_id = ctx.get("thread_id", "anonymous")

        # 🚨 Circuit Breaker: Prevent Infinite Loops
        retry_count = error.get("retry_count", 0)
        MAX_RETRIES = 3

        if retry_count >= MAX_RETRIES:
            logger.warning(f"[ERROR ROUTER] Max retries reached ({MAX_RETRIES}) for node '{error_source}'. Forcing abort.")
            mark_langfuse_trace_error(thread_id, f"Max retries exceeded for node '{error_source}': {error.get('message')}")
            return {"intent": "abort", "error": {**error, "decision": "abort", "reason": "Max retries exceeded"}}

        logger.info(f"[ERROR ROUTER] Analyzing error from node '{error_source}': {error.get('message')} [Attempt {retry_count + 1}]")
        
        last_msg = messages[-1].content if messages else "No previous messages."
        
        try:
            # The JSON parser returns a dict matching the ErrorAnalysis schema
            analysis = await error_chain.ainvoke({
                "error_source": error_source,
                "error_msg": error.get("message"),
                "error_type": error.get("type"),
                "last_msg": f"{last_msg[:500]} (Previous Retries: {retry_count})" 
            })
            
            decision = analysis.get("decision", "abort")
            logger.info(f"[ERROR ROUTER] Decision for '{error_source}': {decision} - {analysis.get('logic', 'N/A')}")
            
            # 🚨 ABORT: Mark the trace as ERROR since the workflow is being stopped
            if decision == "abort":
                mark_langfuse_trace_error(
                    thread_id,
                    f"Workflow aborted at node '{error_source}': {error.get('message')} - {analysis.get('logic', '')}"
                )

            # Increment retry count if we are retrying
            new_retry_count = retry_count
            if decision == "retry":
                new_retry_count += 1

            return {
                "intent": decision,
                "error": {
                    **error,
                    "retry_count": new_retry_count,
                    "decision": decision
                },
                "context": {
                    "last_error_analysis": analysis,
                    "hitl_feedback": analysis.get("suggestion")
                }
            }
            
        except Exception as e:
            logger.error(f"Error Router failed to analyze: {str(e)}")
            # On router failure, default to aborting to avoid infinite loops
            mark_langfuse_trace_error(
                thread_id,
                f"Error router analysis failed for node '{error_source}': {str(e)}"
            )
            return {"intent": "abort", "error": {**error, "decision": "abort", "reason": f"Router analysis error: {str(e)}"}}
            
    return RunnableLambda(execute_router)
