"""
Evaluator Node Builder
Handles automated quality assessment and feedback generation for iterative loops.
"""

import json
import logging
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig, IterationPolicy
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model

logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    """Schema for LLM evaluation response"""
    score: float = Field(..., description="Quality score from 0.0 to 1.0")
    feedback: str = Field(..., description="Constructive feedback for the agent to improve")
    passed: bool = Field(..., description="True if score >= target_score")

def build_evaluator_node(node_id: str, policy: IterationPolicy) -> Callable:
    """
    Build an evaluator node that uses an LLM to assess the last output
    and generate feedback for a specific agent node.
    """
    llm = init_chat_model(policy.evaluator_llm)
    
    # Setup robust JSON parsing instead of brittle tool-calling
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=EvaluationResult)
    
    system_prompt = policy.evaluator_task or (
        "You are a quality control critic. Your job is to evaluate the following output "
        "and provide structural feedback for improvement. "
        "Return a score from 0.0 (unusable) to 1.0 (perfect)."
    )
    
    # Inject format instructions into the system prompt
    full_system_prompt = (
        f"{system_prompt}\n\n"
        "CRITICAL: You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{format_instructions}\n"
        "Example: {{\"score\": 0.8, \"feedback\": \"Good, but needs more detail.\", \"passed\": true}}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        ("human", "LATEST OUTPUT TO EVALUATE:\n\n{output}\n\nPlease evaluate based on the target score: {target_score}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    eval_chain = prompt | llm | parser

    async def execute_evaluator(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"iter_next": "FINISH", "iter_feedback": None}
        
        # Evaluate the last ASSISTANT (AI) output, not an intermediate ToolMessage/
        # orphan tool-call. For tool-calling agent nodes the final message in state can
        # be a ToolMessage, which would produce a meaningless grade.
        last_output = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) or getattr(msg, "type", "") in ("ai", "assistant"):
                content = getattr(msg, "content", None)
                if content:
                    last_output = content if isinstance(content, str) else str(content)
                    break
        if last_output is None:
            last_output = messages[-1].content
        
        # Track iteration count in context
        context = state.get("context", {})
        iter_key = f"iter_count_{node_id}"
        current_iter = context.get(iter_key, 0)
        
        logger.info(f"[EVALUATOR] Evaluating {node_id} (Iteration {current_iter + 1}/{policy.max_iterations})")
        
        try:
            # The JSON parser returns a dict matching the EvaluationResult schema
            result = await eval_chain.ainvoke({
                "output": last_output,
                "target_score": policy.target_score
            })
            
            score = result.get("score", 0.0)
            passed = result.get("passed", False)
            feedback = result.get("feedback", "No feedback provided.")
            
            logger.info(f"[EVALUATOR] Result for {node_id}: score={score}, passed={passed}")
            
            # 🕵️ Langfuse: push the automated accuracy score to the workflow trace.
            # thread_id (from context) maps to the SAME trace id the LangGraph run
            # is pinned to, so the score lands under the actual workflow trace.
            try:
                from Apps.AgentMesh.Utilities.Observer import record_accuracy_score
                thread_id = context.get("thread_id")
                if thread_id:
                    record_accuracy_score(thread_id, score, evaluator="auto_evaluator")
            except Exception:
                logger.debug(f"[EVALUATOR] Skipped Langfuse accuracy score for {node_id}")
            
            # Decide transition: LOOP or FINISH
            if passed or current_iter >= policy.max_iterations - 1:
                return {
                    "context": {**context, iter_key: 0}, # Reset for next time if workflow repeats
                    "iter_next": "FINISH",
                    "iter_feedback": None
                }
            
            # Loop back with feedback
            feedback_content = f"[AUTOMATED FEEDBACK - Iteration {current_iter + 1}]\n{feedback}"
            feedback_msg = HumanMessage(content=feedback_content) if policy.inject_feedback_as == "human" else SystemMessage(content=feedback_content)
            
            return {
                "messages": [feedback_msg],
                "context": {**context, iter_key: current_iter + 1},
                "iter_next": "LOOP",
                "iter_feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for {node_id}: {str(e)}")
            # On failure, default to finishing to avoid infinite loops
            return {"iter_next": "FINISH", "iter_feedback": f"Evaluation Error: {str(e)}"}
            
    return RunnableLambda(execute_evaluator)
