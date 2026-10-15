"""
Follow-up Questions Form Tool (July 2026)
Allows the LLM to ask the user a structured set of questions via a popup form.
The last question always has a "Skip" button.
"""
import json
import logging

from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)


@tool
def followup_form(title: str, questions: list) -> dict:
    """You have access to a Follow-up Form tool.

    Use this tool when you need additional structured information from the user before continuing.

    Instead of asking questions in chat, present a guided form.

    Rules:
    - Ask all missing questions in one form.
    - Keep questions short and clear.
    - For each question, provide suggested answers whenever possible.
    - Suggested answers should cover the most likely user responses.
    - Always include an "Other" or "Custom" option so the user can enter their own answer.
    - Use "text" only when suggestions are not practical.
    - Ask only the minimum number of questions needed.
    - Order questions from most important to least important. The last question can be skipped.

    Args:
        title: Short title or topic for this set of questions.
        questions: Array of question objects. Each object supports:
            - id (str, required): unique identifier
            - question (str, required): the question text
            - type (str, optional): "text", "select", "boolean", or "number". Default "text"
            - suggestions (list, optional): choices for "select" type
            - required (bool, optional): whether answer is mandatory. Default true. The last question is always skippable.
    """
    return {
        "__human_interaction__": True,
        "pause_config": {
            "type": "followup_form",
            "title": title,
            "questions": questions
        }
    }


def register_followup_form():
    registry.register(
        followup_form,
        ToolMeta(
            name="followup_form",
            display_name="Follow-up Questions Form",
            description="Guided forms popup for structured questions.",
            category="Human",
            package="core",
            version="1.0.0",
            ai_visible_fields=["title", "questions"],
        )
    )

register_followup_form()
