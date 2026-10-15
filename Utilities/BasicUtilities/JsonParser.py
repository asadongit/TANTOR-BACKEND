import json
import re
import logging
"""extract JSON content from responses, handling markdown wrapping if present."""
logger = logging.getLogger(__name__)

class JsonParser:
    @staticmethod
    def extract_json(response_text: str) -> dict:
        """Extract and parse JSON from response (handles markdown wrapping)"""
        # Strategy 1: Markdown-wrapped JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Response: {response_text[:200]}")
            raise ValueError(f"Could not parse JSON: {e}")