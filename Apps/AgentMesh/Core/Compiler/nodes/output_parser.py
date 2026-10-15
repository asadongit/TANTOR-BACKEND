"""
Output Parser Node (Auto-fixing JSON/Data Parser)
Inspired by n8n's Auto-fixing Output Parser.
Repairs malformed JSON and structured data using json-repair library.

Follows AgentMesh pattern: Auto-reads from latest message, no field config needed.

Production Features:
- Multi-stage parsing (standard → preprocess → json-repair)
- Intelligent string value cleaning
- Comprehensive error handling & recovery
- Detailed metrics & structured logging
"""

import json
import logging
import re
from typing import Callable, Any, Dict
from json_repair import loads
from langchain_core.messages import AIMessage

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)


def preprocess_malformed_json(raw_str: str) -> str:
    """
    Pre-process extremely malformed JSON to fix common issues before json-repair.
    
    Production-grade preprocessing handles:
    - Unclosed strings at end of lines (closes them intelligently)
    - Missing closing quotes on keys (e.g., "email : -> "email":)
    - Missing commas between object properties
    - Whitespace normalization
    
    This layer tries to fix the most common malformations before json-repair.
    """
    lines = raw_str.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Case 1: Line ends with unclosed quote (e.g., "name" : "Bhumi ,)
        # Pattern: "key" : "value without closing quote
        if re.search(r':\s*"[^"]*$', line):
            line = line + '"'
            logger.debug(f"Preprocessing: Closed unclosed quote")
        
        # Case 2: Missing closing quote on key (e.g., "email : -> "email":)
        # Pattern: "key : (missing closing quote on key)
        match = re.search(r'^(\s*"[^"]+)\s*:\s*(.*)$', line)
        if match:
            key_part = match.group(1)
            value_part = match.group(2)
            # Check if key_part doesn't end with quote (malformed key)
            if not key_part.rstrip().endswith('"'):
                line = key_part + '": ' + value_part
                logger.debug(f"Preprocessing: Fixed missing key closing quote")
        
        # Case 3: If this line ends with a quote and next line starts with a quote (missing comma)
        if i < len(lines) - 1:
            next_line = lines[i + 1].lstrip()
            if (line.rstrip().endswith('"') and 
                next_line.startswith('"') and 
                not line.rstrip().endswith(',') and
                not line.rstrip().endswith('{') and
                not line.rstrip().endswith('[') and
                ':' in next_line):
                line = line + ','
                logger.debug(f"Preprocessing: Added missing comma between properties")
        
        processed_lines.append(line)
    
    result = '\n'.join(processed_lines)
    return result


def clean_string_values(obj: Any) -> Any:
    """
    Production-grade string value cleaning.
    
    Removes erroneous trailing punctuation from string values:
    - Trailing commas, semicolons (likely typos)
    - Excessive trailing whitespace
    
    Also cleans dict keys:
    - Removes trailing whitespace from keys
    
    Preserves:
    - Intentional whitespace inside strings
    - Allowed trailing characters (e.g., @ in emails, / in URLs)
    
    Args:
        obj: Parsed JSON object (dict, list, or primitive)
        
    Returns:
        Cleaned object with refined string values and keys
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Clean the key: remove trailing whitespace
            clean_key = key.rstrip() if isinstance(key, str) else key
            cleaned[clean_key] = clean_string_values(value)
        return cleaned
    
    elif isinstance(obj, list):
        return [clean_string_values(item) for item in obj]
    
    elif isinstance(obj, str):
        # Remove trailing comma, semicolon (likely typos)
        # But preserve @ in emails, / in URLs
        cleaned = obj.rstrip()
        
        # Only remove trailing punctuation if it's obviously wrong
        if cleaned and cleaned[-1] in (',', ';'):
            cleaned = cleaned[:-1].rstrip()
            logger.debug(f"Cleaned string value: removed trailing '{obj[-1]}'")
        
        return cleaned
    
    else:
        # Primitives: numbers, booleans, null - return as-is
        return obj


def build_output_parser(node_config: NodeConfig) -> Callable:
    """
    Build an output parser node that auto-fixes malformed JSON.
    
    Production-grade features:
    - Automatically reads from the latest message (no input_field config needed)
    - Multi-stage parsing: standard → preprocess → json-repair
    - Intelligent string value cleaning (removes trailing punctuation)
    - Comprehensive error handling & recovery strategies
    - Detailed metrics & structured logging
    - Strict/auto/json parsing modes
    
    Config:
    - parse_mode: "auto" (standard→preprocess→repair), "json" (force JSON), "strict" (fail on error)
    - fallback_to_raw: Return original string if all parsing fails (default: True)
    - output_context_key: Context key to store parsed result (default: "parsed_output")
    
    Examples:
        Input:  {name: "John", age: 30, tags: ["a", "b",]}
        Output: {"name": "John", "age": 30, "tags": ["a", "b"]}
        
        Input (malformed): {"name" : "Bhumi , "email : "test@example.com"}
        Output (cleaned):  {"name": "Bhumi", "email": "test@example.com"}
    """
    async def parse_output(state: AgentState):
        messages = state.get("messages", [])
        context = state.get("context", {})
        
        # Get raw input from latest message (automatic, no config needed)
        raw_data = messages[-1].content if messages else None
        
        if not raw_data:
            logger.warning(f"Output Parser Node {node_config.id}: No message content to parse")
            return {"active_agent": node_config.id}
        
        # Get configuration
        parse_mode = getattr(node_config, "parse_mode", "auto")
        fallback_to_raw = getattr(node_config, "fallback_to_raw", True)
        output_context_key = getattr(node_config, "output_context_key", "parsed_output")
        
        # Validate configuration
        if parse_mode not in ("auto", "json", "strict"):
            logger.warning(f"Output Parser Node {node_config.id}: Invalid parse_mode '{parse_mode}', defaulting to 'auto'")
            parse_mode = "auto"
        
        # Convert to string if needed
        raw_str = str(raw_data).strip()
        
        # Metrics
        input_length = len(raw_str)
        logger.info(
            f"Output Parser Node {node_config.id}: Started | "
            f"mode={parse_mode} | length={input_length} chars | fallback={fallback_to_raw}"
        )
        
        parsed_result = None
        parsing_stage = None
        error_msg = None
        
        try:
            # STAGE 1: Try standard JSON parsing first (fastest, no repairs)
            try:
                parsed_result = json.loads(raw_str)
                parsing_stage = "standard"
                logger.info(f"Output Parser Node {node_config.id}: ✅ Standard JSON parsing succeeded")
            
            except json.JSONDecodeError as e:
                # STAGE 2: Pre-process if extremely malformed
                if parse_mode == "strict":
                    raise  # Re-raise in strict mode
                
                logger.debug(f"Output Parser Node {node_config.id}: Standard parsing failed. Attempting preprocessing...")
                preprocessed = preprocess_malformed_json(raw_str)
                
                # Try standard parsing again after preprocessing
                try:
                    parsed_result = json.loads(preprocessed)
                    parsing_stage = "preprocessed"
                    logger.info(f"Output Parser Node {node_config.id}: ✅ Parsing succeeded after preprocessing")
                
                except json.JSONDecodeError:
                    # STAGE 3: Fall back to json-repair for additional auto-fix
                    logger.debug(f"Output Parser Node {node_config.id}: Attempting json-repair as final layer...")
                    parsed_result = loads(preprocessed)
                    parsing_stage = "json-repair"
                    logger.info(f"Output Parser Node {node_config.id}: ✅ Auto-repair succeeded")
            
            # STAGE 4: Clean string values (remove trailing punctuation)
            if parsed_result is not None:
                parsed_result = clean_string_values(parsed_result)
                logger.debug(f"Output Parser Node {node_config.id}: String values cleaned")
            
            # Validate output
            if parsed_result is None:
                raise ValueError("Parsing resulted in None")
            
            # Log success metrics
            output_json = json.dumps(parsed_result)
            logger.info(
                f"Output Parser Node {node_config.id}: ✅ SUCCESS | "
                f"stage={parsing_stage} | "
                f"input={input_length} chars | output={len(output_json)} chars | "
                f"type={type(parsed_result).__name__}"
            )
            
            # Return parsed result to both messages and context
            return {
                "messages": [AIMessage(content=output_json)],
                "context": {output_context_key: parsed_result},
                "active_agent": node_config.id
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Output Parser Node {node_config.id}: ❌ PARSING FAILED | "
                f"stage={parsing_stage} | error={error_msg}"
            )
            
            # RECOVERY STRATEGIES
            
            # Strategy 1: Strict mode - fail hard
            if parse_mode == "strict":
                raise ValueError(f"JSON parsing failed in strict mode: {error_msg}")
            
            # Strategy 2: Fallback to raw string
            if fallback_to_raw:
                logger.warning(f"Output Parser Node {node_config.id}: Using fallback strategy (returning raw string)")
                return {
                    "messages": [AIMessage(content=raw_str)],
                    "context": {output_context_key: raw_str},
                    "active_agent": node_config.id
                }
            
            # Strategy 3: Return error state
            logger.error(f"Output Parser Node {node_config.id}: All parsing strategies exhausted")
            return {
                "error": {
                    "message": error_msg,
                    "type": "JSONParsingError",
                    "stage": parsing_stage,
                    "fallback": False
                },
                "active_agent": node_config.id
            }
    
    return parse_output
