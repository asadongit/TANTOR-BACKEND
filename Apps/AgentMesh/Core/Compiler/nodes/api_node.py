"""
API Node Builder
Handles HTTP requests with dynamic URL/header/body interpolation.
Supports hybrid single/multi-API configurations with sequential/parallel execution.
"""

import json
import logging
import shlex
import asyncio
from typing import Callable, Dict, Any, List

import httpx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Database.Database import AsyncSessionLocal
from Apps.AgentMesh.Core.Connections.OAuthManager import OAuthManager
from Apps.AgentMesh.Utilities.Security import (
    contains_sensitive_data,
    validate_credential_destination,
)
from .utils import interpolate

logger = logging.getLogger(__name__)


def parse_curl_command(curl_string: str) -> Dict[str, Any]:
    """Parse cURL command to extract URL, method, headers, and data (Legacy Parity)"""
    if not curl_string or not isinstance(curl_string, str):
        raise ValueError("cURL command must be a non-empty string")
    
    curl_string = curl_string.strip()
    if curl_string.lower().startswith('curl'):
        curl_string = curl_string[4:].strip()
    
    try:
        parts = shlex.split(curl_string)
    except ValueError as e:
        raise ValueError(f"Invalid cURL syntax: {str(e)}")
    
    result = {"url": None, "method": "GET", "headers": {}, "data": None}
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ['-X', '--request'] and i + 1 < len(parts):
            result['method'] = parts[i + 1].upper()
            i += 2
        elif part in ['-H', '--header'] and i + 1 < len(parts):
            header = parts[i + 1]
            if ':' in header:
                key, value = header.split(':', 1)
                result['headers'][key.strip()] = value.strip()
            i += 2
        elif part in ['-d', '--data', '--data-raw'] and i + 1 < len(parts):
            result['data'] = parts[i + 1]
            if result['method'] == 'GET':
                result['method'] = 'POST'
            i += 2
        elif not part.startswith('-') and result['url'] is None:
            result['url'] = part
            i += 1
        else:
            i += 1
    
    if not result['url']:
        raise ValueError("No URL found in cURL command")
    return result


async def execute_single_api_call(
    client: httpx.AsyncClient,
    config: Any,
    context: Dict[str, Any],
    node_id: str
) -> Any:
    """
    Executes a single API call configuration (supporting cURL parsing if use_curl is True)
    and returns the parsed result or error dict.
    """
    max_retries = getattr(config, "api_max_retries", None)
    if max_retries is None:
        max_retries = getattr(config, "max_retries", 1)
    max_retries = max(1, int(max_retries))

    retry_delay = getattr(config, "api_retry_delay", None)
    if retry_delay is None:
        retry_delay = getattr(config, "retry_delay", 2.0)
    retry_delay = max(0.0, float(retry_delay))

    use_curl = getattr(config, "use_curl", False)
    curl_command = None
    if use_curl:
        curl_template = getattr(config, "curl_command", None)
        if curl_template:
            curl_command = interpolate(curl_template, context)
        else:
            curl_command = context.get("payload", {}).get("curl_command")

    if curl_command:
        logger.info(f"API Node {node_id}: Executing parsed cURL command")
        try:
            parsed_curl = parse_curl_command(curl_command)
        except Exception as ce:
            logger.error(f"API Node {node_id}: Failed to parse cURL - {str(ce)}")
            return {"error": f"Failed to parse cURL: {str(ce)}", "status": "failed"}
        
        url = parsed_curl["url"]
        method = parsed_curl["method"].upper()
        headers = parsed_curl["headers"]
        body = parsed_curl["data"]
        params = {k: interpolate(v, context) for k, v in (getattr(config, "api_params", None) or {}).items()}
        protocol = "REST"
        connection_id = None
        auth_type = "None"
        auth_config = {}
        response_format = getattr(config, "api_response_format", "json")
    else:
        url_template = getattr(config, "api_url", None)
        if not url_template:
            return {"error": "API URL is missing", "status": "failed"}

        method = getattr(config, "api_method", "GET").upper()
        headers_template = getattr(config, "api_headers", None) or {}
        params_template = getattr(config, "api_params", None) or {}
        body_template = getattr(config, "api_body", None)
        connection_id = getattr(config, "connection_id", None)
        auth_type = getattr(config, "api_auth_type", "None")
        auth_config = getattr(config, "api_auth_config", None) or {}
        response_format = getattr(config, "api_response_format", "json")
        protocol = getattr(config, "api_protocol", "REST").upper()
        gql_query = getattr(config, "graphql_query", None)

        logger.info(f"API Node {node_id}: Executing {method} to {url_template}")

        # Interpolate variables
        url = interpolate(url_template, context)
        headers = {k: interpolate(v, context) for k, v in headers_template.items()}
        params = {k: interpolate(v, context) for k, v in params_template.items()}

        # Handle request body
        body = body_template
        if body and isinstance(body, str):
            body = interpolate(body, context)

    # Setup authentication
    auth = None
    if connection_id:
        try:
            async with AsyncSessionLocal() as db:
                fresh_token = await OAuthManager.get_valid_token(db, connection_id)
                validate_credential_destination(url)
                headers["Authorization"] = f"Bearer {fresh_token}"
        except Exception as e:
            logger.error(f"API Node {node_id}: Failed to load connection profile - {str(e)}")
    else:
        if auth_type == "Basic":
            user = auth_config.get("username")
            password = auth_config.get("password")
            if user and password:
                auth = httpx.BasicAuth(user, password)
        elif auth_type == "Bearer":
            token = auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

    # Keep credentials out of URLs; transport follows the configured endpoint.
    credential_payload = {
        "headers": headers,
        "body": body,
        "auth_config": auth_config,
    }
    credential_request = auth is not None or bool(connection_id) or contains_sensitive_data(credential_payload)
    if credential_request:
        validate_credential_destination(url)
    if contains_sensitive_data(params):
        raise ValueError("Credentials must not be placed in API query parameters")

    # Protocol Specific Pre-Processing
    if not curl_command:
        if protocol == "SOAP":
            headers["Content-Type"] = "text/xml"
            method = "POST"
        elif protocol == "GRAPHQL":
            headers["Content-Type"] = "application/json"
            method = "POST"
            if gql_query:
                query_str = interpolate(gql_query, context)
                body = json.dumps({"query": query_str})

    request_kwargs: Dict[str, Any] = {
        "method": method,
        "url": url,
        "headers": headers,
        "params": params,
        "auth": auth
    }
    if credential_request:
        # Do not forward credentials to an unvalidated redirect destination.
        request_kwargs["follow_redirects"] = False

    file_handles_to_close = []
    if body and method in ["POST", "PUT", "PATCH"]:
        if not curl_command and protocol == "SOAP":
            request_kwargs["content"] = body
        else:
            try:
                json_data = json.loads(body)
                files = {}
                has_files = False
                import os

                def extract_files(obj, current_key=""):
                    nonlocal has_files
                    if isinstance(obj, dict):
                        if "filename" in obj and ("vfs_path" in obj or "data" in obj):
                            has_files = True
                            filename = obj["filename"]
                            mime_type = obj.get("mime_type") or "application/octet-stream"
                            if "vfs_path" in obj and os.path.exists(obj["vfs_path"]):
                                try:
                                    fh = open(obj["vfs_path"], "rb")
                                    file_handles_to_close.append(fh)
                                    files[current_key or "file"] = (filename, fh, mime_type)
                                except Exception as fe:
                                    logger.error(f"API Node failed to open file path {obj['vfs_path']}: {fe}")
                            elif "data" in obj and isinstance(obj["data"], str) and "," in obj["data"]:
                                import base64
                                try:
                                    _, encoded = obj["data"].split(",", 1)
                                    file_bytes = base64.b64decode(encoded)
                                    files[current_key or "file"] = (filename, file_bytes, mime_type)
                                except Exception as fe:
                                    logger.error(f"API Node failed to decode base64 file data: {fe}")
                        else:
                            for k, v in obj.items():
                                extract_files(v, k)
                    elif isinstance(obj, list):
                        for idx, item in enumerate(obj):
                            extract_files(item, f"{current_key}[{idx}]" if current_key else "files")

                extract_files(json_data)

                if has_files:
                    def clean_json(obj):
                        if isinstance(obj, dict):
                            if "filename" in obj and ("vfs_path" in obj or "data" in obj):
                                return None
                            cleaned = {k: clean_json(v) for k, v in obj.items()}
                            return {k: v for k, v in cleaned.items() if v is not None}
                        elif isinstance(obj, list):
                            cleaned = [clean_json(x) for x in obj]
                            return [x for x in cleaned if x is not None]
                        return obj

                    request_kwargs["files"] = files
                    request_kwargs["data"] = clean_json(json_data)
                else:
                    request_kwargs["json"] = json_data

            except json.JSONDecodeError:
                request_kwargs["content"] = body

    try:
        for attempt in range(max_retries):
            # Reset file offsets to 0 for retries so file uploads don't send 0 bytes
            for fh in file_handles_to_close:
                try:
                    fh.seek(0)
                except Exception:
                    pass
            try:
                response = await client.request(**request_kwargs)
                response.raise_for_status()
                if response_format == "json":
                    result_data = response.json()
                elif response_format == "binary":
                    result_data = f"__BINARY_DATA__{len(response.content)} bytes"
                else:
                    result_data = response.text
                return result_data
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"API Node {node_id}: HTTP Error {status_code} on attempt {attempt + 1}/{max_retries} - {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"API Node {node_id}: Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                error_msg = f"HTTP Error {status_code}: {e.response.text}"
                if status_code == 401 and not connection_id:
                    error_msg = (
                        "401 Unauthorized. If you are using Manual Authentication, your token "
                        "has likely expired or is invalid. Please update it in the node properties."
                    )
                elif status_code == 401:
                    error_msg = "401 Unauthorized. The Enterprise Connection profile token may be invalid or missing required scopes."
                return {"error": error_msg, "status": "failed"}
            except Exception as e:
                logger.error(f"API Node {node_id}: Failed on attempt {attempt + 1}/{max_retries} - {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"API Node {node_id}: Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                return {"error": str(e), "status": "failed"}
    finally:
        for fh in file_handles_to_close:
            try:
                fh.close()
            except Exception:
                pass


def build_api_node(node_config: NodeConfig) -> Callable:
    """
    Build an API execution node that supports dynamic interpolation
    of URLs, headers, parameters, and request bodies. Supports sequential
    and parallel multi-api orchestration (hybrid mode).
    """
    async def execute_api(state: AgentState):
        context = state.get("context", {})
        additional_apis = getattr(node_config, "additional_apis", []) or []
        enabled_additional = [api for api in additional_apis if getattr(api, "enabled", True)]
        primary_enabled = getattr(node_config, "api_enabled", True)

        timeout = httpx.Timeout(300.0, connect=60.0, read=300.0, write=120.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # Check if this is a standalone single API run
            if not enabled_additional:
                if not primary_enabled:
                    # Skipped single API
                    new_context = {
                        **context,
                        f"api_result_{node_config.id}": {"status": "skipped", "message": "Primary API disabled"},
                        "latest_api_result": {"status": "skipped", "message": "Primary API disabled"}
                    }
                    return {
                        "messages": [AIMessage(content="API Node execution skipped (Primary API is disabled).", response_metadata={"active_agent": node_config.id})],
                        "active_agent": node_config.id,
                        "context": new_context
                    }

                result_data = await execute_single_api_call(client, node_config, context, node_config.id)

                is_failed = isinstance(result_data, dict) and result_data.get("status") == "failed"
                if is_failed and not getattr(node_config, "continue_on_fail", False):
                    error_msg = result_data.get("error", "Unknown error")
                    return {
                        "messages": [AIMessage(content=f"API Request failed: {error_msg}", response_metadata={"active_agent": node_config.id})],
                        "active_agent": node_config.id,
                        "context": {**context, "api_error": error_msg},
                        "error": {"message": error_msg, "type": "HTTPStatusError"},
                        "error_source": node_config.id
                    }

                new_context = {
                    **context,
                    f"api_result_{node_config.id}": result_data,
                    "latest_api_result": result_data
                }

                if isinstance(result_data, (dict, list)):
                    try:
                        content = f"```json\n{json.dumps(result_data, indent=2, default=str)}\n```"
                    except Exception:
                        content = f"{result_data}"
                else:
                    content = f"{result_data}"

                return {
                    "messages": [AIMessage(content=content, response_metadata={"active_agent": node_config.id})],
                    "active_agent": node_config.id,
                    "context": new_context
                }

            else:
                # Multi-API Flow
                execution_mode = getattr(node_config, "api_execution_mode", "sequential").lower()

                if execution_mode == "parallel":
                    # Concurrently execute all enabled API calls
                    configs_to_run = []
                    primary_idx = -1
                    if primary_enabled:
                        configs_to_run.append(node_config)
                        primary_idx = 0

                    # Map sub-API configurations
                    start_additional_idx = len(configs_to_run)
                    configs_to_run.extend(enabled_additional)

                    tasks = [
                        execute_single_api_call(client, cfg, context, f"{node_config.id}_{idx}")
                        for idx, cfg in enumerate(configs_to_run)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    primary_result = "Skipped (Disabled)"
                    if primary_idx != -1:
                        primary_result = results[primary_idx]
                        if isinstance(primary_result, Exception):
                            primary_result = {"error": str(primary_result), "status": "failed"}

                    additional_results = []
                    for idx, res in enumerate(results[start_additional_idx:]):
                        cfg = enabled_additional[idx]
                        if isinstance(res, Exception):
                            res = {"error": str(res), "status": "failed"}
                        additional_results.append({
                            "name": getattr(cfg, "name", f"API #{idx+1}"),
                            "description": getattr(cfg, "description", ""),
                            "result": res
                        })

                    combined_result = {
                        "primary": primary_result,
                        "additional": additional_results
                    }

                else:
                    # Sequential Mode: execute sequentially, updating context step-by-step
                    # so that subsequent calls can interpolate values from preceding results
                    current_context = {**context}

                    # Execute primary first if enabled
                    if primary_enabled:
                        primary_result = await execute_single_api_call(client, node_config, current_context, node_config.id)
                    else:
                        primary_result = "Skipped (Disabled)"

                    # Update context with intermediate result structure
                    current_context[f"api_result_{node_config.id}"] = {
                        "primary": primary_result,
                        "additional": []
                    }
                    current_context["latest_api_result"] = primary_result

                    additional_results = []
                    for idx, cfg in enumerate(enabled_additional):
                        res = await execute_single_api_call(client, cfg, current_context, f"{node_config.id}_{idx+1}")
                        additional_results.append({
                            "name": getattr(cfg, "name", f"API #{idx+1}"),
                            "description": getattr(cfg, "description", ""),
                            "result": res
                        })
                        # Update context step-by-step
                        current_context[f"api_result_{node_config.id}"]["additional"] = additional_results
                        current_context["latest_api_result"] = res

                    combined_result = {
                        "primary": primary_result,
                        "additional": additional_results
                    }

                # Error check across all executed APIs
                primary_failed = (
                    primary_enabled and 
                    isinstance(combined_result["primary"], dict) and 
                    combined_result["primary"].get("status") == "failed"
                )
                additional_failed = any(
                    isinstance(r["result"], dict) and r["result"].get("status") == "failed"
                    for r in combined_result["additional"]
                )

                if (primary_failed or additional_failed) and not getattr(node_config, "continue_on_fail", False):
                    error_msg = "One or more API requests in the multi-API node failed."
                    if primary_failed:
                        error_msg += f" Primary: {combined_result['primary'].get('error', 'Unknown error')}."
                    return {
                        "messages": [AIMessage(content=f"Multi-API Request failed: {error_msg}", response_metadata={"active_agent": node_config.id})],
                        "active_agent": node_config.id,
                        "context": {**context, "api_error": error_msg},
                        "error": {"message": error_msg, "type": "MultiAPIExecutionError"},
                        "error_source": node_config.id
                    }

                new_context = {
                    **context,
                    f"api_result_{node_config.id}": combined_result,
                    "latest_api_result": combined_result
                }

                try:
                    content = f"```json\n{json.dumps(combined_result, indent=2, default=str)}\n```"
                except Exception:
                    content = f"{combined_result}"

                return {
                    "messages": [AIMessage(content=content, response_metadata={"active_agent": node_config.id})],
                    "active_agent": node_config.id,
                    "context": new_context
                }

    return RunnableLambda(execute_api).with_retry(stop_after_attempt=3)
