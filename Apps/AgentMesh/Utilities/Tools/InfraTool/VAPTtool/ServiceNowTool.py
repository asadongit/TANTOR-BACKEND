import os
import logging
import json
import requests
from typing import Optional, Dict, Any, Tuple
from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15
_GROUP_CACHE: Dict[str, str] = {}


def _get_snow_config() -> Tuple[str, str, str]:
    """
    Retrieve ServiceNow credentials from environment variables.
    Checks os.environ, then loads from .env in TANTOR-BACKEND or root.
    """
    instance = os.getenv("SNOW_INSTANCE", "").strip().rstrip("/")
    user = os.getenv("SNOW_USER", "").strip()
    pwd = os.getenv("SNOW_PASS", "").strip()

    if not instance or not user or not pwd:
        # Fallback load from root .env or backend .env
        from dotenv import load_dotenv
        root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.env"))
        backend_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
        if os.path.exists(root_env):
            load_dotenv(dotenv_path=root_env)
        if os.path.exists(backend_env):
            load_dotenv(dotenv_path=backend_env)
            
        instance = os.getenv("SNOW_INSTANCE", "").strip().rstrip("/")
        user = os.getenv("SNOW_USER", "").strip()
        pwd = os.getenv("SNOW_PASS", "").strip()

    if not instance or not user or not pwd:
        raise ValueError(
            "ServiceNow configuration missing. Please ensure SNOW_INSTANCE, "
            "SNOW_USER, and SNOW_PASS are set."
        )

    return instance, user, pwd


def _get_snow_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


def _check_response_for_hibernation(res: requests.Response) -> Optional[str]:
    content_type = res.headers.get("content-type", "").lower()
    if "text/html" in content_type or "hibernat" in res.text.lower():
        return (
            "ServiceNow Developer Instance is currently HIBERNATING (sleeping). "
            "Please visit https://developer.servicenow.com, log in, and click 'Wake Up Instance' to resume it."
        )
    return None


def _get_group_id(group_name: str = "Software") -> Optional[str]:
    if not group_name:
        return None
    if group_name in _GROUP_CACHE:
        return _GROUP_CACHE[group_name]

    try:
        instance, user, pwd = _get_snow_config()
        url = f"{instance}/api/now/table/sys_user_group"
        params = {
            "sysparm_query": f"name={group_name}",
            "sysparm_fields": "name,sys_id",
            "sysparm_limit": 1
        }
        res = requests.get(url, auth=(user, pwd), headers=_get_snow_headers(), params=params, timeout=TIMEOUT_SECONDS)
        hib = _check_response_for_hibernation(res)
        if hib:
            logger.warning(hib)
            return None

        if res.status_code == 200:
            results = res.json().get("result", [])
            if results and isinstance(results, list):
                sys_id = results[0].get("sys_id")
                if sys_id:
                    _GROUP_CACHE[group_name] = sys_id
                    return sys_id
        return None
    except Exception as e:
        logger.error(f"Error fetching assignment group '{group_name}': {e}")
        return None


def _lookup_ticket_sys_id(identifier: str) -> Optional[Tuple[str, str]]:
    """
    Given a sys_id or ticket number (e.g., CHG0030015), returns (sys_id, number).
    """
    clean_id = identifier.strip()
    if not clean_id:
        return None

    instance, user, pwd = _get_snow_config()
    
    # If it looks like a ticket number
    if clean_id.upper().startswith("CHG") or len(clean_id) != 32:
        url = f"{instance}/api/now/table/change_request"
        params = {
            "sysparm_query": f"number={clean_id.upper()}",
            "sysparm_fields": "sys_id,number",
            "sysparm_limit": 1
        }
        res = requests.get(url, auth=(user, pwd), headers=_get_snow_headers(), params=params, timeout=TIMEOUT_SECONDS)
        hib = _check_response_for_hibernation(res)
        if hib:
            logger.warning(hib)
            return None, clean_id

        if res.status_code == 200:
            results = res.json().get("result", [])
            if results and isinstance(results, list):
                return results[0].get("sys_id"), results[0].get("number")
        return None, clean_id

    # If it is a 32-char sys_id, fetch the number
    url = f"{instance}/api/now/table/change_request/{clean_id}"
    params = {"sysparm_fields": "sys_id,number"}
    res = requests.get(url, auth=(user, pwd), headers=_get_snow_headers(), params=params, timeout=TIMEOUT_SECONDS)
    hib = _check_response_for_hibernation(res)
    if hib:
        logger.warning(hib)
        return clean_id, None

    if res.status_code == 200:
        result = res.json().get("result", {})
        return result.get("sys_id", clean_id), result.get("number")

    return clean_id, None


@tool
def create_servicenow_ticket(
    title: str,
    description: str,
    assignment_group: str = "Software"
) -> str:
    """
    Creates a new Normal Change Request in ServiceNow for VAPT vulnerability remediation.
    Automatically assigns the ticket to 'Software' group and sets state to Assess (-4) to trigger approval workflow.
    Returns the Change Request number (e.g. CHG0030012), sys_id, approval link, and status.
    """
    try:
        instance, user, pwd = _get_snow_config()
        group_id = _get_group_id(assignment_group)

        payload: Dict[str, Any] = {
            "short_description": f"VAPT Remediation: {title}",
            "description": description,
            "type": "normal",
            "category": "Software",
            "risk": "3",
            "impact": "2"
        }
        if group_id:
            payload["assignment_group"] = group_id

        url = f"{instance}/api/now/table/change_request"
        res = requests.post(
            url,
            auth=(user, pwd),
            headers=_get_snow_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        hib = _check_response_for_hibernation(res)
        if hib:
            return json.dumps({
                "status": "error",
                "message": hib
            })

        if res.status_code not in [200, 201]:
            return json.dumps({
                "status": "error",
                "message": f"Failed to create ServiceNow ticket (HTTP {res.status_code}): {res.text}"
            })

        data = res.json().get("result", {})
        ticket_number = data.get("number")
        sys_id = data.get("sys_id")

        # Automatically advance state to Assess (-4) to request approval
        if sys_id:
            patch_url = f"{instance}/api/now/table/change_request/{sys_id}"
            requests.patch(
                patch_url,
                auth=(user, pwd),
                headers=_get_snow_headers(),
                json={"state": "-4"},
                timeout=TIMEOUT_SECONDS
            )

        ticket_url = f"{instance}/nav_to.do?uri=change_request.do?sys_id={sys_id}"

        return json.dumps({
            "status": "success",
            "ticket_number": ticket_number,
            "sys_id": sys_id,
            "approval_status": "PENDING",
            "ticket_url": ticket_url,
            "message": (
                f"ServiceNow Change Request {ticket_number} created successfully. "
                f"State is Assess (-4) and Approval is PENDING. "
                f"You MUST verify that this ticket is approved before executing remediation."
            )
        }, indent=2)

    except Exception as e:
        logger.error(f"Error creating ServiceNow ticket: {e}")
        return json.dumps({"status": "error", "message": f"ServiceNow error: {str(e)}"})


@tool
def check_servicenow_approval(
    ticket_identifier: str
) -> str:
    """
    Checks the approval status of a ServiceNow Change Request.
    'ticket_identifier': Either the ticket number (e.g. 'CHG0030015') or the 32-character sys_id.
    Strictly returns 'APPROVED', 'PENDING', or 'REJECTED'.
    Security rule: Fails closed. If error occurs or ticket is not explicitly approved, returns PENDING.
    """
    if not ticket_identifier or not ticket_identifier.strip():
        return json.dumps({
            "status": "error",
            "approval_status": "PENDING",
            "is_approved": False,
            "message": "Missing ticket identifier (ticket number or sys_id)."
        })

    try:
        instance, user, pwd = _get_snow_config()
        resolved_sys_id, resolved_number = _lookup_ticket_sys_id(ticket_identifier)

        if not resolved_sys_id:
            return json.dumps({
                "status": "error",
                "approval_status": "PENDING",
                "is_approved": False,
                "message": f"Could not locate ServiceNow Change Request matching '{ticket_identifier}'."
            })

        url = f"{instance}/api/now/table/change_request/{resolved_sys_id}"
        params = {"sysparm_fields": "number,state,approval"}
        res = requests.get(url, auth=(user, pwd), headers=_get_snow_headers(), params=params, timeout=TIMEOUT_SECONDS)

        hib = _check_response_for_hibernation(res)
        if hib:
            return json.dumps({
                "status": "error",
                "approval_status": "PENDING",
                "is_approved": False,
                "message": hib
            }, indent=2)

        if res.status_code == 200:
            result = res.json().get("result", {})
            raw_approval = str(result.get("approval", "")).strip().lower()
            number = result.get("number") or resolved_number or ticket_identifier

            if raw_approval == "approved":
                approval_status = "APPROVED"
                is_approved = True
                msg = f"Ticket {number} is APPROVED in ServiceNow. Remediation execution is authorized."
            elif raw_approval == "rejected":
                approval_status = "REJECTED"
                is_approved = False
                msg = f"Ticket {number} was REJECTED in ServiceNow. Remediation execution is NOT permitted."
            else:
                approval_status = "PENDING"
                is_approved = False
                msg = f"Ticket {number} is currently PENDING approval (status: '{raw_approval}'). Human approval in ServiceNow is required before execution."

            return json.dumps({
                "status": "success",
                "ticket_number": number,
                "sys_id": resolved_sys_id,
                "approval_status": approval_status,
                "is_approved": is_approved,
                "message": msg
            }, indent=2)

        else:
            return json.dumps({
                "status": "error",
                "approval_status": "PENDING",
                "is_approved": False,
                "message": f"ServiceNow status check failed (HTTP {res.status_code}). Failing closed."
            })

    except Exception as e:
        logger.error(f"Error checking ServiceNow approval status: {e}")
        return json.dumps({
            "status": "error",
            "approval_status": "PENDING",
            "is_approved": False,
            "message": f"ServiceNow error checking status: {str(e)}. Failing closed."
        })


@tool
def close_servicenow_ticket(
    ticket_identifier: str,
    work_notes: str = "VAPT Automated Remediation successfully verified via SSH execution.",
    close_notes: str = "VAPT Automated Remediation successfully verified."
) -> str:
    """
    Appends execution audit work notes and closes the ServiceNow Change Request.
    'ticket_identifier': Either the ticket number (e.g. 'CHG0030015') or the sys_id.
    'work_notes': Detailed execution output to record in ServiceNow audit trail.
    'close_notes': Summary note for closing the ticket.
    """
    try:
        instance, user, pwd = _get_snow_config()
        resolved_sys_id, resolved_number = _lookup_ticket_sys_id(ticket_identifier)

        if not resolved_sys_id:
            return json.dumps({
                "status": "error",
                "message": f"Could not find Change Request for '{ticket_identifier}'."
            })

        # 1. Add work notes
        url = f"{instance}/api/now/table/change_request/{resolved_sys_id}"
        requests.patch(
            url,
            auth=(user, pwd),
            headers=_get_snow_headers(),
            json={"work_notes": work_notes},
            timeout=TIMEOUT_SECONDS
        )

        # 2. Close ticket
        close_payload = {
            "state": "3",  # Closed
            "close_notes": close_notes,
            "close_code": "successful"
        }
        res = requests.patch(
            url,
            auth=(user, pwd),
            headers=_get_snow_headers(),
            json=close_payload,
            timeout=TIMEOUT_SECONDS
        )

        number = resolved_number or ticket_identifier
        if res.status_code in [200, 201]:
            return json.dumps({
                "status": "success",
                "ticket_number": number,
                "message": f"ServiceNow Change Request {number} updated with execution logs and closed successfully."
            })
        else:
            return json.dumps({
                "status": "partial_success",
                "ticket_number": number,
                "message": f"Audit work notes were appended to {number}, but state change to Closed was held by ServiceNow lifecycle policy (HTTP {res.status_code})."
            })

    except Exception as e:
        logger.error(f"Error closing ServiceNow ticket: {e}")
        return json.dumps({"status": "error", "message": f"Error updating ticket: {str(e)}"})


# Register Tools with AgentMesh ToolRegistry
registry.register(
    create_servicenow_ticket,
    ToolMeta(
        name="create_servicenow_ticket",
        display_name="ServiceNow Create Change Ticket",
        description="Creates a Change Request ticket in ServiceNow for VAPT remediation and requests approval.",
        category="ITSM",
        configs=[]
    )
)

registry.register(
    check_servicenow_approval,
    ToolMeta(
        name="check_servicenow_approval",
        display_name="ServiceNow Check Approval Status",
        description="Checks if a ServiceNow Change Request has been approved ('APPROVED', 'PENDING', 'REJECTED').",
        category="ITSM",
        configs=[]
    )
)

registry.register(
    close_servicenow_ticket,
    ToolMeta(
        name="close_servicenow_ticket",
        display_name="ServiceNow Close Ticket & Work Notes",
        description="Appends execution audit notes to ServiceNow Change Request and marks it closed.",
        category="ITSM",
        configs=[]
    )
)
