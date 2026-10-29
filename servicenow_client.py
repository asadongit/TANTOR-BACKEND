import os
import logging
from typing import Optional, Tuple, Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15

# In-memory cache for assignment group IDs: { group_name: sys_id }
_GROUP_CACHE: Dict[str, str] = {}


def _get_config() -> Tuple[str, str, str]:
    """
    Retrieve and validate ServiceNow credentials from environment variables.
    Fails safely without exposing passwords.
    """
    instance = os.getenv("SNOW_INSTANCE", "").strip().rstrip("/")
    user = os.getenv("SNOW_USER", "").strip()
    pwd = os.getenv("SNOW_PASS", "").strip()

    if not instance or not user or not pwd:
        raise ValueError(
            "ServiceNow configuration missing. Please ensure SNOW_INSTANCE, "
            "SNOW_USER, and SNOW_PASS are set in your .env file."
        )

    return instance, user, pwd


def _get_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


def get_assignment_group_id(group_name: str = "Software") -> Optional[str]:
    """
    Query ServiceNow for the sys_id of an assignment group by name.
    Caches the result in-memory to prevent repeated API calls.
    """
    if not group_name:
        return None

    if group_name in _GROUP_CACHE:
        return _GROUP_CACHE[group_name]

    try:
        instance, user, pwd = _get_config()
        url = f"{instance}/api/now/table/sys_user_group"
        params = {
            "sysparm_query": f"name={group_name}",
            "sysparm_fields": "name,sys_id",
            "sysparm_limit": 1
        }
        response = requests.get(
            url,
            auth=(user, pwd),
            headers=_get_headers(),
            params=params,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code == 200:
            results = response.json().get("result", [])
            if results and isinstance(results, list):
                sys_id = results[0].get("sys_id")
                if sys_id:
                    _GROUP_CACHE[group_name] = sys_id
                    return sys_id
            logger.warning(f"Assignment group '{group_name}' not found in ServiceNow.")
            return None
        else:
            logger.error(f"Failed to fetch assignment group '{group_name}': HTTP {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching assignment group '{group_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching assignment group '{group_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_assignment_group_id: {e}")
        return None


def create_change_ticket(
    title: str,
    details: str,
    assignment_group_name: str = "Software",
    auto_request_approval: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Create a new normal change_request in ServiceNow.
    Automatically assigns the assignment group (e.g. Software) and transitions
    state to Assess (-4) to trigger approval workflow if requested.

    Returns:
        (ticket_number, sys_id, error_message)
    """
    try:
        instance, user, pwd = _get_config()
        group_id = get_assignment_group_id(assignment_group_name)

        payload: Dict[str, Any] = {
            "short_description": f"VAPT Remediation: {title}",
            "description": details,
            "type": "normal",
            "category": "Software",
            "risk": "3",
            "impact": "2"
        }

        if group_id:
            payload["assignment_group"] = group_id

        url = f"{instance}/api/now/table/change_request"
        response = requests.post(
            url,
            auth=(user, pwd),
            headers=_get_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code in [200, 201]:
            data = response.json().get("result", {})
            ticket_number = data.get("number")
            sys_id = data.get("sys_id")

            # Move state to Assess (-4) which transitions approval to 'requested'
            if auto_request_approval and sys_id:
                request_approval(sys_id)

            return ticket_number, sys_id, None
        else:
            err_msg = f"Failed to create ServiceNow ticket: HTTP {response.status_code} - {response.text}"
            logger.error(err_msg)
            return None, None, f"ServiceNow ticket creation failed (HTTP {response.status_code})."

    except requests.exceptions.Timeout:
        return None, None, "ServiceNow request timed out after 15 seconds."
    except requests.exceptions.RequestException as e:
        return None, None, f"Unable to reach ServiceNow: {e}"
    except Exception as e:
        return None, None, f"ServiceNow error: {e}"


def request_approval(sys_id: str) -> Tuple[bool, str]:
    """
    Trigger approval request through API by patching state to Assess (-4).
    In ServiceNow Normal Change lifecycle, transitioning to Assess (-4) with an
    assignment_group set automatically moves the approval field to 'requested'.

    Returns:
        (success: bool, message: str)
    """
    if not sys_id:
        return False, "sys_id is required."

    try:
        instance, user, pwd = _get_config()
        url = f"{instance}/api/now/table/change_request/{sys_id}"
        # Setting state to -4 (Assess)
        payload = {"state": "-4"}
        response = requests.patch(
            url,
            auth=(user, pwd),
            headers=_get_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code in [200, 201]:
            result = response.json().get("result", {})
            approval_state = result.get("approval", "")
            return True, f"Ticket moved to Assess. Approval status: {approval_state}"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return False, "Request timed out while requesting approval."
    except requests.exceptions.RequestException as e:
        return False, f"Network error while requesting approval: {e}"
    except Exception as e:
        return False, f"Error requesting approval: {e}"


def get_approval_status(sys_id: str, auth_override: Optional[Tuple[str, str]] = None) -> str:
    """
    Check the approval field of a ServiceNow change request.
    Strictly returns one of:
      - 'APPROVED' : when approval field is exactly 'approved'
      - 'REJECTED' : when approval field is exactly 'rejected'
      - 'PENDING'  : for 'requested', 'not requested', API errors, timeouts, 401s, or unknown values.

    SAFETY RULE: Under NO circumstances does this function return 'APPROVED' on error.
    Fail-closed design.
    """
    if not sys_id:
        logger.error("get_approval_status called without sys_id.")
        return "PENDING"

    try:
        if auth_override:
            instance, _, _ = _get_config()
            auth = auth_override
        else:
            instance, user, pwd = _get_config()
            auth = (user, pwd)

        url = f"{instance}/api/now/table/change_request/{sys_id}"
        params = {"sysparm_fields": "number,state,approval"}

        response = requests.get(
            url,
            auth=auth,
            headers=_get_headers(),
            params=params,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code == 200:
            result = response.json().get("result", {})
            raw_approval = str(result.get("approval", "")).strip().lower()

            if raw_approval == "approved":
                return "APPROVED"
            elif raw_approval == "rejected":
                return "REJECTED"
            else:
                # 'requested', 'not requested', or any other status
                return "PENDING"
        elif response.status_code == 401:
            logger.error("ServiceNow authentication failed (HTTP 401 Unauthorized).")
            return "PENDING"
        else:
            logger.error(f"ServiceNow status check failed with HTTP {response.status_code}.")
            return "PENDING"

    except requests.exceptions.Timeout:
        logger.error(f"ServiceNow approval status check timed out after {TIMEOUT_SECONDS}s.")
        return "PENDING"
    except requests.exceptions.RequestException as e:
        logger.error(f"ServiceNow request network exception: {e}")
        return "PENDING"
    except Exception as e:
        logger.error(f"Unexpected error in get_approval_status: {e}")
        return "PENDING"


def add_work_note(sys_id: str, note_text: str) -> Tuple[bool, str]:
    """
    Appends an audit work note to the change request via PATCH on the 'work_notes' field.
    """
    if not sys_id or not note_text:
        return False, "Missing sys_id or note text."

    try:
        instance, user, pwd = _get_config()
        url = f"{instance}/api/now/table/change_request/{sys_id}"
        payload = {"work_notes": note_text}

        response = requests.patch(
            url,
            auth=(user, pwd),
            headers=_get_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code in [200, 201]:
            return True, "Work note successfully added."
        else:
            return False, f"Failed to add work note (HTTP {response.status_code}): {response.text}"

    except requests.exceptions.Timeout:
        return False, "Timed out adding work note to ServiceNow."
    except requests.exceptions.RequestException as e:
        return False, f"Network error adding work note: {e}"
    except Exception as e:
        return False, f"Error adding work note: {e}"


def close_change_ticket(
    sys_id: str,
    close_notes: str = "VAPT Automated Remediation successfully verified.",
    close_code: str = "successful"
) -> Tuple[bool, str]:
    """
    Attempts to close a ServiceNow change ticket.
    Note: ServiceNow change models typically enforce state transition business rules.
    Normal changes cannot transition directly from New/Assess to Closed without
    advancing through Authorize -> Scheduled -> Implement -> Review.
    """
    if not sys_id:
        return False, "sys_id is required."

    try:
        instance, user, pwd = _get_config()
        url = f"{instance}/api/now/table/change_request/{sys_id}"
        payload = {
            "state": "3",  # 3 = Closed
            "close_notes": close_notes,
            "close_code": close_code
        }

        response = requests.patch(
            url,
            auth=(user, pwd),
            headers=_get_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code in [200, 201]:
            return True, "Ticket successfully closed."
        else:
            # Parse detail message from ServiceNow business rule error
            try:
                err_data = response.json()
                detail = err_data.get("error", {}).get("detail", response.text)
                return False, f"ServiceNow rejected direct close (HTTP {response.status_code}): {detail}"
            except Exception:
                return False, f"HTTP {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return False, "Timed out attempting to close ticket."
    except requests.exceptions.RequestException as e:
        return False, f"Network error closing ticket: {e}"
    except Exception as e:
        return False, f"Error closing ticket: {e}"
