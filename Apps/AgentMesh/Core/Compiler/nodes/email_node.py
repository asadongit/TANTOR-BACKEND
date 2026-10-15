"""
SMTP Email Node Builder
Sends emails via any custom SMTP server (Gmail, Outlook, etc.)
Supports custom templates (HTML/plain text) with Jinja2 interpolation.
Supports CC, BCC, custom headers, and attachments from VFS/disk.
"""

import smtplib
import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import getaddresses
from typing import Callable, Any, Dict, Tuple, List, Optional


from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Config import settings
from .utils import interpolate, get_production_data

logger = logging.getLogger(__name__)

_BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)

_ALLOWED_MOUNTS = [
    os.path.abspath(settings.TEMP_MOUNT),       # covers uploads, crops, audio, temp_images
    os.path.abspath(settings.EXPORTS_MOUNT),
    os.path.abspath(settings.DEEP_AGENTS_MOUNT),
    os.path.abspath(settings.AUDIO_MOUNT),
    os.path.abspath(settings.TEMPLATES_MOUNT),
]


@dataclass
class EmailPayload:
    """Structured payload for SMTP email transmission."""

    to: str
    cc: Optional[str]
    bcc: Optional[str]
    subject: str
    body: str
    html: bool
    smtp_server: str
    smtp_port: int
    user: str
    password: str
    attachments: List[str] = field(default_factory=list)
    headers: Optional[Dict[str, str]] = None
    timeout: int = 15


class RateLimiter:
    """Simple in-memory sliding-window rate limiter (per node)."""

    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - 60
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
        if len(self._buckets[key]) >= self.max_per_minute:
            return False
        self._buckets[key].append(now)
        return True


_rate_limiter = RateLimiter()


def _is_safe_path(path: str) -> bool:
    """Ensure the resolved attachment path is within an allowed mount or the backend root."""
    if not path:
        return False
    resolved = os.path.abspath(os.path.normpath(path))
    if not resolved.startswith(_BACKEND_ROOT):
        return False
    for mount in _ALLOWED_MOUNTS:
        if resolved.startswith(mount):
            return True
    return resolved == _BACKEND_ROOT


# Cache for OAuth access tokens: key -> (access_token, expires_at)
_oauth_token_cache: Dict[str, Tuple[str, float]] = {}


def _build_mime_message(
    sender: str,
    to: str,
    subject: str,
    body: str,
    html: bool,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> MIMEMultipart:
    """Builds a multipart MIME email message supporting standard attachments and CID inline images."""
    import re
    import mimetypes
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

    msg = MIMEMultipart("mixed" if attachments else "alternative")
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject

    if cc:
        msg["Cc"] = cc

    if headers:
        blocked = {"from", "to", "cc", "bcc", "subject", "mime-version", "content-type"}
        for k, v in headers.items():
            if k.lower() not in blocked:
                msg[k] = v

    # Scan for CID inline images in HTML body
    cid_references = []
    if html:
        cid_references = re.findall(r'src=["\']cid:([^"\']+)["\']', body, re.IGNORECASE)

    # Body portion
    body_part = MIMEMultipart("related") if cid_references else MIMEMultipart("alternative")
    subtype = "html" if html else "plain"
    body_part.attach(MIMEText(body, subtype, "utf-8"))
    msg.attach(body_part)

    # Attachments portion
    if attachments:
        for path in attachments:
            if not os.path.exists(path):
                logger.warning(f"Attachment file not found at path: {path}")
                continue
            filename = os.path.basename(path)
            try:
                with open(path, "rb") as f:
                    content_bytes = f.read()

                mime_type, _ = mimetypes.guess_type(path)
                mime_type = mime_type or "application/octet-stream"
                main_type, sub_type = mime_type.split("/", 1)

                part = MIMEBase(main_type, sub_type)
                part.set_payload(content_bytes)
                encoders.encode_base64(part)

                # Check for CID inline referencing
                if filename in cid_references:
                    part.add_header("Content-ID", f"<{filename}>")
                    part.add_header("Content-Disposition", f"inline; filename={filename}")
                else:
                    part.add_header("Content-Disposition", f"attachment; filename={filename}")

                msg.attach(part)
            except Exception as e:
                logger.warning(f"Failed to attach file {path}: {e}")

    return msg


def _send_email_sync(payload: EmailPayload) -> None:
    """Synchronous function to connect and send mail via SMTP."""
    msg = _build_mime_message(
        sender=payload.user,
        to=payload.to,
        subject=payload.subject,
        body=payload.body,
        html=payload.html,
        cc=payload.cc,
        bcc=payload.bcc,
        attachments=payload.attachments,
        headers=payload.headers
    )

    def _extract_clean_emails(raw_str: str) -> List[str]:
        cleaned = []
        for x in raw_str.split(","):
            x = x.strip()
            if not x:
                continue
            parsed = getaddresses([x])
            if parsed and parsed[0][1].strip():
                cleaned.append(parsed[0][1].strip())
            else:
                cleaned.append(x)
        return cleaned

    recipients = _extract_clean_emails(payload.to)
    if payload.cc:
        recipients.extend(_extract_clean_emails(payload.cc))
    if payload.bcc:
        recipients.extend(_extract_clean_emails(payload.bcc))

    if payload.smtp_port == 465:
        server = smtplib.SMTP_SSL(
            payload.smtp_server, payload.smtp_port, timeout=payload.timeout
        )
    else:
        server = smtplib.SMTP(
            payload.smtp_server, payload.smtp_port, timeout=payload.timeout
        )
        server.starttls()

    with server:
        server.login(payload.user, payload.password)
        server.send_message(msg, from_addr=payload.user, to_addrs=recipients)


async def _attempt_email_send(payload: EmailPayload) -> Tuple[bool, str]:
    """Handles the retry loop for SMTP email transmission."""
    retries = 3
    for attempt in range(retries):
        try:
            logger.info(
                f"Attempting SMTP send to {payload.to} via "
                f"{payload.smtp_server}:{payload.smtp_port} (Attempt {attempt + 1})"
            )
            await asyncio.to_thread(_send_email_sync, payload)
            logger.info(f"Email sent successfully to {payload.to}")
            return True, ""
        except smtplib.SMTPAuthenticationError as e:
            msg = f"SMTP Authentication failed. Verify credentials. Error: {str(e)}"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"SMTP transmission failed: {str(e)}"
            logger.warning(f"Failed to send email on attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                logger.exception(
                    f"Final failure sending email to {payload.to}: {str(e)}"
                )
    return False, ""


async def _send_email_oauth_async(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    sender: str,
    to: str,
    subject: str,
    body: str,
    html: bool,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Sends email via Microsoft Graph API using Client Credentials flow (OAuth)."""
    import base64
    import mimetypes
    import httpx
    import time
    import re

    cache_key = f"ms_{client_id}_{tenant_id}"
    now = time.time()
    access_token = None

    if cache_key in _oauth_token_cache:
        token, expires_at = _oauth_token_cache[cache_key]
        if expires_at > now + 60:
            access_token = token

    if not access_token:
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                token_resp = await client.post(token_url, data=token_data)
                if token_resp.status_code != 200:
                    return False, f"Failed to get Microsoft OAuth token: {token_resp.text}"
                token_json = token_resp.json()
                access_token = token_json.get("access_token")
                expires_in = token_json.get("expires_in", 3600)
                if not access_token:
                    return False, "OAuth token missing in Microsoft response"
                _oauth_token_cache[cache_key] = (access_token, now + expires_in)
        except Exception as e:
            return False, f"Microsoft OAuth handshake failed: {e}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            def _parse_recipients(emails_str: Optional[str]) -> List[Dict[str, Any]]:
                if not emails_str:
                    return []
                res = []
                for email in emails_str.split(","):
                    email = email.strip()
                    if email:
                        res.append({"emailAddress": {"address": email}})
                return res

            to_recipients = _parse_recipients(to)
            cc_recipients = _parse_recipients(cc)
            bcc_recipients = _parse_recipients(bcc)

            if not to_recipients:
                return False, "No recipients specified"

            # Scan for CID inline images in HTML body
            cid_references = []
            if html:
                cid_references = re.findall(r'src=["\']cid:([^"\']+)["\']', body, re.IGNORECASE)

            attachments_payload = []
            if attachments:
                for path in attachments:
                    if not os.path.exists(path):
                        continue
                    try:
                        def _read_bytes(p):
                            with open(p, "rb") as f:
                                return f.read()
                        content_bytes = await asyncio.to_thread(_read_bytes, path)
                        content_b64 = base64.b64encode(content_bytes).decode("utf-8")
                        mime_type, _ = mimetypes.guess_type(path)
                        filename = os.path.basename(path)
                        
                        att_data = {
                            "@odata.type": "#microsoft.graph.fileAttachment",
                            "name": filename,
                            "contentType": mime_type or "application/octet-stream",
                            "contentBytes": content_b64
                        }
                        
                        if filename in cid_references:
                            att_data["isInline"] = True
                            att_data["contentId"] = filename
                            
                        attachments_payload.append(att_data)
                    except Exception as e:
                        logger.warning(f"Failed to process attachment {path} for MS Graph OAuth: {e}")

            mail_payload = {
                "message": {
                    "subject": subject,
                    "body": {
                        "contentType": "HTML" if html else "Text",
                        "content": body
                    },
                    "toRecipients": to_recipients,
                },
                "saveToSentItems": "false"
            }

            if cc_recipients:
                mail_payload["message"]["ccRecipients"] = cc_recipients
            if bcc_recipients:
                mail_payload["message"]["bccRecipients"] = bcc_recipients
            if attachments_payload:
                mail_payload["message"]["attachments"] = attachments_payload

            send_url = f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            send_resp = await client.post(send_url, json=mail_payload, headers=headers)
            if send_resp.status_code not in (200, 202):
                return False, f"Graph API sendMail failed: {send_resp.status_code} - {send_resp.text}"

            return True, ""
            
    except Exception as e:
        logger.exception("Unexpected error in MS Graph OAuth mail transmission")
        return False, str(e)


async def _send_email_gmail_oauth_async(
    service_account_json: str,
    delegated_user: str,
    to: str,
    subject: str,
    body: str,
    html: bool,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Sends email via Google Workspace (Gmail API) using Service Account OAuth JWT Delegation."""
    import base64
    import json
    import httpx
    import time
    import google.auth.jwt
    import google.auth.crypt
    
    try:
        sa_info = json.loads(service_account_json)
        client_email = sa_info.get("client_email", "")
    except Exception as e:
        return False, f"Invalid Google Service Account JSON credentials: {e}"
        
    cache_key = f"google_{client_email}_{delegated_user}"
    now = int(time.time())
    access_token = None
    
    if cache_key in _oauth_token_cache:
        token, expires_at = _oauth_token_cache[cache_key]
        if expires_at > now + 60:
            access_token = token
            
    if not access_token:
        try:
            signer = google.auth.crypt.RSASigner.from_service_account_info(sa_info)
            claim_set = {
                "iss": client_email,
                "sub": delegated_user,
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600,
                "scope": "https://www.googleapis.com/auth/gmail.send"
            }
            jwt_assertion = google.auth.jwt.encode(signer, claim_set)
            if isinstance(jwt_assertion, bytes):
                jwt_assertion = jwt_assertion.decode("utf-8")
                
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt_assertion
            }
            
            async with httpx.AsyncClient(timeout=15) as client:
                token_resp = await client.post(token_url, data=token_data)
                if token_resp.status_code != 200:
                    return False, f"Failed to get Google OAuth token: {token_resp.text}"
                token_json = token_resp.json()
                access_token = token_json.get("access_token")
                expires_in = token_json.get("expires_in", 3600)
                if not access_token:
                    return False, "Google OAuth token missing in response"
                _oauth_token_cache[cache_key] = (access_token, now + expires_in)
        except Exception as e:
            return False, f"Google OAuth handshake failed: {e}"
            
    try:
        msg = _build_mime_message(
            sender=delegated_user,
            to=to,
            subject=subject,
            body=body,
            html=html,
            cc=cc,
            bcc=bcc,
            attachments=attachments
        )
        
        raw_bytes = msg.as_bytes()
        raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode("utf-8")
        
        send_url = f"https://gmail.googleapis.com/gmail/v1/users/{delegated_user}/messages/send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            send_resp = await client.post(send_url, json={"raw": raw_b64}, headers=headers)
            if send_resp.status_code != 200:
                return False, f"Gmail API send failed: {send_resp.status_code} - {send_resp.text}"
                
            return True, ""
    except Exception as e:
        logger.exception("Unexpected error in Google Workspace OAuth mail transmission")
        return False, str(e)


def build_email_node(node_config: NodeConfig) -> Callable:
    """Builds a custom SMTP/OAuth email execution node."""

    async def execute_email(state: AgentState):
        context = state.get("context", {}) or {}
        vfs = state.get("vfs", {}) or {}
        vfs_files = vfs.get("files", {}) or {}

        node_id = node_config.id

        # Rate limit check
        if not _rate_limiter.allow(node_id):
            msg = (
                f"Email Node {node_id}: Rate limit exceeded "
                f"(max {_rate_limiter.max_per_minute} emails/minute). "
                "Try adding a Wait node before this node."
            )
            logger.error(msg)
            return {
                "messages": [
                    AIMessage(content=msg, response_metadata={"active_agent": node_id})
                ],
                "active_agent": node_id,
                "context": context,
            }

        # Resolve production-grade variables (fully parsed scope with SmartInput and context/data keys)
        scope = get_production_data(state)

        email_to = interpolate(node_config.email_to or "", scope)
        email_cc = interpolate(node_config.email_cc or "", scope) or None
        email_bcc = interpolate(node_config.email_bcc or "", scope) or None
        email_subject = interpolate(node_config.email_subject or "", scope)
        email_body = interpolate(node_config.email_body or "", scope)
        email_html = getattr(node_config, "email_html", False)

        attachments_list: List[str] = []
        attachments_raw = interpolate(node_config.email_attachments or "", scope)
        raw_paths = []
        if attachments_raw:
            attachments_raw = attachments_raw.strip()
            # If it's a list representation from state (like ['a.pdf', 'b.pdf'])
            if (attachments_raw.startswith("[") and attachments_raw.endswith("]")) or (attachments_raw.startswith("(") and attachments_raw.endswith(")")):
                try:
                    import ast
                    parsed = ast.literal_eval(attachments_raw)
                    if isinstance(parsed, (list, tuple)):
                        raw_paths = [str(x).strip() for x in parsed]
                except Exception:
                    raw_paths = [p.strip() for p in attachments_raw.split(",") if p.strip()]
            else:
                raw_paths = [p.strip() for p in attachments_raw.split(",") if p.strip()]

        for raw_p in raw_paths:
            resolved = vfs_files.get(raw_p, raw_p)
            if not _is_safe_path(resolved):
                logger.warning(
                    f"Email Node {node_id}: Blocked unsafe attachment path '{resolved}'"
                )
                continue
            attachments_list.append(resolved)

        email_auth_method = getattr(node_config, "email_auth_method", "smtp") or "smtp"
        logger.info(f"Email Node {node_id}: Preparing to send email (method={email_auth_method}) to {email_to}")

        if not email_to:
            status_msg = "Error: Recipient email address (email_to) is empty."
            logger.error(status_msg)
            return {
                "messages": [
                    AIMessage(
                        content=status_msg, response_metadata={"active_agent": node_id}
                    )
                ],
                "active_agent": node_id,
                "context": context,
            }

        if email_auth_method == "oauth":
            email_tenant_id = getattr(node_config, "email_tenant_id", "")
            email_client_id = getattr(node_config, "email_client_id", "")
            email_client_secret = getattr(node_config, "email_client_secret", "")
            email_user = node_config.email_user

            if not email_tenant_id or not email_client_id or not email_client_secret or not email_user:
                status_msg = (
                    "Error: Microsoft Graph OAuth configuration is incomplete. "
                    "Please set Tenant ID, Client ID, Client Secret, and Sender Email in node settings."
                )
                logger.error(status_msg)
                return {
                    "messages": [
                        AIMessage(
                            content=status_msg, response_metadata={"active_agent": node_id}
                        )
                    ],
                    "active_agent": node_id,
                    "context": context,
                }

            sent_success, error_details = await _send_email_oauth_async(
                tenant_id=email_tenant_id,
                client_id=email_client_id,
                client_secret=email_client_secret,
                sender=email_user,
                to=email_to,
                subject=email_subject,
                body=email_body,
                html=email_html,
                cc=email_cc,
                bcc=email_bcc,
                attachments=attachments_list
            )
        elif email_auth_method == "gmail_oauth":
            email_google_service_account_json = getattr(node_config, "email_google_service_account_json", "")
            email_google_delegated_user = getattr(node_config, "email_google_delegated_user", "")

            if not email_google_service_account_json or not email_google_delegated_user:
                status_msg = (
                    "Error: Google OAuth configuration is incomplete. "
                    "Please set Google Service Account JSON Key and Sender Email in node settings."
                )
                logger.error(status_msg)
                return {
                    "messages": [
                        AIMessage(
                            content=status_msg, response_metadata={"active_agent": node_id}
                        )
                    ],
                    "active_agent": node_id,
                    "context": context,
                }

            sent_success, error_details = await _send_email_gmail_oauth_async(
                service_account_json=email_google_service_account_json,
                delegated_user=email_google_delegated_user,
                to=email_to,
                subject=email_subject,
                body=email_body,
                html=email_html,
                cc=email_cc,
                bcc=email_bcc,
                attachments=attachments_list
            )
        else:
            smtp_server = node_config.email_smtp_server or "smtp.gmail.com"
            smtp_port = node_config.email_smtp_port or 587
            email_user = node_config.email_user
            email_password = node_config.email_password
            email_timeout = getattr(node_config, "email_smtp_timeout", 15) or 15

            if not email_user or not email_password:
                status_msg = (
                    "Error: SMTP configuration (email_user/email_password) is incomplete. "
                    "Please set them in node settings."
                )
                logger.error(status_msg)
                return {
                    "messages": [
                        AIMessage(
                            content=status_msg, response_metadata={"active_agent": node_id}
                        )
                    ],
                    "active_agent": node_id,
                    "context": context,
                }

            payload = EmailPayload(
                to=email_to,
                cc=email_cc,
                bcc=email_bcc,
                subject=email_subject,
                body=email_body,
                html=email_html,
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                user=email_user,
                password=email_password,
                attachments=attachments_list,
                headers=node_config.email_headers,
                timeout=email_timeout,
            )

            sent_success, error_details = await _attempt_email_send(payload)

        if not sent_success:
            raise RuntimeError(f"Failed to send email: {error_details}")

        status_msg = f"Email sent successfully to {email_to}"
        email_result = {"status": "success", "to": email_to, "subject": email_subject}

        return {
            "messages": [
                AIMessage(
                    content=status_msg, response_metadata={"active_agent": node_id}
                )
            ],
            "active_agent": node_id,
            "context": {**context, "latest_email_result": email_result},
        }

    return RunnableLambda(execute_email).with_retry(stop_after_attempt=3)
