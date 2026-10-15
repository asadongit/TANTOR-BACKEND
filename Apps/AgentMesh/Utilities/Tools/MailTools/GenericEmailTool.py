import smtplib
import time
import logging
import base64
import json
import httpx
import asyncio
import re
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import getaddresses
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# Cache for OAuth access tokens: key -> (access_token, expires_at)
_oauth_token_cache: Dict[str, Tuple[str, float]] = {}


class SMTPInput(BaseModel):
    """Input schema for sending an email with SMTP or OAuth support."""
    to: str = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject of the email.")
    body: str = Field(..., description="The main content (body) of the email.")
    
    # Configuration fields (configured via UI configs)
    auth_method: str = Field("smtp", description="Authentication method: smtp, oauth, or gmail_oauth")
    smtp_server: Optional[str] = Field("smtp.gmail.com", description="SMTP server hostname (for smtp)")
    smtp_port: int = Field(587, description="SMTP server port (usually 587 for TLS, 465 for SSL)")
    user: Optional[str] = Field(None, description="Authenticating username/email (for smtp/oauth)")
    password: Optional[str] = Field(
        None, description="Authenticating password or app/secret (for smtp)", repr=False
    )
    
    # Microsoft Graph OAuth
    tenant_id: Optional[str] = Field(None, description="Azure AD / Entra ID Tenant ID")
    client_id: Optional[str] = Field(None, description="Azure App Registration Client ID")
    client_secret: Optional[str] = Field(
        None, description="Azure App Client Secret", repr=False
    )
    
    # Google Gmail OAuth
    google_service_account_json: Optional[str] = Field(
        None, description="Google Cloud Service Account JSON credentials", repr=False
    )
    google_delegated_user: Optional[str] = Field(None, description="Workspace user email to impersonate")


def _build_mime_message(
    sender: str,
    to: str,
    subject: str,
    body: str,
    html: bool = False,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> MIMEMultipart:
    """Builds a multipart MIME email message supporting standard attachments and CID inline images."""
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
            import os
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


def _send_email_sync(
    to: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    user: str,
    password: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Synchronous function to connect and send mail via SMTP."""
    msg = _build_mime_message(
        sender=user,
        to=to,
        subject=subject,
        body=body,
        html=False,
        cc=cc,
        bcc=bcc,
        attachments=attachments,
        headers=headers
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

    recipients = _extract_clean_emails(to)
    if cc:
        recipients.extend(_extract_clean_emails(cc))
    if bcc:
        recipients.extend(_extract_clean_emails(bcc))

    if smtp_port == 465:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15)
    else:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
        server.starttls()

    with server:
        server.login(user, password)
        server.send_message(msg, from_addr=user, to_addrs=recipients)


async def _attempt_email_send(
    to: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    user: str,
    password: str
) -> Tuple[bool, str]:
    """Handles the retry loop for SMTP email transmission."""
    retries = 3
    for attempt in range(retries):
        try:
            logger.info(
                f"Attempting SMTP send to {to} via "
                f"{smtp_server}:{smtp_port} (Attempt {attempt + 1})"
            )
            await asyncio.to_thread(_send_email_sync, to, subject, body, smtp_server, smtp_port, user, password)
            logger.info(f"Email sent successfully to {to}")
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
                    f"Final failure sending email to {to}: {str(e)}"
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
) -> Tuple[bool, str]:
    """Sends email via Microsoft Graph API using Client Credentials flow (OAuth)."""
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
            if not to_recipients:
                return False, "No recipients specified"

            mail_payload = {
                "message": {
                    "subject": subject,
                    "body": {
                        "contentType": "Text",
                        "content": body
                    },
                    "toRecipients": to_recipients,
                },
                "saveToSentItems": "false"
            }

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
) -> Tuple[bool, str]:
    """Sends email via Google Workspace (Gmail API) using Service Account OAuth JWT Delegation."""
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
            html=False
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


@tool("email", args_schema=SMTPInput)
async def email(
    to: str,
    subject: str,
    body: str,
    smtp_server: Optional[str] = "smtp.gmail.com",
    smtp_port: int = 587,
    user: Optional[str] = None,
    password: Optional[str] = None,
    auth_method: str = "smtp",
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    google_service_account_json: Optional[str] = None,
    google_delegated_user: Optional[str] = None,
) -> str:
    """
    Sends an email using SMTP or OAuth (Microsoft Graph, Google Workspace).
    All configuration should be provided via UI 'Params'.
    """
    auth_method = auth_method.lower()

    if auth_method == "oauth":
        if not tenant_id or not client_id or not client_secret or not user:
            return "Error: Microsoft Graph OAuth configuration is incomplete. Provide tenant_id, client_id, client_secret, and user email."
        success, error_details = await _send_email_oauth_async(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            sender=user,
            to=to,
            subject=subject,
            body=body
        )
    elif auth_method == "gmail_oauth":
        if not google_service_account_json or not google_delegated_user:
            return "Error: Google Workspace OAuth configuration is incomplete. Provide google_service_account_json and google_delegated_user."
        success, error_details = await _send_email_gmail_oauth_async(
            service_account_json=google_service_account_json,
            delegated_user=google_delegated_user,
            to=to,
            subject=subject,
            body=body
        )
    else:
        # Standard SMTP
        if not smtp_server or not user or not password:
            return "Error: SMTP configuration (server/user/password) is incomplete. Please set them in node settings."
        success, error_details = await _attempt_email_send(
            to=to,
            subject=subject,
            body=body,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            user=user,
            password=password
        )

    if success:
        return f"Successfully sent email to {to} using {auth_method}."
    else:
        return f"Error: Failed to send email via {auth_method}. Reason: {error_details}"


# 3. Registry Integration
registry.register(
    email,
    ToolMeta(
        name="email",
        display_name="Email Sender",
        description="Send an email via SMTP or OAuth (Microsoft Graph, Google Workspace).",
        category="Communication",
        ai_visible_fields=["to", "subject", "body"],  # Hidden from AI (frozen by UI)
        configs=[
            {
                "name": "auth_method",
                "type": "string",
                "description": "Authentication method",
                "required": True,
                "options": ["smtp", "oauth", "gmail_oauth"],
                "default": "smtp"
            },
            {
                "name": "smtp_server", 
                "type": "string", 
                "description": "The mail server hostname (SMTP only)", 
                "required": False,
                "options": ["smtp.gmail.com", "smtp.office365.com", "smtp.sendgrid.net", "smtp.mailgun.org"],
                "default": "smtp.gmail.com",
                "visible_if": {"field": "auth_method", "value": "smtp"}
            },
            {
                "name": "smtp_port", 
                "type": "number", 
                "description": "587 (TLS) or 465 (SSL) (SMTP only)", 
                "required": False,
                "options": ["587", "465"],
                "default": 587,
                "visible_if": {"field": "auth_method", "value": "smtp"}
            },
            {
                "name": "user",
                "type": "string",
                "description": "User / Email address (SMTP/Microsoft OAuth)",
                "required": False,
                "visible_if": {"field": "auth_method", "values": ["smtp", "oauth"]}
            },
            {
                "name": "password",
                "type": "password",
                "description": "Password or App Password (SMTP only)",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "smtp"}
            },
            {
                "name": "tenant_id",
                "type": "string",
                "description": "Microsoft Tenant ID",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "oauth"}
            },
            {
                "name": "client_id",
                "type": "string",
                "description": "Microsoft Client ID",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "oauth"}
            },
            {
                "name": "client_secret",
                "type": "password",
                "description": "Microsoft Client Secret",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "oauth"}
            },
            {
                "name": "google_service_account_json",
                "type": "textarea",
                "description": "Google Service Account JSON Key",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "gmail_oauth"}
            },
            {
                "name": "google_delegated_user",
                "type": "string",
                "description": "Google Sender Email (Impersonated User)",
                "required": False,
                "visible_if": {"field": "auth_method", "value": "gmail_oauth"}
            }
        ]
    )
)
