import smtplib
import time
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

class GmailInput(BaseModel):
    """Input schema for sending a Gmail message."""
    to: str = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject of the email.")
    body: str = Field(..., description="The main content (body) of the email.")
    gmail_user: str = Field(..., description="The Gmail address to send from (e.g. user@gmail.com)")
    gmail_password: str = Field(
        ..., description="The 16-character Gmail App Password", repr=False
    )

@tool("send_gmail", args_schema=GmailInput)
async def send_gmail(to: str, subject: str, body: str, gmail_user: str, gmail_password: str) -> str:
    """
    Sends an email using Gmail's SMTP server. 
    The credentials (user/password) are usually provided via UI configuration (frozen).
    """
    if not gmail_user or not gmail_password:
        return "Error: Gmail credentials (user/password) are missing. Please configure them in the node settings."

    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    retries = 3

    # 1. Build the message
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # 2. Robust Execution with Retries
    for attempt in range(retries):
        try:
            logger.info(f"Attempting to send email to {to} from {gmail_user} (Attempt {attempt + 1})")
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.starttls()  # Secure the connection
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
                
            logger.info(f"Email sent successfully to {to}")
            return f"Successfully sent email to {to} with subject: '{subject}'"

        except smtplib.SMTPAuthenticationError:
            logger.error("Authentication failed. Check your App Password.")
            return "Error: Authentication failed. Please verify your Gmail App Password."
        except Exception as e:
            logger.warning(f"Failed to send email on attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"Final failure sending email to {to}: {str(e)}")
                return f"Error: Failed to send email after {retries} attempts. Reason: {str(e)}"
    
    return "Error: Unknown failure in Gmail tool execution."

# 3. Register with AgentMesh Tool Registry
# We use ai_visible_fields to HIDE the credentials from the LLM.
# This forces the user to provide them manually in the 'Params' section of the node in the UI.
registry.register(
    send_gmail,
    ToolMeta(
        name="send_gmail",
        display_name="Gmail Sender",
        description="Send an email via Gmail SMTP. Configure credentials manually in the node settings.",
        category="Communication",
        ai_visible_fields=["to", "subject", "body"],  # ONLY these are seen by the LLM
        configs=[
            {"name": "gmail_user", "type": "string", "description": "Your Gmail address", "required": True},
            {"name": "gmail_password", "type": "string", "description": "Your 16-character App Password", "required": True}
        ]
    )
)
