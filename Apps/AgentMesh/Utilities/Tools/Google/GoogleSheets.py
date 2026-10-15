import requests
import pandas as pd
import io
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Utilities.Tools.Google.google_auth_utils import GoogleSheetsProvider, save_gsheet_config

logger = logging.getLogger(__name__)

# --- Standard Metadata ---
PKG_NAME = "Google Workspace"
CATEGORY = "Google"

# --- Helper Functions ---

def extract_spreadsheet_id(url_or_id: str) -> str:
    if "docs.google.com/spreadsheets/d/" in url_or_id:
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", url_or_id)
        if match:
            return match.group(1)
    return url_or_id

def fetch_sheet_as_df(spreadsheet_id: str, tab_id: str = "0") -> pd.DataFrame:
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={tab_id}"
    response = requests.get(export_url, timeout=20)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

# --- Input Schemas ---

class GSheetConnectInput(BaseModel):
    sheet_link: str = Field(..., description="The shareable URL of the Google Sheet (Ensure 'Anyone with the link can view' is ON).")
    tab_id: str = Field(default="0", description="Optional: The ID of the tab (found as 'gid' in URL). Default is '0' (first tab).")

class GSheetReadInput(BaseModel):
    override_link: Optional[str] = Field(None, description="Optional: A temporary sheet link to read instead of the connected one.")
    max_rows: int = Field(default=10, description="How many rows to read. Use a small number (5-10) for a quick look, or a larger number (up to 100) for deep analysis.")

class GSheetSearchInput(BaseModel):
    query: str = Field(..., description="The search term or question about the data.")

class GSheetQueryInput(BaseModel):
    query: str = Field(..., description="The SQL-like query (e.g. 'SELECT * WHERE C > 1000'). Note: Use column letters like A, B, C.")

# --- Tool Implementations ---

@tool("google_sheets_query", args_schema=GSheetQueryInput)
async def google_sheets_query(query: str, config: RunnableConfig) -> str:
    """
    Runs a SQL-like query on the connected Google Sheet using Google's Visualization API.
    Use this for large datasets (10k+ rows) or when you need math (Greater than, Less than).
    Example query: "SELECT * WHERE G > 100000" (to find salary > 100k).
    """
    try:
        thread_id = GoogleSheetsProvider.get_thread_id(config)
        gs_config = await GoogleSheetsProvider.get_config(thread_id)
        
        if not gs_config or not gs_config.get("sheet_link"):
            return "ERROR: No Google Sheet connected. Use 'google_sheets_connect' first."

        spreadsheet_id = extract_spreadsheet_id(gs_config["sheet_link"])
        tab = gs_config.get("tab_id", "0")
        
        # Build GViz Query URL
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tq={encoded_query}&gid={tab}"
        
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        # Parse the 'google.visualization.Query.setResponse({...})' format
        import json
        import re
        match = re.search(r'google\.visualization\.Query\.setResponse\((.*)\);', response.text, re.DOTALL)
        if not match:
            return "Error: Could not parse Google's response. Check your query syntax."
            
        data = json.loads(match.group(1))
        if data.get('status') == 'error':
            return f"Google Query Error: {str(data.get('errors'))}"
            
        # Extract columns and rows into a clean format
        table = data['table']
        cols = [c['label'] or f"Col_{i}" for i, c in enumerate(table['cols'])]
        rows = []
        for r in table['rows']:
            row_data = {}
            for i, cell in enumerate(r['c']):
                val = cell['v'] if cell else None
                row_data[cols[i]] = val
            rows.append(row_data)
            
        return f"Query Success. Found {len(rows)} matching rows. Result: {str(rows[:50])}"
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return f"Error executing query: {str(e)}"

@tool("google_sheets_connect", args_schema=GSheetConnectInput)
async def google_sheets_connect(sheet_link: str, config: RunnableConfig, tab_id: str = "0") -> str:
    """
    Connects a specific Google Sheet to the current conversation thread.
    This tool stores the link so that subsequent 'read' and 'query' tools know which sheet to use.
    The sheet must be shared as 'Anyone with the link can view'.
    """
    try:
        thread_id = GoogleSheetsProvider.get_thread_id(config)
        await save_gsheet_config(thread_id, {"sheet_link": sheet_link, "tab_id": tab_id})
        return "Successfully connected sheet to this thread. You can now use the 'Read' or 'Query' tools without providing the link again."
    except Exception as e:
        return f"Connection Failed: {str(e)}"

@tool("google_sheets_read", args_schema=GSheetReadInput)
async def google_sheets_read(config: RunnableConfig, override_link: Optional[str] = None, max_rows: int = 10) -> str:
    """
    Reads the data from the currently connected Google Sheet.
    Useful for getting an overview of the data or learning column letters (A, B, C...).
    """
    try:
        thread_id = GoogleSheetsProvider.get_thread_id(config)
        gs_config = await GoogleSheetsProvider.get_config(thread_id)
        
        link = override_link or (gs_config.get("sheet_link") if gs_config else None)
        tab = gs_config.get("tab_id", "0") if gs_config else "0"

        if not link:
            return "ERROR: No Google Sheet connected. Use 'google_sheets_connect' first. (Check your configuration panel for the URL if provided by the human)."

        spreadsheet_id = extract_spreadsheet_id(link)
        df = fetch_sheet_as_df(spreadsheet_id, tab)
        
        # Respect Agent's limit choice, but cap at 100 for safety
        limit = min(max(1, max_rows), 100)
        data = df.head(limit).to_dict(orient='records')
        
        # Add column helping to let the agent know letters
        import string
        col_letters = list(string.ascii_uppercase)
        col_map = {col_letters[i] if i < 26 else f"Col_{i}": col for i, col in enumerate(df.columns)}
        
        return f"Retrieved {len(df)} rows. Column Map: {str(col_map)}. Data Sample: {str(data)}"
    except Exception as e:
        logger.error(f"Read error: {str(e)}")
        return f"Error reading sheet: {str(e)}"

@tool("google_sheets_search", args_schema=GSheetSearchInput)
async def google_sheets_search(query: str, config: RunnableConfig) -> str:
    """
    Searches for information or specific rows within the connected Google Sheet.
    Best for simple keyword searches (Fuzzy matching).
    """
    try:
        thread_id = GoogleSheetsProvider.get_thread_id(config)
        gs_config = await GoogleSheetsProvider.get_config(thread_id)
        
        if not gs_config or not gs_config.get("sheet_link"):
            return "ERROR: No Google Sheet is currently connected to this thread. Call 'google_sheets_connect' with the Spreadsheet URL to start. (Note: If a link is already in your configuration panel, use that one)."

        spreadsheet_id = extract_spreadsheet_id(gs_config["sheet_link"])
        tab = gs_config.get("tab_id", "0")
        df = fetch_sheet_as_df(spreadsheet_id, tab)
        
        # Simple case-insensitive match
        query_lower = query.lower()
        mask = df.apply(lambda row: row.astype(str).str.contains(query_lower, case=False).any(), axis=1)
        matches = df[mask].head(50).to_dict(orient='records')
        
        if not matches:
            return f"No results matching '{query}' found in the sheet."
            
        return f"Found {len(df[mask])} results. Top matches: {str(matches)}"
    except Exception as e:
        return f"Search error: {str(e)}"

# --- Registration ---

registry.register(
    google_sheets_connect,
    ToolMeta(
        name="google_sheets_connect",
        display_name="Connect Google Sheet",
        description="Initializes a Google Sheet session for this thread. (Human or Agent can provide the link).",
        category=CATEGORY,
        package=PKG_NAME,
        version="1.2.0",
        configs=[
            {"name": "sheet_link", "type": "string", "description": "Google Sheet URL", "required": True},
            {"name": "tab_id", "type": "string", "description": "Tab GID (default 0)", "required": False, "default": "0"}
        ]
    )
)

registry.register(
    google_sheets_read,
    ToolMeta(
        name="google_sheets_read",
        display_name="Read Connected Sheet",
        description="Reads data and identifies column letters (A, B, C) for advanced queries.",
        category=CATEGORY,
        package=PKG_NAME,
        version="1.2.0"
    )
)

registry.register(
    google_sheets_query,
    ToolMeta(
        name="google_sheets_query",
        display_name="Advanced Query (Big Data)",
        description="Runs SQL-like queries on the sheet. Best for math, filtering, and large datasets (1 lac rows).",
        category=CATEGORY,
        package=PKG_NAME,
        version="1.2.0",
        ai_visible_fields=["query"]
    )
)

registry.register(
    google_sheets_search,
    ToolMeta(
        name="google_sheets_search",
        display_name="Fuzzy Search",
        description="Search for specific keywords within the connected sheet.",
        category=CATEGORY,
        package=PKG_NAME,
        version="1.2.0",
        ai_visible_fields=["query"]
    )
)
