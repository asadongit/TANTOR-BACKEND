import os
import json
import logging
from typing import List, Optional
from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
import pandas as pd

logger = logging.getLogger(__name__)

@tool
def query_vapt_report(file_path: str, search_column: str, search_value: str, columns_to_return: Optional[List[str]] = None) -> str:
    """
    Search and query specific data from a VAPT report CSV/XLSX file dynamically.
    - file_path: The full absolute file path to the report.
    - search_column: The column to filter by (e.g., 'cve', 'host', 'plugin name', 'severity').
    - search_value: The value to search for within that column (e.g., 'CVE-2021-44228', '192.168.1.5'). It uses a partial case-insensitive match.
    - columns_to_return: A list of specific column names to return (e.g., ['plugin name', 'solution', 'cve']). If empty, returns all columns.
    
    Use this tool when you need specific details (like the remedy/solution for a specific CVE or host) without loading the entire file.
    """
    file_path = file_path.strip().strip('"').strip("'") if file_path else ""

    if not os.path.exists(file_path):
        # Try mnt/temp directory broadly as a fallback
        temp_dir = os.path.join(os.getcwd(), "mnt", "temp")
        if os.path.exists(temp_dir):
            candidates = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(('.csv', '.xlsx')):
                        candidates.append(os.path.join(root, f))
            if candidates:
                # Sort by modification time, newest first
                candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                file_path = candidates[0]
                logger.info(f"VAPT Query Tool: Falling back to most recent upload: {file_path}")
            else:
                return f"Error: File not found at '{file_path}' and no fallback uploads exist in {temp_dir}."
        else:
            return f"Error: File not found at '{file_path}'"

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        if df is None or df.empty:
            return "Error: No data found in the file."

        # Normalize column names to lowercase and strip whitespace
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        search_col_norm = search_column.strip().lower()
        
        # Check if search column exists (try exact or partial match of column name)
        matched_col = None
        for col in df.columns:
            if search_col_norm == col or search_col_norm in col:
                matched_col = col
                break
                
        if not matched_col:
            return f"Error: Column '{search_column}' not found. Available columns: {', '.join(df.columns)}"
            
        # Filter dataframe (case insensitive partial match)
        search_val_str = str(search_value).lower()
        
        # Fill NA with empty string to avoid errors during string matching
        df[matched_col] = df[matched_col].astype(str).fillna('')
        
        if matched_col in ['severity', 'risk', 'risk_factor']:
            numeric_to_label = {"4": "critical", "3": "high", "2": "medium", "1": "low", "0": "info"}
            mapped_search_val = numeric_to_label.get(search_val_str, search_val_str)
            # create a temporary series for matching
            mapped_series = df[matched_col].apply(lambda x: numeric_to_label.get(str(x).strip(), str(x)).lower())
            filtered_df = df[mapped_series.str.contains(mapped_search_val, na=False, regex=False)]
        else:
            filtered_df = df[df[matched_col].str.lower().str.contains(search_val_str, na=False, regex=False)]
        
        if filtered_df.empty:
            return f"No results found where '{matched_col}' contains '{search_value}'."
            
        # Select columns to return
        if columns_to_return and len(columns_to_return) > 0:
            target_cols = []
            for requested_col in columns_to_return:
                req_norm = requested_col.strip().lower()
                for col in df.columns:
                    if req_norm == col or req_norm in col:
                        if col not in target_cols:
                            target_cols.append(col)
                        break
            if target_cols:
                filtered_df = filtered_df[target_cols]
                
        # strictly prevent LLM context window exhaustion, limit the results
        MAX_ROWS = 25
        total_matched = len(filtered_df)
        
        is_truncated = False
        if total_matched > MAX_ROWS:
            filtered_df = filtered_df.head(MAX_ROWS)
            is_truncated = True
            
        # Convert to dictionary for JSON output
        # Fill NA for all columns to prevent unquoted NaN in JSON
        # Cast to string to ensure Pandas Timestamp objects are JSON serializable
        results = filtered_df.fillna('').astype(str).to_dict(orient='records')
        
        summary = {
            "status": "success",
            "total_matches": total_matched,
            "returned_rows": len(results),
            "data": results
        }
        
        if is_truncated:
            summary["warning"] = f"Results truncated to {MAX_ROWS} rows to save LLM context credits. Please refine your search_value if you need more specific data."
            
        return_str = json.dumps(summary, indent=2)
        # Cap output to prevent LLM context window overflow on the second reasoning turn
        MAX_CHARS = 4000
        if len(return_str) > MAX_CHARS:
            summary["data"] = results[:10]
            summary["truncation_notice"] = f"Output capped at {MAX_CHARS} chars. Showing first 10 of {len(results)} rows. Use columns_to_return to narrow columns."
            return_str = json.dumps(summary, indent=2)
        return return_str

    except Exception as e:
        logger.error(f"VAPT Query Tool: Error processing {file_path}: {e}")
        return f"Error querying VAPT file: {str(e)}"

# Register the tool
registry.register(
    query_vapt_report,
    ToolMeta(
        name="query_vapt_report",
        display_name="VAPT Dynamic Query",
        description="Search a VAPT report dynamically by column and value. Limits results to save LLM tokens.",
        category="SecurityTools"
    )
)
