import io
import json
import logging
import asyncio
import pandas as pd
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

router = APIRouter()

# Define the categories for the LLM
CATEGORIES = ["OS", "Network/Infrastructure", "Application", "Other/General"]

async def categorize_batch(vulnerabilities: List[str]) -> Dict[str, str]:
    """Sends a batch of unique vulnerabilities to the LLM for categorization."""
    if not vulnerabilities:
        return {}

    prompt = f"""
You are a cybersecurity expert. Categorize the following list of vulnerabilities into ONE of these specific categories:
{CATEGORIES}

Return ONLY a valid JSON object where the keys are the exact vulnerability names provided, and the values are the category you assigned.

Vulnerabilities to categorize:
{json.dumps(vulnerabilities)}
"""
    try:
        # Use the explicitly requested model
        llm = get_llm_provider(model="openai/gpt-oss-120b", temperature=0.0)
        
        # Call the model using the wrapper's _agenerate to pass kwargs through
        response = await llm._agenerate(
            messages=[HumanMessage(content=prompt)],
            response_format={"type": "json_object"}
        )
        
        result_text = response.generations[0].message.content
        return json.loads(result_text)
    except Exception as e:
        logger.error(f"Error calling LLM for batch: {e}")
        return {"_debug_error": str(e)}

@router.post("/analyze-report")
async def analyze_vapt_report(file: UploadFile = File(...)):
    """
    Endpoint for the React Dashboard. 
    Accepts a VAPT CSV/XLSX, extracts counts deterministically, 
    and uses an LLM to categorize the unique vulnerabilities.
    """
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel files are supported.")

    try:
        contents = await file.read()
        
        # Parse into Pandas DataFrame in-memory
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            
        # Normalize columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    # 1. Deterministic Analysis (Totals & Severity)
    total_vulns = len(df)
    
    # Identify severity column
    severity_col = next((col for col in df.columns if col in ['severity', 'risk', 'risk_factor']), None)
    severity_distribution = []
    
    if severity_col:
        # Normalize numeric severities to strings if needed
        numeric_to_label = {"4": "Critical", "3": "High", "2": "Medium", "1": "Low", "0": "Info"}
        df["_norm_severity"] = df[severity_col].apply(lambda x: numeric_to_label.get(str(x).strip(), str(x)).capitalize())
        
        counts = df["_norm_severity"].value_counts().to_dict()
        for sev, count in counts.items():
            severity_distribution.append({"name": sev, "value": count})
    else:
        severity_distribution = [{"name": "Unknown", "value": total_vulns}]

    # 2. AI Categorization (Unique Vulnerability Names)
    # Identify the vulnerability name column
    name_col = next((col for col in df.columns if col in ['plugin name', 'name', 'vulnerability', 'title']), None)
    
    category_counts = {}
    llm_mapping = {}
    
    if name_col:
        # Extract unique names and drop NA
        unique_vulns = df[name_col].dropna().astype(str).unique().tolist()
        
        # Batching (e.g., chunks of 20)
        BATCH_SIZE = 20
        batches = [unique_vulns[i:i + BATCH_SIZE] for i in range(0, len(unique_vulns), BATCH_SIZE)]
        
        # Run LLM calls in parallel
        tasks = [categorize_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Merge dictionary results
        for res in batch_results:
            llm_mapping.update(res)
            
        # Map back to full dataset to get category counts
        df["_category"] = df[name_col].map(llm_mapping).fillna("Unknown")
        cat_counts = df["_category"].value_counts().to_dict()
        
        for cat, count in cat_counts.items():
            category_counts[cat] = count
    else:
        category_counts = {"Unknown": total_vulns}
        
    # Format categories for Recharts
    formatted_category_dist = [{"name": k, "value": v} for k, v in category_counts.items()]

    # Return structured JSON
    return {
        "summary": {
            "total_vulnerabilities": total_vulns,
            "unique_vulnerabilities": len(llm_mapping) if llm_mapping else 0
        },
        "severity_distribution": severity_distribution,
        "category_distribution": formatted_category_dist,
        "llm_mapping": llm_mapping,
        "debug": llm_mapping.get("_debug_error", "No errors")
    }
