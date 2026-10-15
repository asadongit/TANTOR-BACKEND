"""
Workflow Variables API Router
=============================
Generic, workflow-scoped key-value configuration system.

Endpoints:
  GET    /workflow/{workflow_id}/variables           - List all variables
  POST   /workflow/{workflow_id}/variables           - Create/update a single variable
  POST   /workflow/{workflow_id}/variables/bulk      - Create/update multiple variables
  DELETE /workflow/{workflow_id}/variables/{key}      - Delete a single variable
  DELETE /workflow/{workflow_id}/variables            - Delete all variables
  POST   /workflow/{workflow_id}/variables/extract-json-keys - Upload JSON, flatten, return keys

This is 100% generic — not tied to any specific workflow type (CD, HR, etc.).
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging
import re

from Database.Database import get_db
from Database.AgentMeshCrud import (
    list_workflow_variables,
    get_workflow_variable_by_key,
    upsert_workflow_variable,
    bulk_upsert_workflow_variables,
    delete_workflow_variable,
    delete_all_workflow_variables,
    VALID_VAR_TYPES,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Variable Explorer (Workflow Variables)"])


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class VariablePayload(BaseModel):
    """Schema for creating/updating a single variable."""
    key: str = Field(..., min_length=1, max_length=128, description="Unique variable name within the workflow")
    value: Any = Field(..., description="Variable value — type depends on var_type")
    var_type: str = Field("text", description=f"One of: {', '.join(sorted(VALID_VAR_TYPES))}")
    description: Optional[str] = Field(None, max_length=500, description="Optional UI hint")
    source_filename: Optional[str] = Field(None, description="Original filename for json_keys type")

class BulkVariablesPayload(BaseModel):
    """Schema for bulk-creating/updating multiple variables."""
    variables: List[VariablePayload] = Field(..., min_length=1, max_length=100)


# ─── List Variables ────────────────────────────────────────────────────────────

@router.get("/workflow/{workflow_id}/variables")
async def get_workflow_variables(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns all variables for the specified workflow.
    Used by the frontend Variables Panel and by nodes to populate dropdowns.
    """
    try:
        variables = await list_workflow_variables(db, workflow_id)
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "count": len(variables),
            "variables": variables,
        }
    except Exception as e:
        logger.error(f"Failed to list variables for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Create / Update Single Variable ──────────────────────────────────────────

@router.post("/workflow/{workflow_id}/variables")
async def create_or_update_variable(
    workflow_id: str,
    payload: VariablePayload,
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new variable or updates an existing one (upsert by key).
    If a variable with the same key already exists for this workflow, it's updated.
    """
    try:
        # Sanitize key: lowercase, strip whitespace, replace spaces with underscores
        sanitized_key = payload.key.strip().lower().replace(" ", "_")
        
        result = await upsert_workflow_variable(
            db=db,
            workflow_id=workflow_id,
            key=sanitized_key,
            value=payload.value,
            var_type=payload.var_type,
            description=payload.description,
            source_filename=payload.source_filename,
        )
        return {
            "status": "success",
            "message": f"Variable '{sanitized_key}' saved successfully",
            "variable": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upsert variable '{payload.key}' for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Bulk Create / Update ─────────────────────────────────────────────────────

@router.post("/workflow/{workflow_id}/variables/bulk")
async def bulk_create_or_update_variables(
    workflow_id: str,
    payload: BulkVariablesPayload,
    db: AsyncSession = Depends(get_db),
):
    """
    Creates or updates multiple variables in a single request.
    Each variable is upserted independently; partial failures are reported.
    """
    try:
        # Sanitize all keys
        sanitized = []
        for var in payload.variables:
            sanitized.append({
                "key": var.key.strip().lower().replace(" ", "_"),
                "value": var.value,
                "var_type": var.var_type,
                "description": var.description,
                "source_filename": var.source_filename,
            })
        
        results = await bulk_upsert_workflow_variables(db, workflow_id, sanitized)
        return {
            "status": "success",
            "message": f"Saved {len(results)} variable(s)",
            "count": len(results),
            "variables": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to bulk upsert variables for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Delete Single Variable ───────────────────────────────────────────────────

@router.delete("/workflow/{workflow_id}/variables/{key}")
async def remove_variable(workflow_id: str, key: str, db: AsyncSession = Depends(get_db)):
    """Deletes a single variable by key."""
    try:
        deleted = await delete_workflow_variable(db, workflow_id, key)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Variable '{key}' not found")
        return {"status": "success", "message": f"Variable '{key}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete variable '{key}' for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Delete All Variables ─────────────────────────────────────────────────────

@router.delete("/workflow/{workflow_id}/variables")
async def remove_all_variables(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Deletes all variables for a workflow."""
    try:
        count = await delete_all_workflow_variables(db, workflow_id)
        return {"status": "success", "message": f"Deleted {count} variable(s)", "deleted_count": count}
    except Exception as e:
        logger.error(f"Failed to delete all variables for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Extract JSON Keys (Upload & Flatten) ─────────────────────────────────────

def _flatten_dict(nested: dict, parent_key: str = "", sep: str = "/") -> dict:
    """
    Recursively flattens a nested dictionary.
    Mirrors the flatten logic used by json_transform_node.py.
    
    Example:
        {"a": {"b": 1, "c": [{"d": 2}]}}
        → {"a/b": 1, "a/c/0/d": 2}
    """
    items = []
    for k, v in nested.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def _collapse_repeats(keys: List[str], sep: str = "/") -> tuple:
    """Paths that are the same field repeated per group, listed once.

    A sample document usually repeats its whole structure - once per year, per
    line item, per array entry - so flattening it yields the same field many
    times over:

        FY2023/BalanceSheet/Assets/CashOnHand
        FY2024/BalanceSheet/Assets/CashOnHand
        FY2025/BalanceSheet/Assets/CashOnHand

    That is one field, not three, and handing all three to an extraction agent
    asks it for the same thing three times.

    The hard part is telling a REPEATED GROUP from a genuine branch. Dropping
    any segment that varies would merge Assets with Liabilities, which are
    different fields that merely sit at the same depth. What separates them is
    what lies BENEATH: every year holds an identical structure, while assets and
    liabilities do not. So a position is treated as a repeated group only when
    its values all carry the same set of paths below them.

    Structure alone is not enough to decide this. Under one parent,

        Inventories/Total   Cash/Total

    has exactly the shape of

        FY2023/CashOnHand   FY2024/CashOnHand

    yet the first is two fields and the second is one field twice. What tells
    them apart is the NAMES: years and array indices form a series - the same
    wording with a number that advances - while Inventories and Cash are simply
    different words. So a group is collapsed only when its members both look
    like a series AND carry the same structure beneath them.

    Repetition is also local: it happens under a particular parent, not at a
    depth across the whole file. Siblings are therefore compared with siblings.

    Returns the collapsed list and a note of what was collapsed, because a list
    silently a third of its expected size is worse than no list at all.
    """
    if not keys:
        return [], []

    rows = [str(k).split(sep) for k in keys]

    def shape_of(name: str):
        """What a name looks like with its number taken out, or None.

        FY2024 and FY2025 share a shape; 0 and 1 share one; CashFlowStatement
        has none because it carries no number.
        """
        if name.isdigit():
            return ("#", "")
        m = re.fullmatch(r"(\D*?)(\d+)(\D*)", name)
        return (m.group(1).lower(), m.group(3).lower()) if m else None

    # what sits under each parent, and what lies beneath each of those children
    children: Dict[str, set] = {}
    beneath: Dict[str, set] = {}
    for r in rows:
        for i in range(len(r) - 1):
            parent = sep.join(r[:i])
            children.setdefault(parent, set()).add(r[i])
            beneath.setdefault(sep.join(r[:i + 1]), set()).add(sep.join(r[i + 1:]))

    # A repeated group is a SUBSET of a parent's children, not necessarily all
    # of them: a document commonly holds fixed sections and repeating ones side
    # by side - FinancialStatement carries CashFlowStatement and Ratios next to
    # FY2024, FY2025, FY2026. Demanding that every child be a series would
    # reject the whole parent because of one fixed sibling.
    repeats: Dict[str, set] = {}
    note = []
    for parent, kids in children.items():
        by_shape: Dict[Any, list] = {}
        for kid in kids:
            shape = shape_of(kid)
            if shape:
                by_shape.setdefault(shape, []).append(kid)
        for names in by_shape.values():
            if len(names) < 2:
                continue
            names = sorted(names)
            shapes = [beneath.get(f"{parent}{sep}{k}" if parent else k, set())
                      for k in names]
            widest = max((len(s) for s in shapes), default=0)
            if not widest:
                continue
            shared = set.intersection(*shapes)
            if len(shared) / widest >= 0.9:
                repeats.setdefault(parent, set()).update(names)
                note.append({"under": parent or "(root)",
                             "distinct_values": len(names),
                             "examples": names[:8]})

    if not repeats:
        return sorted(set(keys)), []

    out, seen = [], set()
    for r in rows:
        kept = []
        for i, part in enumerate(r):
            # drop a segment only where it is one of a repeating set of
            # siblings - never a fixed section that merely sits beside them
            if i < len(r) - 1 and part in repeats.get(sep.join(r[:i]), ()):
                continue
            kept.append(part)
        path = sep.join(kept)
        if path and path not in seen:
            seen.add(path)
            out.append(path)

    return sorted(out), note


@router.post("/workflow/{workflow_id}/variables/extract-json-keys")
async def extract_json_keys(
    workflow_id: str,
    file: UploadFile = File(...),
    variable_key: Optional[str] = Form(None),
    collapse_repeats: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a JSON file upload, flattens the structure, and:
    1. Returns the sorted list of flattened keys
    2. Auto-saves as a workflow variable with var_type='json_keys'

    Handles both JSON objects and arrays of objects.
    This is the same flatten logic used by json_transform_node at runtime,
    ensuring the keys you see here match exactly what the pipeline processes.

    Set collapse_repeats when the sample repeats its whole structure - once per
    year, per entry, per array element - and you want each field named once
    rather than once per repetition. What was collapsed is reported back, so a
    list arriving a third of its expected size is explained rather than a
    mystery. Off by default: the raw flattened keys are what json_transform
    works with, and that must keep being what this returns.
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        allowed_ext = (".json", ".txt")
        if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_ext)}")
        
        # Read and parse
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding error. Please provide UTF-8 encoded file")
        
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Flatten and extract keys
        if isinstance(json_data, list):
            # Array of objects — collect unique keys across all items
            all_keys = set()
            for item in json_data:
                if isinstance(item, dict):
                    flattened = _flatten_dict(item)
                    all_keys.update(flattened.keys())
            keys = sorted(list(all_keys))
        elif isinstance(json_data, dict):
            flattened = _flatten_dict(json_data)
            keys = sorted(list(flattened.keys()))
        else:
            raise HTTPException(status_code=400, detail="JSON must be an object or array of objects")
        
        raw_count = len(keys)
        collapsed_note = []
        if collapse_repeats:
            keys, collapsed_note = _collapse_repeats(keys)

        # Auto-save as a workflow variable. The name is the caller's to choose:
        # a fixed name means a second upload silently destroys the first, so two
        # sources cannot be held side by side and compared.
        variable_key = (variable_key or "").strip().lower().replace(" ", "_") \
            or "sample_json_keys"
        how = (f"{len(keys)} field names (from {raw_count} flattened keys, "
               f"repetitions collapsed)" if collapse_repeats else f"{len(keys)} keys")
        await upsert_workflow_variable(
            db=db,
            workflow_id=workflow_id,
            key=variable_key,
            value=keys,
            var_type="json_keys",
            description=f"Flattened keys from {file.filename} ({how})",
            source_filename=file.filename,
        )

        logger.info(f"Extracted {len(keys)} keys from {file.filename} for workflow {workflow_id}"
                    + (f" (collapsed from {raw_count})" if collapse_repeats else ""))

        message = f"Extracted {len(keys)} keys and saved as workflow variable '{variable_key}'"
        if collapse_repeats and collapsed_note:
            where = ", ".join(f"{n['distinct_values']} under {n['under']}"
                              for n in collapsed_note)
            message = (f"Found {raw_count} flattened keys, which are "
                       f"{len(keys)} distinct fields repeated ({where}). "
                       f"Saved the {len(keys)} as '{variable_key}'.")
        elif collapse_repeats:
            message = (f"Extracted {len(keys)} keys and saved as '{variable_key}'. "
                       f"Nothing repeated, so nothing was collapsed.")

        return {
            "status": "success",
            "file_name": file.filename,
            "total_keys": len(keys),
            "raw_key_count": raw_count,
            "collapsed": collapsed_note,
            "keys": keys,
            "variable_key": variable_key,
            "message": message,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting JSON keys: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Working out what a document type contains
#
# Deciding which fields a workflow must produce is set-up work, done once by a
# person - not something to repeat on every run. These three do that job, and
# each is usable ON ITS OWN: every one takes a named input and writes a name the
# caller chooses, so none depends on the others having been run. Someone who
# simply wants to type a list still can; these only save the typing.
#
# Nothing here knows what a balance sheet is. They compare lists and group names
# by patterns the CALLER supplies - and where no patterns are supplied, they
# offer suggestions drawn from the sample itself, so the vocabulary comes from
# the user's own documents rather than being shipped in the code.
#
# Values are never stored - only the names of things. A sample document belongs
# to a real customer, and the names are all that is needed.
# ═══════════════════════════════════════════════════════════════════════════════

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 20000


def _normalise(text: str) -> str:
    """A name reduced to the thing it names, for comparison only.

    Sheets decorate labels - an enumerator in front, a gloss or a formula in
    brackets behind - and JSON squashes the words together. Neither difference
    means the two are different fields, so both are stripped before comparing.
    The ORIGINAL wording is always what gets reported back.
    """
    s = re.sub(r"^[\(\-\s]*[a-zivx]{1,3}[\)\.]\s*", "", (text or "").strip())
    s = re.sub(r"\([^)]*\)", " ", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _leaf(name: str) -> str:
    """The last part of a path - a flattened key's own name."""
    return re.split(r"[./]", str(name))[-1]


def _words(name: str) -> List[str]:
    """A name broken into the words it is made of.

    Handles the three ways names are written - CamelCase, snake_case and plain
    spacing - so patterns can be suggested from whichever a document type uses.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


class CompareSourcesPayload(BaseModel):
    """Two saved lists to line up against each other."""
    left: str = Field(..., description="Variable holding the first list")
    right: str = Field(..., description="Variable holding the second list")
    save_as: Optional[str] = Field(
        None, description="Variable to save the combined list under. Omit to compare without saving.")


class ClassifyKeysPayload(BaseModel):
    """One saved list, split in two by the caller's patterns."""
    source: str = Field(..., description="Variable holding the list to split")
    patterns: List[str] = Field(
        default_factory=list,
        description="Words or regular expressions. A name matching any of them goes to the matched group. Leave empty to get suggestions instead of a split.")
    match_whole_path: bool = Field(
        False, description="Match against the whole path rather than the name's last part.")
    save_matched_as: Optional[str] = Field(None, description="Variable for the names that matched")
    save_rest_as: Optional[str] = Field(None, description="Variable for everything else")


async def _load_list(db: AsyncSession, workflow_id: str, key: str) -> List[Any]:
    """A saved variable's value, insisted upon being a list."""
    record = await get_workflow_variable_by_key(db, workflow_id, key)
    if not record:
        raise HTTPException(status_code=404, detail=f"No variable named '{key}' in this workflow")
    value = record.get("value")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    if not isinstance(value, list):
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{key}' holds {type(value).__name__}, not a list of names")
    return value


# ─── Sample Spreadsheet → the rows that carry a figure ────────────────────────

@router.post("/workflow/{workflow_id}/variables/extract-sheet-rows")
async def extract_sheet_rows(
    workflow_id: str,
    file: UploadFile = File(...),
    variable_key: Optional[str] = Form(None),
    sheet_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a spreadsheet and returns the row labels that actually carry a value.

    The mirror of extract-json-keys, for the case where a deliverable is a
    workbook rather than JSON. Rows that are only headings are reported
    separately rather than dropped, because whether a heading matters is the
    caller's judgement, not ours.

    Only labels are read. No amount is stored or returned.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        if not file.filename.lower().endswith((".xlsx", ".xlsm")):
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: .xlsx, .xlsm")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File is {len(content) // 1048576} MB; the limit is {MAX_UPLOAD_BYTES // 1048576} MB")

        try:
            import openpyxl
        except ImportError:
            raise HTTPException(status_code=500, detail="Spreadsheet support is not installed on this server")

        import io as _io
        try:
            wb = openpyxl.load_workbook(_io.BytesIO(content), data_only=False, read_only=True)
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Could not open the workbook: {err}")

        sheets = wb.sheetnames
        if sheet_name and sheet_name not in sheets:
            wb.close()
            raise HTTPException(
                status_code=400,
                detail=f"No sheet named '{sheet_name}'. This workbook has: {', '.join(sheets)}")
        ws = wb[sheet_name] if sheet_name else wb[sheets[0]]

        carries_value, headings, truncated = [], [], False
        for n, row in enumerate(ws.iter_rows(values_only=True)):
            if n >= MAX_ROWS:
                truncated = True
                break
            # A formula is not a label. Sheets often build their headings with
            # =CONCATENATE(...), and the text of the formula reads like words,
            # so taking the first cell containing letters picks up the formula
            # and offers it as a field name.
            label = next((c for c in row
                          if isinstance(c, str)
                          and not c.lstrip().startswith("=")
                          and re.search(r"[A-Za-z]", c)), None)
            if not label:
                continue
            label = re.sub(r"\s+", " ", label).strip()
            # a bare enumerator - "(I)", "a)" - names nothing on its own
            if not _normalise(label):
                continue
            # a heading names a group; a field sits beside a number or a formula
            has_value = any(
                isinstance(c, (int, float))
                or (isinstance(c, str) and c.startswith("="))
                for c in row)
            (carries_value if has_value else headings).append(label)
        wb.close()

        labels = sorted(set(carries_value))
        variable_key = (variable_key or "").strip().lower().replace(" ", "_") or "sample_sheet_rows"
        await upsert_workflow_variable(
            db=db,
            workflow_id=workflow_id,
            key=variable_key,
            value=labels,
            var_type="json_keys",
            description=f"Rows carrying a value, from {file.filename} sheet '{ws.title}' ({len(labels)} labels)",
            source_filename=file.filename,
        )

        logger.info(f"Extracted {len(labels)} valued rows from {file.filename} for workflow {workflow_id}")
        return {
            "status": "success",
            "file_name": file.filename,
            "sheet": ws.title,
            "available_sheets": sheets,
            "total_labels": len(labels),
            "labels": labels,
            "headings_ignored": len(set(headings)),
            "headings": sorted(set(headings))[:200],
            "truncated": truncated,
            "variable_key": variable_key,
            "message": f"Found {len(labels)} rows carrying a value and saved them as '{variable_key}'",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting sheet rows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ─── Two lists, lined up ──────────────────────────────────────────────────────

@router.post("/workflow/{workflow_id}/variables/compare-sources")
async def compare_sources(
    workflow_id: str,
    payload: CompareSourcesPayload,
    db: AsyncSession = Depends(get_db),
):
    """
    Lines up two saved lists and reports what is in both, and what is in only one.

    Where a deliverable has two faces - a JSON contract and a workbook, say -
    building the field list from one alone means whatever the other adds is
    quietly missing, and nobody discovers it until a customer does. This makes
    that difference visible before it is built into anything.

    Comparison ignores decoration; the wordings reported back are the originals.
    """
    try:
        left = await _load_list(db, workflow_id, payload.left)
        right = await _load_list(db, workflow_id, payload.right)

        # keep the first spelling seen for each normalised name, to report back
        def index(items):
            out = {}
            for item in items:
                name = _normalise(_leaf(item))
                if name:            # an entry that names nothing is not a field
                    out.setdefault(name, str(item))
            return out

        li, ri = index(left), index(right)
        both = sorted(li.keys() & ri.keys())
        only_left = sorted(li.keys() - ri.keys())
        only_right = sorted(ri.keys() - li.keys())

        combined = [li[k] for k in sorted(li)] + [ri[k] for k in only_right]
        saved = None
        if payload.save_as:
            saved = payload.save_as.strip().lower().replace(" ", "_")
            await upsert_workflow_variable(
                db=db, workflow_id=workflow_id, key=saved, value=combined,
                var_type="json_keys",
                description=(f"Everything in '{payload.left}' or '{payload.right}' "
                             f"({len(combined)} names)"))

        return {
            "status": "success",
            "left": {"variable": payload.left, "count": len(left)},
            "right": {"variable": payload.right, "count": len(right)},
            "in_both": len(both),
            "only_in_left": [li[k] for k in only_left],
            "only_in_right": [ri[k] for k in only_right],
            "combined_count": len(combined),
            "variable_key": saved,
            "message": (f"{len(both)} names appear in both; "
                        f"{len(only_left)} only in '{payload.left}'; "
                        f"{len(only_right)} only in '{payload.right}'"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── One list, split by the caller's patterns ─────────────────────────────────

@router.post("/workflow/{workflow_id}/variables/classify-keys")
async def classify_keys(
    workflow_id: str,
    payload: ClassifyKeysPayload,
    db: AsyncSession = Depends(get_db),
):
    """
    Splits a saved list in two, by patterns the caller gives.

    Some of what a deliverable contains is worked out rather than read - totals,
    ratios, anything computed from other values. Asking a model to read those
    invites it to invent them, and an invented total agrees with itself, which
    is worse than having no check at all. So the two groups are separated before
    either is used.

    Send no patterns and nothing is split: instead the words that recur across
    the list are returned as SUGGESTIONS, with the count each would catch. The
    vocabulary is therefore learnt from the caller's own sample rather than
    built into this server, which is what lets the same endpoint serve a
    financial statement, a policy schedule or a shipping manifest.
    """
    try:
        names = await _load_list(db, workflow_id, payload.source)
        target = (lambda n: str(n)) if payload.match_whole_path else (lambda n: _leaf(n))

        # ── no patterns: suggest some, drawn from the list itself ──
        if not payload.patterns:
            counts: Dict[str, int] = {}
            for name in names:
                for word in set(w.lower() for w in _words(target(name))):
                    counts[word] = counts.get(word, 0) + 1
            total = max(1, len(names))
            # a word in nearly everything separates nothing; one in almost
            # nothing is not a pattern. What is useful lies in between.
            useful = [(w, c) for w, c in counts.items()
                      if 2 <= c <= total * 0.75 and len(w) > 2]
            useful.sort(key=lambda x: -x[1])
            suggestions = [
                {"pattern": w, "would_match": c,
                 "examples": [str(n) for n in names if w in target(n).lower()][:3]}
                for w, c in useful[:25]
            ]
            return {
                "status": "success",
                "source": payload.source,
                "total": len(names),
                "suggestions": suggestions,
                "message": (f"No patterns given, so nothing was split. "
                            f"{len(suggestions)} recurring words found in "
                            f"'{payload.source}' - choose any as patterns."),
            }

        # ── patterns given: split, and say what each one caught ──
        compiled, invalid = [], []
        for p in payload.patterns:
            try:
                compiled.append((p, re.compile(p, re.I)))
            except re.error as err:
                invalid.append({"pattern": p, "error": str(err)})
        if invalid and not compiled:
            raise HTTPException(status_code=400,
                                detail=f"No usable patterns: {invalid}")

        matched, rest = [], []
        per_pattern = {p: 0 for p, _ in compiled}
        for name in names:
            hit = next((p for p, rx in compiled if rx.search(target(name))), None)
            if hit:
                per_pattern[hit] += 1
                matched.append(str(name))
            else:
                rest.append(str(name))

        saved = {}
        for key_name, values, note in (
            (payload.save_matched_as, matched, "matched"),
            (payload.save_rest_as, rest, "did not match"),
        ):
            if key_name:
                clean = key_name.strip().lower().replace(" ", "_")
                await upsert_workflow_variable(
                    db=db, workflow_id=workflow_id, key=clean, value=values,
                    var_type="json_keys",
                    description=(f"Names from '{payload.source}' that {note} "
                                 f"{payload.patterns} ({len(values)} names)"))
                saved[note] = clean

        return {
            "status": "success",
            "source": payload.source,
            "total": len(names),
            "matched_count": len(matched),
            "rest_count": len(rest),
            "per_pattern": per_pattern,
            "invalid_patterns": invalid,
            "matched": matched[:500],
            "rest": rest[:500],
            "saved_as": saved,
            "message": (f"{len(matched)} of {len(names)} matched; "
                        f"{len(rest)} did not"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying keys: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
