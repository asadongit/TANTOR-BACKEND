"""
Variable Explorer (Workflow Variables) CRUD Operations
───────────────────────────────────────────────────────
Provides async database access for workflow-scoped variables.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..Models import WorkflowVariableModel
from Apps.AgentMesh.Utilities.Security import (
    is_secret_mask,
    protect_sensitive_data,
    redact_sensitive_data,
    reveal_sensitive_data,
)

VALID_VAR_TYPES = {"text", "number", "boolean", "list", "schema_list", "json_keys", "json", "variable_ref"}

def _validate_variable_value(value: Any, var_type: str) -> None:
    """
    Validates that the value matches the expected var_type.
    Raises ValueError on mismatch.
    """
    if value is None:
        raise ValueError("Variable value cannot be None")
    
    if var_type in ("text", "variable_ref"):
        if not isinstance(value, str):
            raise ValueError(f"Variable of type '{var_type}' expects a string value, got {type(value).__name__}")
    
    elif var_type == "number":
        if not isinstance(value, (int, float)):
            raise ValueError(f"Variable of type 'number' expects int or float, got {type(value).__name__}")
    
    elif var_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"Variable of type 'boolean' expects True/False, got {type(value).__name__}")
    
    elif var_type in ("list", "json_keys"):
        if not isinstance(value, list):
            raise ValueError(f"Variable of type '{var_type}' expects a list, got {type(value).__name__}")
    
    elif var_type == "schema_list":
        if not isinstance(value, list):
            raise ValueError(f"Variable of type 'schema_list' expects a list of objects, got {type(value).__name__}")
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"schema_list item [{i}] must be an object, got {type(item).__name__}")
            if "name" not in item:
                raise ValueError(f"schema_list item [{i}] must have a 'name' field")
    
    elif var_type == "json":
        # JSON accepts any serializable value (dict, list, str, number, bool)
        pass

def _variable_value(key: str, value: Any, *, mode: str) -> Any:
    wrapped = {key: value}
    if mode == "encrypt":
        return protect_sensitive_data(wrapped)[key]
    if mode == "decrypt":
        return reveal_sensitive_data(value)
    if mode == "redact":
        return redact_sensitive_data(wrapped)[key]
    raise ValueError("mode must be one of: encrypt, decrypt, redact")


def _serialize_variable(v: WorkflowVariableModel, *, reveal_secrets: bool = False) -> Dict[str, Any]:
    mode = "decrypt" if reveal_secrets else "redact"
    return {
        "id": v.id,
        "workflow_id": v.workflow_id,
        "key": v.key,
        "value": _variable_value(v.key, v.value, mode=mode),
        "var_type": v.var_type,
        "description": v.description,
        "source_filename": v.source_filename,
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "updated_at": v.updated_at.isoformat() if v.updated_at else None,
    }


async def list_workflow_variables(
    db: AsyncSession,
    workflow_id: str,
    *,
    reveal_secrets: bool = False,
) -> List[Dict[str, Any]]:
    """
    Returns all variables for a specific workflow, sorted by creation time.
    """
    stmt = (
        select(WorkflowVariableModel)
        .where(WorkflowVariableModel.workflow_id == workflow_id)
        .order_by(WorkflowVariableModel.created_at.asc())
    )
    result = await db.execute(stmt)
    variables = result.scalars().all()
    return [_serialize_variable(v, reveal_secrets=reveal_secrets) for v in variables]

async def get_workflow_variable_by_key(
    db: AsyncSession,
    workflow_id: str,
    key: str,
    *,
    reveal_secrets: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single variable by its workflow_id + key combination.
    """
    stmt = (
        select(WorkflowVariableModel)
        .where(
            WorkflowVariableModel.workflow_id == workflow_id,
            WorkflowVariableModel.key == key
        )
    )
    result = await db.execute(stmt)
    v = result.scalar_one_or_none()
    if not v:
        return None
    return _serialize_variable(v, reveal_secrets=reveal_secrets)

async def upsert_workflow_variable(
    db: AsyncSession,
    workflow_id: str,
    key: str,
    value: Any,
    var_type: str = "text",
    description: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates or updates a workflow variable.
    If a variable with (workflow_id, key) already exists, it updates the value.
    Otherwise, it creates a new record.
    
    Returns the created/updated variable as a dict.
    Raises ValueError for invalid var_type.
    """
    if var_type not in VALID_VAR_TYPES:
        raise ValueError(f"Invalid var_type '{var_type}'. Must be one of: {', '.join(sorted(VALID_VAR_TYPES))}")
    
    stmt = (
        select(WorkflowVariableModel)
        .where(
            WorkflowVariableModel.workflow_id == workflow_id,
            WorkflowVariableModel.key == key
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if is_secret_mask(value):
        if existing is None:
            raise ValueError("A masked secret cannot be used when creating a variable")
        protected_value = existing.value
    else:
        _validate_variable_value(value, var_type)
        protected_value = _variable_value(key, value, mode="encrypt")
    
    if existing:
        # Update
        existing.value = protected_value
        existing.var_type = var_type
        if description is not None:
            existing.description = description
        if source_filename is not None:
            existing.source_filename = source_filename
        existing.updated_at = datetime.utcnow()
    else:
        # Create
        existing = WorkflowVariableModel(
            workflow_id=workflow_id,
            key=key,
            value=protected_value,
            var_type=var_type,
            description=description,
            source_filename=source_filename,
        )
        db.add(existing)
    
    await db.commit()
    await db.refresh(existing)
    
    return _serialize_variable(existing, reveal_secrets=False)

async def bulk_upsert_workflow_variables(
    db: AsyncSession,
    workflow_id: str,
    variables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Creates or updates multiple workflow variables in a single transaction.
    Each item in 'variables' must have: key, value, var_type.
    Optional: description, source_filename.
    
    Returns the list of all created/updated variables.
    """
    results = []
    for var in variables:
        key = var.get("key")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"Each variable must have a non-empty 'key'. Got: {key!r}")
        
        result = await upsert_workflow_variable(
            db=db,
            workflow_id=workflow_id,
            key=key.strip(),
            value=var.get("value"),
            var_type=var.get("var_type", "text"),
            description=var.get("description"),
            source_filename=var.get("source_filename"),
        )
        results.append(result)
    
    return results

async def delete_workflow_variable(db: AsyncSession, workflow_id: str, key: str) -> bool:
    """
    Deletes a single workflow variable by workflow_id + key.
    Returns True if deleted, False if not found.
    """
    stmt = (
        select(WorkflowVariableModel)
        .where(
            WorkflowVariableModel.workflow_id == workflow_id,
            WorkflowVariableModel.key == key
        )
    )
    result = await db.execute(stmt)
    v = result.scalar_one_or_none()
    if v:
        await db.delete(v)
        await db.commit()
        return True
    return False

async def delete_all_workflow_variables(db: AsyncSession, workflow_id: str) -> int:
    """
    Deletes all variables for a workflow. Returns count of deleted items.
    """
    stmt = (
        select(WorkflowVariableModel)
        .where(WorkflowVariableModel.workflow_id == workflow_id)
    )
    result = await db.execute(stmt)
    variables = result.scalars().all()
    count = len(variables)
    for v in variables:
        await db.delete(v)
    if count > 0:
        await db.commit()
    return count
