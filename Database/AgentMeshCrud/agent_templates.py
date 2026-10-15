from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..Models import AgentTemplateModel
from .workflows import process_workflow_credentials
from .helpers import _apply_template_filters

async def save_agent_template(
    db: AsyncSession,
    name: str,
    description: Optional[str],
    group: str,
    category: str,
    config: dict
):
    """
    Saves a new agent configuration as a template.
    """
    # Scrub tool credentials and API keys as a backend safety failsafe
    clean_config = {**config}
    clean_config["tool_configs"] = {}
    clean_config["api_auth_config"] = {}
    if "llm" in clean_config and isinstance(clean_config["llm"], dict):
        clean_config["llm"] = {**clean_config["llm"]}
        if "api_key_override" in clean_config["llm"]:
            clean_config["llm"]["api_key_override"] = ""

    # ENCRYPT sensitive parts of the config
    # We use a wrapper since process_workflow_credentials expects a workflow-like structure with a 'nodes' list
    wrapped_config = {"nodes": [clean_config]}
    processed_wrapped = process_workflow_credentials(wrapped_config, mode="encrypt")
    encrypted_config = processed_wrapped["nodes"][0]

    db_template = AgentTemplateModel(
        name=name,
        description=description,
        group=group,
        category=category,
        config=encrypted_config
    )
    db.add(db_template)
    await db.commit()
    await db.refresh(db_template)
    return db_template

async def list_agent_templates(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists agent template SUMMARIES with pagination and filtering.
    Excludes the 'config' column to avoid loading large JSON blobs and
    running per-row decryption for list views.
    Use get_agent_template() for the full config of a single template.
    """
    stmt = select(
        AgentTemplateModel.id,
        AgentTemplateModel.name,
        AgentTemplateModel.description,
        AgentTemplateModel.group,
        AgentTemplateModel.category,
        AgentTemplateModel.created_at,
        AgentTemplateModel.updated_at,
    ).order_by(AgentTemplateModel.created_at.desc()).offset(skip).limit(limit)

    stmt = _apply_template_filters(stmt, group, category, search)
    result = await db.execute(stmt)
    rows = result.mappings().all()
    return [dict(row) for row in rows]

async def count_agent_templates(
    db: AsyncSession,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> int:
    """Returns total count of agent templates matching the given filters."""
    stmt = select(func.count()).select_from(AgentTemplateModel)
    stmt = _apply_template_filters(stmt, group, category, search)
    result = await db.execute(stmt)
    return result.scalar()

async def delete_agent_template(db: AsyncSession, template_id: str):
    """
    Deletes an agent template.
    """
    stmt = select(AgentTemplateModel).where(AgentTemplateModel.id == template_id)
    result = await db.execute(stmt)
    db_template = result.scalar_one_or_none()
    
    if db_template:
        await db.delete(db_template)
        await db.commit()
        return True
    return False
