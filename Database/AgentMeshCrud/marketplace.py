from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, String as SQLString
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging
from ..Models import WorkflowModel, MarketplaceUsecaseModel


def _safe_persisted(value: Any) -> Any:
    """Remove recognizable credential material from public marketplace data."""
    return sanitize_for_logging(redact_sensitive_data(value))

async def publish_usecase(
    db: AsyncSession,
    workflow_id: str,
    tags: Optional[List[str]] = None,
    marketplace_name: Optional[str] = None,
    marketplace_description: Optional[str] = None,
    launch_url: Optional[str] = None,
    launch_mode: Optional[str] = None,
    navigation_path: Optional[str] = None
):
    """
    Publishes or updates a workflow in the marketplace usecases table.
    """
    # 1. Fetch the workflow to ensure it exists
    wf_stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
    wf_result = await db.execute(wf_stmt)
    workflow = wf_result.scalar_one_or_none()
    if not workflow:
        raise ValueError("Workflow not found")

    # 2. Extract configuration & Auto-fetch trigger node metadata
    config = workflow.config or {}
    nodes = config.get("nodes", [])
    trigger_node = None
    for n in nodes:
        node_type = n.get("type") or (n.get("data") and n.get("data", {}).get("type"))
        if node_type == "button_trigger":
            trigger_node = n
            break

    metadata_json = {}
    if trigger_node:
        metadata_json = {
            "trigger_type": "button",
            "trigger_node_id": trigger_node.get("id"),
            "button_label": trigger_node.get("data", {}).get("button_label") or trigger_node.get("data", {}).get("label") or "Run",
            "button_input": trigger_node.get("data", {}).get("button_input") or "Start"
        }

    # Choose name/description from request params or fallback to workflow defaults
    final_name = _safe_persisted(marketplace_name if marketplace_name else workflow.name)
    final_description = _safe_persisted(
        marketplace_description if marketplace_description else workflow.description
    )
    safe_group = _safe_persisted(workflow.group)
    safe_category = _safe_persisted(workflow.category)
    safe_launch_url = _safe_persisted(launch_url)
    safe_launch_mode = _safe_persisted(launch_mode)
    safe_navigation_path = _safe_persisted(navigation_path)
    safe_metadata = _safe_persisted(metadata_json)
    safe_tags = _safe_persisted(tags) if tags is not None else None

    # Validate name uniqueness across the marketplace (exclude own workflow if updating)
    name_check_stmt = select(MarketplaceUsecaseModel).where(
        MarketplaceUsecaseModel.marketplace_name == final_name,
        MarketplaceUsecaseModel.workflow_id != workflow_id
    )
    name_check_res = await db.execute(name_check_stmt)
    if name_check_res.scalar_one_or_none():
        raise ValueError(f"Marketplace name '{final_name}' is already taken by another usecase.")

    # 3. Check if usecase already exists
    stmt = select(MarketplaceUsecaseModel).where(MarketplaceUsecaseModel.workflow_id == workflow_id)
    result = await db.execute(stmt)
    db_usecase = result.scalar_one_or_none()

    if db_usecase:
        # Increment version and update metadata
        db_usecase.version += 1
        db_usecase.marketplace_name = final_name
        db_usecase.marketplace_description = final_description
        db_usecase.group = safe_group
        db_usecase.category = safe_category
        db_usecase.launch_url = safe_launch_url
        if safe_launch_mode is not None:
            db_usecase.launch_mode = safe_launch_mode
        if safe_navigation_path is not None:
            db_usecase.navigation_path = safe_navigation_path
        db_usecase.metadata_json = safe_metadata
        if safe_tags is not None:
            db_usecase.tags = safe_tags
        db_usecase.publish = True
        db_usecase.admin_status = "pending"
        db_usecase.updated_at = datetime.utcnow()
    else:
        # Create new
        db_usecase = MarketplaceUsecaseModel(
            workflow_id=workflow_id,
            marketplace_name=final_name,
            marketplace_description=final_description,
            group=safe_group,
            category=safe_category,
            launch_url=safe_launch_url,
            launch_mode=safe_launch_mode or "default",
            navigation_path=safe_navigation_path,
            metadata_json=safe_metadata,
            tags=safe_tags or [],
            publish=True,
            admin_status="pending",
            version=1
        )
        db.add(db_usecase)

    await db.commit()
    await db.refresh(db_usecase)
    return db_usecase

async def list_marketplace_usecases(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    admin_status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists published marketplace usecases by joining with workflows to get the configuration.
    """
    stmt = select(
        MarketplaceUsecaseModel.workflow_id.label("id"),
        MarketplaceUsecaseModel.marketplace_name.label("name"),
        MarketplaceUsecaseModel.marketplace_description.label("description"),
        MarketplaceUsecaseModel.group,
        MarketplaceUsecaseModel.version,
        MarketplaceUsecaseModel.tags,
        MarketplaceUsecaseModel.launch_url,
        MarketplaceUsecaseModel.launch_mode,
        MarketplaceUsecaseModel.navigation_path,
        MarketplaceUsecaseModel.created_at,
        MarketplaceUsecaseModel.updated_at,
        MarketplaceUsecaseModel.metadata_json.label("metadata"),
        MarketplaceUsecaseModel.admin_status,
        MarketplaceUsecaseModel.remarks,
        WorkflowModel.is_active
    ).join(WorkflowModel, MarketplaceUsecaseModel.workflow_id == WorkflowModel.id)

    if admin_status and admin_status != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.admin_status == admin_status)
    else:
        if admin_status is None:
            stmt = stmt.where(MarketplaceUsecaseModel.admin_status == "approved")

    stmt = stmt.where(MarketplaceUsecaseModel.publish == True)

    # Filters
    if group and group != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.group == group)
    if category and category != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.category == category)
    if tag and tag != "All":
        pattern = f'%"{tag}"%'
        stmt = stmt.where(cast(MarketplaceUsecaseModel.tags, SQLString).ilike(pattern))
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (MarketplaceUsecaseModel.marketplace_name.ilike(pattern, escape="~")) |
            (MarketplaceUsecaseModel.marketplace_description.ilike(pattern, escape="~"))
        )

    stmt = stmt.order_by(MarketplaceUsecaseModel.updated_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    rows = result.mappings().all()

    # Format result lists
    formatted = []
    for r in rows:
        formatted.append(_safe_persisted(dict(r)))
    return formatted

async def count_marketplace_usecases(
    db: AsyncSession,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    admin_status: Optional[str] = None
) -> int:
    """Returns the total count of published usecases matching filters."""
    stmt = select(func.count()).select_from(MarketplaceUsecaseModel)\
        .where(MarketplaceUsecaseModel.publish == True)

    if admin_status and admin_status != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.admin_status == admin_status)
    else:
        if admin_status is None:
            stmt = stmt.where(MarketplaceUsecaseModel.admin_status == "approved")

    # Filters
    if group and group != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.group == group)
    if category and category != "All":
        stmt = stmt.where(MarketplaceUsecaseModel.category == category)
    if tag and tag != "All":
        pattern = f'%"{tag}"%'
        stmt = stmt.where(cast(MarketplaceUsecaseModel.tags, SQLString).ilike(pattern))
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (MarketplaceUsecaseModel.marketplace_name.ilike(pattern, escape="~")) |
            (MarketplaceUsecaseModel.marketplace_description.ilike(pattern, escape="~"))
        )

    result = await db.execute(stmt)
    return result.scalar()

async def update_marketplace_usecase_status(
    db: AsyncSession,
    workflow_id: str,
    admin_status: str,
    remarks: Optional[str] = None
) -> Optional[MarketplaceUsecaseModel]:
    """Updates the admin status of a published marketplace use case."""
    stmt = select(MarketplaceUsecaseModel).where(MarketplaceUsecaseModel.workflow_id == workflow_id)
    result = await db.execute(stmt)
    usecase = result.scalar_one_or_none()
    if usecase:
        usecase.admin_status = admin_status
        if remarks is not None:
            usecase.remarks = _safe_persisted(remarks)
        usecase.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(usecase)
        return usecase
    return None

async def delete_marketplace_usecase(db: AsyncSession, workflow_id: str) -> bool:
    """Deletes a marketplace use case by workflow ID."""
    stmt = select(MarketplaceUsecaseModel).where(MarketplaceUsecaseModel.workflow_id == workflow_id)
    result = await db.execute(stmt)
    usecase = result.scalar_one_or_none()
    if usecase:
        await db.delete(usecase)
        await db.commit()
        return True
    return False
