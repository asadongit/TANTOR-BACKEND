from typing import Optional
from sqlalchemy import Select
from ..Models import WorkflowModel, AgentTemplateModel

def _apply_workflow_filters(
    stmt: Select,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> Select:
    """Applies group/category/search filters to a WorkflowModel query."""
    if group:
        stmt = stmt.where(WorkflowModel.group == group)
    if category:
        stmt = stmt.where(WorkflowModel.category == category)
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (WorkflowModel.name.ilike(pattern, escape="~")) |
            (WorkflowModel.description.ilike(pattern, escape="~"))
        )
    return stmt


def _apply_template_filters(
    stmt: Select,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> Select:
    """Applies group/category/search filters to an AgentTemplateModel query."""
    if group:
        stmt = stmt.where(AgentTemplateModel.group == group)
    if category:
        stmt = stmt.where(AgentTemplateModel.category == category)
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (AgentTemplateModel.name.ilike(pattern, escape="~")) |
            (AgentTemplateModel.description.ilike(pattern, escape="~"))
        )
    return stmt
