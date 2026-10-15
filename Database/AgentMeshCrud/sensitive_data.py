"""Bounded, one-time migration for data created before secret encryption."""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Utilities.Security import (
    protect_sensitive_data,
    redact_sensitive_data,
    sanitize_for_logging,
)
from Database.Models import (
    AgentTemplateModel,
    ComponentTemplateModel,
    ExecutionLogModel,
    MCPServerModel,
    MarketplaceUsecaseModel,
    NotificationModel,
    SensitiveDataMigrationModel,
    ToolStateModel,
    WorkflowModel,
    WorkflowRunHistoryModel,
    WorkflowTemplateModel,
    WorkflowVariableModel,
)
from .connections import migrate_plaintext_connection_credentials
from .mcp import process_mcp_args, process_mcp_command, process_mcp_env

logger = logging.getLogger(__name__)

SENSITIVE_DATA_MIGRATION_NAME = "common-sensitive-data-v1"


def _redact_persisted(value: Any) -> Any:
    return sanitize_for_logging(redact_sensitive_data(value))


def _protect_fields(row: Any, fields: tuple[str, ...]) -> bool:
    changed = False
    for field in fields:
        current = getattr(row, field)
        if current is None:
            continue
        protected = protect_sensitive_data(current)
        if protected != current:
            setattr(row, field, protected)
            changed = True
    return changed


def _migrate_mcp(row: MCPServerModel) -> bool:
    safe_command = process_mcp_command(row.command, mode="encrypt")
    safe_env = process_mcp_env(row.env, mode="encrypt")
    safe_args = process_mcp_args(row.args, mode="encrypt")
    changed = (
        safe_command != row.command
        or safe_env != (row.env or {})
        or safe_args != (row.args or [])
    )
    if changed:
        row.command = safe_command
        row.env = safe_env
        row.args = safe_args
    return changed


def _migrate_tool_state(row: ToolStateModel) -> bool:
    protected = protect_sensitive_data(row.state, force=True)
    if protected == row.state:
        return False
    row.state = protected
    return True


def _migrate_variable(row: WorkflowVariableModel) -> bool:
    protected = protect_sensitive_data({row.key: row.value})[row.key]
    if protected == row.value:
        return False
    row.value = protected
    return True


def _migrate_execution_log(row: ExecutionLogModel) -> bool:
    safe_input = _redact_persisted(row.input_data)
    safe_output = _redact_persisted(row.output_data)
    safe_error = sanitize_for_logging(row.error_message)
    if (safe_input, safe_output, safe_error) == (
        row.input_data,
        row.output_data,
        row.error_message,
    ):
        return False
    row.input_data = safe_input
    row.output_data = safe_output
    row.error_message = safe_error
    return True


def _migrate_history(row: WorkflowRunHistoryModel) -> bool:
    safe_output = _redact_persisted(row.output)
    safe_messages = _redact_persisted(row.messages)
    safe_context = _redact_persisted(row.context)
    if (safe_output, safe_messages, safe_context) == (
        row.output,
        row.messages,
        row.context,
    ):
        return False
    row.output = safe_output
    row.messages = safe_messages
    row.context = safe_context
    return True


def _migrate_notification(row: NotificationModel) -> bool:
    safe_title = sanitize_for_logging(row.title)
    safe_message = sanitize_for_logging(row.message)
    safe_metadata = _redact_persisted(row.metadata_json)
    if (safe_title, safe_message, safe_metadata) == (
        row.title,
        row.message,
        row.metadata_json,
    ):
        return False
    row.title = safe_title
    row.message = safe_message
    row.metadata_json = safe_metadata
    return True


def _migrate_marketplace(row: MarketplaceUsecaseModel) -> bool:
    fields = (
        "marketplace_name",
        "marketplace_description",
        "group",
        "category",
        "tags",
        "launch_url",
        "launch_mode",
        "navigation_path",
        "metadata_json",
        "remarks",
    )
    changed = False
    for field in fields:
        current = getattr(row, field)
        safe_value = _redact_persisted(current)
        if safe_value != current:
            setattr(row, field, safe_value)
            changed = True
    return changed


async def _migrate_model_in_batches(
    db: AsyncSession,
    model: type,
    transform: Callable[[Any], bool],
    *,
    batch_size: int,
) -> int:
    """Keyset-page a table, committing and releasing each bounded batch."""
    primary_keys = sa_inspect(model).primary_key
    if len(primary_keys) != 1:
        raise RuntimeError(f"{model.__name__} must have one primary key for batched migration")
    primary_key = primary_keys[0]
    last_key = None
    changed_rows = 0
    processed_rows = 0

    while True:
        stmt = select(model).order_by(primary_key).limit(batch_size)
        if last_key is not None:
            stmt = stmt.where(primary_key > last_key)
        result = await db.execute(stmt)
        rows = result.scalars().all()
        if not rows:
            break

        last_key = getattr(rows[-1], primary_key.key)
        changed_rows += sum(int(transform(row)) for row in rows)
        processed_rows += len(rows)
        await db.commit()
        db.expunge_all()

        if processed_rows % max(batch_size * 10, 1000) == 0:
            logger.info(
                "Sensitive-data migration scanned %d %s row(s)",
                processed_rows,
                model.__tablename__,
            )

    return changed_rows


async def migrate_legacy_sensitive_data(
    db: AsyncSession,
    *,
    batch_size: int | None = None,
) -> int:
    """Protect legacy rows once without loading a complete table into memory.

    Each batch is committed independently. If startup is interrupted, completed
    batches stay protected and the idempotent migration safely resumes on the
    next startup. A database marker avoids repeating the scan after completion.
    """
    marker = await db.get(SensitiveDataMigrationModel, SENSITIVE_DATA_MIGRATION_NAME)
    if marker is not None:
        logger.info("Sensitive-data migration already completed; skipping legacy scan")
        return 0

    resolved_batch_size = batch_size or settings.SENSITIVE_DATA_MIGRATION_BATCH_SIZE
    changed_rows = await migrate_plaintext_connection_credentials(
        db,
        batch_size=resolved_batch_size,
    )

    migrations: tuple[tuple[type, Callable[[Any], bool]], ...] = (
        (WorkflowModel, lambda row: _protect_fields(row, ("config", "draft_config"))),
        (AgentTemplateModel, lambda row: _protect_fields(row, ("config",))),
        (WorkflowTemplateModel, lambda row: _protect_fields(row, ("config",))),
        (ComponentTemplateModel, lambda row: _protect_fields(row, ("config",))),
        (MCPServerModel, _migrate_mcp),
        (ToolStateModel, _migrate_tool_state),
        (WorkflowVariableModel, _migrate_variable),
        (ExecutionLogModel, _migrate_execution_log),
        (WorkflowRunHistoryModel, _migrate_history),
        (NotificationModel, _migrate_notification),
        (MarketplaceUsecaseModel, _migrate_marketplace),
    )
    for model, transform in migrations:
        changed_rows += await _migrate_model_in_batches(
            db,
            model,
            transform,
            batch_size=resolved_batch_size,
        )

    db.add(SensitiveDataMigrationModel(
        name=SENSITIVE_DATA_MIGRATION_NAME,
        changed_rows=changed_rows,
    ))
    try:
        await db.commit()
    except IntegrityError:
        # Another application instance may have finished the same idempotent
        # migration concurrently. Accept that only when its marker exists.
        await db.rollback()
        marker = await db.get(SensitiveDataMigrationModel, SENSITIVE_DATA_MIGRATION_NAME)
        if marker is None:
            raise

    logger.info(
        "Sensitive-data migration completed; protected or redacted %d row(s)",
        changed_rows,
    )
    return changed_rows
