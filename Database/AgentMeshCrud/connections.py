"""Persistence boundary for encrypted connection credentials."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from Apps.AgentMesh.Utilities.Security import encrypt_string, is_encrypted
from Database.Models import ConnectionProfileModel

logger = logging.getLogger(__name__)

SENSITIVE_CONNECTION_FIELDS = ("access_token", "refresh_token", "oauth_client_secret")


def protect_connection_values(values: dict) -> dict:
    """Return connection values with every secret encrypted for persistence."""
    protected = dict(values)
    for field in SENSITIVE_CONNECTION_FIELDS:
        if protected.get(field) is not None:
            protected[field] = encrypt_string(protected[field])
    return protected


async def migrate_plaintext_connection_credentials(
    db: AsyncSession,
    *,
    batch_size: int = 100,
) -> int:
    """Encrypt legacy plaintext connection secrets in-place.

    Safe to run repeatedly. Corrupt ciphertext fails closed rather than being
    overwritten, which preserves evidence and prevents accidental data loss.
    """
    migrated_rows = 0
    last_id = None
    while True:
        stmt = select(ConnectionProfileModel).order_by(ConnectionProfileModel.id).limit(batch_size)
        if last_id is not None:
            stmt = stmt.where(ConnectionProfileModel.id > last_id)
        result = await db.execute(stmt)
        connections = result.scalars().all()
        if not connections:
            break

        last_id = connections[-1].id
        for connection in connections:
            changed = False
            for field in SENSITIVE_CONNECTION_FIELDS:
                value = getattr(connection, field)
                if value:
                    protected = encrypt_string(value)  # also validates existing ciphertext
                    if protected != value:
                        setattr(connection, field, protected)
                        changed = True
            migrated_rows += int(changed)

        await db.commit()
        db.expunge_all()

    if migrated_rows:
        logger.info("Encrypted legacy credentials for %d connection profile(s)", migrated_rows)
    return migrated_rows
