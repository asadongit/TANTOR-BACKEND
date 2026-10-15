"""Reusable role extraction and authorization helpers for Keycloak users."""

import os
from collections.abc import Iterable

from fastapi import HTTPException


ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin").strip().lower()
USER_ROLE = os.getenv("USER_ROLE", "user").strip().lower()


def _normalize_roles(roles: Iterable[object]) -> set[str]:
    """Return non-empty roles in a consistent, case-insensitive form."""
    return {
        str(role).strip().lower()
        for role in roles
        if role is not None and str(role).strip()
    }


def extract_roles(decoded_token: dict, client_id: str | None = None) -> list[str]:
    """Extract client and realm roles from a decoded Keycloak access token."""
    configured_client_id = client_id or os.getenv("KEYCLOAK_CLIENT_ID", "")
    client_roles = (
        decoded_token.get("resource_access", {})
        .get(configured_client_id, {})
        .get("roles", [])
        if configured_client_id
        else []
    )
    realm_roles = decoded_token.get("realm_access", {}).get("roles", [])
    existing_roles = decoded_token.get("roles", [])
    return sorted(_normalize_roles([*client_roles, *realm_roles, *existing_roles]))


def attach_roles(decoded_token: dict, client_id: str | None = None) -> dict:
    """Return the decoded user claims with a normalized ``roles`` field."""
    user = dict(decoded_token)
    user["roles"] = extract_roles(decoded_token, client_id=client_id)
    return user


def has_any_role(user: dict, allowed_roles: Iterable[str]) -> bool:
    """Check whether a user has at least one of the supplied roles."""
    user_roles = _normalize_roles(user.get("roles", []))
    required_roles = _normalize_roles(allowed_roles)
    return bool(user_roles.intersection(required_roles))


def require_roles(user: dict, allowed_roles: Iterable[str]) -> dict:
    """Allow the user or raise HTTP 403 when none of the roles match."""
    required_roles = _normalize_roles(allowed_roles)
    if not has_any_role(user, required_roles):
        raise HTTPException(
            status_code=403,
            detail=f"Required role: {', '.join(sorted(required_roles))}",
        )
    return user
