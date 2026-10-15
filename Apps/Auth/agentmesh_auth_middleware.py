"""Central authentication guard for AgentMesh HTTP APIs."""

import re
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from Apps.Auth.auth_utils import decode_token
from Apps.Auth.rbac import ADMIN_ROLE, attach_roles, has_any_role


AGENTMESH_API_PREFIX = "/api/v1/agentmesh"
PUBLIC_AGENTMESH_PATHS = {
    f"{AGENTMESH_API_PREFIX}/health",
    f"{AGENTMESH_API_PREFIX}/execution/upload-temp",
}
WEBHOOK_PATH_PREFIX = f"{AGENTMESH_API_PREFIX}/webhook/"
VAPT_PATH_PREFIX = f"{AGENTMESH_API_PREFIX}/vapt/"



@dataclass(frozen=True)
class RolePolicy:
    """Declarative RBAC rule that can be extended when new roles are introduced."""

    methods: frozenset[str]
    path_pattern: re.Pattern[str]
    allowed_roles: frozenset[str]

    def matches(self, method: str, path: str) -> bool:
        return method.upper() in self.methods and bool(self.path_pattern.fullmatch(path))


AGENTMESH_ROLE_POLICIES = (
    RolePolicy(
        methods=frozenset({"GET", "POST", "DELETE"}),
        path_pattern=re.compile(rf"{AGENTMESH_API_PREFIX}/connections(?:/[^/]+)?/?"),
        allowed_roles=frozenset({ADMIN_ROLE}),
    ),
    RolePolicy(
        methods=frozenset({"PATCH", "DELETE"}),
        path_pattern=re.compile(
            rf"{AGENTMESH_API_PREFIX}/workflow/[^/]+/marketplace(?:/status)?/?"
        ),
        allowed_roles=frozenset({ADMIN_ROLE}),
    ),
    RolePolicy(
        methods=frozenset({"PATCH", "DELETE"}),
        path_pattern=re.compile(
            rf"{AGENTMESH_API_PREFIX}/workflow-template/[^/]+(?:/status)?/?"
        ),
        allowed_roles=frozenset({ADMIN_ROLE}),
    ),
)


def required_agentmesh_roles(method: str, path: str) -> frozenset[str]:
    """Return roles required by the first RBAC policy matching this request."""
    for policy in AGENTMESH_ROLE_POLICIES:
        if policy.matches(method, path):
            return policy.allowed_roles
    return frozenset()


def requires_agentmesh_authentication(method: str, path: str) -> bool:
    """Return True when an HTTP request must provide a valid Keycloak token."""
    if method.upper() == "OPTIONS":
        return False
    if not path.startswith(AGENTMESH_API_PREFIX):
        return False

    normalized_path = path.rstrip("/") or "/"
    if normalized_path in PUBLIC_AGENTMESH_PATHS:
        return False
    if path.startswith(WEBHOOK_PATH_PREFIX):
        return False
    if path.startswith(VAPT_PATH_PREFIX):
        return False
    if path.startswith(f"{AGENTMESH_API_PREFIX}/workflow/") and path.endswith("/execute"):
        return False
    return True


def _auth_error(status_code: int, detail: str) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers=headers,
    )


class AgentMeshAuthenticationMiddleware(BaseHTTPMiddleware):
    """Validate Keycloak Bearer tokens before protected AgentMesh routes run."""

    async def dispatch(self, request: Request, call_next):
        if not requires_agentmesh_authentication(request.method, request.url.path):
            return await call_next(request)

        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return _auth_error(401, "Not authenticated")

        try:
            decoded_token = await run_in_threadpool(
                decode_token,
                token.strip(),
            )
            request.state.current_user = attach_roles(decoded_token)
        except HTTPException as error:
            return _auth_error(error.status_code, str(error.detail))
        except Exception:
            return _auth_error(401, "Invalid authentication token")

        required_roles = required_agentmesh_roles(
            request.method,
            request.url.path,
        )
        if required_roles and not has_any_role(
            request.state.current_user,
            required_roles,
        ):
            return _auth_error(
                403,
                f"Required role: {', '.join(sorted(required_roles))}",
            )

        return await call_next(request)
