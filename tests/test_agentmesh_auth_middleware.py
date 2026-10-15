import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from Apps.Auth.agentmesh_auth_middleware import (
    AgentMeshAuthenticationMiddleware,
    requires_agentmesh_authentication,
    required_agentmesh_roles,
)
from Apps.Auth.rbac import ADMIN_ROLE, USER_ROLE, extract_roles, has_any_role


class AgentMeshAuthenticationRouteTests(unittest.TestCase):
    def test_route_classification(self):
        self.assertTrue(
            requires_agentmesh_authentication("GET", "/api/v1/agentmesh/workflows")
        )
        self.assertFalse(
            requires_agentmesh_authentication("GET", "/api/v1/agentmesh/health")
        )
        self.assertFalse(
            requires_agentmesh_authentication(
                "POST", "/api/v1/agentmesh/webhook/example"
            )
        )
        self.assertFalse(
            requires_agentmesh_authentication("OPTIONS", "/api/v1/agentmesh/workflows")
        )

    def test_admin_route_classification(self):
        self.assertEqual(
            required_agentmesh_roles("GET", "/api/v1/agentmesh/connections"),
            frozenset({ADMIN_ROLE}),
        )
        self.assertEqual(
            required_agentmesh_roles(
                "PATCH", "/api/v1/agentmesh/workflow/abc/marketplace/status"
            ),
            frozenset({ADMIN_ROLE}),
        )
        self.assertEqual(
            required_agentmesh_roles("GET", "/api/v1/agentmesh/workflows"),
            frozenset(),
        )

    def setUp(self):
        app = FastAPI()
        app.add_middleware(AgentMeshAuthenticationMiddleware)

        @app.get("/api/v1/agentmesh/workflows")
        async def protected_route():
            return {"status": "success"}

        @app.get("/api/v1/agentmesh/health")
        async def public_health():
            return {"status": "healthy"}

        @app.get("/api/v1/agentmesh/connections")
        async def admin_route():
            return {"status": "success"}

        self.client = TestClient(app)

    def test_missing_token_returns_401(self):
        response = self.client.get("/api/v1/agentmesh/workflows")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

    def test_public_health_does_not_require_token(self):
        response = self.client.get("/api/v1/agentmesh/health")
        self.assertEqual(response.status_code, 200)

    @patch("Apps.Auth.agentmesh_auth_middleware.decode_token")
    def test_valid_token_allows_request(self, decode_token_mock):
        decode_token_mock.return_value = {"sub": "user-123"}
        response = self.client.get(
            "/api/v1/agentmesh/workflows",
            headers={"Authorization": "Bearer valid-token"},
        )
        self.assertEqual(response.status_code, 200)
        decode_token_mock.assert_called_once_with("valid-token")

    @patch("Apps.Auth.agentmesh_auth_middleware.decode_token")
    def test_admin_role_allows_admin_route(self, decode_token_mock):
        decode_token_mock.return_value = {"roles": [ADMIN_ROLE]}
        response = self.client.get(
            "/api/v1/agentmesh/connections",
            headers={"Authorization": "Bearer admin-token"},
        )
        self.assertEqual(response.status_code, 200)

    @patch("Apps.Auth.agentmesh_auth_middleware.decode_token")
    def test_user_role_cannot_access_admin_route(self, decode_token_mock):
        decode_token_mock.return_value = {"roles": [USER_ROLE]}
        response = self.client.get(
            "/api/v1/agentmesh/connections",
            headers={"Authorization": "Bearer user-token"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], f"Required role: {ADMIN_ROLE}")

    @patch("Apps.Auth.agentmesh_auth_middleware.decode_token")
    def test_invalid_token_returns_401(self, decode_token_mock):
        from fastapi import HTTPException

        decode_token_mock.side_effect = HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )
        response = self.client.get(
            "/api/v1/agentmesh/workflows",
            headers={"Authorization": "Bearer invalid-token"},
        )
        self.assertEqual(response.status_code, 401)

    @patch("Apps.Auth.agentmesh_auth_middleware.decode_token")
    def test_missing_keycloak_configuration_returns_503(self, decode_token_mock):
        from fastapi import HTTPException

        decode_token_mock.side_effect = HTTPException(
            status_code=503,
            detail="Authentication service is not configured",
        )
        response = self.client.get(
            "/api/v1/agentmesh/workflows",
            headers={"Authorization": "Bearer some-token"},
        )
        self.assertEqual(response.status_code, 503)


class RoleHelperTests(unittest.TestCase):
    def test_extracts_and_normalizes_client_and_realm_roles(self):
        token = {
            "resource_access": {"darch": {"roles": ["Admin"]}},
            "realm_access": {"roles": ["USER"]},
        }
        self.assertEqual(extract_roles(token, client_id="darch"), ["admin", "user"])

    def test_has_any_role_is_case_insensitive(self):
        self.assertTrue(has_any_role({"roles": ["Admin"]}, ["admin"]))


if __name__ == "__main__":
    unittest.main()
