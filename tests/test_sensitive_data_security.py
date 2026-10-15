import unittest
from types import SimpleNamespace

from Apps.AgentMesh.Utilities.Security import (
    SensitiveDataError,
    decrypt_string,
    encrypt_string,
    is_encrypted,
    protect_fields,
    require_secure_transport,
    reveal_fields,
)
from Database.AgentMeshCrud.connections import (
    SENSITIVE_CONNECTION_FIELDS,
    protect_connection_values,
)
from Apps.AgentMesh.Api.routers.connections import ConnectionCreate, list_connections, save_connection
from Apps.AgentMesh.Api.routers.system import notifications_websocket


class SensitiveDataSecurityTests(unittest.TestCase):
    def test_round_trip_and_idempotent_encryption(self):
        encrypted = encrypt_string("top-secret")

        self.assertTrue(is_encrypted(encrypted))
        self.assertNotIn("top-secret", encrypted)
        self.assertEqual(decrypt_string(encrypted), "top-secret")
        self.assertEqual(encrypt_string(encrypted), encrypted)

    def test_legacy_plaintext_can_be_read_for_migration(self):
        self.assertEqual(decrypt_string("legacy-secret"), "legacy-secret")

    def test_corrupt_ciphertext_fails_closed(self):
        with self.assertRaises(SensitiveDataError):
            decrypt_string("ENC:not-valid-fernet")
        with self.assertRaises(SensitiveDataError):
            encrypt_string("ENC:not-valid-fernet")

    def test_mapping_helpers_only_transform_declared_fields(self):
        source = {"secret": "value", "label": "public"}
        protected = protect_fields(source, ("secret",))
        revealed = reveal_fields(protected, ("secret",))

        self.assertEqual(source, {"secret": "value", "label": "public"})
        self.assertTrue(is_encrypted(protected["secret"]))
        self.assertEqual(revealed, source)

    def test_connection_secret_registry_is_complete(self):
        source = {
            "access_token": "access",
            "refresh_token": "refresh",
            "oauth_client_secret": "client-secret",
            "oauth_client_id": "public-client-id",
        }
        protected = protect_connection_values(source)

        for field in SENSITIVE_CONNECTION_FIELDS:
            self.assertTrue(is_encrypted(protected[field]), field)
            self.assertNotEqual(protected[field], source[field])
        self.assertEqual(protected["oauth_client_id"], source["oauth_client_id"])

    def test_credentials_require_tls_except_loopback_development(self):
        require_secure_transport("https://api.example.com/token")
        require_secure_transport("http://localhost:8000/test")

        with self.assertRaises(SensitiveDataError):
            require_secure_transport("http://api.example.com/token")


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.value


class _FakeSession:
    def __init__(self, query_results):
        self.query_results = list(query_results)
        self.added = None
        self.committed = False
        self.rolled_back = False

    async def execute(self, _statement):
        return _ScalarResult(self.query_results.pop(0))

    def add(self, value):
        self.added = value

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


class ConnectionApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_plaintext_websocket_is_rejected_before_authentication(self):
        class FakeWebSocket:
            headers = {}
            url = SimpleNamespace(scheme="ws", hostname="api.example.test")
            client = SimpleNamespace(host="198.51.100.10")

            def __init__(self):
                self.closed = None

            async def close(self, *, code, reason):
                self.closed = (code, reason)

        websocket = FakeWebSocket()
        await notifications_websocket(websocket)
        self.assertEqual(websocket.closed[0], 4403)

    async def test_save_encrypts_before_orm_persistence(self):
        db = _FakeSession([None])
        payload = ConnectionCreate(
            platform_name="example",
            auth_type="OAuth2",
            access_token="access",
            refresh_token="refresh",
            oauth_client_id="public-id",
            oauth_client_secret="secret",
            oauth_token_url="https://id.example.com/token",
        )

        await save_connection(payload, db)

        self.assertTrue(db.committed)
        self.assertEqual(db.added.oauth_client_id, "public-id")
        for field in SENSITIVE_CONNECTION_FIELDS:
            stored = getattr(db.added, field)
            self.assertTrue(is_encrypted(stored), field)
            self.assertEqual(decrypt_string(stored), getattr(payload, field))

    async def test_list_never_serializes_connection_secrets(self):
        connection = type(
            "Connection",
            (),
            {
                "id": "conn_1",
                "platform_name": "example",
                "auth_type": "Bearer",
                "access_token": "must-not-leak",
                "refresh_token": "must-not-leak",
                "oauth_client_secret": "must-not-leak",
                "oauth_client_id": "public-id",
                "oauth_token_url": None,
                "enable_heartbeat": False,
                "test_url": None,
                "last_used_at": None,
                "created_at": None,
            },
        )()
        db = _FakeSession([[connection]])

        response = await list_connections(db)

        serialized = response["connections"][0]
        for field in SENSITIVE_CONNECTION_FIELDS:
            self.assertNotIn(field, serialized)


if __name__ == "__main__":
    unittest.main()
