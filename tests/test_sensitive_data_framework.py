import io
import logging
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
import Apps.AgentMesh.Utilities.Security as security_module

from Apps.AgentMesh.Utilities.Security import (
    SECRET_MASK,
    SensitiveDataError,
    contains_sensitive_data,
    install_sensitive_data_log_filter,
    is_encrypted,
    is_sensitive_field,
    merge_masked_sensitive_data,
    protect_sensitive_data,
    redact_sensitive_data,
    require_secure_database_transport,
    rotate_encrypted_string,
    reveal_sensitive_data,
    sanitize_for_logging,
    validate_credential_destination,
)
from Apps.AgentMesh.Utilities.State.ToolStateManager import process_state_data
from Apps.AgentMesh.Utilities.Observer import _sanitize_capture
from Apps.AgentMesh.Api.routers.execution import _safe_api_payload
from langchain_core.messages import AIMessage
from Database.AgentMeshCrud.mcp import process_mcp_args, process_mcp_command, process_mcp_env
from Database.AgentMeshCrud.workflows import process_workflow_credentials


class CommonSensitiveDataFrameworkTests(unittest.TestCase):
    def test_shared_registry_covers_current_and_future_credential_names(self):
        for field in (
            "access_token",
            "oauth_client_secret",
            "email_password",
            "oci_private_key",
            "vendor_api_key",
            "custom_refresh_token",
            "credential_value",
        ):
            self.assertTrue(is_sensitive_field(field), field)
        self.assertFalse(is_sensitive_field("username"))
        self.assertFalse(is_sensitive_field("max_tokens"))

    def test_nested_workflow_secrets_are_protected_without_mutating_source(self):
        source = {
            "triggers": {"webhook": {"secret": "hook-secret", "path": "incoming"}},
            "nodes": [{
                "id": "api-1",
                "task": "safe prompt",
                "llm": {"api_key_override": "provider-key", "model": "safe-model"},
                "api_auth_config": {"username": "public-user", "password": "db-pass"},
                "api_headers": {"Content-Type": "application/json", "Authorization": "Bearer abc.def"},
                "tool_configs": {"gitlab": {"private_token": "git-token", "project": "safe-project"}},
            }],
        }

        protected = process_workflow_credentials(source, mode="encrypt")

        self.assertEqual(source["nodes"][0]["llm"]["api_key_override"], "provider-key")
        self.assertTrue(is_encrypted(protected["triggers"]["webhook"]["secret"]))
        self.assertTrue(is_encrypted(protected["nodes"][0]["llm"]["api_key_override"]))
        self.assertTrue(is_encrypted(protected["nodes"][0]["api_auth_config"]["password"]))
        self.assertTrue(is_encrypted(protected["nodes"][0]["api_headers"]["Authorization"]))
        self.assertTrue(is_encrypted(protected["nodes"][0]["tool_configs"]["gitlab"]["private_token"]))
        self.assertEqual(protected["nodes"][0]["api_auth_config"]["username"], "public-user")
        self.assertEqual(protected["nodes"][0]["task"], "safe prompt")
        self.assertEqual(reveal_sensitive_data(protected), source)
        self.assertTrue(contains_sensitive_data(protected))
        self.assertFalse(contains_sensitive_data({"username": "public-user"}))
        self.assertEqual(
            rotate_encrypted_string(protected["triggers"]["webhook"]["secret"]),
            protected["triggers"]["webhook"]["secret"],
        )

    def test_commands_and_json_with_embedded_credentials_are_sealed(self):
        source = {"curl_command": 'curl -H "Authorization: Bearer token-value" https://example.com'}
        protected = protect_sensitive_data(source)
        self.assertTrue(is_encrypted(protected["curl_command"]))
        self.assertEqual(reveal_sensitive_data(protected), source)

    def test_previous_key_ciphertext_is_rewrapped_with_primary_key(self):
        old_cipher = Fernet(Fernet.generate_key())
        new_cipher = Fernet(Fernet.generate_key())
        legacy = f"ENC:{old_cipher.encrypt(b'test-value').decode()}"

        with patch.object(security_module, "_primary_cipher", new_cipher), patch.object(
            security_module,
            "_cipher_suite",
            MultiFernet([new_cipher, old_cipher]),
        ):
            rotated = security_module.rotate_encrypted_string(legacy)

        self.assertEqual(new_cipher.decrypt(rotated[4:].encode()), b"test-value")
        with self.assertRaises(InvalidToken):
            old_cipher.decrypt(rotated[4:].encode())

    def test_api_redaction_and_masked_update_preserve_ciphertext(self):
        stored = protect_sensitive_data({"password": "existing", "label": "safe"})
        public = redact_sensitive_data(stored)
        self.assertEqual(public, {"password": SECRET_MASK, "label": "safe"})

        merged = merge_masked_sensitive_data(
            {"password": SECRET_MASK, "label": "changed"},
            stored,
        )
        self.assertEqual(merged["password"], stored["password"])
        self.assertEqual(merged["label"], "changed")

    def test_mcp_env_and_secret_arguments_are_never_returned_plaintext(self):
        env = process_mcp_env({"UNUSUAL_VENDOR_CREDENTIAL": "value"}, mode="encrypt")
        args = process_mcp_args(["--token", "token-value", "--mode=read"], mode="encrypt")
        command = process_mcp_command("runner --api-key key-value", mode="encrypt")
        self.assertTrue(is_encrypted(env["UNUSUAL_VENDOR_CREDENTIAL"]))
        self.assertTrue(is_encrypted(args[1]))
        self.assertTrue(is_encrypted(command))
        self.assertEqual(process_mcp_env(env, mode="redact")["UNUSUAL_VENDOR_CREDENTIAL"], SECRET_MASK)
        self.assertEqual(process_mcp_args(args, mode="redact")[1], SECRET_MASK)
        self.assertEqual(process_mcp_env(env, mode="decrypt")["UNUSUAL_VENDOR_CREDENTIAL"], "value")
        self.assertEqual(process_mcp_args(args, mode="decrypt")[1], "token-value")
        self.assertEqual(process_mcp_command(command, mode="redact"), SECRET_MASK)
        self.assertEqual(process_mcp_command(command, mode="decrypt"), "runner --api-key key-value")

    def test_tool_state_encrypts_every_string_and_reveals_only_on_read(self):
        state = {"session": "session-value", "nested": ["private-value", 1]}
        protected = process_state_data(state, mode="encrypt")
        self.assertTrue(is_encrypted(protected["session"]))
        self.assertTrue(is_encrypted(protected["nested"][0]))
        self.assertEqual(process_state_data(protected, mode="decrypt"), state)

    def test_credential_destinations_require_tls_and_clean_urls(self):
        validate_credential_destination("https://api.example.com/token")
        require_secure_database_transport("postgresql://user:pass@db.example.com/app?sslmode=verify-full")
        require_secure_database_transport("neo4j+s://graph.example.com:7687")

        with self.assertRaises(SensitiveDataError):
            validate_credential_destination("http://api.example.com/token")
        with self.assertRaises(SensitiveDataError):
            validate_credential_destination("https://user:pass@api.example.com/token")
        with self.assertRaises(SensitiveDataError):
            validate_credential_destination("https://api.example.com/token?access_token=value")
        with self.assertRaises(SensitiveDataError):
            validate_credential_destination("https://api.example.com/token?q=api_key=value")
        with self.assertRaises(SensitiveDataError):
            require_secure_database_transport("postgresql://user:pass@db.example.com/app")

    def test_log_sanitizer_removes_structured_and_embedded_secrets(self):
        private_key = "-----BEGIN PRIVATE KEY-----\nprivate-key-material\n-----END PRIVATE KEY-----"
        safe = sanitize_for_logging({
            "password": "plain-password",
            "message": 'client_secret="secret-value" Authorization: Bearer token-value',
            "dsn": "postgresql://user:password@db.example.com/app",
            "unlabelled_message": f"Failure while using {private_key}",
        })
        serialized = str(safe)
        self.assertNotIn("plain-password", serialized)
        self.assertNotIn("secret-value", serialized)
        self.assertNotIn("token-value", serialized)
        self.assertNotIn("user:password@", serialized)
        self.assertNotIn("private-key-material", serialized)

    def test_log_record_factory_sanitizes_handlers_added_after_install(self):
        original_factory = logging.getLogRecordFactory()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        test_logger = logging.getLogger("agentmesh.security.factory-test")
        test_logger.handlers = [handler]
        test_logger.propagate = False
        test_logger.setLevel(logging.INFO)
        try:
            install_sensitive_data_log_filter()
            test_logger.info("Authorization: Bearer token-value")
        finally:
            logging.setLogRecordFactory(original_factory)
            test_logger.handlers = []
        self.assertNotIn("token-value", stream.getvalue())
        self.assertIn(SECRET_MASK, stream.getvalue())

    def test_external_observability_and_api_boundaries_redact_runtime_results(self):
        message = AIMessage(
            content='client_secret="secret-value"',
            additional_kwargs={"api_key": "key-value"},
        )
        safe_message = _sanitize_capture(message)
        safe_response = _safe_api_payload({
            "access_token": "token-value",
            "messages": [message],
        })

        self.assertNotIn("secret-value", str(safe_message))
        self.assertNotIn("key-value", str(safe_message))
        self.assertNotIn("token-value", str(safe_response))
        self.assertNotIn("secret-value", str(safe_response))


if __name__ == "__main__":
    unittest.main()
