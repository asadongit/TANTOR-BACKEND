import copy
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlparse

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

# Use the MASTER_KEY from settings
if not hasattr(settings, "MASTER_KEY") or not settings.MASTER_KEY:
    raise ValueError("CRITICAL SECURITY ERROR: AGENTMESH_MASTER_KEY is not set in environment.")

class SensitiveDataError(ValueError):
    """Raised when sensitive data cannot be protected or safely recovered."""


def _build_keyring() -> MultiFernet:
    raw_keys = [settings.MASTER_KEY]
    if settings.PREVIOUS_MASTER_KEYS:
        raw_keys.extend(k.strip() for k in settings.PREVIOUS_MASTER_KEYS.split(",") if k.strip())
    try:
        return MultiFernet([Fernet(key.encode()) for key in raw_keys])
    except (TypeError, ValueError) as exc:
        logger.critical("Invalid AgentMesh encryption key configuration")
        raise SensitiveDataError(
            "AGENTMESH_MASTER_KEY/AGENTMESH_PREVIOUS_MASTER_KEYS must contain valid Fernet keys"
        ) from exc


_cipher_suite = _build_keyring()
_primary_cipher = Fernet(settings.MASTER_KEY.encode())

ENCRYPTION_PREFIX = "ENC:"
SECRET_MASK = "********"

# Central registry used by workflow configs, templates, variables, logs and
# checkpoints.  New credential field names can be added here without changing
# every persistence/API path individually.
_SENSITIVE_FIELD_NAMES = {
    "access_key",
    "access_key_id",
    "access_token",
    "admin_password",
    "api_key",
    "api_key_override",
    "authorization",
    "client_secret",
    "credential",
    "credential_value",
    "credentials",
    "connection_string",
    "cookie",
    "database_url",
    "db_connection_string",
    "email_client_secret",
    "email_google_service_account_json",
    "email_password",
    "gmail_password",
    "id_token",
    "oauth_client_secret",
    "oauth_token",
    "personal_access_token",
    "passphrase",
    "password",
    "private_key",
    "private_token",
    "proxy_authorization",
    "redis_url",
    "refresh_token",
    "secret",
    "secret_key",
    "secret_access_key",
    "service_account_json",
    "session_token",
    "signing_key",
    "sas_token",
    "ssh_key",
    "token",
    "webhook_secret",
    "x_api_key",
}
_SENSITIVE_FIELD_SUFFIXES = (
    "_access_key",
    "_api_key",
    "_client_secret",
    "_connection_string",
    "_credentials",
    "_passphrase",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_SENSITIVE_CONTAINERS = {
    "env",
    "secrets",
}
_MASK_VALUES = {SECRET_MASK, "__AGENTMESH_SECRET_SET__"}


def _normalize_field_name(field_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(field_name).strip().lower()).strip("_")


def is_sensitive_field(field_name: Any) -> bool:
    """Return whether a field name represents credential/sensitive data."""
    normalized = _normalize_field_name(field_name)
    return normalized in _SENSITIVE_FIELD_NAMES or normalized.endswith(_SENSITIVE_FIELD_SUFFIXES)


def is_sensitive_container(field_name: Any) -> bool:
    """Return whether every scalar value in a mapping is credential material."""
    return _normalize_field_name(field_name) in _SENSITIVE_CONTAINERS


def is_secret_mask(value: Any) -> bool:
    """Recognize the API sentinel used instead of returning an existing secret."""
    return isinstance(value, str) and value in _MASK_VALUES

def encrypt_string(plain_text: str) -> str:
    """Encrypt a value at rest. Existing ciphertext is validated and left unchanged."""
    if not plain_text:
        return plain_text
    if not isinstance(plain_text, str):
        raise TypeError("Sensitive values must be strings")

    if plain_text.startswith(ENCRYPTION_PREFIX):
        # Validate existing ciphertext and rewrap old-key values when rotating.
        return rotate_encrypted_string(plain_text)

    try:
        cipher_text = _cipher_suite.encrypt(plain_text.encode()).decode()
        return f"{ENCRYPTION_PREFIX}{cipher_text}"
    except Exception as exc:
        logger.exception("Sensitive value encryption failed")
        raise SensitiveDataError("Sensitive value encryption failed") from exc

def decrypt_string(cipher_text: str) -> str:
    """
    Decrypts a string if it has the ENC: prefix.
    If it doesn't, it returns it as-is (handling legacy plain-text data).
    """
    if not cipher_text:
        return cipher_text
    if not isinstance(cipher_text, str):
        raise TypeError("Encrypted values must be strings")
        
    if not cipher_text.startswith(ENCRYPTION_PREFIX):
        # Legacy plain-text data
        return cipher_text
        
    try:
        actual_cipher = cipher_text[len(ENCRYPTION_PREFIX):]
        return _cipher_suite.decrypt(actual_cipher.encode()).decode()
    except (InvalidToken, ValueError, TypeError) as exc:
        # Do not log ciphertext or return it to a caller as though it were plaintext.
        logger.error("Sensitive value decryption failed; key mismatch or corrupt ciphertext")
        raise SensitiveDataError("Sensitive value decryption failed") from exc


def is_encrypted(value: Any) -> bool:
    """Return whether a value uses the application's encrypted-at-rest envelope."""
    return isinstance(value, str) and value.startswith(ENCRYPTION_PREFIX)


def rotate_encrypted_string(value: str) -> str:
    """Rewrap previous-key ciphertext with the primary key exactly once."""
    if not is_encrypted(value):
        return encrypt_string(value)
    token = value[len(ENCRYPTION_PREFIX):].encode()
    try:
        _primary_cipher.decrypt(token)
        return value
    except InvalidToken:
        try:
            return f"{ENCRYPTION_PREFIX}{_cipher_suite.rotate(token).decode()}"
        except (InvalidToken, ValueError, TypeError) as exc:
            logger.error("Sensitive value rotation failed; key mismatch or corrupt ciphertext")
            raise SensitiveDataError("Sensitive value rotation failed") from exc


def protect_fields(data: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Return a copy with the named string fields encrypted."""
    protected = dict(data)
    for field in fields:
        value = protected.get(field)
        if value is not None:
            protected[field] = encrypt_string(value)
    return protected


def reveal_fields(data: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Return a short-lived copy with the named fields decrypted for point-of-use only."""
    revealed = dict(data)
    for field in fields:
        value = revealed.get(field)
        if value is not None:
            revealed[field] = decrypt_string(value)
    return revealed


def protect_sensitive_data(value: Any, *, force: bool = False) -> Any:
    """Return a deep protected copy of an arbitrary JSON-like structure.

    `force=True` is intended for stores whose entire value is private (for
    example MCP environments and tool state).  Otherwise protection is driven
    by the shared sensitive-field registry above.
    """
    if isinstance(value, Mapping):
        protected = {}
        for key, item in value.items():
            field_is_sensitive = is_sensitive_field(key)
            child_force = force or field_is_sensitive or is_sensitive_container(key)
            protected[key] = protect_sensitive_data(item, force=child_force)
        return protected
    if isinstance(value, list):
        return [protect_sensitive_data(item, force=force) for item in value]
    if isinstance(value, tuple):
        return tuple(protect_sensitive_data(item, force=force) for item in value)
    if is_encrypted(value):
        return rotate_encrypted_string(value)
    if (
        isinstance(value, str)
        and value
        and not is_secret_mask(value)
        and (force or contains_sensitive_text(value))
    ):
        return encrypt_string(value)
    return copy.deepcopy(value)


def reveal_sensitive_data(value: Any) -> Any:
    """Return a short-lived decrypted copy for an authorized runtime consumer."""
    if isinstance(value, Mapping):
        return {key: reveal_sensitive_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [reveal_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(reveal_sensitive_data(item) for item in value)
    if is_encrypted(value):
        return decrypt_string(value)
    return copy.deepcopy(value)


def redact_sensitive_data(value: Any, *, force: bool = False) -> Any:
    """Return an API/log-safe copy in which secrets are replaced by a sentinel."""
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            field_is_sensitive = is_sensitive_field(key)
            child_force = force or field_is_sensitive or is_sensitive_container(key)
            redacted[key] = redact_sensitive_data(item, force=child_force)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item, force=force) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item, force=force) for item in value)
    if (force or is_encrypted(value) or contains_sensitive_text(value)) and value not in (None, ""):
        return SECRET_MASK
    return copy.deepcopy(value)


def contains_sensitive_data(value: Any, *, force: bool = False) -> bool:
    """Return whether a structured value contains registered credential data."""
    return redact_sensitive_data(value, force=force) != value


def merge_masked_sensitive_data(incoming: Any, stored: Any) -> Any:
    """Preserve stored ciphertext when an edit round-trips an API secret mask."""
    if is_secret_mask(incoming):
        return copy.deepcopy(stored) if stored is not None else None
    if isinstance(incoming, Mapping):
        stored_mapping = stored if isinstance(stored, Mapping) else {}
        return {
            key: merge_masked_sensitive_data(item, stored_mapping.get(key))
            for key, item in incoming.items()
        }
    if isinstance(incoming, list):
        stored_list = stored if isinstance(stored, list) else []
        return [
            merge_masked_sensitive_data(item, stored_list[index] if index < len(stored_list) else None)
            for index, item in enumerate(incoming)
        ]
    return copy.deepcopy(incoming)


_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"(?i)(?P<prefix>(?:access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|"
    r"client[_-]?secret|password|private[_-]?key|authorization|cookie)['\"]?\s*[=:]\s*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s,;'\"]+)(?P=quote)"
)
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_URL_PASSWORD_PATTERN = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s/@]+)(@)")
_FLAG_SECRET_PATTERN = re.compile(
    r"(?i)((?:^|\s)--?(?:access[-_]?token|refresh[-_]?token|api[-_]?key|"
    r"client[-_]?secret|password|private[-_]?key|secret|token)\s+)([^\s]+)"
)
_PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA )?PRIVATE KEY-----.*?-----END (?:RSA )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact_text(value: Any) -> Any:
    """Redact common secret representations embedded inside free-form text."""
    if not isinstance(value, str):
        return value
    result = _PEM_PRIVATE_KEY_PATTERN.sub(SECRET_MASK, value)
    result = _BEARER_PATTERN.sub(r"\1" + SECRET_MASK, result)
    result = _URL_PASSWORD_PATTERN.sub(r"\1" + SECRET_MASK + r"\3", result)
    result = _FLAG_SECRET_PATTERN.sub(r"\1" + SECRET_MASK, result)
    return _KEY_VALUE_SECRET_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{SECRET_MASK}{match.group('quote')}",
        result,
    )


def contains_sensitive_text(value: Any) -> bool:
    """Detect credential material embedded in commands, JSON or connection URLs."""
    if not isinstance(value, str):
        return False
    if "-----BEGIN PRIVATE KEY-----" in value or "-----BEGIN RSA PRIVATE KEY-----" in value:
        return True
    return redact_text(value) != value


def sanitize_for_logging(value: Any) -> Any:
    """Recursively sanitize structured or free-form values before logging."""
    if isinstance(value, Mapping):
        return {
            key: (SECRET_MASK if is_sensitive_field(key) and item not in (None, "")
                  else sanitize_for_logging(item))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(sanitize_for_logging(item) for item in value)
    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]
    return redact_text(value)


class SensitiveDataFilter(logging.Filter):
    """Last-line defense preventing credential values from reaching log sinks."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_for_logging(record.msg)
        if record.args:
            record.args = sanitize_for_logging(record.args)
        return True


def install_sensitive_data_log_filter() -> None:
    """Install redaction for current handlers and all subsequently created records."""
    current_factory = logging.getLogRecordFactory()
    if not getattr(current_factory, "_agentmesh_sensitive_data_filter", False):
        original_factory = current_factory

        def safe_record_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            record.msg = sanitize_for_logging(record.msg)
            if record.args:
                record.args = sanitize_for_logging(record.args)
            return record

        safe_record_factory._agentmesh_sensitive_data_filter = True
        logging.setLogRecordFactory(safe_record_factory)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(item, SensitiveDataFilter) for item in handler.filters):
            handler.addFilter(SensitiveDataFilter())


def validate_credential_destination(url: str) -> None:
    """Forbid secrets in HTTP URLs without imposing a transport policy.

    The configured endpoint and client determine HTTP/HTTPS transport. Storage
    encryption and credential redaction apply independently of that choice.
    """
    parsed = urlparse(url)
    if parsed.password or contains_sensitive_text(url):
        raise SensitiveDataError("Credentials must not be embedded in destination URLs")
    if any(is_sensitive_field(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise SensitiveDataError("Credentials must not be placed in URL query parameters")
