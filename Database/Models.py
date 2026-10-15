from sqlalchemy import (
    Column,
    String,
    JSON,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    BigInteger,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class WorkflowModel(Base):
    """
    SQLAlchemy Model for storing compiled workflow JSON architectures.
    """
    __tablename__ = "Workflows"

    id = Column(String, primary_key=True, index=True, comment="UUID of the workflow")
    name = Column(String, nullable=False, comment="Human readable name")
    description = Column(String, nullable=True)
    group = Column(String, default="Default", index=True)
    category = Column(String, default="General", index=True)
    
    # Store the entire Pydantic WorkflowConfig model as JSON
    config = Column(JSON, nullable=False, comment="Frontend JSON/YAML string configuration")
    draft_config = Column(JSON, nullable=True, comment="Uncompiled sandbox/draft version of the workflow")

    is_active = Column(Boolean, default=True)

    # If True, this workflow is pre-compiled at server startup for zero cold-start latency.
    # Use for webhook-critical or time-sensitive workflows.
    # All other workflows are compiled lazily on first request and cached thereafter.
    warmup = Column(Boolean, default=False, comment="Pre-compile on startup if True")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExecutionLogModel(Base):
    """
    SQLAlchemy Model for storing execution traces of the AgentMesh Engine.
    Enables local observability and dashboarding.
    """
    __tablename__ = "execution_logs"

    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, index=True, nullable=False)
    node_id = Column(String, nullable=False)
    workflow_id = Column(String, index=True)
    
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    
    latency_ms = Column(Float, nullable=True) # Duration in ms
    tokens_used = Column(JSON, nullable=True) # {input, output, total}
    
    model_name = Column(String, nullable=True) # Model used for this run
    cost = Column(Float, nullable=True)        # Calculated cost in USD
    
    status = Column(String, default="success", index=True) # success, error
    error_message = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class SensitiveDataMigrationModel(Base):
    """Records completed one-time sensitive-data migrations.

    Keeping this marker in the database prevents large historical tables from
    being scanned on every application startup.
    """

    __tablename__ = "sensitive_data_migrations"

    name = Column(String, primary_key=True)
    completed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    changed_rows = Column(BigInteger, default=0, nullable=False)

class ArtifactModel(Base):
    """
    Stores metadata for files kept in artifact storage.

    The actual file is stored on the configured local/NFS filesystem.
    This table stores only the information required to locate it.
    """
    __tablename__ = "artifacts"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    thread_id = Column(
        String,
        nullable=False,
        index=True,
    )

    workflow_id = Column(
        String,
        nullable=True,
        index=True,
    )

    original_filename = Column(
        String,
        nullable=False,
    )

    relative_path = Column(
        String,
        nullable=False,
        unique=True,
    )

    content_type = Column(
        String,
        nullable=True,
    )

    size_bytes = Column(
        BigInteger,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )


class CheckpointArtifactReferenceModel(Base):
    """Tracks which LangGraph threads currently use a checkpoint artifact.

    A single content-addressed artifact may be referenced by several threads.
    The physical file is safe to delete only after the final reference is gone.
    Uploaded/user-owned artifacts are not recorded here and are therefore never
    removed by checkpoint lifecycle cleanup.
    """

    __tablename__ = "checkpoint_artifact_refs"

    thread_id = Column(String, primary_key=True)
    artifact_id = Column(
        String,
        ForeignKey("artifacts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    workflow_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class WorkflowRunHistoryModel(Base):
    """Compact terminal workflow history retained independently of checkpoints."""

    __tablename__ = "workflow_run_history"

    thread_id = Column(String, primary_key=True)
    workflow_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    output = Column(String, nullable=True)
    messages = Column(JSON, nullable=False, default=list)
    context = Column(JSON, nullable=False, default=dict)
    active_agent = Column(String, nullable=True)
    completed_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)

class MCPServerModel(Base):
    """
    SQLAlchemy Model for storing Model Context Protocol (MCP) server configurations.
    """
    __tablename__ = "mcp_servers"

    name = Column(String, primary_key=True, index=True, comment="Unique name for the MCP server")
    command = Column(String, nullable=False, comment="Executable command (e.g., npx, python)")
    args = Column(JSON, default=list, comment="List of arguments for the command")
    env = Column(JSON, default=dict, comment="Dictionary of environment variables")
    transport = Column(String, default="stdio", comment="Communication transport (usually stdio)")
    enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkflowNoteModel(Base):
    """
    SQLAlchemy Model for storing notes associated with a workflow.
    """
    __tablename__ = "workflow_notes"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("Workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    content = Column(String, nullable=False)
    category = Column(String, default="General", index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkflowVariableModel(Base):
    """
    SQLAlchemy Model for storing workflow-level variables.
    
    This is a generic, workflow-scoped key-value store that any workflow can use.
    Variables are defined at the workflow level (before any nodes) and can be
    consumed by any node in the workflow via dropdowns or runtime injection.
    
    Supported var_types:
      - text:        Single string value (e.g., a URL, a name)
      - number:      Numeric value (int or float)
      - boolean:     True/False toggle
      - list:        Array of strings (e.g., tags, keywords)
      - schema_list: Array of objects with 'name' and 'description' 
                     (e.g., agent classes: [{name: "FA", description: "..."}])
      - json_keys:   Auto-generated list of flattened JSON keys from a file upload
                     (read-only at runtime, refreshed by re-uploading)
      - json:        Freeform JSON object or array
    
    Example usage across workflows:
      - CD Workflow:  key="agent_classes", type="schema_list", value=[{name:"FA",...}]
      - HR Workflow:  key="departments",   type="list",        value=["Eng","Sales"]
      - Any Workflow: key="api_base_url",  type="text",        value="http://..."
    """
    __tablename__ = "workflow_variables"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("Workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    
    key = Column(String, nullable=False, index=True, comment="User-defined variable name (e.g., 'agent_classes')")
    value = Column(JSON, nullable=False, comment="Variable value — type depends on var_type")
    var_type = Column(String, nullable=False, default="text", comment="text|number|boolean|list|schema_list|json_keys|json")
    description = Column(String, nullable=True, comment="Optional hint displayed in the UI")
    
    # Metadata for json_keys type — stores the source filename
    source_filename = Column(String, nullable=True, comment="Original filename for json_keys type variables")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite unique constraint: one key per workflow
    __table_args__ = (
        UniqueConstraint('workflow_id', 'key', name='uq_workflow_variable_key'),
    )


class NotificationModel(Base):
    """
    SQLAlchemy Model for storing in-app notifications generated by workflow nodes.
    """
    __tablename__ = "notifications"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("Workflows.id", ondelete="CASCADE"), index=True, nullable=True)
    thread_id = Column(String, index=True, nullable=True)
    
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    type = Column(String, default="info") # info, success, warning, error
    
    is_read = Column(Boolean, default=False, index=True)
    
    # Optional metadata (JSON)
    metadata_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ToolStateModel(Base):
    """
    Generic model for storing tool-specific context (session IDs, credentials) per thread.
    Used to achieve persistent tool memory across horizontal scales.
    """
    __tablename__ = "tool_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, index=True, nullable=False)
    tool_key = Column(String, index=True, nullable=False) # e.g., 'azure_creds', 'aiml_session'
    
    # Context data stored as JSON (e.g., credentials, session IDs)
    state = Column(JSON, nullable=False)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AgentTemplateModel(Base):
    """
    SQLAlchemy Model for storing reusable Agent configurations (Templates).
    """
    __tablename__ = "agent_templates"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    group = Column(String, default="Default", index=True)
    category = Column(String, default="General", index=True)
    
    # Store the specific agent node configuration (type, task, llm, tools, etc.)
    config = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ConnectionProfileModel(Base):
    """
    SQLAlchemy Model for securely storing OAuth tokens and API keys for external platforms.
    This serves as the central registry for the Advanced API Connector.
    """
    __tablename__ = "api_connections"

    id = Column(String, primary_key=True, index=True, default=lambda: f"conn_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, index=True, nullable=True)              # RBAC ownership
    platform_name = Column(String, index=True, nullable=False)       # e.g., 'salesforce', 'custom_erp'
    
    # Core Auth Data
    auth_type = Column(String, nullable=False, default="None")       # OAuth2, Bearer, Basic, None
    access_token = Column(String, nullable=True)                     # Fernet ciphertext; never plaintext
    refresh_token = Column(String, nullable=True)                    # Fernet ciphertext; never plaintext
    expires_at = Column(DateTime, nullable=True)                     # Expiration for Lazy Refresh logic
    
    # Dynamic Enterprise OAuth2 Configurations
    oauth_client_id = Column(String, nullable=True)                  # Dynamic App ID
    oauth_client_secret = Column(String, nullable=True)              # Fernet ciphertext; never plaintext
    oauth_token_url = Column(String, nullable=True)                  # e.g., https://oauth2.googleapis.com/token
    
    # Heartbeat config
    enable_heartbeat = Column(Boolean, default=False)
    test_url = Column(String, nullable=True)
    last_used_at = Column(DateTime, default=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SkillModel(Base):
    """
    SQLAlchemy Model for storing reusable skills / Standard Operating Procedures (SOPs).
    """
    __tablename__ = "skills"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True, unique=True, comment="Shorthand unique identifier / filename")
    description = Column(String, nullable=True)
    content = Column(String, nullable=False, comment="Markdown SOP content")
    tags = Column(JSON, default=list, comment="List of tags associated with the skill")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class DocumentTemplateModel(Base):
    """
    SQLAlchemy Model for storing Document Templates used by the Report Generation Node.
    This is completely isolated and does not touch existing tables like Workflows or Execution Logs.
    """
    __tablename__ = "document_templates"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    category = Column(String, default="General", index=True)
    format_type = Column(String, nullable=False, index=True) # e.g., '.docx', '.pdf'
    storage_path = Column(String, nullable=False) # Path in the VFS or cloud
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CategoryModel(Base):
    """
    SQLAlchemy Model for storing global dictionary of categories (e.g., API categories).
    """
    __tablename__ = "api_categories"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)
    type = Column(String, nullable=False, default="api_node") # Component type this category belongs to
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ComponentTemplateModel(Base):
    """
    SQLAlchemy Model for storing reusable components like API Nodes.
    """
    __tablename__ = "api_component_templates"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False, index=True) # e.g., 'api_node'
    description = Column(String, nullable=True)
    
    group = Column(String, default="Default", index=True)
    category_id = Column(String, ForeignKey("api_categories.id"), nullable=True)
    category_name = Column(String, nullable=True) # Denormalized for fast reads
    
    config = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MarketplaceUsecaseModel(Base):
    """
    SQLAlchemy Model for storing published marketplace use cases.
    """
    __tablename__ = "marketplace_usecases"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    marketplace_name = Column(String, nullable=False)
    marketplace_description = Column(String, nullable=True)
    group = Column(String, default="Default", index=True)
    category = Column(String, default="General", index=True)
    owner = Column(String, default="system")
    workflow_id = Column(String, ForeignKey("Workflows.id", ondelete="CASCADE"), index=True, nullable=False)
    launch_url = Column(String, nullable=True, default=None)
    launch_mode = Column(String, default="default", index=True)
    navigation_path = Column(String, nullable=True, default=None)
    metadata_json = Column(JSON, nullable=True, name="metadata")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)
    publish = Column(Boolean, default=True, index=True)
    admin_status = Column(String, default="approved", index=True)
    remarks = Column(String, nullable=True)
    tags = Column(JSON, default=list)

class WorkflowTemplateModel(Base):
    """
    SQLAlchemy Model for storing Workflow Templates.
    NOTE: Owner is currently hardcoded/defaulted to 'system'. 
    Comments will highlight places to transition to role-based access controls (RBAC) later.
    """
    __tablename__ = "workflow_templates"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    group = Column(String, default="Default", index=True)
    category = Column(String, default="General", index=True)
    
    # Store the entire Pydantic WorkflowConfig model as JSON
    config = Column(JSON, nullable=False)
    
    # FUTURE TODO: Transition from hardcoded 'system' to token-derived user identity for RBAC
    owner = Column(String, default="system", index=True, nullable=False)
    
    # Scope: 'private', 'member', 'public'
    scope = Column(String, default="private", index=True, nullable=False)
    
    # FUTURE TODO: Add admin approval states ('pending', 'approved', 'rejected') and check role-based permissions
    status = Column(String, default="approved", index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
