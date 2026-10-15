import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class WorkflowRegistry:
    """
    Runtime registry for compiled LangGraph graphs.

    Strategy: Lazy Load + Cache
    ─────────────────────────────────────────────────────────────────────────────
    - On startup: only pre-compile workflows flagged with warmup=True.
    - On first request for any other workflow: fetch config from DB, compile,
      cache. All subsequent requests hit the cache instantly.
    - This keeps startup fast and RAM lean regardless of how many workflows exist.
    ─────────────────────────────────────────────────────────────────────────────

    Checkpointer Strategy (2026-08-06):
    ─────────────────────────────────────────────────────────────────────────────
    Uses a shared **AsyncPostgresSaver** (backed by a global AsyncConnectionPool)
    so every node superstep is durably persisted to Postgres asynchronously.
    Initialized via checkpointer.init_checkpointer() in the FastAPI lifespan;
    shutdown via close_checkpointer(). Checkpoint TTL cleanup is owned by the
    Scheduler's SQL pruning job (AGENTMESH_CHECKPOINT_TTL_DAYS).
    ─────────────────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        # Compiled LangGraph app objects keyed by workflow_id
        self._compiled_graphs: Dict[str, Any] = {}
        # Parsed WorkflowConfig objects keyed by workflow_id
        self._configs: Dict[str, Any] = {}
        # Flat cache for node metadata, keyed by node_id: { "label": str, "mute_chat_bubble": bool }
        self._node_metadata_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("[Registry] Initialized using shared production checkpointer.")

    @property
    def checkpointer(self):
        from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer
        return get_checkpointer()

    def shutdown(self):
        """Checkpointer resources managed via FastAPI lifespan close_checkpointer()."""
        pass

    def clear_thread_checkpoint(self, thread_id: str) -> None:
        """No-op. Checkpoints are durably stored in PostgreSQL and pruned via periodic SQL TTL schedules."""
        pass

    # ─── Fast Node Metadata Caching ───────────────────────────────────────────

    def get_node_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fast O(1) lookup of node metadata (label, mute_chat_bubble) across registered workflows."""
        return self._node_metadata_cache.get(node_id)

    # ─── Compile & Cache ──────────────────────────────────────────────────────

    def register_and_compile(self, config) -> Any:
        """
        Compiles a WorkflowConfig and caches it.
        Replaces any existing cached version for this workflow_id.
        """
        from Apps.AgentMesh.Core.Compiler.GraphCompiler import GraphCompiler
        logger.info(f"[Registry] Compiling: '{config.name}' ({config.id})")
        app = GraphCompiler.compile(config, checkpointer=self.checkpointer)
        self._compiled_graphs[config.id] = app
        self._configs[config.id] = config
        
        # Populate flat node metadata cache
        for node in getattr(config, "nodes", []) or []:
            self._node_metadata_cache[node.id] = {
                "label": getattr(node, "label", None) or getattr(node, "name", None) or node.id,
                "mute_chat_bubble": getattr(node, "mute_chat_bubble", False)
            }

        logger.debug(f"[Registry] Cached. Total compiled: {len(self._compiled_graphs)}")
        return app

    async def get_executable_graph(self, workflow_id: str, db: Optional[AsyncSession] = None) -> Optional[Any]:
        """
        Returns the compiled graph for a workflow.

        Cache hit  → instant return.
        Cache miss → fetch config from DB, compile, cache, return.
                     Requires db session. Returns None if workflow not found.
        """
        # ── Cache hit ─────────────────────────────────────────────────────────
        if workflow_id in self._compiled_graphs:
            return self._compiled_graphs[workflow_id]

        # ── Cache miss: lazy compile ───────────────────────────────────────────
        if db is None:
            logger.warning(
                f"[Registry] Cache miss for '{workflow_id}' but no db session provided. "
                "Cannot lazy-compile. Was this workflow ever compiled?"
            )
            return None

        logger.info(f"[Registry] Cache miss — lazy compiling '{workflow_id}' from DB...")
        try:
            from Database.AgentMeshCrud import get_workflow
            from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig

            db_workflow = await get_workflow(db, workflow_id, reveal_secrets=True)
            if not db_workflow:
                logger.error(f"[Registry] Workflow '{workflow_id}' not found in DB.")
                return None

            config = WorkflowConfig.model_validate(db_workflow.config)
            return self.register_and_compile(config)

        except Exception as e:
            logger.error(f"[Registry] Lazy compile failed for '{workflow_id}': {e}", exc_info=True)
            return None

    def get_config(self, workflow_id: str) -> Optional[Any]:
        return self._configs.get(workflow_id)

    def get_all_configs(self) -> list:
        return list(self._configs.values())

    def unregister(self, workflow_id: str):
        """Remove a workflow from the cache (e.g. after delete or recompile)."""
        removed = self._compiled_graphs.pop(workflow_id, None)
        config = self._configs.pop(workflow_id, None)
        if config:
            for node in getattr(config, "nodes", []) or []:
                self._node_metadata_cache.pop(node.id, None)
        if removed:
            logger.info(f"[Registry] Unregistered '{workflow_id}'.")

    # ─── Startup Warmup ───────────────────────────────────────────────────────

    def load_from_db(self, workflow_models: list):
        """
        Pre-compiles only warmup=True workflows at startup.
        All other workflows are lazily compiled on first request.

        Args:
            workflow_models: Full WorkflowModel SQLAlchemy objects (must include config).
        """
        warmup_models = [m for m in workflow_models if getattr(m, 'warmup', False)]
        total = len(workflow_models)
        warmup_count = len(warmup_models)

        logger.info(
            f"[Registry] Startup: {total} workflows found. "
            f"Pre-compiling {warmup_count} warmup workflow(s). "
            f"{total - warmup_count} will be lazily compiled on first request."
        )

        from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
        for model in warmup_models:
            try:
                config = WorkflowConfig.model_validate(model.config)
                self.register_and_compile(config)
            except Exception as e:
                logger.error(f"[Registry] Warmup compile failed for '{model.id}': {e}")


# Global singleton
workflow_registry = WorkflowRegistry()
