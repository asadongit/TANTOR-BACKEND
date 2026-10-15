import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

def register_agent_mesh_mounts(app: FastAPI):
    """
    Handles directory creation and static file mounting for AgentMesh.
    Moves the 'Mounting' complexity out of main.py.
    """
    # 1. Mount Temporary Files — ALL session data lives here
    if not os.path.exists(settings.TEMP_MOUNT):
        os.makedirs(settings.TEMP_MOUNT, exist_ok=True)
    app.mount("/temp", StaticFiles(directory=settings.TEMP_MOUNT), name="temp")
    logger.info(f"Mounted AgentMesh temp directory: {settings.TEMP_MOUNT}")

    # Ensure durable artifact storage exists.
    # This is local during development and may be an NFS mount in production.
    os.makedirs(settings.ARTIFACT_MOUNT, exist_ok=True)
    logger.info(f"AgentMesh artifact storage directory: {settings.ARTIFACT_MOUNT}")

    # 2. Mount Exported Files (under /temp/exports/{thread_id}/, also accessible via /exports)
    if not os.path.exists(settings.EXPORTS_MOUNT):
        os.makedirs(settings.EXPORTS_MOUNT, exist_ok=True)
    app.mount("/exports", StaticFiles(directory=settings.EXPORTS_MOUNT), name="exports")
    logger.info(f"Mounted AgentMesh exports directory: {settings.EXPORTS_MOUNT}")

@asynccontextmanager
async def agent_mesh_lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifecycle for AgentMesh services.
    Encapsulates Registry Hydration, Scheduler control, and Tool Discovery.
    """
    from Apps.AgentMesh.Utilities.Scheduler import scheduler
    from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
    from Apps.AgentMesh.Core.ToolRegistry import registry
    from Apps.AgentMesh.Core.RedisManager import redis_manager
    from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
    from Apps.AgentMesh.Utilities.Tunnel import TunnelManager
    from Database.Database import AsyncSessionLocal
    from Database import AgentMeshCrud

    from Apps.AgentMesh.Utilities.Memory.checkpointer import init_checkpointer, close_checkpointer

    logger.info("Starting AgentMesh domain services...")

    # 0. Initialize persistent checkpointer (AsyncPostgresSaver)
    await init_checkpointer()

    # 0. Connect Redis (concurrency guards)
    await redis_manager.connect()

    # 0. Start Webhook Tunnel (External Access) — only when explicitly enabled
    if settings.NGROK_ENABLED:
        asyncio.create_task(TunnelManager.start(port=8000))
    else:
        logger.info("Ngrok tunnel disabled (NGROK_ENABLED=false). Skipping.")

    # 1. Registry Hydration — only pre-compile warmup=True workflows
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            from Database.Models import WorkflowModel
            # One-time/idempotent upgrade for all credential-bearing rows
            # created before the common sensitive-data framework.
            await AgentMeshCrud.migrate_legacy_sensitive_data(db)

            # Fetch warmup workflows OR active workflows with enabled triggers (webhook/cron/polling)
            from sqlalchemy import or_
            warmup_stmt = select(WorkflowModel).where(
                WorkflowModel.is_active == True,
                or_(
                    WorkflowModel.warmup == True,
                    WorkflowModel.config["triggers"]["webhook"]["enabled"].as_boolean() == True,
                    WorkflowModel.config["triggers"]["cron"]["enabled"].as_boolean() == True,
                    WorkflowModel.config["triggers"]["polling"]["enabled"].as_boolean() == True
                )
            )
            try:
                warmup_result = await db.execute(warmup_stmt)
                warmup_models = warmup_result.scalars().all()
            except Exception:
                # Fallback to loading all active workflows if json path query is unsupported
                all_stmt = select(WorkflowModel).where(WorkflowModel.is_active == True)
                all_res = await db.execute(all_stmt)
                warmup_models = all_res.scalars().all()

            # Decrypt credentials before compiling
            from Database.AgentMeshCrud import process_workflow_credentials
            for m in warmup_models:
                if m.config:
                    # Keep runtime plaintext detached so autoflush can never
                    # persist it during the following scheduler query.
                    db.expunge(m)
                    m.config = process_workflow_credentials(m.config, mode="decrypt")

            workflow_registry.load_from_db(warmup_models)

            # For the scheduler we need trigger configs — fetch all active workflow triggers
            # using a lightweight query (only id + config columns needed for trigger parsing)
            trigger_stmt = select(WorkflowModel.id, WorkflowModel.config).where(
                WorkflowModel.is_active == True
            )
            trigger_result = await db.execute(trigger_stmt)
            trigger_rows = trigger_result.all()

            scheduler.start()
            
            # Register daily database pruning maintenance task
            try:
                scheduler.add_pruning_schedule()
            except Exception as pe:
                logger.error(f"Failed to register database pruning task: {pe}")

            for wf_id, wf_config_raw in trigger_rows:
                try:
                    decrypted = process_workflow_credentials(wf_config_raw or {}, mode="decrypt")
                    config = WorkflowConfig.model_validate(decrypted)
                    if config.triggers:
                        if config.triggers.cron and config.triggers.cron.enabled:
                            scheduler.add_workflow_schedule(
                                wf_id,
                                config.triggers.cron.cron_expression,
                                config.triggers.cron.timezone
                            )
                        if config.triggers.polling and config.triggers.polling.enabled:
                            poll = config.triggers.polling
                            scheduler.add_polling_schedule(
                                workflow_id=wf_id,
                                tool_name=poll.poller_tool,
                                tool_args=poll.tool_arguments,
                                destination_key=poll.destination_key,
                                cron_expression=poll.cron_expression,
                                timezone=poll.timezone
                            )
                except Exception as ex:
                    logger.warning(f"Scheduler: skipped triggers for '{wf_id}': {ex}")

            total_active = len(trigger_rows)
            logger.info(
                f"AgentMesh Scheduler initialized. "
                f"{total_active} active workflow(s) scanned for triggers."
            )
    except Exception as e:
        # Migration/key failures must stop startup. Continuing could leave
        # legacy plaintext accessible while giving a false secure state.
        logger.exception("AgentMesh hydration or sensitive-data migration failed")
        raise RuntimeError("AgentMesh secure startup failed") from e

    # 2. Tool Discovery (Independent of DB)
    try:
        # Local Tool Discovery
        registry.discover_local_tools(settings.TOOLS_PATH)
        
        # Sync MCP Servers from DB
        async with AsyncSessionLocal() as db:
            mcp_servers = await AgentMeshCrud.list_mcp_servers(db, sealed=True)
            for s in mcp_servers:
                registry.add_mcp_server(s.name, {
                    "command": s.command,
                    "args": s.args,
                    "env": s.env,
                    "transport": s.transport
                })
            
            if mcp_servers:
                logger.info(f"Syncing {len(mcp_servers)} MCP servers...")
                await registry.sync_mcp_tools()
                
        logger.info(f"Tool discovery complete. Total tools available: {len(registry.get_all_meta())}")
    except Exception as e:
        logger.error(f"AgentMesh Tool discovery failed: {e}")

    # --- STARTUP COMPLETE ---
    yield
    # --- SHUTDOWN LOGIC ---

    logger.info("Shutting down AgentMesh domain services...")
    try:
        scheduler.stop()
        logger.info("AgentMesh Scheduler stopped.")

        # Disconnect Redis
        await redis_manager.close()

        # Close the shared checkpointer pool
        await close_checkpointer()
        workflow_registry.shutdown()

        # Shutdown all active MCP sessions
        asyncio.create_task(registry.shutdown())

        # Shutdown Webhook Tunnel
        asyncio.create_task(TunnelManager.stop())

        logger.info("MCP and Tunnel sessions shutdown initiated.")
    except Exception as e:
        logger.error(f"Error during AgentMesh shutdown: {e}")
