from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import logging
from typing import Dict, Any, Optional
import datetime
import httpx
from sqlalchemy import select
from Database.Database import AsyncSessionLocal
from Database.Models import ConnectionProfileModel
from Apps.AgentMesh.Core.Connections.OAuthManager import OAuthManager
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

class WorkflowScheduler:
    """
    Manages scheduled execution of AgentMesh workflows using APScheduler.
    """
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._jobs: Dict[str, Any] = {}

    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            # Register Heartbeat Monitor (Runs every 5 minutes)
            self.scheduler.add_job(
                self._monitor_heartbeats,
                IntervalTrigger(minutes=5),
                id="api_heartbeat_monitor",
                replace_existing=True
            )
            self.scheduler.add_job(
                self._execute_history_cleanup,
                IntervalTrigger(seconds=settings.HISTORY_CLEANUP_INTERVAL_SECONDS),
                id="agentmesh_history_ttl_cleanup",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self.scheduler.start()
            logger.info(
                "Workflow Scheduler started; completed history cleanup runs every %s seconds.",
                settings.HISTORY_CLEANUP_INTERVAL_SECONDS,
            )

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Workflow Scheduler stopped.")

    async def _execute_history_cleanup(self):
        """Delete compact workflow history after its configured expiration."""
        from Apps.AgentMesh.Core.Services.WorkflowHistoryService import WorkflowHistoryService

        try:
            async with AsyncSessionLocal() as db:
                deleted = await WorkflowHistoryService.delete_expired(db)
            if deleted:
                logger.info("Deleted %s expired workflow history row(s).", deleted)
        except Exception:
            logger.exception("Workflow history TTL cleanup failed; it will retry.")

    def add_workflow_schedule(self, workflow_id: str, cron_expression: str, timezone: str = "UTC"):
        """
        Adds or updates a schedule for a specific workflow.
        """
        job_id = f"job_{workflow_id}"
        
        # Remove existing job if any
        if job_id in self._jobs:
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass

        # Add new job
        try:
            self.scheduler.add_job(
                self._execute_workflow,
                CronTrigger.from_crontab(cron_expression, timezone=timezone),
                id=job_id,
                args=[workflow_id],
                replace_existing=True,
                misfire_grace_time=3600
            )
            self._jobs[job_id] = cron_expression
            logger.info(f"Scheduled workflow '{workflow_id}' with cron '{cron_expression}'")
        except Exception as e:
            logger.error(f"Failed to schedule workflow '{workflow_id}': {e}")

    def pause_workflow_schedule(self, workflow_id: str):
        """Pauses a scheduled workflow/poller."""
        cron_job_id = f"job_{workflow_id}"
        poll_job_id = f"poll_{workflow_id}"
        for job_id in [cron_job_id, poll_job_id]:
            if job_id in self._jobs:
                try:
                    self.scheduler.pause_job(job_id)
                    logger.info(f"Paused job '{job_id}'")
                except Exception as e:
                    logger.warning(f"Could not pause job '{job_id}': {e}")

    def resume_workflow_schedule(self, workflow_id: str):
        """Resumes a paused scheduled workflow/poller."""
        cron_job_id = f"job_{workflow_id}"
        poll_job_id = f"poll_{workflow_id}"
        for job_id in [cron_job_id, poll_job_id]:
            if job_id in self._jobs:
                try:
                    self.scheduler.resume_job(job_id)
                    logger.info(f"Resumed job '{job_id}'")
                except Exception as e:
                    logger.warning(f"Could not resume job '{job_id}': {e}")

    def remove_workflow_schedule(self, workflow_id: str):
        """Removes a schedule/polling for a specific workflow."""
        cron_job_id = f"job_{workflow_id}"
        poll_job_id = f"poll_{workflow_id}"
        for job_id in [cron_job_id, poll_job_id]:
            if job_id in self._jobs:
                try:
                    self.scheduler.remove_job(job_id)
                    del self._jobs[job_id]
                    logger.info(f"Removed scheduled job '{job_id}'")
                except Exception as e:
                    logger.error(f"Failed to remove scheduled job '{job_id}': {e}")

    def add_polling_schedule(
        self, 
        workflow_id: str, 
        tool_name: str, 
        tool_args: dict, 
        destination_key: str,
        cron_expression: str,
        timezone: str = "UTC"
    ):
        """
        Adds or updates an active polling schedule for a specific workflow.
        Runs via cron_expression.
        """
        job_id = f"poll_{workflow_id}"
        
        # Remove existing job if any
        if job_id in self._jobs:
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass

        # Build cron trigger
        trigger = CronTrigger.from_crontab(cron_expression, timezone=timezone)
        schedule_desc = f"cron '{cron_expression}'"

        # Add new job
        try:
            self.scheduler.add_job(
                self._execute_polling,
                trigger,
                id=job_id,
                args=[workflow_id, tool_name, tool_args, destination_key],
                replace_existing=True,
                misfire_grace_time=3600,
                next_run_time=datetime.datetime.now() # Execute once immediately for responsive startup
            )
            
            self._jobs[job_id] = schedule_desc
            logger.info(f"Scheduled active polling for workflow '{workflow_id}' with tool '{tool_name}' using {schedule_desc}")
        except Exception as e:
            logger.error(f"Failed to schedule polling for workflow '{workflow_id}': {e}")

    async def _execute_workflow(self, workflow_id: str):
        """
        Target function for the scheduler. 
        Calls the workflow registry to get the graph and runs it.
        """
        from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
        from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer
        from Apps.AgentMesh.Utilities.Observer import get_observer
        
        logger.info(f"Triggering scheduled workflow: {workflow_id}")

        # Scheduler runs outside HTTP context — no db session available.
        # Scheduled workflows should have warmup=True set to ensure they're
        # always pre-compiled. db=None means cache-hit only.
        import asyncio
        app_instance = await workflow_registry.get_executable_graph(workflow_id, db=None)
        if not app_instance:
            logger.warning(
                f"Scheduled workflow '{workflow_id}' not in registry. "
                "Consider setting warmup=True for scheduled workflows to ensure "
                "they are always pre-compiled at startup."
            )
            return

        request_id = f"cron_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Attach observer and langfuse handler so cron runs produce traces in Observability / Langfuse
        observer = get_observer(workflow_id=workflow_id)
        callbacks = [observer]
        
        from Apps.AgentMesh.Utilities.Observer import get_langfuse_handler, flush_langfuse
        lf_handler = get_langfuse_handler(trace_id=request_id)
        if lf_handler:
            callbacks.append(lf_handler)
        
        config = {
            "configurable": {"thread_id": request_id},
            "callbacks": callbacks,
        }
        state = {
            "messages": [("user", "System scheduled trigger start")],
            "context": {"source": "cron_trigger", "trigger_time": datetime.datetime.now().isoformat()}
        }
            
        try:
            await app_instance.ainvoke(state, config=config)
            flush_langfuse()
            logger.info(f"Scheduled workflow {workflow_id} completed successfully.")
        except Exception as e:
            flush_langfuse()
            logger.error(f"Error in scheduled workflow {workflow_id}: {e}")

    async def _execute_polling(self, workflow_id: str, tool_name: str, tool_args: dict, destination_key: str):
        """
        Target function for background polling trigger. Runs programmatic tools and
        triggers the workflowService if new items are detected.
        """
        from Apps.AgentMesh.Core.ToolRegistry import registry
        from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
        import inspect

        logger.debug(f"Executing poller trigger for workflow '{workflow_id}' using tool '{tool_name}'")
        
        try:
            # 1. Fetch tool from registry
            tool = registry.get_tool(tool_name)
            if not tool:
                logger.error(f"Polling Scheduler: Tool '{tool_name}' not found for workflow '{workflow_id}'")
                return

            # --- Safe Parameter Type Casting using tool's args_schema ---
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    if hasattr(tool.args_schema, "model_validate"):
                        validated_model = tool.args_schema.model_validate(tool_args)
                        tool_args = validated_model.model_dump()
                    else:
                        validated_model = tool.args_schema(**tool_args)
                        tool_args = getattr(validated_model, "dict", lambda: dict(validated_model))()
                except Exception as e:
                    logger.warning(f"Polling Scheduler: Dynamic parameter coercion warning for '{tool_name}': {e}. Using raw parameters.")

            # 2. Extract underlying executable function
            wrapped_func = getattr(tool, 'coroutine', None) or getattr(tool, 'func', None)
            if not wrapped_func:
                logger.error(f"Polling Scheduler: Tool '{tool_name}' lacks executable attributes.")
                return

            # 3. Create RunnableConfig to inject thread isolation
            thread_id = f"poll_run_{workflow_id}"
            config = {"configurable": {"context": {}, "thread_id": thread_id}}
            
            # 4. Invoke tool programmatically
            final_input = {**tool_args, "config": config}
            
            is_async = getattr(tool, 'coroutine', None) is not None or inspect.iscoroutinefunction(wrapped_func)
            if is_async:
                result = await wrapped_func(**final_input)
            else:
                result = wrapped_func(**final_input)

            # 5. Check if we received new records/data (truthy result, non-empty list, dict, or string)
            has_data = False
            if result is not None:
                if isinstance(result, list):
                    has_data = len(result) > 0
                elif isinstance(result, dict):
                    has_data = len(result) > 0
                elif isinstance(result, str):
                    has_data = len(result.strip()) > 0
                else:
                    has_data = bool(result)

            logger.info(f"Polling Scheduler: Tool '{tool_name}' returned result of type '{type(result).__name__}' (has_data={has_data})")
            
            if has_data:
                logger.info(f"Polling Scheduler: Triggering workflow '{workflow_id}' with new items!")
                
                # 6. Trigger the Workflow directly with the seeded payload
                # Use source = "webhook" to match GraphCompiler's entry router
                run_thread_id = f"poll_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                await WorkflowService.run(
                    workflow_id=workflow_id,
                    input_text=f"System Poller Trigger: New data found via {tool_name}",
                    thread_id=run_thread_id,
                    context={
                        "source": "polling",
                        destination_key: result
                    }
                )
        except Exception as e:
            logger.error(f"Polling Scheduler: Job failed for workflow '{workflow_id}': {e}")

    async def _monitor_heartbeats(self):
        """
        Background task to monitor API connection health.
        Implements Idle Suspend: Only checks connections used within the last 24 hours.
        """
        try:
            async with AsyncSessionLocal() as db:
                # 1. Fetch active connections needing heartbeats
                twenty_four_hours_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
                stmt = select(ConnectionProfileModel).where(
                    ConnectionProfileModel.enable_heartbeat == True,
                    ConnectionProfileModel.test_url.isnot(None),
                    ConnectionProfileModel.last_used_at > twenty_four_hours_ago
                )
                result = await db.execute(stmt)
                connections = result.scalars().all()
                
                if not connections:
                    return

                logger.info(f"[Heartbeat Monitor] Checking health for {len(connections)} active connections...")
                
                for conn in connections:
                    try:
                        # 2. Lazy Refresh Token if needed
                        token = None
                        if conn.auth_type in ["OAuth2", "Bearer"]:
                            token = await OAuthManager.get_valid_token(db, conn.id)
                        
                        # 3. Ping the test URL
                        headers = {"Authorization": f"Bearer {token}"} if token else {}
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            res = await client.get(conn.test_url, headers=headers)
                            res.raise_for_status()
                            
                    except Exception as e:
                        # 4. Alert on Failure
                        logger.error(f"[Heartbeat Monitor] 🚨 FAILURE for '{conn.platform_name}' (ID: {conn.id}): {e}")
                        # FUTURE ENHANCEMENT: Integrate with NotificationModel here to alert users
                        
        except Exception as e:
            logger.error(f"[Heartbeat Monitor] Critical failure in monitor loop: {e}")

    def add_pruning_schedule(self, cron_expression: str = "0 2 * * *", timezone: str = "UTC"):
        """
        Adds a daily maintenance job to prune checkpoints and apply database TTL.
        Default: Run daily at 2:00 AM.
        """
        import os
        enabled = os.getenv("AGENTMESH_PRUNING_ENABLED", "true").lower() == "true"
        if not enabled:
            logger.info("Database pruning is disabled by AGENTMESH_PRUNING_ENABLED env var.")
            return

        job_id = "agentmesh_pruning_job"
        
        # Remove existing job if any
        if job_id in self._jobs:
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass
                
        cron_expression = os.getenv("AGENTMESH_PRUNING_CRON", cron_expression)
        
        try:
            self.scheduler.add_job(
                self._execute_pruning,
                CronTrigger.from_crontab(cron_expression, timezone=timezone),
                id=job_id,
                replace_existing=True,
                misfire_grace_time=3600
            )
            self._jobs[job_id] = cron_expression
            logger.info(f"Scheduled daily database pruning with cron '{cron_expression}'")
        except Exception as e:
            logger.error(f"Failed to schedule database pruning: {e}")

    async def _execute_pruning(self):
        """
        Prunes intermediate checkpoints and deletes expired thread histories.
        Uses optimized database queries for PostgreSQL or native savers for fallback.
        """
        logger.info("Executing scheduled database pruning...")
        try:
            import os
            from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer
            from Database.Database import AsyncSessionLocal
            from sqlalchemy import text
            
            db_url = os.getenv("DATABASE_URL", "")
            is_postgres = "postgres" in db_url.lower()

            if is_postgres:
                logger.info("Postgres database checkpointer detected. Using high-performance SQL pruning...")
                try:
                    days_old = int(os.getenv("AGENTMESH_CHECKPOINT_TTL_DAYS", "30"))
                    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_old)
                    
                    async with AsyncSessionLocal() as db:
                        async with db.begin():
                            # 1. HARD TTL: Delete entire threads older than TTL
                            logger.info(f"Deleting entire threads older than {days_old} days (cutoff: {cutoff_date})...")
                            find_stmt = text('''
                                SELECT thread_id
                                FROM checkpoints
                                GROUP BY thread_id
                                HAVING MAX((checkpoint->>'ts')::timestamptz) < :cutoff_date
                            ''')
                            res = await db.execute(find_stmt, {"cutoff_date": cutoff_date})
                            expired_threads = [row[0] for row in res.all()]
                            
                            if expired_threads:
                                logger.info(f"Found {len(expired_threads)} expired threads to prune: {expired_threads}")
                                
                                # Delete expired checkpoints
                                await db.execute(
                                    text("DELETE FROM checkpoints WHERE thread_id = ANY(:threads)"),
                                    {"threads": expired_threads}
                                )
                                # Delete orphan writes and blobs
                                await db.execute(text('''
                                    DELETE FROM checkpoint_writes
                                    WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM checkpoints)
                                '''))
                                await db.execute(text('''
                                    DELETE FROM checkpoint_blobs
                                    WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM checkpoints)
                                '''))

                                # 1b. SYNC HISTORY TAB WITH CHECKPOINT TTL: delete the execution_logs
                                # rows for the SAME expired threads so the History list and the
                                # checkpointer expire together (no more "shows in History but reopens empty").
                                await db.execute(
                                    text("DELETE FROM execution_logs WHERE thread_id = ANY(:threads)"),
                                    {"threads": expired_threads}
                                )
                                logger.info("Successfully deleted expired thread checkpoints, writes, blobs, and execution_logs.")
                            else:
                                logger.info("No expired threads found.")

                            # 2. INTERMEDIATE PRUNING: For surviving active threads, prune intermediate checkpoints, writes, and blobs
                            logger.info("Pruning intermediate checkpoints for remaining active threads...")
                            
                            # 2a. Delete checkpoints
                            del_checkpoints = await db.execute(text('''
                                DELETE FROM checkpoints
                                WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (
                                    WITH latest_checkpoints AS (
                                        SELECT DISTINCT ON (thread_id, checkpoint_ns)
                                            thread_id,
                                            checkpoint_ns,
                                            checkpoint_id
                                        FROM checkpoints
                                        ORDER BY thread_id, checkpoint_ns, (checkpoint->>'ts')::timestamptz DESC
                                    )
                                    SELECT c.thread_id, c.checkpoint_ns, c.checkpoint_id
                                    FROM checkpoints c
                                    LEFT JOIN latest_checkpoints l
                                        ON c.thread_id = l.thread_id
                                        AND c.checkpoint_ns = l.checkpoint_ns
                                        AND c.checkpoint_id = l.checkpoint_id
                                    WHERE l.checkpoint_id IS NULL
                                )
                            '''))
                            
                            # 2b. Delete intermediate writes
                            del_writes = await db.execute(text('''
                                DELETE FROM checkpoint_writes
                                WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                                    SELECT DISTINCT thread_id, checkpoint_ns, checkpoint_id
                                    FROM checkpoints
                                )
                            '''))

                            # 2c. Delete intermediate/unused blobs
                            del_blobs = await db.execute(text('''
                                DELETE FROM checkpoint_blobs b
                                WHERE NOT EXISTS (
                                    WITH surviving_checkpoints AS (
                                        SELECT DISTINCT ON (thread_id, checkpoint_ns)
                                            thread_id,
                                            checkpoint_ns,
                                            checkpoint
                                        FROM checkpoints
                                        ORDER BY thread_id, checkpoint_ns, (checkpoint->>'ts')::timestamptz DESC
                                    )
                                    SELECT 1
                                    FROM surviving_checkpoints s,
                                    LATERAL jsonb_each_text(s.checkpoint->'channel_versions') kv
                                    WHERE s.thread_id = b.thread_id
                                      AND s.checkpoint_ns = b.checkpoint_ns
                                      AND kv.value = b.version
                                )
                            '''))

                            logger.info(
                                f"Postgres pruning stats: "
                                f"Deleted {del_checkpoints.rowcount} intermediate checkpoints, "
                                f"{del_writes.rowcount} intermediate writes, "
                                f"{del_blobs.rowcount} intermediate blobs."
                            )

                        # PostgreSQL and NFS cannot share one transaction. Retry
                        # any artifact release that failed during run completion,
                        # and release artifacts for threads removed by the TTL SQL.
                        from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService

                        async with AsyncSessionLocal() as artifact_db:
                            artifact_stats = (
                                await ArtifactService.release_orphaned_checkpoint_artifacts(
                                    artifact_db
                                )
                            )
                        logger.info(
                            "Checkpoint artifact cleanup: "
                            f"threads={artifact_stats['threads']}, "
                            f"deleted={artifact_stats['deleted']}, "
                            f"preserved={artifact_stats['preserved']}, "
                            f"failed={artifact_stats['failed']}"
                        )
                except Exception as e:
                    logger.error(f"Postgres database pruning failed: {e}")
            else:
                logger.info("Non-Postgres checkpointer detected. Falling back to native/saver pruning...")
                # Fetch thread IDs
                thread_ids = []
                try:
                    async with AsyncSessionLocal() as db:
                        result = await db.execute(text("SELECT DISTINCT thread_id FROM checkpoints"))
                        thread_ids = [row[0] for row in result.all()]
                except Exception as e:
                    logger.debug(f"Could not fetch thread IDs for native saver: {e}")
                    return

                if thread_ids:
                    try:
                        saver = get_checkpointer()
                        if hasattr(saver, "aprune"):
                            await saver.aprune(thread_ids, strategy="keep_latest")
                            logger.info("Native intermediate checkpoints pruning (aprune) completed.")
                        elif hasattr(saver, "prune"):
                            saver.prune(thread_ids, strategy="keep_latest")
                            logger.info("Native intermediate checkpoints pruning (prune) completed.")
                        else:
                            logger.warning("Checkpointer does not support native pruning methods.")
                    except NotImplementedError:
                        logger.debug("Native pruning is not implemented by the active saver.")
                    except Exception as e:
                        logger.error(f"Error during native checkpointer pruning: {e}")
                        
            # 3. DIRECTORY SWEEP: Prune VFS directories in mnt/deep_agents older than TTL days
            try:
                import time
                import shutil
                from Apps.AgentMesh.Config import settings
                days_old = int(os.getenv("AGENTMESH_CHECKPOINT_TTL_DAYS", "30"))
                deep_agent_root = settings.DEEP_AGENTS_MOUNT
                if os.path.exists(deep_agent_root):
                    logger.info(f"Sweeping VFS root '{deep_agent_root}' to prune directories older than {days_old} days...")
                    cutoff_ts = time.time() - (days_old * 86400)
                    pruned_count = 0
                    for item in os.listdir(deep_agent_root):
                        item_path = os.path.join(deep_agent_root, item)
                        if os.path.isdir(item_path):
                            mtime = os.path.getmtime(item_path)
                            if mtime < cutoff_ts:
                                logger.info(f"Pruning expired DeepAgent VFS directory: {item_path}")
                                shutil.rmtree(item_path, ignore_errors=True)
                                pruned_count += 1
                    if pruned_count > 0:
                        logger.info(f"Successfully pruned {pruned_count} expired DeepAgent VFS directories.")
                    else:
                        logger.info("No expired DeepAgent VFS directories found.")
            except Exception as fe:
                logger.error(f"Failed to prune DeepAgent VFS directories: {fe}")

            # 4. SESSION SWEEP: Prune expired temp session directories under TEMP_MOUNT
            # Only prune {thread_id}/ folders — NEVER touch known mount subdirs
            try:
                protected = {sub.split("/")[-1] for sub in settings._MOUNT_SUBDIRS.values()}
                temp_root = settings.TEMP_MOUNT
                if os.path.exists(temp_root):
                    logger.info(f"Sweeping temp root '{temp_root}' to prune session dirs older than {days_old} days...")
                    cutoff_ts = time.time() - (days_old * 86400)
                    pruned_count = 0
                    for item in os.listdir(temp_root):
                        if item in protected or item.startswith("."):
                            continue
                        item_path = os.path.join(temp_root, item)
                        if os.path.isdir(item_path):
                            mtime = os.path.getmtime(item_path)
                            if mtime < cutoff_ts:
                                logger.info(f"Pruning expired session directory: {item_path}")
                                shutil.rmtree(item_path, ignore_errors=True)
                                pruned_count += 1
                    if pruned_count > 0:
                        logger.info(f"Successfully pruned {pruned_count} expired temp session directories.")
                    else:
                        logger.info("No expired temp session directories found.")
            except Exception as fe:
                logger.error(f"Failed to prune temp session directories: {fe}")
                
        except Exception as e:
            logger.error(f"Database pruning job failed: {e}")

# Global singleton
scheduler = WorkflowScheduler()
