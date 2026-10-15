import unittest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
from Apps.AgentMesh.Utilities.Security import SECRET_MASK, is_encrypted
from Database.AgentMeshCrud.mcp import list_mcp_servers, save_mcp_server
from Database.AgentMeshCrud.notifications import create_notification
from Database.AgentMeshCrud.variables import list_workflow_variables, upsert_workflow_variable
from Database.AgentMeshCrud.workflows import get_workflow, save_workflow
from Database.Models import MCPServerModel, NotificationModel, WorkflowModel, WorkflowVariableModel
from Apps.AgentMesh.Core.WebSocketManager import ConnectionManager


class SensitiveDataPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(WorkflowModel.__table__.create)
            await connection.run_sync(MCPServerModel.__table__.create)
            await connection.run_sync(WorkflowVariableModel.__table__.create)
            await connection.run_sync(NotificationModel.__table__.create)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_workflow_secret_is_encrypted_masked_on_edit_and_revealed_for_runtime(self):
        config = WorkflowConfig.model_validate({
            "id": "workflow-secure",
            "name": "Secure workflow",
            "nodes": [],
            "edges": [],
            "triggers": {"webhook": {"secret": "hook-value"}},
        })
        async with self.sessions() as db:
            await save_workflow(db, config)
            raw = (await db.execute(
                select(WorkflowModel).where(WorkflowModel.id == config.id)
            )).scalar_one()
            stored_secret = raw.draft_config["triggers"]["webhook"]["secret"]
            self.assertTrue(is_encrypted(stored_secret))
            self.assertNotIn("hook-value", stored_secret)

            masked_config = config.model_copy(deep=True)
            masked_config.triggers.webhook.secret = SECRET_MASK
            await save_workflow(db, masked_config)
            raw_after_edit = (await db.execute(
                select(WorkflowModel).where(WorkflowModel.id == config.id)
            )).scalar_one()
            self.assertEqual(
                raw_after_edit.draft_config["triggers"]["webhook"]["secret"],
                stored_secret,
            )

        async with self.sessions() as db:
            runtime = await get_workflow(db, config.id, reveal_secrets=True)
            self.assertEqual(
                runtime.draft_config["triggers"]["webhook"]["secret"],
                "hook-value",
            )

    async def test_mcp_values_are_encrypted_in_database_and_masked_by_default(self):
        async with self.sessions() as db:
            await save_mcp_server(
                db,
                name="secure-mcp",
                command="runner --token command-value",
                args=["--api-key", "argument-value"],
                env={"CUSTOM_CREDENTIAL": "environment-value"},
            )
            raw = await db.get(MCPServerModel, "secure-mcp")
            self.assertTrue(is_encrypted(raw.command))
            self.assertTrue(is_encrypted(raw.args[1]))
            self.assertTrue(is_encrypted(raw.env["CUSTOM_CREDENTIAL"]))

        async with self.sessions() as db:
            public = (await list_mcp_servers(db))[0]
            self.assertEqual(public.command, SECRET_MASK)
            self.assertEqual(public.args[1], SECRET_MASK)
            self.assertEqual(public.env["CUSTOM_CREDENTIAL"], SECRET_MASK)

        async with self.sessions() as db:
            sealed = (await list_mcp_servers(db, sealed=True))[0]
            self.assertTrue(is_encrypted(sealed.command))
            self.assertTrue(is_encrypted(sealed.args[1]))
            self.assertTrue(is_encrypted(sealed.env["CUSTOM_CREDENTIAL"]))

    async def test_sensitive_workflow_variable_is_encrypted_and_api_masked(self):
        async with self.sessions() as db:
            response = await upsert_workflow_variable(
                db,
                workflow_id="workflow-secure",
                key="vendor_api_token",
                value="variable-value",
                var_type="text",
            )
            self.assertEqual(response["value"], SECRET_MASK)
            raw = (await db.execute(select(WorkflowVariableModel))).scalar_one()
            self.assertTrue(is_encrypted(raw.value))

        async with self.sessions() as db:
            public = await list_workflow_variables(db, "workflow-secure")
            runtime = await list_workflow_variables(
                db,
                "workflow-secure",
                reveal_secrets=True,
            )
            self.assertEqual(public[0]["value"], SECRET_MASK)
            self.assertEqual(runtime[0]["value"], "variable-value")

    async def test_notifications_and_websocket_payloads_are_redacted(self):
        async with self.sessions() as db:
            notification = await create_notification(
                db,
                workflow_id=None,
                title="Credential alert",
                message="api_key=key-value",
                metadata_json={"access_token": "token-value"},
            )
            self.assertNotIn("key-value", notification.message)
            self.assertEqual(notification.metadata_json["access_token"], SECRET_MASK)

        class FakeConnection:
            def __init__(self):
                self.payload = None

            async def send_json(self, payload):
                self.payload = payload

        connection = FakeConnection()
        manager = ConnectionManager()
        await manager._send_to_connections(
            [connection],
            {"access_token": "token-value", "message": "client_secret=secret-value"},
        )
        self.assertNotIn("token-value", str(connection.payload))
        self.assertNotIn("secret-value", str(connection.payload))


if __name__ == "__main__":
    unittest.main()
