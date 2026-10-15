import asyncio
import os
import sys
import json

sys.path.append(os.getcwd())
from Apps.AgentMesh.Utilities.Tools.SqlTool.sql_tools import sql_configure_auth, sql_query_executor

async def run_final():
    sql_password = os.environ.get("AGENTMESH_TEST_SQL_PASSWORD")
    if not sql_password:
        raise RuntimeError("AGENTMESH_TEST_SQL_PASSWORD must be set to run this integration test")
    test_config = {"configurable": {"thread_id": "test_sql_final_output"}}
    schema = "aiml_predictions_destination"
    table = "insurance_multiclass_pred_dest_copy"
    results = {}

    await sql_configure_auth.ainvoke({
        "host": os.environ.get("AGENTMESH_TEST_SQL_HOST", "127.0.0.1"),
        "user": os.environ.get("AGENTMESH_TEST_SQL_USER", "postgres"),
        "password": sql_password,
        "database": os.environ.get("AGENTMESH_TEST_SQL_DATABASE", "postgres"),
        "schema_name": schema, "table_name": table
    }, config=test_config)

    # 1. Info of CUST_00551
    res_1 = await sql_query_executor.ainvoke({
        "sql_query": f"SELECT * FROM \"{schema}\".\"{table}\" WHERE \"customer_id\" = 'CUST_00551'"
    }, config=test_config)
    results["customer_info"] = json.loads(res_1)

    # 2. Count Score > 500
    res_2 = await sql_query_executor.ainvoke({
        "sql_query": f"SELECT COUNT(*) as total FROM \"{schema}\".\"{table}\" WHERE \"credit_score\" > 500"
    }, config=test_config)
    results["high_score_count"] = json.loads(res_2)

    with open("sql_test_final.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_final())
