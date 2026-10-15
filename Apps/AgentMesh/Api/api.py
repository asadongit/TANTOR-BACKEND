from fastapi import APIRouter
from Apps.AgentMesh.Api.routers import workflow, execution, tool, rag, system, deep_agent, agent_template, connections, category, component_template, workflow_template, workflow_variables, observability, node_testing

# Main API Router
router = APIRouter()

# Include Sub-Modules (Modular Architecture)
# Each module handles a specific Domain of the application

# 1. Workflow Management Domain
router.include_router(workflow.router)

# 1b. Workflow Variables (Generic workflow-level config)
router.include_router(workflow_variables.router)

# 2. Execution & Orchestration Domain
router.include_router(execution.router)

# 3. Tool Discovery & MCP Domain
router.include_router(tool.router)

# 4. RAG & Data Ingestion Domain
router.include_router(rag.router)

# 5. System, Health & Notifications Domain
router.include_router(system.router)
router.include_router(node_testing.router)

# 6. Deep Agent Specialized Protocols (VFS, Skills)
router.include_router(deep_agent.router)
router.include_router(agent_template.router)
router.include_router(workflow_template.router)
from Apps.AgentMesh.Api.routers import report_templates, dummy_apis
router.include_router(report_templates.router)
router.include_router(dummy_apis.router)

# 7. Enterprise Connections Vault
router.include_router(connections.router)

# 8. API Globalisation and Component Templates
router.include_router(category.router)
router.include_router(component_template.router)

# 9. Observability & Metrics (Langfuse proxy)
router.include_router(observability.router)

# 10. VAPT Dashboard API
from Apps.AgentMesh.Utilities.Tools.InfraTool.VAPTtool import vapt_dashboard_api
router.include_router(vapt_dashboard_api.router, prefix="/vapt", tags=["VAPT Dashboard"])

# Discovery Endpoint (A2A)
@router.get("/.well-known/agent-mesh.json", tags=["Discovery"])
async def get_agent_mesh_card():
    """Exposes Mesh capabilities and protocols."""
    return {
        "name": "AgentMesh Platform",
        "description": "High-fidelity Autonomous Agent Mesh with Deep Agent orchestration.",
        "version": "1.1.0",
        "protocols": ["MCP", "A2A", "LangGraph"],
        "capabilities": [
            "Autonomous Planning",
            "Multi-modal VFS",
            "Human-in-the-Loop Resumption",
            "Dynamic Subagent Delegation"
        ]
    }
