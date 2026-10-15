from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

from Apps.AgentMesh.Core.ToolRegistry import registry
from Database.Database import get_db
from Database import AgentMeshCrud

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tools"])

class MCPRegistrationRequest(BaseModel):
    name: str
    command: str
    args: Optional[List[str]] = []
    env: Optional[Dict[str, str]] = {}
    transport: Optional[str] = "stdio"

class MCPRegistrationResponse(BaseModel):
    status: str
    name: str

@router.get("/tools")
async def list_tools(
    page: int = 1,
    limit: int = 1000,
    search: Optional[str] = None,
    category: Optional[str] = None,
    package: Optional[str] = None
):
    all_tools = registry.get_all_meta()
    filtered = all_tools
    if package:
        filtered = [t for t in filtered if t.package and t.package.lower() == package.lower()]
    if category:
        filtered = [t for t in filtered if t.category and t.category.lower() == category.lower()]
    if search:
        s = search.lower()
        filtered = [
            t for t in filtered 
            if s in t.name.lower() 
            or (t.description and s in t.description.lower()) 
            or (t.category and s in t.category.lower())
        ]
    
    skip = (page - 1) * limit
    return {
        "status": "success",
        "total": len(filtered),
        "page": page,
        "limit": limit,
        "tools": filtered[skip : skip + limit]
    }

@router.get("/tools/packages")
async def list_tool_packages(page: int = 1, limit: int = 10, search: Optional[str] = None):
    all_tools = registry.get_all_meta()
    packages_dict = {}
    for t in all_tools:
        pkg_name = t.package or "built-in"
        if pkg_name not in packages_dict:
            packages_dict[pkg_name] = {"name": pkg_name, "version": t.version or "1.0.0", "tools": []}
        packages_dict[pkg_name]["tools"].append(t)
    
    package_list = list(packages_dict.values())
    if search:
        s = search.lower()
        package_list = [
            p for p in package_list 
            if s in p["name"].lower()
            or any(
                s in t.name.lower() 
                or (t.description and s in t.description.lower()) 
                or (t.category and s in t.category.lower())
                for t in p["tools"]
            )
        ]
    
    package_list.sort(key=lambda x: x["name"].lower())
    skip = (page - 1) * limit
    return {
        "status": "success",
        "total": len(package_list),
        "page": page,
        "limit": limit,
        "packages": package_list[skip : skip + limit]
    }

@router.post("/tools/mcp/register", response_model=MCPRegistrationResponse)
async def register_mcp_server(request: MCPRegistrationRequest, db: AsyncSession = Depends(get_db)):
    saved = await AgentMeshCrud.save_mcp_server(db, name=request.name, command=request.command, args=request.args, env=request.env, transport=request.transport)
    registry.add_mcp_server(request.name, {
        "command": saved.command,
        "args": saved.args,
        "env": saved.env,
        "transport": saved.transport,
    })
    return {"status": "registered", "name": request.name}

@router.get("/tools/mcp/list")
async def list_mcp_servers(db: AsyncSession = Depends(get_db)):
    servers = await AgentMeshCrud.list_mcp_servers(db)
    return {
        "status": "success",
        "servers": [
            {
                "name": s.name,
                "command": s.command,
                "args": s.args,
                "env": s.env,
                "transport": s.transport,
                "enabled": s.enabled
            } for s in servers
        ]
    }

@router.delete("/tools/mcp/{name}")
async def delete_mcp_server(name: str, db: AsyncSession = Depends(get_db)):
    if await AgentMeshCrud.delete_mcp_server(db, name):
        registry.remove_mcp_server(name)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="MCP server not found")

@router.post("/tools/mcp/sync")
async def sync_mcp_tools():
    await registry.sync_mcp_tools()
    return {"status": "synced"}
