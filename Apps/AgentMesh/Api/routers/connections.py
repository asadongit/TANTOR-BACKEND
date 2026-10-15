from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import datetime

from Database.Database import get_db
from Database.Models import ConnectionProfileModel
from Database.AgentMeshCrud.connections import protect_connection_values
from Apps.AgentMesh.Utilities.Security import validate_credential_destination

router = APIRouter()

class ConnectionCreate(BaseModel):
    id: Optional[str] = None
    platform_name: str
    auth_type: str = "None"
    access_token: Optional[str] = Field(None, repr=False)
    refresh_token: Optional[str] = Field(None, repr=False)
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = Field(None, repr=False)
    oauth_token_url: Optional[str] = None
    enable_heartbeat: bool = False
    test_url: Optional[str] = None

@router.get("/connections", tags=["Connections"])
async def list_connections(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ConnectionProfileModel))
    connections = result.scalars().all()
    
    # Do not return actual tokens to the frontend for security
    return {
        "status": "success",
        "connections": [
            {
                "id": c.id,
                "platform_name": c.platform_name,
                "auth_type": c.auth_type,
                "oauth_client_id": c.oauth_client_id,
                "oauth_token_url": c.oauth_token_url,
                "enable_heartbeat": c.enable_heartbeat,
                "test_url": c.test_url,
                "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in connections
        ]
    }


@router.post("/connections", tags=["Connections"])
async def save_connection(conn: ConnectionCreate, db: AsyncSession = Depends(get_db)):
    try:
        if conn.oauth_token_url:
            validate_credential_destination(conn.oauth_token_url)
        if conn.test_url and (conn.access_token or conn.refresh_token or conn.auth_type != "None"):
            validate_credential_destination(conn.test_url)
        conn_id = conn.id or f"conn_{uuid.uuid4().hex[:8]}"
        protected = protect_connection_values(conn.model_dump())
        
        # Check if exists
        result = await db.execute(select(ConnectionProfileModel).where(ConnectionProfileModel.id == conn_id))
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.platform_name = conn.platform_name
            existing.auth_type = conn.auth_type
            if conn.access_token:
                existing.access_token = protected["access_token"]
            if conn.refresh_token:
                existing.refresh_token = protected["refresh_token"]
            if conn.oauth_client_id:
                existing.oauth_client_id = conn.oauth_client_id
            if conn.oauth_client_secret:
                existing.oauth_client_secret = protected["oauth_client_secret"]
            if conn.oauth_token_url:
                existing.oauth_token_url = conn.oauth_token_url
            existing.enable_heartbeat = conn.enable_heartbeat
            existing.test_url = conn.test_url
            existing.updated_at = datetime.datetime.utcnow()
        else:
            new_conn = ConnectionProfileModel(
                id=conn_id,
                platform_name=conn.platform_name,
                auth_type=conn.auth_type,
                access_token=protected["access_token"],
                refresh_token=protected["refresh_token"],
                oauth_client_id=conn.oauth_client_id,
                oauth_client_secret=protected["oauth_client_secret"],
                oauth_token_url=conn.oauth_token_url,
                enable_heartbeat=conn.enable_heartbeat,
                test_url=conn.test_url
            )
            db.add(new_conn)
            
        await db.commit()
        return {"status": "success", "id": conn_id}
    except Exception:
        await db.rollback()
        # Internal crypto/DB details can contain operational secrets.
        raise HTTPException(status_code=500, detail="Unable to save connection securely")

@router.delete("/connections/{conn_id}", tags=["Connections"])
async def delete_connection(conn_id: str, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(ConnectionProfileModel).where(ConnectionProfileModel.id == conn_id))
        existing = result.scalar_one_or_none()
        if not existing:
            raise HTTPException(status_code=404, detail="Connection not found")
            
        await db.delete(existing)
        await db.commit()
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
