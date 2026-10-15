from fastapi import APIRouter, HTTPException, Depends
import os
import logging
from datetime import datetime
from typing import List, Optional

from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
from Apps.AgentMesh.Config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from Database.Database import get_db
from Database import AgentMeshCrud
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deep Agent Protocols"])

@router.get("/deep-agent/vfs/list")
async def list_vfs_files_endpoint(root_dir: str, thread_id: str = "default_thread"):
    """Unified Workspace VFS Scanners."""
    files = []
    seen_paths = set()

    # 1. SCAN POSTGRES (Virtual VFS)
    try:
        service = WorkflowService()
        state = await service.get_thread_state(thread_id)
        if state and "vfs" in state:
            vfs_data = state["vfs"]
            for file_path, info in vfs_data.get("files", {}).items():
                size = sum(len(str(line)) + 1 for line in info.get("content", [])) if isinstance(info, dict) else len(str(info))
                path_clean = file_path.replace("\\", "/").lstrip("/")
                if path_clean not in seen_paths:
                    files.append({
                        "name": os.path.basename(file_path),
                        "path": path_clean,
                        "size": size,
                        "modified": datetime.now().isoformat(),
                        "type": "virtual",
                        "category": "Knowledge"
                    })
                    seen_paths.add(path_clean)
    except Exception: pass

    # 2. SCAN PHYSICAL DISK (Knowledge)
    deep_path = os.path.join(root_dir, thread_id)
    if os.path.exists(deep_path):
        for root, _, filenames in os.walk(deep_path):
            for f in filenames:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, deep_path).replace("\\", "/").lstrip("/")
                if rel_path not in seen_paths:
                    stat = os.stat(full_path)
                    files.append({
                        "name": f, "path": rel_path, "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "disk", "category": "Knowledge"
                    })
                    seen_paths.add(rel_path)

    # 3. SCAN EXPORTS (Artifacts)
    exports_path = os.path.join(settings.EXPORTS_MOUNT, thread_id)
    if os.path.exists(exports_path):
        for root, _, filenames in os.walk(exports_path):
            for f in filenames:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, exports_path).replace("\\", "/").lstrip("/")
                if rel_path not in seen_paths:
                    stat = os.stat(full_path)
                    files.append({
                        "name": f, "path": rel_path, "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "disk", "category": "Artifact"
                    })
                    seen_paths.add(rel_path)

    return {"status": "success", "files": files, "thread_id": thread_id}

@router.get("/deep-agent/vfs/read")
async def read_vfs_file_endpoint(root_dir: str, file_path: str, thread_id: str = "default_thread"):
    # 1. Try Virtual State
    try:
        service = WorkflowService()
        state = await service.get_thread_state(thread_id)
        if state and "vfs" in state:
            vfs_data = state["vfs"]
            files = vfs_data.get("files", {})
            content = files.get(file_path) or files.get(file_path.lstrip("/")) or files.get("/" + file_path)
            if content is not None:
                if isinstance(content, dict):
                    raw_val = content.get("content", "")
                    if isinstance(raw_val, list):
                        final_text = "\n".join(str(l) for l in raw_val)
                    else:
                        final_text = str(raw_val)
                else:
                    final_text = str(content)
                return {"status": "success", "content": final_text, "source": "virtual"}
    except Exception: pass

    # 2. Try Physical Disk (Knowledge)
    knowledge_path = os.path.join(root_dir, thread_id, file_path)
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r', encoding='utf-8', errors='replace') as f:
            return {"status": "success", "content": f.read(), "source": "knowledge"}

    # 3. Try Exports (Artifact)
    exports_path = os.path.join(settings.EXPORTS_MOUNT, thread_id, file_path)
    if os.path.exists(exports_path):
        with open(exports_path, 'r', encoding='utf-8', errors='replace') as f:
            return {"status": "success", "content": f.read(), "source": "artifact"}

    raise HTTPException(status_code=404, detail="File not found")

class SkillCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    content: str
    tags: Optional[List[str]] = []

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        import re
        name = v.strip()
        if not name:
            raise ValueError('Name is required.')
        if len(name) > 64:
            raise ValueError('Name must be 64 characters or fewer.')
        if re.search(r'[A-Z]', name):
            raise ValueError('Uppercase letters are not allowed. Use lowercase only.')
        if name.startswith('-') or name.endswith('-'):
            raise ValueError('Name must not start or end with a hyphen.')
        if '--' in name:
            raise ValueError('Consecutive hyphens (--) are not allowed.')
        if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', name):
            raise ValueError(
                'Name may only contain lowercase letters (a-z), digits (0-9), and hyphens (-).'
            )
        return name

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            raise ValueError('Description is required.')
        desc = v.strip()
        if not desc:
            raise ValueError('Description is required.')
        if len(desc) > 1024:
            raise ValueError('Description must be 1024 characters or fewer.')
        if len(desc) < 20:
            raise ValueError(
                'Description is too short. Describe what the skill does and when to use it (min 20 chars).'
            )
        return desc

@router.get("/skills/list")
async def list_available_skills_endpoint(
    page: int = 1,
    limit: int = 1000,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Fetch all skills from the database with pagination and search."""
    skip = (page - 1) * limit
    db_skills = await AgentMeshCrud.list_skills(db, skip=skip, limit=limit, search=search)
    total = await AgentMeshCrud.count_skills(db, search=search)
    skills = []
    for s in db_skills:
        skills.append({
            "id": s.id,
            "name": s.name,
            "description": s.description or "Professional Agentic SOP",
            "content": s.content,
            "tags": s.tags or []
        })
    return {"status": "success", "skills": skills, "total": total}

@router.post("/skills/create")
async def create_skill_endpoint(request: SkillCreateRequest, db: AsyncSession = Depends(get_db)):
    """Create or update a skill in the database."""
    try:
        db_skill = await AgentMeshCrud.save_skill(
            db,
            name=request.name,
            description=request.description,
            content=request.content,
            tags=request.tags
        )
        return {"status": "success", "skill": {
            "id": db_skill.id,
            "name": db_skill.name,
            "description": db_skill.description,
            "content": db_skill.content,
            "tags": db_skill.tags
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/skills/{skill_id}")
async def delete_skill_endpoint(skill_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a skill from the database."""
    deleted = await AgentMeshCrud.delete_skill(db, skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "success"}
