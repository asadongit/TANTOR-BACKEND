import uuid
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..Models import SkillModel

async def save_skill(
    db: AsyncSession,
    name: str,
    description: Optional[str],
    content: str,
    tags: Optional[List[str]] = None,
    skill_id: Optional[str] = None
):
    """Saves or updates a skill in the database."""
    if skill_id:
        stmt = select(SkillModel).where(SkillModel.id == skill_id)
        result = await db.execute(stmt)
        db_skill = result.scalar_one_or_none()
    else:
        # Check if same name exists to update instead of creating duplicate
        stmt = select(SkillModel).where(SkillModel.name == name)
        result = await db.execute(stmt)
        db_skill = result.scalar_one_or_none()

    if db_skill:
        db_skill.name = name
        db_skill.description = description
        db_skill.content = content
        if tags is not None:
            db_skill.tags = tags
    else:
        db_skill = SkillModel(
            id=skill_id or str(uuid.uuid4()),
            name=name,
            description=description,
            content=content,
            tags=tags or []
        )
        db.add(db_skill)

    await db.commit()
    await db.refresh(db_skill)
    return db_skill

async def get_skill(db: AsyncSession, skill_id: str):
    """Retrieves a skill by ID."""
    stmt = select(SkillModel).where(SkillModel.id == skill_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_skill_by_name(db: AsyncSession, name: str):
    """Retrieves a skill by name."""
    stmt = select(SkillModel).where(SkillModel.name == name)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def list_skills(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None
) -> List[SkillModel]:
    """Lists skills from database with search filtering."""
    stmt = select(SkillModel).offset(skip).limit(limit)
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (SkillModel.name.ilike(pattern, escape="~")) |
            (SkillModel.description.ilike(pattern, escape="~")) |
            (SkillModel.content.ilike(pattern, escape="~"))
        )
    result = await db.execute(stmt)
    return result.scalars().all()

async def count_skills(
    db: AsyncSession,
    search: Optional[str] = None
) -> int:
    """Returns the total count of skills matching search term."""
    stmt = select(func.count()).select_from(SkillModel)
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (SkillModel.name.ilike(pattern, escape="~")) |
            (SkillModel.description.ilike(pattern, escape="~")) |
            (SkillModel.content.ilike(pattern, escape="~"))
        )
    result = await db.execute(stmt)
    return result.scalar()

async def delete_skill(db: AsyncSession, skill_id: str) -> bool:
    """Deletes a skill by ID."""
    stmt = select(SkillModel).where(SkillModel.id == skill_id)
    result = await db.execute(stmt)
    db_skill = result.scalar_one_or_none()
    if db_skill:
        await db.delete(db_skill)
        await db.commit()
        return True
    return False
