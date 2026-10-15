from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel
from Database.Database import get_db
from Database.Models import CategoryModel
import uuid
from datetime import datetime

router = APIRouter(tags=["Categories"])

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str = "api_node"

@router.post("/categories")
async def create_category(category: CategoryCreate, db: AsyncSession = Depends(get_db)):
    # Check if category with same name and type already exists
    result = await db.execute(
        select(CategoryModel).filter(
            CategoryModel.name == category.name,
            CategoryModel.type == category.type
        )
    )
    existing = result.scalars().first()
    
    if existing:
        return {"id": existing.id, "name": existing.name, "description": existing.description, "type": existing.type}
    
    new_cat = CategoryModel(
        id=str(uuid.uuid4()),
        name=category.name,
        description=category.description,
        type=category.type
    )
    db.add(new_cat)
    await db.commit()
    await db.refresh(new_cat)
    return {"id": new_cat.id, "name": new_cat.name, "description": new_cat.description, "type": new_cat.type}

@router.get("/categories")
async def list_categories(type: Optional[str] = Query(None), db: AsyncSession = Depends(get_db)):
    query = select(CategoryModel)
    if type:
        query = query.filter(CategoryModel.type == type)
    
    query = query.order_by(CategoryModel.name)
    result = await db.execute(query)
    categories = result.scalars().all()
    
    return {
        "categories": [
            {"id": c.id, "name": c.name, "description": c.description, "type": c.type} 
            for c in categories
        ]
    }
