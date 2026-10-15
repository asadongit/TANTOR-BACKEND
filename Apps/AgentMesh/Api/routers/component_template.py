from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel
from Database.Database import get_db
from Database.Models import ComponentTemplateModel, CategoryModel
from Apps.AgentMesh.Utilities.Security import protect_sensitive_data, redact_sensitive_data
import uuid
from datetime import datetime

router = APIRouter(tags=["Component Templates"])

class ComponentTemplateCreate(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    group: str = "Default"
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    config: dict

@router.post("/component-templates")
async def save_component_template(template: ComponentTemplateCreate, db: AsyncSession = Depends(get_db)):
    # Optional logic to dynamically create a category if only category_name is provided
    cat_id = template.category_id
    cat_name = template.category_name
    
    if cat_name:
        # Check for exact duplicate in this category
        duplicate_check = await db.execute(
            select(ComponentTemplateModel).filter(
                ComponentTemplateModel.name == template.name,
                ComponentTemplateModel.category_name == cat_name,
                ComponentTemplateModel.type == template.type
            )
        )
        if duplicate_check.scalars().first():
            raise HTTPException(
                status_code=409, 
                detail=f"A template named '{template.name}' already exists in the '{cat_name}' category."
            )
    
    if cat_name and not cat_id:
        result = await db.execute(
            select(CategoryModel).filter(
                CategoryModel.name == cat_name
            )
        )
        existing_cat = result.scalars().first()
        
        if existing_cat:
            cat_id = existing_cat.id
        else:
            new_cat = CategoryModel(
                id=str(uuid.uuid4()),
                name=cat_name,
                type=template.type
            )
            db.add(new_cat)
            await db.commit()
            await db.refresh(new_cat)
            cat_id = new_cat.id

    new_template = ComponentTemplateModel(
        id=str(uuid.uuid4()),
        name=template.name,
        type=template.type,
        description=template.description,
        group=template.group,
        category_id=cat_id,
        category_name=cat_name,
        config=protect_sensitive_data(template.config)
    )
    
    db.add(new_template)
    await db.commit()
    await db.refresh(new_template)
    
    return {"message": "Component template saved successfully", "id": new_template.id}

@router.get("/component-templates/filters")
async def list_component_template_filters(
    type: Optional[str] = Query(None),
    exclude_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    # Fetch distinct groups
    group_query = select(ComponentTemplateModel.group).filter(ComponentTemplateModel.group.isnot(None)).distinct()
    if type:
        group_query = group_query.filter(ComponentTemplateModel.type == type)
    if exclude_type:
        group_query = group_query.filter(ComponentTemplateModel.type != exclude_type)
    group_result = await db.execute(group_query)
    groups = sorted(list(set(group_result.scalars().all())))
    
    # Fetch categories
    cat_query = select(CategoryModel)
    if type:
        cat_query = cat_query.filter(CategoryModel.type == type)
    if exclude_type:
        cat_query = cat_query.filter(CategoryModel.type != exclude_type)
    cat_result = await db.execute(cat_query)
    categories = [{"id": c.id, "name": c.name, "description": c.description, "type": c.type} for c in cat_result.scalars().all()]

    # Fetch distinct types
    type_query = select(ComponentTemplateModel.type).filter(ComponentTemplateModel.type.isnot(None)).distinct()
    if type:
        type_query = type_query.filter(ComponentTemplateModel.type == type)
    if exclude_type:
        type_query = type_query.filter(ComponentTemplateModel.type != exclude_type)
    type_result = await db.execute(type_query)
    types = sorted(list(set(type_result.scalars().all())))
    
    return {
        "groups": groups,
        "categories": categories,
        "types": types
    }
 
@router.get("/component-templates")
async def list_component_templates(
    type: Optional[str] = Query(None),
    exclude_type: Optional[str] = Query(None),
    group: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    category_name: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    query = select(ComponentTemplateModel)
    
    if type:
        query = query.filter(ComponentTemplateModel.type == type)
    if exclude_type:
        query = query.filter(ComponentTemplateModel.type != exclude_type)
    if group and group != "All":
        query = query.filter(ComponentTemplateModel.group == group)
    if category_id and category_id != "All":
        query = query.filter(ComponentTemplateModel.category_id == category_id)
    if category_name and category_name != "All":
        query = query.filter(ComponentTemplateModel.category_name == category_name)
    if search:
        query = query.filter(ComponentTemplateModel.name.ilike(f"%{search}%"))
        
    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    query = query.order_by(ComponentTemplateModel.created_at.desc()).offset((page - 1) * limit).limit(limit)
    result_exec = await db.execute(query)
    templates = result_exec.scalars().all()
    
    result = []
    for t in templates:
        result.append({
            "id": t.id,
            "name": t.name,
            "type": t.type,
            "description": t.description,
            "group": t.group,
            "category_id": t.category_id,
            "category_name": t.category_name,
            "config": redact_sensitive_data(t.config),
            "created_at": t.created_at
        })
        
    return {
        "templates": result,
        "total": total,
        "page": page,
        "limit": limit
    }

@router.delete("/component-template/{template_id}")
async def delete_component_template(template_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ComponentTemplateModel).filter(ComponentTemplateModel.id == template_id))
    template = result.scalars().first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
        
    await db.delete(template)
    await db.commit()
    
    return {"message": "Component template deleted successfully"}
