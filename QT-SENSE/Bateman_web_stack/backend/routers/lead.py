from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import schemas, crud
from ..database import SessionLocal, engine
from ..utils import (
    generate_leads_with_llm,
    search_leads_via_conference
)

# Initialize router
router = APIRouter(
    prefix="/leads",
    tags=["Leads"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.Lead)
def create_lead(lead: schemas.LeadCreate, db: Session = Depends(get_db)):
    return crud.create_lead(db, lead)

@router.get("/{lead_id}", response_model=schemas.Lead)
def read_lead(lead_id: int, db: Session = Depends(get_db)):
    db_lead = crud.get_lead(db, lead_id=lead_id)
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return db_lead

@router.get("/", response_model=List[schemas.Lead])
def read_leads(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    leads = crud.get_leads(db, skip=skip, limit=limit)
    return leads

@router.put("/{lead_id}", response_model=schemas.Lead)
def update_lead(lead_id: int, lead: schemas.LeadCreate, db: Session = Depends(get_db)):
    db_lead = crud.update_lead(db, lead_id, lead)
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return db_lead

@router.delete("/{lead_id}", response_model=schemas.Lead)
def delete_lead(lead_id: int, db: Session = Depends(get_db)):
    db_lead = crud.delete_lead(db, lead_id)
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return db_lead

# Additional Endpoints for Dynamic Lead Management

@router.post("/generate", response_model=List[schemas.Lead])
def generate_leads(context: str, num_leads: int = 10, lead_types: List[str] = ["Research Groups"], db: Session = Depends(get_db)):
    if not context.strip():
        raise HTTPException(status_code=400, detail="Context cannot be empty")
    if not lead_types:
        raise HTTPException(status_code=400, detail="At least one lead type must be specified")
    
    # Generate leads using utility function
    generated_leads = generate_leads_with_llm(context, num_leads)
    
    created_leads = []
    for lead in generated_leads:
        lead_create = schemas.LeadCreate(
            type=schemas.LeadType.Company,
            entity=lead["Entity"],
            source_urls=lead.get("Source URLs", ""),
            dynamic_fields={}  # Populate if necessary
        )
        db_lead = crud.create_lead(db, lead_create)
        created_leads.append(db_lead)
    
    return created_leads

@router.post("/search-conference", response_model=List[schemas.Lead])
def search_conference_leads(conference_input: str, context: str, db: Session = Depends(get_db)):
    if not context.strip():
        raise HTTPException(status_code=400, detail="Context cannot be empty")
    if not conference_input.strip():
        raise HTTPException(status_code=400, detail="Conference input cannot be empty")
    
    # Search leads via conference using utility function
    leads_found = search_leads_via_conference(conference_input, context)
    
    created_leads = []
    for lead in leads_found:
        lead_create = schemas.LeadCreate(
            type=schemas.LeadType.Company,
            entity=lead["Entity"],
            source_urls=lead.get("Source URLs", ""),
            dynamic_fields={}
        )
        db_lead = crud.create_lead(db, lead_create)
        created_leads.append(db_lead)
    
    return created_leads
