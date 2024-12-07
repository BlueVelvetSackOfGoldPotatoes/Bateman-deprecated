from sqlalchemy.orm import Session
from . import models, schemas
from typing import List, Optional

# CRUD for Leads

def get_leads(db: Session, skip: int = 0, limit: int = 100) -> List[models.Lead]:
    return db.query(models.Lead).offset(skip).limit(limit).all()

def get_lead(db: Session, lead_id: int) -> Optional[models.Lead]:
    return db.query(models.Lead).filter(models.Lead.id == lead_id).first()

def create_lead(db: Session, lead: schemas.LeadCreate) -> models.Lead:
    db_lead = models.Lead(
        type=lead.type,
        entity=lead.entity,
        category=lead.category,
        ceo_pi=lead.ceo_pi,
        country=lead.country,
        university=lead.university,
        summary=lead.summary,
        recommendations=lead.recommendations,
        source_urls=lead.source_urls,
        dynamic_fields=lead.dynamic_fields
    )
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    return db_lead

def update_lead(db: Session, lead_id: int, lead: schemas.LeadCreate) -> Optional[models.Lead]:
    db_lead = get_lead(db, lead_id)
    if db_lead:
        for key, value in lead.dict().items():
            setattr(db_lead, key, value)
        db.commit()
        db.refresh(db_lead)
    return db_lead

def delete_lead(db: Session, lead_id: int) -> Optional[models.Lead]:
    db_lead = get_lead(db, lead_id)
    if db_lead:
        db.delete(db_lead)
        db.commit()
    return db_lead

# CRUD for Personnel

def get_personnel(db: Session, skip: int = 0, limit: int = 100) -> List[models.Personnel]:
    return db.query(models.Personnel).offset(skip).limit(limit).all()

def get_personnel_by_id(db: Session, personnel_id: int) -> Optional[models.Personnel]:
    return db.query(models.Personnel).filter(models.Personnel.id == personnel_id).first()

def create_personnel(db: Session, personnel: schemas.PersonnelCreate) -> models.Personnel:
    db_personnel = models.Personnel(
        lead_id=personnel.lead_id,
        personnel_name=personnel.personnel_name,
        personnel_title=personnel.personnel_title,
        personnel_email=personnel.personnel_email,
        personnel_phone=personnel.personnel_phone,
        source_urls=personnel.source_urls,
        dynamic_fields=personnel.dynamic_fields
    )
    db.add(db_personnel)
    db.commit()
    db.refresh(db_personnel)
    return db_personnel

def update_personnel(db: Session, personnel_id: int, personnel: schemas.PersonnelCreate) -> Optional[models.Personnel]:
    db_personnel = get_personnel_by_id(db, personnel_id)
    if db_personnel:
        for key, value in personnel.dict().items():
            setattr(db_personnel, key, value)
        db.commit()
        db.refresh(db_personnel)
    return db_personnel

def delete_personnel(db: Session, personnel_id: int) -> Optional[models.Personnel]:
    db_personnel = get_personnel_by_id(db, personnel_id)
    if db_personnel:
        db.delete(db_personnel)
        db.commit()
    return db_personnel
