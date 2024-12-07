from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict
from enum import Enum

class LeadType(str, Enum):
    Company = "Company"
    Personnel = "Personnel"

class LeadBase(BaseModel):
    type: LeadType
    entity: str
    category: Optional[str] = None
    ceo_pi: Optional[str] = None
    country: Optional[str] = None
    university: Optional[str] = None
    summary: Optional[str] = None
    recommendations: Optional[str] = None
    source_urls: Optional[str] = None
    dynamic_fields: Optional[Dict[str, str]] = None  # For dynamic lead fields

class LeadCreate(LeadBase):
    pass

class Lead(LeadBase):
    id: int

    class Config:
        orm_mode = True

class PersonnelBase(BaseModel):
    lead_id: int
    personnel_name: str
    personnel_title: Optional[str] = None
    personnel_email: Optional[str] = None
    personnel_phone: Optional[str] = None
    source_urls: Optional[str] = None
    dynamic_fields: Optional[Dict[str, str]] = None  # For dynamic personnel fields

class PersonnelCreate(PersonnelBase):
    pass

class Personnel(PersonnelBase):
    id: int

    class Config:
        orm_mode = True
