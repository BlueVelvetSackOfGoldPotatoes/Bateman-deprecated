# backend/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://user:password@localhost/dbname"

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL found in environment variables.")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# backend/models.py

from sqlalchemy import Column, Integer, String, Text, JSON, Enum, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import enum

class LeadType(enum.Enum):
    Company = "Company"
    Personnel = "Personnel"

class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(Enum(LeadType), nullable=False)
    entity = Column(String, nullable=False, index=True)
    category = Column(String, nullable=True)
    ceo_pi = Column(String, nullable=True)
    country = Column(String, nullable=True)
    university = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    source_urls = Column(Text, nullable=True)
    # Dynamic fields stored as JSON
    dynamic_fields = Column(JSON, nullable=True)

    # Relationships
    personnel = relationship("Personnel", back_populates="lead")

class Personnel(Base):
    __tablename__ = "personnel"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    personnel_name = Column(String, nullable=False)
    personnel_title = Column(String, nullable=True)
    personnel_email = Column(String, nullable=True)
    personnel_phone = Column(String, nullable=True)
    source_urls = Column(Text, nullable=True)
    # Dynamic fields stored as JSON
    dynamic_fields = Column(JSON, nullable=True)

    # Relationships
    lead = relationship("Lead", back_populates="personnel")
 
# backend/schemas.py --------------------------------------------------------------------------------------------

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

# backend/crud.py --------------------------------------------------------------------------------------------

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

# backend/routers/lead.py --------------------------------------------------------------------------------------------

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

# backend/routers/personnel.py ---------------------------------------------------------------------------------------

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import schemas, crud
from ..database import SessionLocal, engine

# Initialize router
router = APIRouter(
    prefix="/personnel",
    tags=["Personnel"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.Personnel)
def create_personnel(personnel: schemas.PersonnelCreate, db: Session = Depends(get_db)):
    return crud.create_personnel(db, personnel)

@router.get("/{personnel_id}", response_model=schemas.Personnel)
def read_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = crud.get_personnel_by_id(db, personnel_id=personnel_id)
    if not db_personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel

@router.get("/", response_model=List[schemas.Personnel])
def read_personnel_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    personnel = crud.get_personnel(db, skip=skip, limit=limit)
    return personnel

@router.put("/{personnel_id}", response_model=schemas.Personnel)
def update_personnel(personnel_id: int, personnel: schemas.PersonnelCreate, db: Session = Depends(get_db)):
    db_personnel = crud.update_personnel(db, personnel_id, personnel)
    if not db_personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel

@router.delete("/{personnel_id}", response_model=schemas.Personnel)
def delete_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = crud.delete_personnel(db, personnel_id)
    if not db_personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel


# backend/main.py -----------------------------------------------------------------------------------------------

from fastapi import FastAPI
from .database import engine, Base
from .routers import lead, personnel
from fastapi.middleware.cors import CORSMiddleware
import os

# Create all tables in the database. This is equivalent to "Create Table" statements in raw SQL.
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="BATEMAN API",
    description="API for managing leads and personnel",
    version="1.0.0"
)

# CORS configuration
origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),  # React frontend
    # Add other allowed origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(lead.router)
app.include_router(personnel.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the BATEMAN API"}


# alembic.ini --------------------------------------------------------------------------------------------------------

[alembic]
script_location = alembic
sqlalchemy.url = postgresql://user:password@localhost/dbname


# backend/alembic/env.py ----------------------------------------------------------------------------------------------

from __future__ import with_statement
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your models here
from database import Base  # noqa
import models  # noqa

# this is the Alembic Config object, which provides access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py, can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    return os.getenv("DATABASE_URL")


def run_migrations_offline():
    """Run migrations in 'offline' mode.
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    Calls to context.execute() here emit the given string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # To detect type changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # To detect type changes
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

// frontend/src/services/api.js ----------------------------------------------------------------------------------------

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Leads API
export const createLead = (leadData) => api.post('/leads/', leadData);
export const generateLeads = (context, num_leads, lead_types) =>
  api.post('/leads/generate', { context, num_leads, lead_types });
export const searchConferenceLeads = (conference_input, context) =>
  api.post('/leads/search-conference', { conference_input, context });
export const getLeads = (skip = 0, limit = 100) => api.get(`/leads/?skip=${skip}&limit=${limit}`);
export const updateLead = (id, leadData) => api.put(`/leads/${id}`, leadData);
export const deleteLead = (id) => api.delete(`/leads/${id}`);

// Personnel API
export const createPersonnel = (personnelData) => api.post('/personnel/', personnelData);
export const getPersonnel = (skip = 0, limit = 100) => api.get(`/personnel/?skip=${skip}&limit=${limit}`);
export const updatePersonnel = (id, personnelData) => api.put(`/personnel/${id}`, personnelData);
export const deletePersonnel = (id) => api.delete(`/personnel/${id}`);

// Additional API functions can be added here as needed

export default api;


// frontend/src/components/DynamicFields/ManageFields.js --------------------------------------------------------------

import React, { useState } from 'react';
import styled from 'styled-components';

const Container = styled.div`
  margin-bottom: 2rem;
`;

const ManageFields = ({ sectionType, fields, setFields }) => {
  const [newField, setNewField] = useState('');
  const [fieldToRemove, setFieldToRemove] = useState('');

  const handleAddField = () => {
    const trimmedField = newField.trim();
    if (trimmedField && !fields.includes(trimmedField)) {
      setFields([...fields, trimmedField]);
      setNewField('');
    }
  };

  const handleRemoveField = () => {
    if (fieldToRemove) {
      setFields(fields.filter((field) => field !== fieldToRemove));
      setFieldToRemove('');
    }
  };

  return (
    <Container>
      <h3>Manage {sectionType === 'lead' ? 'Lead' : 'Personnel'} Fields</h3>
      
      {/* Add Field */}
      <div>
        <input
          type="text"
          placeholder={`Add new ${sectionType === 'lead' ? 'Lead' : 'Personnel'} field`}
          value={newField}
          onChange={(e) => setNewField(e.target.value)}
        />
        <button onClick={handleAddField}>Add Field</button>
      </div>
      
      {/* Remove Field */}
      <div style={{ marginTop: '1rem' }}>
        <select value={fieldToRemove} onChange={(e) => setFieldToRemove(e.target.value)}>
          <option value="">Select field to remove</option>
          {fields.map((field, idx) => (
            <option key={idx} value={field}>{field}</option>
          ))}
        </select>
        <button onClick={handleRemoveField}>Remove Field</button>
      </div>
    </Container>
  );
};

export default ManageFields;


// frontend/src/components/InputLeads/InputLeads.js -------------------------------------------------------------------

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { st_tags } from 'streamlit-tags'; // Assuming you have a similar tagging component
import ManageFields from '../DynamicFields/ManageFields';
import { createLead, generateLeads, searchConferenceLeads, getLeads } from '../../services/api';
import LeadTable from '../common/LeadTable';
import DownloadButtons from '../common/DownloadButtons';

const Container = styled.div`
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
`;

const InputLeads = () => {
  const [context, setContext] = useState('');
  const [option, setOption] = useState('Generate Leads');
  const [numLeads, setNumLeads] = useState(10);
  const [leadTypes, setLeadTypes] = useState(['Research Groups']);
  const [conferenceInput, setConferenceInput] = useState('');
  const [manualLeads, setManualLeads] = useState('');
  const [leads, setLeads] = useState([]);
  const [leadFields, setLeadFields] = useState([
    "Entity", "Category", "CEO/PI", "Country",
    "University", "Summary", "Recommendations", "Source URLs"
  ]);
  const [personnelFields, setPersonnelFields] = useState([
    "Personnel Name", "Personnel Title",
    "Personnel Email", "Personnel Phone"
  ]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Fetch existing leads on component mount
  useEffect(() => {
    fetchLeads();
  }, []);
  
  const fetchLeads = async () => {
    try {
      const response = await getLeads();
      setLeads(response.data);
    } catch (error) {
      console.error(error);
      setMessage('Failed to fetch leads.');
    }
  };

  const handleGenerateLeads = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead generation.');
      return;
    }
    if (leadTypes.length === 0) {
      setMessage('Please specify at least one lead type.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await generateLeads(context, numLeads, leadTypes);
      setLeads([...leads, ...response.data]);
      setMessage(`Generated ${response.data.length} leads successfully!`);
    } catch (error) {
      console.error(error);
      setMessage('Failed to generate leads.');
    }
    setIsLoading(false);
  };

  const handleAddManualLeads = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead addition.');
      return;
    }
    const lines = manualLeads.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) {
      setMessage('No valid leads entered.');
      return;
    }
    setIsLoading(true);
    try {
      const createPromises = lines.map(line => {
        const [entity] = line.split(',');
        return createLead({ 
          type: "Company",
          entity: entity.trim(), 
          dynamic_fields: {}
        });
      });
      const results = await Promise.all(createPromises);
      setLeads([...leads, ...results.map(res => res.data)]);
      setMessage(`Added ${results.length} new leads successfully!`);
      setManualLeads('');
    } catch (error) {
      console.error(error);
      setMessage('Failed to add leads.');
    }
    setIsLoading(false);
  };

  const handleSearchConference = async () => {
    if (!context.trim()) {
      setMessage('Please provide a context for lead search.');
      return;
    }
    if (!conferenceInput.trim()) {
      setMessage('Please enter a conference name or URL.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await searchConferenceLeads(conferenceInput, context);
      setLeads([...leads, ...response.data]);
      setMessage(`Added ${response.data.length} new conference leads successfully!`);
      setConferenceInput('');
    } catch (error) {
      console.error(error);
      setMessage('Failed to search conference leads.');
    }
    setIsLoading(false);
  };

  return (
    <Container>
      <h2>Input Leads</h2>
      
      {/* Context Input */}
      <div>
        <label>Context:</label>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          placeholder="Provide context for lead generation or scraping..."
          rows="4"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Lead Input Options */}
      <div style={{ marginTop: '1rem' }}>
        <label>Choose how to input leads:</label>
        <select value={option} onChange={(e) => setOption(e.target.value)} style={{ width: '100%', padding: '0.5rem' }}>
          <option value="Generate Leads">Generate Leads</option>
          <option value="Add Leads Manually">Add Leads Manually</option>
          <option value="Search Leads via Conference">Search Leads via Conference</option>
        </select>
      </div>
      
      {/* Manage Dynamic Fields */}
      <ManageFields sectionType="lead" fields={leadFields} setFields={setLeadFields} />
      <ManageFields sectionType="personnel" fields={personnelFields} setFields={setPersonnelFields} />
      
      {/* Conditional Rendering Based on Option */}
      {option === "Generate Leads" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Number of leads per type:</label>
          <input
            type="number"
            value={numLeads}
            onChange={(e) => setNumLeads(parseInt(e.target.value))}
            min="1"
            max="100"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          
          <label>Lead Types:</label>
          <ManageFields 
            sectionType="lead" 
            fields={leadTypes} 
            setFields={setLeadTypes} 
          />
          
          <button onClick={handleGenerateLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Generating...' : 'Generate Leads'}
          </button>
        </div>
      )}
      
      {option === "Add Leads Manually" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Enter one lead per line, in the format 'Entity':</label>
          <textarea
            value={manualLeads}
            onChange={(e) => setManualLeads(e.target.value)}
            placeholder="Entity Name"
            rows="6"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          <button onClick={handleAddManualLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Adding...' : 'Add Leads'}
          </button>
        </div>
      )}
      
      {option === "Search Leads via Conference" && (
        <div style={{ marginTop: '1rem' }}>
          <label>Enter the conference name or URL:</label>
          <input
            type="text"
            value={conferenceInput}
            onChange={(e) => setConferenceInput(e.target.value)}
            placeholder="Conference Name or URL"
            style={{ width: '100%', padding: '0.5rem' }}
          />
          <button onClick={handleSearchConference} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
            {isLoading ? 'Searching...' : 'Search Leads'}
          </button>
        </div>
      )}
      
      {/* Display Message */}
      {message && <p style={{ marginTop: '1rem', color: 'green' }}>{message}</p>}
      
      {/* Leads Table */}
      {leads.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Leads</h3>
          <LeadTable leads={leads} leadFields={leadFields} personnelFields={personnelFields} />
          <DownloadButtons leads={leads} />
        </div>
      )}
    </Container>
  );
};

export default InputLeads;


// frontend/src/components/ScrapeLeadInformation/ScrapeLeadInformation.js -----------------------------------------------

import React, { useState } from 'react';
import styled from 'styled-components';
import { extractLeadInformation } from '../../services/api'; // Assuming you have such an API
import LeadTable from '../common/LeadTable';
import DownloadButtons from '../common/DownloadButtons';

const Container = styled.div`
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
`;

const ScrapeLeadInformation = () => {
  const [leadsToProcess, setLeadsToProcess] = useState([]);
  const [columnsToRetrieve, setColumnsToRetrieve] = useState([
    "Entity", "CEO/PI", "Researchers", "Grants",
    "Phone Number", "Email", "Country", "University",
    "Summary", "Contacts"
  ]);
  const [personKeywords, setPersonKeywords] = useState([
    "Education",
    "Current Position",
    "Expertise",
    "Email",
    "Phone Number",
    "Faculty",
    "University",
    "Bio",
    "Academic/Work Website Profile Link",
    "LinkedIn/Profile Link",
    "Facebook/Profile Link",
    "Grant",
    "Curriculum Vitae"
  ]);
  const [maxPersons, setMaxPersons] = useState(10);
  const [leadsInfo, setLeadsInfo] = useState([]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleScrapeLeads = async () => {
    if (leadsToProcess.length === 0) {
      setMessage('Please provide leads to process.');
      return;
    }
    if (columnsToRetrieve.length === 0) {
      setMessage('Please select at least one information field to retrieve.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await extractLeadInformation(leadsToProcess, columnsToRetrieve, personKeywords, maxPersons);
      setLeadsInfo(response.data);
      setMessage('Lead information scraped successfully!');
    } catch (error) {
      console.error(error);
      setMessage('Failed to scrape lead information.');
    }
    setIsLoading(false);
  };

  return (
    <Container>
      <h2>Analyze Lead Information</h2>
      
      {/* Leads to Process Input */}
      <div>
        <label>Leads to Process (comma-separated IDs):</label>
        <input
          type="text"
          value={leadsToProcess.join(', ')}
          onChange={(e) => setLeadsToProcess(e.target.value.split(',').map(id => id.trim()))}
          placeholder="e.g., 1,2,3"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Information Fields to Retrieve */}
      <div style={{ marginTop: '1rem' }}>
        <label>Information Fields to Retrieve:</label>
        <select multiple value={columnsToRetrieve} onChange={(e) => {
          const options = e.target.options;
          const selected = [];
          for (let i = 0; i < options.length; i++) {
            if (options[i].selected) {
              selected.push(options[i].value);
            }
          }
          setColumnsToRetrieve(selected);
        }} style={{ width: '100%', padding: '0.5rem', height: '150px' }}>
          <option value="Entity">Entity</option>
          <option value="CEO/PI">CEO/PI</option>
          <option value="Researchers">Researchers</option>
          <option value="Grants">Grants</option>
          <option value="Phone Number">Phone Number</option>
          <option value="Email">Email</option>
          <option value="Country">Country</option>
          <option value="University">University</option>
          <option value="Summary">Summary</option>
          <option value="Contacts">Contacts</option>
          {/* Add dynamic fields as options */}
        </select>
      </div>
      
      {/* Person Search Keywords */}
      <div style={{ marginTop: '1rem' }}>
        <label>Person Search Keywords (comma-separated):</label>
        <input
          type="text"
          value={personKeywords.join(', ')}
          onChange={(e) => setPersonKeywords(e.target.value.split(',').map(k => k.trim()))}
          placeholder="e.g., Education, Expertise, Email"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Max Persons Input */}
      <div style={{ marginTop: '1rem' }}>
        <label>Maximum number of persons to process per lead:</label>
        <input
          type="number"
          value={maxPersons}
          onChange={(e) => setMaxPersons(parseInt(e.target.value))}
          min="1"
          max="100"
          style={{ width: '100%', padding: '0.5rem' }}
        />
      </div>
      
      {/* Scrape Button */}
      <button onClick={handleScrapeLeads} disabled={isLoading} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
        {isLoading ? 'Scraping...' : 'Search and Analyse Leads'}
      </button>
      
      {/* Display Message */}
      {message && <p style={{ marginTop: '1rem', color: 'green' }}>{message}</p>}
      
      {/* Leads Information Table */}
      {leadsInfo.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Leads Information</h3>
          <LeadTable leads={leadsInfo} />
          <DownloadButtons leads={leadsInfo} />
        </div>
      )}
    </Container>
  );
};

export default ScrapeLeadInformation;


// frontend/src/components/common/LeadTable.js -----------------------------------------------------------------------

import React from 'react';
import styled from 'styled-components';

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  
  th, td {
    border: 1px solid #ddd;
    padding: 0.75rem;
    text-align: left;
  }
  
  th {
    background-color: #f2f2f2;
  }
  
  tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  
  tr:hover {
    background-color: #f1f1f1;
  }
`;

const LeadTable = ({ leads, leadFields = [], personnelFields = [] }) => {
  return (
    <div>
      <Table>
        <thead>
          <tr>
            <th>ID</th>
            {leadFields.map((field, idx) => (
              <th key={`lead-${idx}`}>{field}</th>
            ))}
            {personnelFields.map((field, idx) => (
              <th key={`personnel-${idx}`}>{field}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {leads.map((lead, idx) => (
            <tr key={idx}>
              <td>{lead.id}</td>
              {leadFields.map((field, index) => (
                <td key={`lead-${idx}-${index}`}>{lead[field] || 'Not Available'}</td>
              ))}
              {personnelFields.map((field, index) => (
                <td key={`personnel-${idx}-${index}`}>{lead[field] || 'Not Available'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
};

export default LeadTable;
