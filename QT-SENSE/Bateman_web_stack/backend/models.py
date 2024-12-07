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
