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
