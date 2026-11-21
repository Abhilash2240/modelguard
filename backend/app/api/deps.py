from app.db.session import get_db
from app.services.dataset_service import DatasetService
from fastapi import Depends
from sqlalchemy.orm import Session


def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    return DatasetService(db=db)
