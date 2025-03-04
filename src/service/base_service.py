from typing import Type, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session


T = TypeVar('TEntity')
R = TypeVar('ResponseDto', bound=BaseModel)

def get_dto_by_id(db: Session, model: Type[T], entity_id: int, response_dto: Type[R]) -> R | None:
    entity: T | None = db.query(model).filter(model.id == entity_id).first()

    if entity is  None:
        return None
    return response_dto.model_validate(entity)
