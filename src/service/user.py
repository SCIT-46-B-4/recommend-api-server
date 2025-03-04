from sqlalchemy.orm import Session

from src.entity.user import UserEntity
from src.dto.response.user_response import UserResponse
from src.service.base_service import get_dto_by_id


def get_user_by_id(db: Session, user_id: int) -> UserResponse | None:
    return get_dto_by_id(db, UserEntity, user_id, UserResponse)
