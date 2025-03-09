from fastapi import APIRouter, Depends, Request
import json
from sqlalchemy.orm import Session

from src.core.exception.not_found_exceptions import UserNotFoundExceiption
from src.service.schedule import get_schedule_by_id
from src.service.user import get_user_by_id
from src.db.connection import get_db
from src.dto.response.user_response import UserResponse
from src.core.model.rankmodel import recommend_destinations


router = APIRouter(prefix="/recommend")

@router.get("")
async def get_recommend_schedule(
    request: Request,
    userId: int = None,
    db: Session = Depends(get_db)
    ):

    if not userId:
        raise UserNotFoundExceiption()

    user: UserResponse|None = get_user_by_id(db, userId)

    if not user:
        raise UserNotFoundExceiption()

    recommendations = recommend_destinations(userId)

    if recommendations is None:
        raise UserNotFoundExceiption()

    return recommendations
