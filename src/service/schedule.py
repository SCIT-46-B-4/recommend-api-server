from sqlalchemy.orm import Session

from src.entity.schedule import ScheduleEntity
from src.dto.response.schedule_response import ScheduleResponse
from src.service.base_service import get_dto_by_id


def get_schedule_by_id(db: Session, schedule_id: int) -> ScheduleResponse | None:
    return get_dto_by_id(db, ScheduleEntity, schedule_id, ScheduleResponse)
