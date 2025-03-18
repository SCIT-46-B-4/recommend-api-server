from datetime import date
from typing import List

from src.dto.base_model import ResponseBaseModel
from src.dto.response.detail_schedule_response import DetailScheduleResponse


class ScheduleResponse(ResponseBaseModel):
    user_id: int

    name: str | None = None
    city_id: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    city_name: str | None = None

    detail_schedules: List[DetailScheduleResponse] = []
