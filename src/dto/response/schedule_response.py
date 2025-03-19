from datetime import date
from typing import List

from src.dto.base_model import ResponseBaseModel
from src.dto.response.detail_schedule_response import DetailScheduleResponse


class ScheduleResponse(ResponseBaseModel):
    user_id: int

    name: str | None = None
    city_id: int
    start_date: date
    end_date: date
    city_name: str

    detail_schedules: List[DetailScheduleResponse] = []
