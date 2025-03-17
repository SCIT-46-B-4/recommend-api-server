from datetime import date
from typing import List

from src.dto.base_model import ResponseBaseModel
from src.dto.response.detail_schedule_response import DetailScheduleResponse


class ScheduleResponse(ResponseBaseModel):
    user_id: int
    country_id: int | None = None
    city_id: int | None = None

    name: str | None = None
    start_date: date
    end_date: date
    country_name: str
    city_name: str

    detail_schedules: List[DetailScheduleResponse] = []
