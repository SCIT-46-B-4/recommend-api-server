from datetime import datetime, date
from typing import List, Optional

from pydantic import BaseModel

from src.dto.response.detail_schedule_response import DetailScheduleResponse


class ScheduleResponse(BaseModel):
    id: int
    user_id: int
    country_id: Optional[int] = None
    city_id: Optional[int] = None

    name: Optional[str] = None
    start_date: date
    end_date: date
    country_name: str
    city_name: str

    created_at: datetime
    updated_at: Optional[datetime] = None

    detail_schedules: List[DetailScheduleResponse] = []

    class Config:
        from_attributes = True
