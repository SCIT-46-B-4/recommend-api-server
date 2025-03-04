from datetime import datetime, date
from pydantic import BaseModel
from typing import List, Optional

from src.dto.response.route_response import RouteResponse


class DetailScheduleResponse(BaseModel):
    id: int
    date: date

    created_at: datetime
    updated_at: Optional[datetime] = None

    route: List[RouteResponse] = []

    class Config:
        from_attributes = True
