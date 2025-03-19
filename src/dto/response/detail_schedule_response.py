from datetime import datetime
from typing import List

from src.dto.response.route_response import RouteResponse
from src.dto.base_model import ResponseBaseModel


class DetailScheduleResponse(ResponseBaseModel):
    date: datetime | None = None
    day: int
    routes: List[RouteResponse] = []
