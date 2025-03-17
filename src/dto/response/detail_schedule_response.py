from datetime import date
from typing import List

from src.dto.response.route_response import RouteResponse
from src.dto.base_model import ResponseBaseModel


class DetailScheduleResponse(ResponseBaseModel):
    date: date
    day: int
    routes: List[RouteResponse] = []
