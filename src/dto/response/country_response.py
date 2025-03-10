from typing import List

from src.dto.base_model import ResponseBaseModel
from src.dto.response.city_response import CityResponse
from src.dto.response.schedule_response import ScheduleResponse


class CountryResponse(ResponseBaseModel):
    kr_name: str | None = None
    eng_name: str | None = None
    iso3: str | None = None
    iso2: str | None = None
    continent: str | None = None
    continent_code: str | None = None
    currency_code: str | None = None

    cities: List[CityResponse] = []
    schedules: List[ScheduleResponse] = []
