from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from src.dto.response.city_response import CityResponse
from src.dto.response.schedule_response import ScheduleResponse


class CountryResponse(BaseModel):
    id: int
    kr_name: Optional[str] = None
    eng_name: Optional[str] = None
    iso3: Optional[str] = None
    iso2: Optional[str] = None
    continent: Optional[str] = None
    continent_code: Optional[str] = None
    currency_code: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    cities: List[CityResponse] = []
    schedules: List[ScheduleResponse] = []

    class Config:
        from_attributes = True
