from typing import Optional

from src.dto.base_model import ResponseBaseModel


class CityResponse(ResponseBaseModel):
    kr_name: Optional[str] = None
    eng_name: Optional[str] = None
    city_code: Optional[str] = None
