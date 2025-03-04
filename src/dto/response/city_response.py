from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CityResponse(BaseModel):
    id: int
    kr_name: Optional[str] = None
    eng_name: Optional[str] = None
    city_code: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
