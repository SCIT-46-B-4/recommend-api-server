from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from src.dto.response.destination_response import DestinationResponse


class RouteResponse(BaseModel):
    id: int
    order_number: int

    created_at: datetime
    updated_at: Optional[datetime] = None

    destination: DestinationResponse = None

    class Config:
        from_attributes = True
