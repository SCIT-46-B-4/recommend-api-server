from src.dto.base_model import ResponseBaseModel
from src.dto.response.destination_response import DestinationResponse


class RouteResponse(ResponseBaseModel):
    order_number: int

    destination: DestinationResponse = None
