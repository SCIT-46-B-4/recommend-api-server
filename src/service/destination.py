from sqlalchemy.orm import Session

from src.entity.destination import DestinationEntity
from src.dto.response.destination_response import DestinationResponse
from src.service.base_service import get_dto_by_id


def get_destination_by_id(db: Session, destination_id: int) -> DestinationResponse | None:
    return get_dto_by_id(db, DestinationEntity, destination_id, DestinationResponse)
