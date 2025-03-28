from datetime import datetime
from pydantic import BaseModel, ConfigDict

from src.utils import snake_to_camel


class ResponseBaseModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=snake_to_camel,
        populate_by_name=True,
    )

    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
