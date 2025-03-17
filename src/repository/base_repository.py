from typing import Type

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.type.common_type import E
from src.db.connection import get_db


class BaseRepository:
    def __init__(self, db: AsyncSession=Depends(get_db)):
        self.db = db

    async def get_entity_by_id(self, model: Type[E], entity_id: int) -> E | None:
        result = await self.db.execute(select(model).where(model.id == entity_id))
        entity: E | None = result.scalar_one_or_none()

        return entity
