
from src.entity.city import CityEntity
from src.repository.base_repository import BaseRepository


class CityRepository(BaseRepository):
    
    async def get_city_by_id(self, city_id: int) -> CityEntity | None:

        return await self.get_entity_by_id(CityEntity, city_id)