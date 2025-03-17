from src.entity.user import UserEntity
from src.repository import BaseRepository


class UserRepository(BaseRepository):

    async def get_user_by_id(self, user_id: int) -> UserEntity | None:

        return await self.get_entity_by_id(UserEntity, user_id)
