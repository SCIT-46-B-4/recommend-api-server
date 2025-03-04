from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserResponse(BaseModel):
    id: int
    name: str
    nickname: str
    email: str
    phone: str
    is_agree_loc: bool
    is_agree_news_noti: bool
    is_agree_marketing_noti: bool
    join_date: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    profile_img: Optional[str] = None

    class Config:
        from_attributes = True
