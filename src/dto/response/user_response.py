from datetime import datetime
from typing import Optional

from src.dto.base_model import ResponseBaseModel


class UserResponse(ResponseBaseModel):
    name: str
    nickname: str
    email: str
    phone: str
    is_agree_loc: bool
    is_agree_news_noti: bool
    is_agree_marketing_noti: bool
    join_date: datetime
    deleted_at: Optional[datetime] = None
    profile_img: Optional[str] = None

    @classmethod
    def model_validate(cls, obj):
        obj_dict = obj.__dict__.copy() if hasattr(obj, "__dict__") else obj.copy()

        if "created_at" in obj_dict:
            obj_dict["join_date"] = obj_dict.pop("created_at")

        return cls.model_construct(**obj_dict)
