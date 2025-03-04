from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel


class DestinationResponse(BaseModel):
    id: int

    # ToDo: Enum으로 교체
    # 예: '1', '2' 등 문자형으로 저장된 경우
    type: str
    kr_name: str
    loc_name: str
    title: str
    content: str
    address: Optional[str] = None
    contact: Optional[str] = None
    homepage: Optional[str] = None
    how_to_go: str
    available_time: Optional[str] = None
    feature: Optional[Dict] = None
    score: float
    title_img: Optional[str] = None

    coordinate: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

    @classmethod
    def from_orm_custom(cls, orm_obj):
        """
        ORM 객체에서 기본 Pydantic 모델로 변환한 후,
        coordinate 필드(WKT 문자열)를 파싱하여 latitude와 longitude를 재할당합니다.
        """
        instance = cls.model_validate(orm_obj)
        if orm_obj.coordinate:
            coord_str = orm_obj.coordinate
            if coord_str.startswith("POINT(") and coord_str.endswith(")"):
                coord_values = coord_str[6:-1].split()
                if len(coord_values) == 2:
                    try:
                        instance.latitude = float(coord_values[0])
                        instance.longitude = float(coord_values[1])
                    except ValueError:
                        pass
        return instance
