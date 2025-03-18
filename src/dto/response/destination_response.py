from typing import Dict
import enum
from typing import Dict

from src.dto.base_model import ResponseBaseModel

class DestinationType(enum.Enum):
    """
    1: 관광지, 2: 식당, 3: 쇼핑센터, 4: 숙박업소, 5: 대중교통
    """
    SIGHTSEEING = "1"
    RESTAURANT = "2"
    SHOPPING = "3"
    ACCOMMODATION = "4"
    TRANSPORT = "5"

    def __str__(self):
        mapping = {
            "1": "관광",
            "2": "식당",
            "3": "쇼핑센터",
            "4": "숙박업소",
            "5": "대중교통"
        }
        return mapping[self.value]

class DestinationResponse(ResponseBaseModel):
    # ToDo: Enum으로 교체
    # 예: '1', '2' 등 문자형으로 저장된 경우
    type: DestinationType
    id: int
    kr_name: str
    title: str
    title_img: str | None = None

    latitude: float | None = None
    longitude: float | None = None

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
