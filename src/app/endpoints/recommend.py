from datetime import datetime
from fastapi import APIRouter, Depends

from typing import Dict

from fastapi import APIRouter, Depends, status

from src.dto.request.survey import SurveyRequest
from src.core.exception import BadReqException, RequiredQueryParameterException
from src.core.model.rankmodel import recommend_destinations
from src.dto.response.schedule_response import ScheduleResponse
from src.utils import convert_query_params


router = APIRouter(prefix="/recommend")

@router.post(
    path="",
    status_code=status.HTTP_200_OK,
    # response_model=ScheduleResponse
)
async def get_recommend_schedule(
    survey: SurveyRequest,
    params: Dict[str, str]=Depends(convert_query_params)
):
    user_id: str | None = params.get("user_id", None)
    if not user_id:
        raise RequiredQueryParameterException()
    user_id: int = int(user_id)

    recommendations = recommend_destinations(survey, user_id)

    # if recommendations is None:
    #     raise BadReqException()
    example_response = {
        "user_id": user_id,
        "country_id": 1,
        "city_id": 1,
        "name": "도쿄 여행",
        "start_date": datetime.today(),
        "end_date": datetime.today(),
        "country_name": "일본",
        "city_name": "도쿄",
        "detail_schedules": [
            {
                "date": datetime.today(),
                "routes": [
                    {
                        "order_number": 1,
                        "destination": {
                            "type": 1,
                            "kr_name": "어딜까",
                            "loc_name": "abx",
                            "title": "자 이제 떠나보자",
                            "content": "안녕!",
                            "address": "우리집이에요",
                            "contact": "010-1234-5678",
                            "homepage": "안알랴줌",
                            "how_to_go": "잘 와봐",
                            "feature": {
                                "이건": "어떻게 써먹어야 햐나"
                            },
                            "available_time": "7/24/365 가능",
                            "score": 10.0,
                            "title_img": "abc/asdasdasdasd",
                            "latitude": 36.12345,
                            "longitude": 127.123456,
                        }
                    },
                    {
                        "order_number": 2,
                        "destination": {
                            "type": 2,
                            "kr_name": "어딜까2",
                            "loc_name": "abx2",
                            "title": "자 이제 떠나보자2",
                            "content": "안녕2!",
                            "address": "우2리집이에요",
                            "contact": "0102-1234-5678",
                            "homepage": "2안알랴줌",
                            "how_to_go": "2잘 와봐",
                            "feature": {
                                "이건": "어22떻게 써먹어야 햐나"
                            },
                            "available_time": "722/24/365 가능",
                            "score": 9.0,
                            "title_img": "abc/asd222asdasdasd",
                            "latitude": 36.12345,
                            "longitude": 127.123456,
                        }
                    },
                ]
            }
        ]
    }
    return example_response

    """
    전처리 할 때 데이터를 일본어 영어 한국어를 한 언어로 통일.

    동행인 및 사용자에 따라 가중치를 어떻게 줄지 고민.
        연인, 가족 등) -> 놀이동산 박물관()
        부모님, 아이인 경우 접근성, 안전한 관광지 우선에 가중치

    입력 받는 설문 결과에 따라 가중치 분배 + 좋아요 및 평점은 우선 임시 데이터 추가해서 돌리는 걸로

    region: 지역 제한이니까
    duration + schedule style와 빽뺵 => 아침먹고 2개, 점심먹고 2개, 저녁먹고 1개 / 널널 => 아침먹고 1개, 점심먹고 2개, 저녁먹고 0개
    companion: destination feature 안에 분위기/아이와 함께/인기메뉴/주요 방문자 등의 정보가 있음 -> 이걸 쓰면 좋을 듯
    style: 어떤 여행을 가고 싶은가 -> title, contents 내에 있는 정보를 이용해 필터링 + 데이터 확인해서 featrue에 세부 분류 넣는 걸 생각해봐야함.
    transport: 주차 가능 여부만 자동차 렌탈 고객에게 가중치 주는 걸로 + 범위 제한 결과 필터링이 가능한지 찾아보고, 가능하다면 추가해보는 걸로
        => 자동차 렌탈 고객: 주차 가능 여행지에 가중치 +
        => 걷기 선호 고객: input dataset에 제한을 미리 필터링해서 ml에 넣기
        => 나머지 고객: 어떤 필터링도 걸지 않기 식으로?
    """

