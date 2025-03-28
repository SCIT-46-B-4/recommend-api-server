from datetime import timedelta
from typing import Dict

from fastapi import APIRouter, Depends, status

from src.core.model import stf
from src.core.preprocessing.user_preprocessing import user_preprocessing
from src.core.model.kmeans import k_means
from src.core.exception import BadReqException, RequiredQueryParameterException
from src.core.model.rankmodel import ranking_model
from src.dto.request.survey import SurveyRequest
from src.dto.response import ScheduleResponse
from src.utils import convert_query_params


router = APIRouter(prefix="/recommend")

@router.post(path="", status_code=status.HTTP_200_OK)
async def get_recommend_schedule(survey: SurveyRequest, params: Dict[str, str]=Depends(convert_query_params)):
    user_id: str | None = params.get("user_id", None)
    if not user_id:
        raise RequiredQueryParameterException("user id not given")

    survey = survey.model_dump()
    city_name = survey["city"]
    survey["user_id"] = int(user_id)

    user_preprocessing(survey)
    k_means()
    stf.stf()
    recommendation = ranking_model()
    if recommendation is None:
        raise BadReqException()

    recommendation["user_id"] = user_id
    recommendation["city_id"] = survey["city_id"]
    recommendation["start_date"] = survey["start_date"]
    recommendation["end_date"] = survey["end_date"]
    recommendation["city_name"] = city_name
    recommendation["detail_schedules"] = [
        {
            **{k: v for k, v in detail.items() if k != "day"},
            "date": recommendation["start_date"] + timedelta(days=detail["day"] - 1)
        } for detail in recommendation["detail_schedules"]
    ]
    print(recommendation)
    return ScheduleResponse.model_validate(recommendation)
