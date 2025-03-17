from datetime import datetime

from sklearn.cluster import k_means
from fastapi import APIRouter, Depends

from typing import Dict

from fastapi import APIRouter, Depends, status

from src.core.model import stf
from src.core.preprocessing import user_preprocessing
from src.dto.request.survey import SurveyRequest
from src.core.exception import BadReqException, RequiredQueryParameterException
from src.core.model.rankmodel import ranking_model
from src.dto.response.schedule_response import ScheduleResponse
from src.utils import convert_query_params


router = APIRouter(prefix="/recommend")

@router.post(path="", status_code=status.HTTP_200_OK, response_model=ScheduleResponse)
async def get_recommend_schedule(survey: SurveyRequest, params: Dict[str, str]=Depends(convert_query_params)):
    user_id: str | None = params.get("user_id", None)
    if not user_id:
        raise RequiredQueryParameterException("user id not given")

    survey["user_id"] = int(user_id)
    user_preprocessing(survey)
    k_means()
    stf()
    recommendations = ranking_model()
    if recommendations is None:
        raise BadReqException()

    return recommendations
