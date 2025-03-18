from datetime import datetime

from fastapi import APIRouter, Depends

from typing import Dict

from fastapi import APIRouter, Depends, status

from src.core.model import stf
from src.core.preprocessing import user_preprocessing
from src.dto.request.survey import SurveyRequest
from src.core.model.kmeans import k_means
from src.core.exception import BadReqException, RequiredQueryParameterException
from src.core.model.rankmodel import ranking_model
from src.dto.response.schedule_response import ScheduleResponse
from src.utils import convert_query_params


router = APIRouter(prefix="/recommend")

@router.post(path="", status_code=status.HTTP_200_OK)
async def get_recommend_schedule(survey: SurveyRequest, params: Dict[str, str]=Depends(convert_query_params)):
    user_id: str | None = params.get("user_id", None)
    if not user_id:
        raise RequiredQueryParameterException("user id not given")

    survey = survey.model_dump()
    survey["user_id"] = int(user_id)
    print("#################")
    user_preprocessing.user_preprocessing(survey)
    print("@@@@@@@@@@@@@@@@@")
    k_means()
    print("$$$$$$$$$$$$$$$$$$$")
    stf.stf()
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    recommendation = ranking_model()
    if recommendation is None:
        raise BadReqException()

    # recommendation["city_id"] = survey["city_id"]
    # recommendation["start_date"] = survey["start_date"]
    # recommendation["end_date"] = survey["end_date"]
    # recommendation["city_name"] = survey["city"]


    return recommendation
