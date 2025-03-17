from typing import List

from pydantic import BaseModel, ConfigDict

from src.utils import snake_to_camel


class SurveyRequest(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=snake_to_camel,
        populate_by_name=True,
    )

    city: str
    period: str
    companion: List[str]
    travel_style: List[str]
    transport: List[str]
    schedule_style: str

"""
surveyDto: SurveyDto(
    city=tokyo, 
    period=2n3d, 
    companion=[couple, spouse], 
    travelStyle=[experience, nature, food], 
    transport=[public, flexible], 
    scheduleStyle=relaxed
)
"""