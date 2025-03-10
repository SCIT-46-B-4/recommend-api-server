from typing import TypeVar

from pydantic import BaseModel


E = TypeVar('Entity') # Define Genertic Entity
RESPONSEDTO = TypeVar('ResponseDto', bound=BaseModel) # Define Generic Response Dto
V = TypeVar('V') # Define Generic Value in Dict