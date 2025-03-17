from fastapi import HTTPException, status


class NotFoundException(HTTPException):
    def __init__(self, status_code=status.HTTP_404_NOT_FOUND, detail: str="Not Found"):
        super().__init__(status_code, detail)

class UserNotFoundExceiption(NotFoundException):
    def __init__(self, detail: str="User Not Found"):
        super().__init__(detail=detail)
