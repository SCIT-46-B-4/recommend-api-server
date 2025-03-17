from fastapi import HTTPException, status


class BadReqException(HTTPException):
    def __init__(self, status_code=status.HTTP_400_BAD_REQUEST, detail: str="Bad Request"):
        super().__init__(status_code=status_code, detail=detail)

class RequiredQueryParameterException(BadReqException):
    def __init__(self, detail="Required Query Params Not Provided"):
        super().__init__(detail=detail)
