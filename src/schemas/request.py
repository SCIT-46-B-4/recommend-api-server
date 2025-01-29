from pydantic import BaseModel


class CreateTodoRequest(BaseModel):
    id: int
    contents: str
    is_done: bool