from typing import Dict, List

from fastapi import Body, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.connection import get_db
from src.db.orm import Todo
from src.db import repository
from src.schemas.request import CreateTodoRequest
from src.schemas.response import TodoListSchemas, TodoSchema


app: FastAPI = FastAPI()

@app.get("/")
async def health_check_handler() -> Dict[str, str]:
    return {"ping": "pong"}

todo_data: Dict[int, CreateTodoRequest] = {
    1: {
        "id": 1,
        "contents": "실전! FastAPI 섹션 0 수강",
        "id_done": True
    },
    2: {
        "id": 2,
        "contents": "실전! FastAPI 섹션 1 수강",
        "id_done": False
    },
    3: {
        "id": 3,
        "contents": "실전! FastAPI 섹션 2 수강",
        "id_done": False
    }
}

@app.get("/todos")
async def get_todos_handler(order: str = "ASC", session: Session=Depends(get_db)) -> TodoListSchemas:
    todos: List[Todo] = await repository.get_todos(session=session)
    if order == "DESC":
        todos = todos[::-1]

    return TodoListSchemas(todos=[TodoSchema.from_orm(todo) for todo in todos])

@app.get("/todos/{todo_id}")
async def get_todo_handler(todo_id: int, session: Session=Depends(get_db)) -> TodoSchema:
    todo: Todo|None = await repository.get_todo_by_id(session, todo_id)
    if todo is None:
        raise HTTPException(status_code=404, detail="ToDo Not Found")
    return TodoSchema.from_orm(todo)

@app.post("/todos", status_code=201)
async def create_todo(request: CreateTodoRequest):
    todo_data[request.id] = request.model_dump()
    return todo_data[request.id]

@app.patch("/todos/{todo_id}")
async def update_todo(todo_id: int, is_done: bool = Body(..., embed=True)):
    todo = todo_data.get(todo_id)
    if todo:
        todo["is_done"] = is_done
        return todo
    raise HTTPException(status_code=404, detail="ToDo Not Found")

@app.delete("/todos/{todo_id}", status_code=204)
async def delete_todo(todo_id: int):
    if todo_data.pop(todo_id, None):
        return 
    raise HTTPException(status_code=404, detail="ToDo Not Found")  