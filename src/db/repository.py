from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.orm import Todo

async def get_todos(session: Session) -> List[Todo]:
    return list(session.scalars(select(Todo)))

async def get_todo_by_id(session: Session, todo_id: int) -> Todo|None:
    return session.scalar(select(Todo).where(Todo.id == todo_id))