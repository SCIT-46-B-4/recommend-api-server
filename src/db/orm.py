from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Todo(Base):
    __tablename__ = "todos"
    
    id = Column(Integer, primary_key=True, index=True)
    contents = Column(String, nullable=False)
    is_done = Column(Boolean, nullable=False, default=False)

    def __repr__(self):
        # toString
        return f"Todo(id={self.id}, contents={self.contents}, is_done={self.is_done})"
