from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker


DATABASE_URL = "mysql+pymysql://root:@localhost:3306/todo"
 
engine:Engine = create_engine(DATABASE_URL, echo=True)
SessionFactory:sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()