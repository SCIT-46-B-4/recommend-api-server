import os
import urllib

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_USERNAME: str = os.getenv("DB_USERNAME", "scit")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "scit")
DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME: str = os.getenv("DB_NAME", "")
DB_PORT: str = os.getenv("DB_PORT", "3306")
DB_ECHO: bool = os.getenv("DB_ECHO", "true").lower() == "true"

if not DB_NAME:
    raise ValueError("DB_NAME 환경변수가 설정되지 않았습니다.")

# 비밀번호 특수문자 허용
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

DATABASE_URL: str = f"mysql+pymysql://{DB_USERNAME}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine: Engine = create_engine(DATABASE_URL, echo=DB_ECHO)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_db():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
