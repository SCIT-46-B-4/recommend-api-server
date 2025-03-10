import os
import urllib

from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

load_dotenv()

DB_USERNAME: str = os.getenv("DB_USERNAME", "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root")
DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME: str = os.getenv("DB_NAME", "")
DB_PORT: str = os.getenv("DB_PORT", "3306")
DB_ECHO: bool = os.getenv("DB_ECHO", "true").lower() == "true"

if not DB_NAME:
    raise ValueError("DB_NAME 환경변수가 설정되지 않았습니다.")

# 비밀번호 특수문자 허용
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

DATABASE_URL: str = f"mysql+asyncmy://{DB_USERNAME}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=DB_ECHO,
    pool_size=10,
    max_overflow=0,
    pool_timeout=30, # second
    pool_recycle=60,  # second
    pool_pre_ping=True,
)
SessionFactory = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session = SessionFactory()
    try:
        yield session
    finally:
        await session.close()
