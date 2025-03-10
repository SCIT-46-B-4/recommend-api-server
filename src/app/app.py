from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import anyio
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.endpoints.recommend import router
from src.db.connection import get_db


load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 200
    yield

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router, prefix="/api")
    # 다른 설정들(예: 미들웨어, 이벤트 핸들러 등)을 추가할 수 있습니다.

    @app.get("/")
    async def health_check_handler(db: AsyncSession=Depends(get_db)) -> Dict[str, str]:
        try:
            result = await db.execute(text("SELECT 1"))
            value = result.scalar()

            if value == 1:
                return {"statusMsg": "good"}
            else:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database is not responding")

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database connection failed: {str(e)}")

    return app
