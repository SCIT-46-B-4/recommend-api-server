from fastapi import FastAPI

from src.app.endpoints.recommend import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    # 다른 설정들(예: 미들웨어, 이벤트 핸들러 등)을 추가할 수 있습니다.
    return app
