from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI

from src.app.app import create_app


load_dotenv()

app: FastAPI = create_app()

@app.get("/")
async def health_check_handler() -> Dict[str, str]:
    return {"statusMsg": "good"}
