FROM python:3.12.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
