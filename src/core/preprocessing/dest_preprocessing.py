import asyncio
import pandas as pd
import os
import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from src.db.connection import get_db
from src.repository.city import CityRepository

# Call csv
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_dir = os.path.join(base_dir, "data")
file_path = os.path.join(data_dir, "destinations.csv")

destinations_df = pd.read_csv(file_path, encoding='utf-8-sig')

# Extract meaningful features from `feature` column
def extract_features(feature_col):
    try:
        if isinstance(feature_col, str):
            feature_dict = json.loads(feature_col.replace("'", '"'))
            if isinstance(feature_dict, dict):
                return feature_dict.get("facilities", [])  # Ensure facilities is always a list
        return []  # Default empty list if parsing fails or format is incorrect
    except json.JSONDecodeError:
        return []

def extract_atmosphere(feature_col):
    try:
        if isinstance(feature_col, str):
            feature_dict = json.loads(feature_col.replace("'", '"'))
            if isinstance(feature_dict, dict):
                return feature_dict.get("분위기")  # Extract atmosphere
        return []
    except json.JSONDecodeError:
        return []

destinations_df["facilities"] = destinations_df["feature"].apply(extract_features)
destinations_df["atmosphere"] = destinations_df["feature"].apply(extract_atmosphere)

del destinations_df["feature"]  # Remove raw feature column

# Encoding 'type' 2 elements into specific categories
cafe_keywords = {"카페", "커피숍/커피 전문점", "아트카페", "코스프레 카페", "에스프레소 바", "동물카페", "아이스크림 가게", "제과점", "카페테리아"}
shisha_keywords = {"물담배 바", "물담뱃대 판매점"}
bar_keywords = {"이자카야", "술집", "모던 이자카야 레스토랑", "여성 접대 술집"}

def categorize_type_2(row):
    if row["type"] == 2:
        if row["content"] in cafe_keywords:
            return "카페"
        elif row["content"] in shisha_keywords:
            return "물담배"
        elif row["content"] in bar_keywords:
            return "이자카야"
        elif row["content"] == "선물 가게":
            return "선물가게"
        else:
            return "음식점"
    return row["content"]

destinations_df["content"] = destinations_df.apply(categorize_type_2, axis=1)

# Extract Latitude & Longitude from `wkt_coordinate`
def extract_lat_long(wkt):
    try:
        if pd.isna(wkt):
            return None, None
        coords = wkt.replace("POINT(", "").replace(")", "").split()
        return float(coords[1]), float(coords[0])
    except Exception:
        return None, None

destinations_df[["latitude", "longitude"]] = destinations_df["wkt_coordinate"].apply(lambda x: pd.Series(extract_lat_long(x)))
del destinations_df["wkt_coordinate"]  # Remove original column

# Process `how_to_go` column for transportation-based recommendations
def get_transport_accessibility(how_to_go):
    if pd.isna(how_to_go):
        return "unknown"
    if "駅" in how_to_go:
        return "대중교통"
    return how_to_go

destinations_df["how_to_go"] = destinations_df["how_to_go"].apply(get_transport_accessibility)

# 비동기 DB 엔진 생성 (PostgreSQL 기준, MySQL은 'mysql+aiomysql://...' 사용)
DATABASE_URL = "mysql+asyncmy://scit:scit@localhost:3306/letsleave"

# Async SQLAlchemy 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# 비동기 세션 생성
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# 비동기 함수: 데이터 가져오기
async def fetch_data(id):
    async with AsyncSessionLocal() as session:
        result = await session.execute(text(f"SELECT kr_name FROM cities where id={id}"))
        row = result.scalar_one_or_none()
    return {"kr_name": row}

# 모든 city_id에 대해 fetch_data 실행하는 함수
async def fetch_all_data(city_ids):
    tasks = [fetch_data(city_id) for city_id in city_ids]
    return await asyncio.gather(*tasks)

# destinations_df의 city_id 컬럼을 리스트로 변환
city_ids = destinations_df["city_id"].tolist()

# 비동기 작업 실행하여 결과 리스트 얻기
kr_name_dict_list = asyncio.run(fetch_all_data(city_ids))

# 만약 단순히 kr_name 문자열만 필요하다면, fetch_data에서 return {"kr_name": row} 대신 row를 반환하거나,
# 아래와 같이 리스트 컴프리헨션을 이용해 값을 추출할 수 있습니다.
# 각 행에 대해 {"city_id": city_id, "kr_name": kr_name} 형태의 딕셔너리 생성
destinations_df["city"] = [
    {"kr_name": res["kr_name"]}
    for res in kr_name_dict_list
    ]

# Save processed data
cleaned_file_path = os.path.join(data_dir, "destinations_cleaned.csv")
destinations_df.to_csv(cleaned_file_path, encoding="utf-8-sig")

