import json
import pickle
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# file_path 지정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_dir = os.path.join(base_dir, "data")
pkl_dir = os.path.join(base_dir, "pkl")

user_path = os.path.join(data_dir, "exuser_cleaned.csv")

# 사용자 정보 로드
df_users = pd.read_csv(user_path, encoding='utf-8')

# 시간이 오래 소요되는 파트 ~~
# SentenceTransformer 모델 로드
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 사용자의 feature 결합
def combine_user_features(row):
    companion_text = " ".join(row["companion"]) if isinstance(row["companion"], list) else row["companion"]
    travel_style_text = " ".join(row["travel_style"]) if isinstance(row["travel_style"], list) else row["travel_style"]
    return f"{row['region']} {row['travel_duration']} {companion_text} {travel_style_text} {row['transport']} {row['schedule_style']}"


# 여행지의 feature 결합
def combine_dest_features(row):
    text_parts = []
    for col in ['city_id', 'kr_name', 'title', 'content', 'address', 'how_to_go', 'score', 'available_time',
                'facilities', 'atmosphere']:
        value = row.get(col, "")
        if pd.notnull(value) and value != "":
            text_parts.append(str(value))
    return " ".join(text_parts)

# 최종 점수 계산 후, 카페에 가중치 적용하는 함수
def adjust_cafe_weight(df, cafe_weight):
    """
    df: 추천 대상 DataFrame
    cafe_weight: 카페에 부여할 가중치 (예: 1.2이면 20% 증가)
    """
    # type이 2이고 content가 "카페"인 경우에 final_score를 cafe_weight로 곱합니다.
    df.reset_index(drop=True, inplace=True)
    mask = (df["type"] == 2) & (df["content"] == "카페")
    df.loc[mask, "final_score"] = df.loc[mask, "final_score"] * cafe_weight
    return df

def adjust_food_weight(df, food_weight):
    """
    df: 추천 대상 DataFrame
    food_weight: 음식에 부여할 가중치 (예: 1.2이면 20% 증가)
    """
    df.reset_index(drop=True, inplace=True)
    # type이 2이고 content가 "카페"인 경우에 final_score를 cafe_weight로 곱합니다.
    mask = (df["type"] == 2) & (df["content"] == "음식점")
    df.loc[mask, "final_score"] = df.loc[mask, "final_score"] * food_weight
    return df

# User_id로 추천 진행
user_id = df_users.iloc[0]["user_id"]
user_row = df_users[df_users["user_id"] == user_id].iloc[0]

user_region = int(user_row["region"])  # 사용자의 지역
num_days = int(user_row["day"])  # 사용자의 여행 일정 (Day 수)
recommendations = {}

# ✅ Day별 추천 진행
for day in range(1, num_days + 1):
    day_file_path = os.path.join(pkl_dir, f"day_{day}.pkl")

    # Day별 데이터 로드
    if not os.path.exists(day_file_path):
        print(f"❌ Day {day}의 클러스터링 파일이 존재하지 않습니다. Skipping...")
        continue

    with open(day_file_path, "rb") as data_file:
        df_dest = pickle.load(data_file)

    # 사용자의 주요 클러스터 추출
    if f"cluster_{day}" in df_dest.columns and not df_dest[f"cluster_{day}"].isnull().all():
        filtered_cluster = df_dest[f"cluster_{day}"].mode()
        user_cluster = filtered_cluster.iloc[0] if not filtered_cluster.empty else None
    else:
        print(f"❌ Day {day}: 클러스터 정보 없음")
        continue

    # 사용자의 region_id와 일치하는 city_id만 필터링
    filtered_dest = df_dest[(df_dest["city_id"] == user_region) & (df_dest[f"cluster_{day}"] == user_cluster)]

    if filtered_dest.empty:
        print(f"❌ Day {day}: 지역 {user_region}, 클러스터 {user_cluster}에 해당하는 여행지가 없습니다.")
        continue

    user_feature = combine_user_features(user_row)
    filtered_dest["combined_features"] = filtered_dest.apply(lambda row: combine_dest_features(row), axis=1)

    # SentenceTransformer 임베딩 생성
    user_embedding = model.encode([user_feature])
    dest_embeddings = model.encode(filtered_dest["combined_features"].tolist())

    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(user_embedding, dest_embeddings)[0]

    # score 가중치 반영
    if "score" in filtered_dest.columns:
        filtered_dest["score"] = pd.to_numeric(filtered_dest["score"], errors="coerce").fillna(0.3)
        min_score, max_score = filtered_dest["score"].min(), filtered_dest["score"].max()
        filtered_dest["norm_score"] = (filtered_dest["score"] - min_score) / (max_score - min_score) \
            if max_score > min_score else 0.3
    else:
        filtered_dest["norm_score"] = 0.3

    # 최종 점수 계산
    final_score = 0.8 * similarity_scores + 0.2 * filtered_dest["norm_score"].values
    filtered_dest["final_score"] = final_score
    filtered_dest = adjust_cafe_weight(filtered_dest, cafe_weight=1.3)
    filtered_dest = adjust_food_weight(filtered_dest, food_weight=1.0)

    # 상위 추천 목록 선정 : RankingModel에 넘길 데이터의 개수
    top_n = 30
    top_indices = np.argsort(filtered_dest["final_score"].values)[::-1][:top_n]

    # 추천 결과 저장
    day_recommendations = []
    type_6_set = set()

    for j in top_indices:
        dest_id = int(filtered_dest.iloc[j]["id"])
        dest_type = int(filtered_dest.iloc[j]["type"])

        if dest_type == 6 and dest_id in type_6_set:
            continue

        rec = {
            "dest_id": dest_id,
            "name": str(filtered_dest.iloc[j]["kr_name"]),
            "title": str(filtered_dest.iloc[j]["title"]),
            "content": str(filtered_dest.iloc[j]["content"]),
            "address": str(filtered_dest.iloc[j]["address"]),
            "city_id": int(filtered_dest.iloc[j]["city_id"]),
            "type" : dest_type,
            "latitude" : float(filtered_dest.iloc[j]["latitude"]),
            "longitude" : float(filtered_dest.iloc[j]["longitude"]),
            "facilities": str(filtered_dest.iloc[j]["facilities"]),
            "atmosphere": str(filtered_dest.iloc[j]["atmosphere"]),
            "score" : float(filtered_dest.iloc[j]["score"]),
            "similarity": float(similarity_scores[j]),
            "normalized_score": float(filtered_dest.iloc[j]["norm_score"]),
            "final_score": float(final_score[j])
        }
        day_recommendations.append(rec)

        if dest_type == 6:
            type_6_set.add(dest_id)

    sites_count = len(type_6_set)

    # 추천된 관광지가 부족할 시 fallback으로 extra_tourist_sites추천
    if sites_count < 5:
        extra_candidates = filtered_dest[(filtered_dest["type"] == 6) & (~filtered_dest["id"].isin(list(type_6_set)))]
        extra_candidates = extra_candidates.sort_values("final_score", ascending=False)
        needed = 5 - sites_count
        for _, extra in extra_candidates.head(needed).iterrows():
            rec_extra = {
                "dest_id": int(extra["id"]),
                "kr_name": str(extra["kr_name"]),
                "title": str(extra["title"]),
                "content": str(extra["content"]),
                "address": str(extra["address"]),
                "city_id": int(extra["city_id"]),
                "type": int(extra["type"]),
                "latitude": float(extra["latitude"]),
                "longitude": float(extra["longitude"]),
                "facilities": str(extra["facilities"]),
                "atmosphere": str(extra["atmosphere"]),
                "score": float(extra["score"]),
                "similarity": float(
                    cosine_similarity(user_embedding, model.encode([combine_dest_features(extra)]))[0][0]),
                "normalized_score": float(extra["norm_score"]),
                "final_score": float(extra["final_score"])
            }
            day_recommendations.append(rec_extra)
            type_6_set.add(int(extra["id"]))

    recommendations[f"day_{day}"] = day_recommendations

# 전체 Day 추천 결과 JSON 저장
final_output_path = os.path.join(data_dir, f"recommendations_{user_id}.json")
with open(final_output_path, "w", encoding="utf-8") as json_file:
    json.dump(recommendations, json_file, ensure_ascii=False, indent=4)

print(f"✅ 유저 {user_id}의 전체 추천 결과가 {final_output_path}에 저장되었습니다.")
