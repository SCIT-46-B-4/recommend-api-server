import json
import random
import pandas as pd
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def combine_user_features(row):
    """
    사용자 데이터의 여러 피처를 하나의 문자열로 결합합니다.
    companion과 travel_style은 리스트인 경우 공백으로 결합합니다.
    """
    companion_text = " ".join(row["companion"]) if isinstance(row["companion"], list) else row["companion"]
    travel_style_text = " ".join(row["travel_style"]) if isinstance(row["travel_style"], list) else row["travel_style"]
    return f"{row['region']} {row['travel_duration']} {companion_text} {travel_style_text} {row['transport']} {row['schedule_style']}"

def combine_dest_features(row):
    """
    destination 데이터의 텍스트 필드를 하나의 문자열로 결합합니다.
    japanese_name, foreign_name, content, address, 그리고 detail_info 내 '영업요일', '영업시간' 등을 포함합니다.
    """
    text_parts = []
    for col in ['city_id', 'kr_name', 'loc_name','title', 'content', 'address', 'how_to_go',
                'score', 'available_time']:
        value = row.get(col, "")
        if pd.notnull(value) and value != "":
            # 만약 value가 이미 문자열이라면 그대로, 아니면 문자열로 변환
            text_parts.append(str(value))

        # JSON 형식으로 저장된 feature 컬럼 처리
        feature_value = row.get("feature", "")
        if pd.notnull(feature_value) and feature_value != "":
            try:
                # feature 컬럼의 값이 JSON 문자열이라면 파싱
                feature_json = json.loads(feature_value)
                # 만약 dict 형태라면 key와 value를 "key: value" 형태로 추가
                if isinstance(feature_json, dict):
                    for key, value in feature_json.items():
                        text_parts.append(f"{key}: {value}")
                # 만약 리스트라면 리스트의 모든 항목을 추가
                elif isinstance(feature_json, list):
                    text_parts.extend([str(item) for item in feature_json])
                else:
                    text_parts.append(str(feature_json))
            except Exception as e:
                # JSON 파싱이 실패하면 원본 문자열 그대로 추가
                text_parts.append(str(feature_value))

    return " ".join(text_parts)

# file_path 지정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_dir = os.path.join(base_dir, "data")

user_path = os.path.join(data_dir, "exuser.json")
dest_path = os.path.join(data_dir, "destinations.csv")

# 사용자 정보: recommend-api-server/data 폴더 내의 exuser.json 파일을 읽어 DataFrame으로 생성
with open(user_path, 'r', encoding='utf-8') as f:
    user_data = json.load(f)
df_users = pd.DataFrame(user_data)

# destination 정보: recommend-api-server/data 폴더 내의 exdata.csv 파일을 읽어 DataFrame으로 생성
df_dest = pd.read_csv(dest_path, encoding='utf-8')

# user, destination의 feature를 하나의 문자열로 결합
df_users["combined_features"] = df_users.apply(combine_user_features, axis=1)
df_dest["combined_features"] = df_dest.apply(lambda row: combine_dest_features(row), axis=1)

# SentenceTransformer 모델 로드
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#사용자와 destination의 결합된 텍스트를 리스트로 추출하여 임베딩 생성
user_embeddings = model.encode(df_users["combined_features"].tolist())
dest_embeddings = model.encode(df_dest["combined_features"].tolist())

# 결과는 (사용자수 x destination수) 형태의 행렬 / 코사인 유사도 계산
similarity_matrix = cosine_similarity(user_embeddings, dest_embeddings)

# score을 이용한 가중치 (임의)
if 'score' in df_dest.columns:
    df_dest["score"] = pd.to_numeric(df_dest["score"], errors="coerce")

    if df_dest["score"].isnull().all():  # score 컬럼이 있지만 모든 값이 NaN인 경우
        df_dest["norm_score"] = 1.0  # 기본값 설정
    else:
        min_score = df_dest['score'].min()
        max_score = df_dest['score'].max()

        if max_score - min_score == 0:  # 모든 점수가 동일할 경우
            df_dest["norm_score"] = 1.0
        else:
            df_dest["norm_score"] = (df_dest["score"] - min_score) / (max_score - min_score)
else:
    df_dest["norm_score"] = 1.0  # score가 아예 없는 경우 기본값 설정

top_n = 5
recommendations = {}

for i, user_id in enumerate(df_users["user_id"]):
    user_region = int(df_users.iloc[i]["region"])  # 사용자의 여행 지역

    # ✅ 사용자의 region_id와 일치하는 city_id만 필터링
    filtered_dest = df_dest[df_dest["city_id"] == user_region]

    if filtered_dest.empty:
        print(f"❌ 지역 ID {user_region}에 해당하는 여행지가 없습니다.")
        recommendations[user_id] = []
        continue

    # ✅ 해당 여행지의 인덱스를 가져오기
    filtered_indices = filtered_dest.index.tolist()
    scores = similarity_matrix[i, filtered_indices]  # 해당 여행지들의 유사도 값만 가져오기
    norm_scores = filtered_dest["norm_score"].values

    # ✅ 최종 점수 계산 및 정렬
    final_score = 0.8 * scores + 0.2 * norm_scores
    top_indices = np.argsort(final_score)[::-1][:top_n]

    # ✅ 추천 결과 저장
    recommendations[user_id] = []
    for j in top_indices:
        dest_id = filtered_dest.iloc[j]["id"]
        rec = {
            "dest_id": dest_id,
            "name": filtered_dest.iloc[j]["kr_name"],
            "location": filtered_dest.iloc[j]["loc_name"],
            "title": filtered_dest.iloc[j]["title"],
            "content": filtered_dest.iloc[j]["content"],
            "address": filtered_dest.iloc[j]["address"],
            "city_id": filtered_dest.iloc[j]["city_id"],
            "similarity": scores[j],
            "normalized_score": filtered_dest.iloc[j]["norm_score"],
            "final_score": final_score[j]
        }
        recommendations[user_id].append(rec)

# ✅ 추천된 여행지 출력
for user_id, recs in recommendations.items():
    print(
        f"\n🔹 Recommendations for User {user_id} (지역 ID: {df_users[df_users['user_id'] == user_id]['region'].values[0]}) 🔹")
    for rec in recs:
        print(f"📍 여행지: {rec['name']} ({rec['location']}) [도시 ID: {rec['city_id']}]")
        print(f"🏷️ 제목: {rec['title']}")
        print(f"📝 설명: {rec['content']}")
        print(f"📍 주소: {rec['address']}")
        print(f"⭐ 추천 점수: {rec['final_score']:.4f}")
        print("-" * 50)


