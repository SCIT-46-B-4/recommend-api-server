import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def construct_pair_features(df_users, df_dest, user_embeddings, dest_embeddings):
    pair_features = []
    # 코사인 유사도를 미리 계산해도 되지만, 여기서는 한 쌍씩 계산
    # (더 빠른 계산을 위해서는 cosine_similarity(user_embeddings, dest_embeddings)를 미리 구할 수 있음)
    # 미리 계산된 행렬를 사용한다면:
    sim_matrix = cosine_similarity(user_embeddings, dest_embeddings)  # shape: (n_users, n_dest)

    for i, user_row in df_users.iterrows():
        for j, dest_row in df_dest.iterrows():
            # 1. Cosine Similarity
            cos_sim = sim_matrix[i, j]
            # 2. Normalized Rating
            rating = dest_row["norm_rating"]
            # 3. Region Match: 사용자의 region이 destination의 address 내에 포함되면 1, 아니면 0
            region_match = 1 if user_row["region"] in dest_row.get("address", "") else 0
            # 4. Text Length Difference: 결합된 텍스트의 단어 수 차이 (절대값)
            user_text_len = len(user_row["combined_features"].split())
            dest_text_len = len(dest_row["combined_features"].split())
            text_len_diff = abs(user_text_len - dest_text_len)

            # 최종 피처 벡터: 예를 들어 4차원 벡터로 구성
            feature_vector = np.array([cos_sim, rating, region_match, text_len_diff])
            # (라벨이 있는 경우 label도 추가해야 합니다. 여기서는 예시로 라벨을 1로 둠)
            # Label : 각 사용자-아이템 쌍에 대해 "해당 아이템이 사용자의 선호와 얼마나 관련이 있는지"를 나타내는 정답 값.
            # 실제 진행 시 사용자 피드백을 기반으로 라벨을 정해야함. -> Avg rating?
            label = 1  # Example label

            pair_features.append({
                "user_id": user_row["user_id"],
                "dest_id": dest_row["id"] if "id" in dest_row else j,
                "features": feature_vector,
                "label": label
            })
    return pair_features

# 사용자-아이템 쌍별 피처 생성
# pair_data = construct_pair_features(df_users, df_dest, user_embeddings, dest_embeddings)

# pair_data는 사용자-아이템 쌍별 피처 데이터를 담은 리스트라고 가정합니다.
# XGBoost의 학습에서 label은 Integer만 받기 때문에, rating을 받는다고 가정하면 소수점 처리가 요구됨.
# for d in pair_data:
#    d['label'] = int(round(random.uniform(0.0, 5.0)))

# 예시로 첫 몇 개 항목의 라벨을 출력해 확인
#for i in range(5):
#    print(f"Pair {i} label:", pair_data[i]['label'])

# 예시로 DataFrame으로 변환하여 확인
#df_pair = pd.DataFrame(pair_data)
#print(df_pair.head())

def recommend_destinations(pair_data, top_n=5):
    return None