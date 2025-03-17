import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic

# Call csv
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_dir = os.path.join(base_dir, "data")
dest_file_path = os.path.join(data_dir, "destinations_cleaned.csv")
user_file_path = os.path.join(data_dir, "exuser_cleaned.csv")

# PKL 저장 폴더 설정
pkl_dir = os.path.join(base_dir, "pkl")
os.makedirs(pkl_dir, exist_ok=True)

destinations_df = pd.read_csv(dest_file_path, encoding='utf-8-sig')
user_df = pd.read_csv(user_file_path, encoding='utf-8-sig')

# User's region info
user_region = user_df.loc[0, 'region']  # 예제로 첫 유저를 선택. input값 고려
# Iterate
num_nights = user_df.loc[0, 'night']
num_days = user_df.loc[0, 'day']

# filtering destinations
filtered_df = destinations_df[(destinations_df['city_id'] == user_region)]

# KMeans Clustering
# k = Clustering의 개수. / 추천할 일정의 수 : int(user_df.loc[0, "schedule_style"])

# 숙소 중복 방지를 위한 list 설정
# TODO :  숙소 근처의 대표 관광지 박아넣기
# 군집의 범위를 제한하는 방법론. ->
# 숙소와 관광지의 중간값 (euclidean, har머시기.) 해당 값을 기준으로 클러스터링 -> 모델 쪼개서 각 개수를 지정.
# 중간지점 기점으로 클러스터링 진행 -> 해당 중간기점에서 가장 가까운(근사치이용) 관광지 추출.
# 해당 filtered_dest를 군집.
# 군집의 개수는 제한없이.
# TODO : 반복문은 나중에 제일 마지막
selected_acc = []
selected_closest_places = set()
all_clustered_data = {}
base_info = {}
successful_clustered_days = 0

# Clustering한 결과 Row 수가 150개 미만이면 다시 Clustering
while successful_clustered_days < num_days:
    # 숙박지 랜덤 샘플링 (중복 방지)
    if successful_clustered_days < num_nights:  # 숙박지 개수만큼만 선택
        available_acc = filtered_df[(filtered_df["type"] == 4) &
                                    (~filtered_df[["latitude", "longitude"]].apply(tuple, axis=1).isin(selected_acc))]

        if available_acc.empty:
            print(" 더 이상 선택할 숙박지가 없습니다. 중복 허용하여 선택 진행")
            available_acc = filtered_df[filtered_df["type"] == 4]  # 중복 허용

        # threshold 지정
        threshold = available_acc["score"].quantile(0.75)  # 예: 상위 25%의 score만 사용
        available_acc = available_acc[available_acc["score"] >= threshold]

        acc = available_acc.sample(n=1, weights="score").iloc[0]  # 1개 선택
        acc_location = (acc["latitude"], acc["longitude"])

        # Add selected acc. to list
        selected_acc.append(acc_location)

        print(f"🔹 Day {successful_clustered_days + 1}선택된 랜덤 좌표: {acc_location}")

    def calculate_distance(row):
        places = (float(row['latitude']), float(row['longitude']))
        return geodesic((acc_location[1], acc_location[0]), (places[1], places[0])).kilometers


    tourist_places = filtered_df[filtered_df["type"] == 6].copy()
    tourist_places["distance"] = tourist_places.apply(calculate_distance, axis=1)
    tourist_places = tourist_places.sort_values(by="distance")

    if tourist_places.empty:
        print(f"Day {successful_clustered_days + 1} : No type 6 places found.")
        selected_acc.pop()
        continue

    # Closest type 6 place from accomodation
    # closest_place: 이미 선택된 것과 중복되지 않는 후보를 선택
    closest_place = None
    for idx, row in tourist_places.iterrows():
        candidate_id = int(row["id"])
        if candidate_id not in selected_closest_places:
            closest_place = row
            break
    if closest_place is None:
        print(f"Day {successful_clustered_days + 1} : 모든 closest_place가 이미 선택되었습니다. 가장 가까운 곳을 재사용합니다.")
        closest_place = tourist_places.iloc[0]
    selected_closest_places.add(int(closest_place["id"]))
    closest_location = (closest_place["latitude"], closest_place["longitude"])
    print(
        f"Closest Place Day {successful_clustered_days + 1} 선택: {closest_place['kr_name']}, 좌표: {closest_location}")

    # Midpoint Calc.
    midpoint = ((acc_location[0] + closest_location[0]) / 2, (acc_location[1] + closest_location[1]) / 2)
    print(f"MidPoint Day {successful_clustered_days + 1} = {midpoint}")

    # Radius for KMeans
    # @params : radius
    def within_radius(row, center, radius=3):
        location = (float(row["latitude"]), float(row["longitude"]))
        return geodesic((center[1], center[0]), (location[1], location[0])).kilometers <= radius


    clustered_places = filtered_df.loc[(filtered_df["type"].isin([1, 2, 3, 5, 6])) &
                                   filtered_df.apply(lambda row: within_radius(row, midpoint), axis=1)].copy()

    # 관광지가 2개 이상 포함되도록 조정
    min_tourist_sites = 10
    if (clustered_places["type"] == 6).sum() < min_tourist_sites:
        additional_tourist_sites = tourist_places.iloc[:min_tourist_sites]
        clustered_places = pd.concat([clustered_places, additional_tourist_sites]).drop_duplicates(subset=["id"])

    # @params : clustered_places
    row_limit = 200
    if len(clustered_places) < row_limit:
        print(f"ReClustering Day {successful_clustered_days + 1}; Rows less then {row_limit}")
        selected_acc.pop()
        continue

    # 클러스터링의 개수 제한 해제.
    """
    @params : radius, len(clustered_places)
    radius : 군집의 반경 범위, 크게 하면 더 많은 Rows를 가져올 수 있음
    len(clustered_places) : Row의 개수 제한, 너무 적을 경우 다시 clustering하도록 세팅.
    """
    k = max(2, min(5, len(clustered_places)))
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=200, n_init=10, random_state=0)
    clustered_places[f"cluster_{successful_clustered_days + 1}"] = kmeans.fit_predict(
        clustered_places[["latitude", "longitude"]])

    base_info[f"day_{successful_clustered_days+1}"] = {
        "accommodation" : acc.to_dict(),
        "closest_place" : closest_place.to_dict()
    }

    all_clustered_data[successful_clustered_days + 1] = clustered_places
    successful_clustered_days += 1

# ✅ 루프가 끝난 후 base_info 저장
if base_info:
    base_info_path = os.path.join(pkl_dir, 'base_info.pkl')
    print(f"📝 base_info 저장 중... ({len(base_info)}개의 데이터 포함)")

    try:
        with open(base_info_path, 'wb') as info_file:
            pickle.dump(base_info, info_file)
        print(f"✅ base_info.pkl 저장 완료: {base_info_path}")

    except Exception as e:
        print(f"❌ base_info.pkl 저장 실패: {e}")
else:
    print("⚠️ base_info가 비어 있어 저장되지 않음.")

if all_clustered_data:
    # Base가 되는 accomodation, closest_place 정보 저장.
    # base_info_path = os.path.join(pkl_dir, f'base_info.pkl')
    # with open(base_info_path, 'wb') as info_file:
    #     pickle.dump(base_info, info_file)

    # print(f"Iter : {num} - KMeans Clustering Done (Num of clusters : {num})")

    for day, df in all_clustered_data.items():
        data_path = os.path.join(pkl_dir, f'day_{day}.pkl')
        with open(data_path, 'wb') as data_file:
            pickle.dump(df, data_file)

        # 클러스터링 결과 출력
        print(f"\n✅ Day {day} 최종 클러스터링 결과:")
        print(df[['kr_name', 'latitude', 'longitude', 'city_id'] +
                 [f"cluster_{i}" for i in range(1, num_days + 1) if
                  f"cluster_{i}" in df.columns]])

    # 클러스터링 결과 시각화
    plt.figure(figsize=(8, 6))
    unique_days = list(all_clustered_data.keys())

    palette = sns.color_palette("husl", len(unique_days))  # 선명한 색상 팔레트

    for i, (day, df) in enumerate(all_clustered_data.items()):
        plt.scatter(df["longitude"], df["latitude"],
                    c=palette[i], cmap="viridis", edgecolors="k", alpha=0.7,
                    label=f"Day {day}")

    # ✅ 기준이 되는 숙박지 강조
    for i, (lat, lon) in enumerate(selected_acc):
        if i < len(unique_days):  # Day 정보 범위 체크
            plt.scatter(lon, lat, color="red", s=200, marker="X", label=f"Acc_{unique_days[i]}")
        else:
            plt.scatter(lon, lat, color="red", s=200, marker="X", label="Acc_Unlabeled")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"KMeans Clustering (city_id = {user_region}, 기준: Midpoint 중심)")
    plt.legend()
    plt.colorbar(label="Cluster ID")
    plt.show()

else:
    print("❌ 클러스터링된 데이터가 없습니다.")

