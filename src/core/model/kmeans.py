import pandas as pd
import os
from sklearn.cluster import KMeans
import pickle
from geopy.distance import geodesic
from src.core.exception.bad_request_exceptions import BadReqException

def k_means() -> None:
    # Call csv
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    dest_file_path = os.path.join(data_dir, "destinations_cleaned.csv")
    user_file_path = os.path.join(data_dir, "preprocessed_user.csv")

    # PKL 저장 폴더 설정
    pkl_dir = os.path.join(base_dir, "pkl")

    destinations_df = pd.read_csv(dest_file_path, encoding='utf-8-sig')
    user_df = pd.read_csv(user_file_path, encoding='utf-8-sig')

    # User's region info
    user_city = user_df.loc[0, 'city']
    num_nights = user_df.loc[0, 'night']
    num_days = user_df.loc[0, 'day']

    # filtering destinations
    filtered_df = destinations_df[(destinations_df['city_id'] == user_city)]

    # KMeans Clustering
    selected_acc = []
    selected_closest_places = set()
    all_clustered_data = {}
    base_info = {}
    successful_clustered_days = 0

    while successful_clustered_days < num_days:
        # 숙박지 랜덤 샘플링 (중복 방지)
        if successful_clustered_days < num_nights:  # 숙박지 개수만큼만 선택
            available_acc = filtered_df[
                (filtered_df["type"] == 4) &
                (~filtered_df[["latitude", "longitude"]].apply(tuple, axis=1).isin(selected_acc))
            ]

            if available_acc.empty:
                available_acc = filtered_df[filtered_df["type"] == 4]  # 중복 허용

            # threshold 지정
            # threshold = score의 상위 25%만 사용, min_distance_threshold 이상인 숙박지 선정
            threshold = available_acc["score"].quantile(0.75)  # 예: 상위 25%의 score만 사용
            min_distance_threshold = 1
            available_acc = available_acc[available_acc["score"] >= threshold]

            # 이전 숙박지가 이미 선택되어 있다면, 거리를 계산하여 min_distance_threshold 이상인 숙박지를 선택
            if selected_acc:
                prev_location = selected_acc[-1]  # 마지막에 선택된 숙박지 (latitude, longitude)
                available_acc = available_acc.copy()
                available_acc["distance_from_prev"] = available_acc.apply(
                    lambda row: geodesic(
                        (prev_location[1], prev_location[0]),
                        (row["longitude"], row["latitude"])
                    ).kilometers,
                    axis=1
                )

                candidates = available_acc[available_acc["distance_from_prev"] >= min_distance_threshold]
                if not candidates.empty:
                    acc = candidates.sample(n=1, weights="score").iloc[0]
                else:
                    acc = available_acc.sort_values(by="distance_from_prev", ascending=False).iloc[0]
            else:
                acc = available_acc.sample(n=1, weights="score").iloc[0]

            acc_location = (acc["latitude"], acc["longitude"])
            # 숙박지 좌표를 한 번만 추가
            selected_acc.append(acc_location)

        def calculate_distance(row):
            places = (float(row['latitude']), float(row['longitude']))
            return geodesic((acc_location[1], acc_location[0]), (places[1], places[0])).kilometers

        tourist_places = filtered_df[filtered_df["type"] == 6].copy()
        tourist_places["distance"] = tourist_places.apply(calculate_distance, axis=1)
        tourist_places = tourist_places.sort_values(by="distance")

        if tourist_places.empty:
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
            closest_place = tourist_places.iloc[0]
        selected_closest_places.add(int(closest_place["id"]))
        closest_location = (closest_place["latitude"], closest_place["longitude"])

        # Midpoint Calc.
        midpoint = ((acc_location[0] + closest_location[0]) / 2, (acc_location[1] + closest_location[1]) / 2)

        # Radius for KMeans
        # @params : radius
        # Add Logic : city_id에 따른 radius 설정
        city_radius_mapping = {
            1: 3,
            2: 3,
            3: 2.5,
            4: 4
        }

        radius = city_radius_mapping.get(user_city, 3)
        def within_radius(row, center, radius):
            location = (float(row["latitude"]), float(row["longitude"]))
            return geodesic((center[1], center[0]), (location[1], location[0])).kilometers <= radius

        clustered_places = filtered_df.loc[(filtered_df["type"].isin([1, 2, 3, 5, 6])) &
            filtered_df.apply(lambda row: within_radius(row, midpoint, radius), axis=1)].copy()

        min_tourist_sites = 10
        if (clustered_places["type"] == 6).sum() < min_tourist_sites:
            additional_tourist_sites = tourist_places.iloc[:min_tourist_sites]
            clustered_places = pd.concat([clustered_places, additional_tourist_sites]).drop_duplicates(subset=["id"])

        # @params : clustered_places
        row_limit = 200
        if len(clustered_places) < row_limit:
            selected_acc.pop()
            continue

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

    if base_info:
        base_info_path = os.path.join(pkl_dir, 'base_info.pkl')

        try:
            with open(base_info_path, 'wb') as info_file:
                pickle.dump(base_info, info_file)

        except Exception as e:
            raise BadReqException("Base Info not saved")
    else:
        raise BadReqException("Base Info is empty")

    if all_clustered_data:
        for day, df in all_clustered_data.items():
            data_path = os.path.join(pkl_dir, f'day_{day}.pkl')
            with open(data_path, 'wb') as data_file:
                pickle.dump(df, data_file)