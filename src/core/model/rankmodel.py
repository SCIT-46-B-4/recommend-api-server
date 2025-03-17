import pickle
import pandas as pd
import os
import numpy as np
import json

import xgboost as xgb
from geopy.distance import geodesic

def ranking_model():
    def classify_df_user_day_by_type(df_user_day):
        """
        Memory efficient 방식으로 df_user_day를 type별로 분류합니다.
        - type 2인 경우, 각 행의 "content" 값을 기준으로 인덱스를 저장합니다.
        - 그 외의 type은 해당 행의 인덱스를 리스트에 저장합니다.

        반환 예시:
        {
            2: {
                "음식점": [0, 3, 7],
                "카페": [1, 5],
                "이자카야": [2]
            },
            6: [4, 8],
            4: [6, 9]
        }
        """
        categorized = {}
        for idx, row in df_user_day.iterrows():
            t = row["type"]
            if t == 2:
                content = row["content"]
                categorized.setdefault(2, {})
                categorized[2].setdefault(content, [])
                categorized[2][content].append(idx)
            else:
                categorized.setdefault(t, [])
                categorized[t].append(idx)
        return categorized


    # file_path 지정
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    pkl_dir = os.path.join(base_dir, "pkl")
    data_dir = os.path.join(base_dir, "data")
    base_info_path = os.path.join(pkl_dir, "base_info.pkl")
    user_info_path = os.path.join(data_dir, "exuser_cleaned.csv")

    # Base info 로드 : 정상 동작 확인
    with open(base_info_path, "rb") as data_file:
        base_info = pickle.load(data_file)

    # 딕셔너리를 DataFrame으로 변환
    df_base = pd.DataFrame.from_dict(base_info, orient="index")

    # json_normalize()를 사용하여 accommodation과 closest_place를 각각 풀어주기
    df_accommodation = pd.json_normalize(df_base["accommodation"])
    df_closest_place = pd.json_normalize(df_base["closest_place"])

    # 컬럼명 변경 (충돌 방지)
    df_accommodation = df_accommodation.add_prefix("accommodation_")
    df_closest_place = df_closest_place.add_prefix("closest_place_")

    # 최종적으로 두 개의 DataFrame을 병합
    df_final_base = pd.concat([df_accommodation, df_closest_place], axis=1)

    # print(tabulate(df_accommodation, headers="keys", tablefmt="fancy_grid"))
    # print(tabulate(df_closest_place, headers="keys", tablefmt="fancy_grid"))
    # print(tabulate(df_final_base, headers="keys", tablefmt="fancy_grid"))

    # stf에서 처리된 recommendations 파일를 로드
    recommendations = {}
    for file_name in os.listdir(data_dir):
        if file_name.startswith("recommendations") and file_name.endswith(".json"):
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
                user_recs = json.load(file)

    df_dest = []
    for day, rec in user_recs.items():
        df_day = pd.json_normalize(user_recs, record_path=[f"{day}"])
        df_day["day"] = day
        df_dest.append(df_day)

    df_final_dest = pd.concat(df_dest, ignore_index=True)
    # print(tabulate(df_final_dest, headers="keys", tablefmt="fancy_grid"))

    # 사용자 일정 스타일 로드
    user_info = pd.read_csv(user_info_path, encoding='utf-8')
    mapping_user_style = dict(zip(user_info["user_id"], user_info["schedule_style"]))

    # TODO : 일정 패턴 적용, user schedule_style에 따라 개수 조정, Nan값으로 되어있는 값들.
    # 2. 일정 패턴에 따른 일정 구성
    # 일정 패턴 정의
    itinerary = {
        5 : {
            "first_day" : [6, 2, 6, 2, 4],  # "관광지", "식당", "관광지", "식당", "숙박지"
            # 앞의 숙박지 : day1의 숙박지 -> 뒤의 숙박지 : day2의 숙박지.
            "middle_day" : [4, 2, 6, 2, 4], # "숙박지", "식당", "관광지", "식당", "숙박지"
            "last_day" : [4, 6, 2, 2] # "숙박지", "관광지", "식당", "식당(카페)"
            },
        7 : {
            "first_day" : [6, 2, 6, 2, 2, 4],  # "관광지", "식당", "식당(카페)", "관광지","식당(이자카야)", "숙박지"
            # 앞의 숙박지 : day1의 숙박지 -> 뒤의 숙박지 : day2의 숙박지.
            "middle_day" : [4, 6, 2, 2, 6, 2, 4], # "숙박지", "관광지", "식당", "식당(카페)", "관광지", "식당(이자카야)", "숙박지"
            "last_day" : [4, 6, 2, 6, 2] # "숙박지", "관광지", "식당", "관광지", "식당(카페)"
        }
    }

    food_type_order = {
        5 : {
            "first_day": ["음식점", "음식점"],
            "middle_day": ["음식점", "음식점"],
            "last_day": ["음식점", "카페"]
            },
        7 : {
            "first_day": ["음식점", "카페", "이자카야"],
            "middle_day": ["음식점", "카페", "이자카야"],
            "last_day": ["음식점", "카페"]
        }
    }

    def preprocess_base(df_final_base):
        """
        Preprocessing final_base
        """

        # dataframe의 index 초기화, drop: 기존 idx컬럼 drop, inplace: 본 데이터프레임에 반영.
        df_final_base.reset_index(drop=True, inplace=True)

        df_final_base.index = [int(i) + 1 for i in df_final_base.index]
        extra_row = []

        for day in df_final_base.index:
            base_row = df_final_base.loc[day]

            # 숙박지 데이터 추가
            acc_row = {
                "day": f"day_{str(day)}",
                "dest_id": base_row["accommodation_id"],
                "city_id": base_row["accommodation_city_id"],
                "type": base_row["accommodation_type"],
                "name": base_row["accommodation_kr_name"],
                "loc_name": base_row["accommodation_loc_name"],
                "title": base_row["accommodation_title"],
                "content": base_row["accommodation_content"],
                "latitude": base_row["accommodation_latitude"],
                "longitude": base_row["accommodation_longitude"],
                "address": base_row["accommodation_address"],
                "contact": base_row["accommodation_contact"],
                "how_to_go": base_row["accommodation_how_to_go"],
                "available_time": base_row["accommodation_available_time"],
                "created_at": base_row["accommodation_created_at"],
                "updated_at": base_row["accommodation_updated_at"],
                "title_img": base_row["accommodation_title_img"],
                "facilities": base_row["accommodation_facilities"],
                "final_score": base_row["accommodation_score"]
            }

            # 숙박지 관련 모든 컬럼 추가
            for col in df_final_base.columns:
                if col.startswith("accommodation_"):
                    acc_row[col] = base_row[col]
            extra_row.append(acc_row)

            # 가까운 관광지 데이터 추가
            place_row = {
                "day": f"day_{day}",
                "dest_id": base_row["closest_place_id"],
                "city_id": base_row["closest_place_city_id"],
                "type": base_row["closest_place_type"],
                "name": base_row["closest_place_kr_name"],
                "loc_name": base_row["closest_place_loc_name"],
                "title": base_row["closest_place_title"],
                "content": base_row["closest_place_content"],
                "latitude": base_row["closest_place_latitude"],
                "longitude": base_row["closest_place_longitude"],
                "address": base_row["closest_place_address"],
                "contact": base_row["closest_place_contact"],
                "how_to_go": base_row["closest_place_how_to_go"],
                "available_time": base_row["closest_place_available_time"],
                "created_at": base_row["closest_place_created_at"],
                "updated_at": base_row["closest_place_updated_at"],
                "title_img": base_row["closest_place_title_img"],
                "facilities": base_row["closest_place_facilities"],
                "final_score": base_row["closest_place_score"]
            }

            # 관광지 관련 모든 컬럼 추가
            for col in df_final_base.columns:
                if col.startswith("closest_place_"):
                    place_row[col] = base_row[col]
            extra_row.append(place_row)

            df_extra_row = pd.DataFrame(extra_row)
            cols_to_drop = [col for col in df_extra_row.columns if
                            col.startswith("accommodation_") or col.startswith("closest_place_")]
            df_extra_row.drop(columns=cols_to_drop, inplace=True)

        return df_extra_row

    # 0. df_final_base와 df_final_dest를 병합 : Clear
    def merge_base_with_days(df_final_base, df_final_dest):
        """
        숙박지와 가장 가까운 관광지를 day_n별로 매칭하여 df_final_dest에 추가
        """
        df_final_base = preprocess_base(df_final_base)

        # 원본 데이터와 합치기
        df_final_dest = pd.concat([df_final_dest, df_final_base], ignore_index=True)

        cols_to_drop = [col for col in df_final_dest.columns if
                        col.startswith("accommodation_") or col.startswith("closest_place_")]
        df_final_dest.drop(columns=cols_to_drop, inplace=True)

        return df_final_dest

    # Step 0 df_combined : All data
    df_combined = merge_base_with_days(df_final_base, df_final_dest)

    # print(tabulate(df_combined, headers="keys"))

    # 1. Xgboost Ranker 적용

    # 1.1 거리 / 점수(score)에 따른 가중치 조정
    def weighted_score(df_combined):
        """
        거리 계산 및 점수 계산을 수행하는 공통 함수, 가중치 적용 목적
        """

        # :param weight 가중치 : 해당 km이내에 존재하면 가중치 부여
        weight = 3

        results = []

        # Day별로 수행.
        for day in df_combined["day"].unique():
            df_day = df_combined[df_combined["day"] == day].copy()

            # 숙박지(type == 4) 찾기 (day별 기준점)
            accommodation = df_day[df_day["type"] == 4]

            if accommodation.empty:
                print(f"⚠ Warning: No accommodation found for {day}. Skipping distance calculation.")
                df_day["distance"] = np.nan
                df_day["ranked_score"] = df_day["final_score"] * df_day["score"]
                results.append(df_day)
                continue

            # 숙박지는 1개만 있어야 하므로 첫 번째 행 선택
            acc_row = accommodation.iloc[0]
            acc_lat, acc_lon = float(acc_row["longitude"]), float(acc_row["latitude"])

            # 거리 계산 수행 (숙박지와 각 장소 간의 거리)
            df_day["distance"] = df_day.apply(lambda row: geodesic(
                (acc_lat, acc_lon), (float(row["longitude"]), float(row["latitude"]))
            ).kilometers if row["type"] != 4 else 0, axis=1)  # 숙박지 자체는 거리 0

            # 거리 가중치 적용 (5km 이상일 경우 가중치 감소)
            df_day["distance_factor"] = (1 - (df_day["distance"] / weight)).clip(lower=0)  # 최소값 0으로 제한

            # 최종 점수 계산 (거리 가중치 반영)
            df_day["ranked_score"] = df_day["final_score"] * df_day["score"] * df_day["distance_factor"]

            results.append(df_day)

            # 모든 day 데이터를 다시 합치기
        return pd.concat(results, ignore_index=True)

    def prepare_data_for_xgb(df_combined):
        """
        XGBoost 학습을 위한 데이터 전처리 (feature selection)
        """
        df_combined = weighted_score(df_combined)

        data, labels, group = [], [], []

        for day in df_combined["day"].unique():
            # Day에 해당하는 데이터만 추출
            df_day = df_combined[df_combined["day"] == day].copy()
            print(f"Processing day {day}, number of records: {len(df_day)}")

            if df_day.empty:
                print(f"distance_weight() returned None for day {day}, skipping.")
                continue

            # XGBoost 학습을 위한 feature 선택
            features = ["final_score", "distance", "score", "ranked_score"]

            # 결측치 추가 처리
            x = df_day[features].values
            y = df_day["ranked_score"].values

            # ranked_score가 Nan이나 결측치일 경우 1.0으로 변환
            y = np.nan_to_num(y , nan=1.0, posinf=1.0, neginf=1.0)

            data.append(x)
            labels.append(y)
            group.append(len(df_day))

        return np.vstack(data), np.hstack(labels), np.array(group)

    def train_xgb_model(data, label, group):
        """
        XGBoost 모델 학습 및 저장 (feature 반영)
        """

        dtrain = xgb.DMatrix(data, label=labels)
        dtrain.set_group(group)

        # hyperparam for XGBoost
        """
        hyperparam for XGBoost    
        :param objective : Objective function
        :param eta (Learning rate) : 트리 가중치를 업데이트할 때의 축소율. (0.01 ~ 0.3)
        :param gamma (Regularization) : 정규화 기준. 값이 클수록 정규적인 트리구조 형성, 오버피팅 방지.
        :param max_depth : 트리의 최대 높이. (2 ~ 10) 값이 클수록 더 복잡한 패턴 학습. 오버피팅 가능성도 있음.
        :param min_child_weight : 트리의 최소 높이. (1 ~ 10) 값이 작을수록 샘플을 더 잘 분류. (트리의 성장 제한)
        :param eval_metric (Evaluation metric.) : 트리 학습을 위한 평가지표. RankModel에선 주로 NDCG를 사용. 
        """
        params = {
            "objective": "rank:pairwise",
            "eta": 0.1,
            "gamma": 1.0,
            "max_depth": 6,
            "min_child_weight": 1,
            "eval_metric": "ndcg",
            "lambda": 10

        }
        num_round = 200
        model = xgb.train(params, dtrain, num_round)

        model_path = os.path.join(pkl_dir, "xgb_rank_model.pkl")
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        print(f"✅ XGBoost 모델이 {model_path}에 저장되었습니다.")
        return model

    def apply_xgb_rank_model(df_combined, xgb_model):
        """
        XGBoost Ranker 모델을 적용하여 최적화
        """
        ranked_result = {}

        for day in df_combined["day"].unique():
            df_day = df_combined[df_combined["day"] == day].copy()
            print(f"📊 Processing {day} - DataFrame shape: {df_day.shape}")

            df_day = weighted_score(df_day)

            # XGBoost Model 적용
            dtest = xgb.DMatrix(df_day[["final_score", "score", "distance", "ranked_score"]])
            df_day["ranked_score"] = xgb_model.predict(dtest)
            df_day = df_day.sort_values(by="ranked_score", ascending=False)

            ranked_result[day] = df_day

        return ranked_result


    def map_itinerary_slots(classified, itinerary_pattern, food_type_order, user_schedule_style, day_key, df_user_day,
                            prev_accommodation, current_accommodation, current_tourism, used_places):
        """
        classified: classify_df_user_day_by_type의 결과
            예: {2: {'음식점': [idx1, idx2, ...], '카페': [...], '이자카야': [...]}, 4: [idx, ...], 6: [idx, ...]}
        itinerary_pattern: 각 슬롯에 넣어야 할 type 리스트 (예, [6, 2, 6, 2, 4])
        food_type_order: 전역 food_type_order 딕셔너리 (예, {5: {"first_day": [...], ...}, 7: {...}})
        user_schedule_style: int (예: 5 또는 7)
        day_key: "first_day", "middle_day", "last_day" 중 하나
        df_user_day: 해당 day의 DataFrame (인덱스는 원래 df_user_day의 행 인덱스)
        prev_accommodation: 이전 날 숙소 (dict, type 4)
        current_accommodation: 현재 날 숙소 (dict, type 4)
        current_tourism: 현재 날 관광지 후보 (dict, type 6)
        used_places: set, 이미 선택된 dest_id들을 담고 있음

        반환:
            itinerary_pattern 각 슬롯에 대응하는 후보 행(dict)들의 리스트
        """
        schedule_result = []
        # food_type_order 딕셔너리에서 현재 day_key에 해당하는 순서 리스트 추출 (예, ["음식점", "음식점"] 등)
        order_list = food_type_order.get(user_schedule_style, {}).get(day_key, [])
        food_index = 0
        total_four = itinerary_pattern.count(4)
        four_count = 0

        for slot in itinerary_pattern:
            if slot == 2:
                candidate = None
                # food 슬롯: order_list에 따른 우선순위로 후보 선택
                if order_list:
                    target_food = order_list[food_index] if food_index < len(order_list) else order_list[-1]
                    if 2 in classified and target_food in classified[2]:
                        # 후보 목록에서 순차적으로 pop하여 used_places에 없는 항목 선택
                        while classified[2][target_food]:
                            idx = classified[2][target_food].pop(0)
                            candidate_candidate = df_user_day.loc[idx].to_dict()
                            if candidate_candidate["dest_id"] not in used_places:
                                candidate = candidate_candidate
                                break
                # fallback: 만약 candidate가 None이고, target_food가 '카페'인 경우,
                # df_user_day 내에서 content가 "카페"인 다른 후보를 찾아봅니다.
                if candidate is None and target_food == "카페":
                    fallback_df = df_user_day[df_user_day["content"] == "카페"]
                    for idx, row in fallback_df.iterrows():
                        if row["dest_id"] not in used_places:
                            candidate = row.to_dict()
                            break

                schedule_result.append(candidate)
                if candidate is not None:
                    used_places.add(candidate["dest_id"])
                food_index += 1

            elif slot == 4:
                # 숙박 슬롯: 만약 총 슬롯이 1개이면 current_accommodation,
                # 여러 개면 첫 슬롯은 prev_accommodation, 마지막은 current_accommodation, 중간은 classified[4]에서 선택
                four_count += 1
                candidate = None
                if total_four == 1:
                    candidate = current_accommodation
                else:
                    if four_count == 1:
                        candidate = prev_accommodation
                    elif four_count == total_four:
                        candidate = current_accommodation
                    else:
                        if 4 in classified and classified[4]:
                            while classified[4]:
                                idx = classified[4].pop(0)
                                candidate_candidate = df_user_day.loc[idx].to_dict()
                                if candidate_candidate["dest_id"] not in used_places:
                                    candidate = candidate_candidate
                                    break
                schedule_result.append(candidate)
                if candidate is not None:
                    used_places.add(candidate["dest_id"])

            elif slot == 6:
                candidate = None
                # 우선 current_tourism 후보 (이미 사용되지 않은 경우)
                if current_tourism is not None and current_tourism["dest_id"] not in used_places:
                    candidate = current_tourism
                # 그 다음 classified에서 type 6 후보 선택 (used_places 체크)
                if candidate is None and 6 in classified and classified[6]:
                    while classified[6]:
                        idx = classified[6].pop(0)
                        candidate_candidate = df_user_day.loc[idx].to_dict()
                        if candidate_candidate["dest_id"] not in used_places:
                            candidate = candidate_candidate
                            break
                # fallback: df_user_day에서 type==6 행 중 used_places에 없는 항목 탐색
                if candidate is None:
                    for idx, r in df_user_day[df_user_day["type"] == 6].iterrows():
                        if r["dest_id"] not in used_places:
                            candidate = r.to_dict()
                            break
                schedule_result.append(candidate)
                if candidate is not None:
                    used_places.add(candidate["dest_id"])

            else:
                # 그 외 슬롯: 해당 type의 후보를 classified에서 순차적으로 선택 (used_places 체크)
                candidate = None
                if slot in classified and classified[slot]:
                    while classified[slot]:
                        idx = classified[slot].pop(0)
                        candidate_candidate = df_user_day.loc[idx].to_dict()
                        if candidate_candidate["dest_id"] not in used_places:
                            candidate = candidate_candidate
                            break
                schedule_result.append(candidate)
                if candidate is not None:
                    used_places.add(candidate["dest_id"])
        return schedule_result


    def map_food_type_slots(classified, food_type_order, user_schedule_style, day_key, df_user_day):
        """
        classified: classify_df_user_day_by_type의 결과 (예, {2: {'음식점': [indices,...], '카페': [...], ...}, 4: [...], 6: [...]})
        food_type_order: 위에 정의된 food_type_order dict
        user_schedule_style: 예, 5 또는 7
        day_key: "first_day", "middle_day", "last_day" 중 하나
        df_user_day: 해당 day의 DataFrame

        반환: food_type_order에 따라 각 슬롯에 해당하는 후보 행(dict)의 리스트를 반환.
                예를 들어, order_list가 ["음식점", "음식점"]이면, classified["2"]["음식점"]에서 순서대로 인덱스 0번, 1번을 pop하여 해당 행 dict로 반환.
                만약 후보가 부족하면 None을 반환.
        """
        # TODO : 중복 제거 및 빈 리스트에 대한 처리.
        # order_list에 해당하는 food type 순서를 가져옵니다.
        order_list = food_type_order.get(user_schedule_style, {}).get(day_key, [])
        result = []
        # classified가 type 2 항목을 포함하지 않으면 빈 리스트 반환
        if 2 not in classified:
            return result

        classified_type2 = classified[2]
        for food_type in order_list:
            if food_type in classified_type2 and classified_type2[food_type]:
                # 동일한 food_type이 중복되면, pop(0)을 통해 순서대로 후보(인덱스)를 꺼냅니다.
                idx = classified_type2[food_type].pop(0)
                candidate = df_user_day.loc[idx].to_dict()
                result.append(candidate)
            else:
                result.append(None)
        return result

    # 사용자 일정 스타일 맵핑 (user_id -> schedule_style)
    def apply_itinerary_pattern(df_ranked_dest, df_final_base, user_info):
        """
        일정 패턴을 적용하여 사용자별 최적의 여행 일정을 생성
        """
        # 사용자 일정 스타일 맵핑 (user_id -> schedule_style)
        user_schedule_map = dict(zip(user_info["user_id"], user_info["schedule_style"]))

        # user_id 추가 (day 기준으로 user_id 매핑)
        df_ranked_dest["user_id"] = df_ranked_dest["day"].map(lambda x: user_info["user_id"].iloc[0])

        df_final_base = preprocess_base(df_final_base)

        final_schedule = {}
        used_places = set()  # (날짜별) 장소 중복 방지.

        # Day별 정렬
        unique_days = sorted(df_ranked_dest["day"].unique(), key=lambda x: int(x.split("_")[1]))

        for day in unique_days:
            df_day = df_ranked_dest[df_ranked_dest["day"] == day].copy()

            # 현재 day 인덱스 가져오기
            day_index = int(day.split('_')[1])

            # (여러 사용자가 있다면 groupby를 사용했지만, 단일 일정으로 처리하므로 전체 df_day를 사용)
            # 해당 사용자의 schedule_style 가져오기 (기본값 5)
            # 예시에서는 첫 번째 사용자 정보로 처리
            user_id = user_info["user_id"].iloc[0]
            user_schedule_style = user_schedule_map.get(user_id, 5)

            # 사용자 일정 스타일에 따른 패턴 선택
            pattern = itinerary.get(user_schedule_style, itinerary[5])  # 기본값 5일 일정

            # Day 키 자동 선택
            if day_index == 1:
                day_key = "first_day"
            elif day_index == len(unique_days):
                day_key = "last_day"
            else:
                day_key = "middle_day"

            # 선택된 키에 맞는 패턴 할당
            day_pattern = pattern[day_key]

            # 장소 분류 (type 값 기반)
            categorized = classify_df_user_day_by_type(df_day)

            # 이전날/다음날 숙박지 설정
            current_accommodation = None  # Day1에 적용할 숙박지
            prev_accommodation = None
            current_tourism = None
            prev_day = f"day_{day_index - 1}"

            if prev_day in df_final_base['day'].values:
                prev_accommodation = df_final_base[df_final_base['day'] == prev_day].iloc[0].to_dict()
            if day in df_final_base['day'].values:
                df_temp = df_final_base[df_final_base['day'] == day]
                if len(df_temp) >= 1:
                    current_accommodation = df_temp.iloc[0].to_dict()
                if len(df_temp) >= 2:
                    current_tourism = df_temp.iloc[1].to_dict()

            # 🔹 일정 구성
            itinerary_list = map_itinerary_slots(categorized, day_pattern, food_type_order, user_schedule_style,
                                                 day_key, df_day,
                                                 prev_accommodation, current_accommodation, current_tourism,
                                                 used_places)

            # 사용된 장소 업데이트
            for cand in itinerary_list:
                if cand is not None:
                    used_places.add(cand["dest_id"])

            # 최종 일정은 day별로 저장 (user_id 단계 없이)
            final_schedule[day] = itinerary_list

        return final_schedule

    # Model Execute
    data, labels, group = prepare_data_for_xgb(df_combined)
    xgb_model = train_xgb_model(data, labels, group)
    model_itinerary = apply_xgb_rank_model(df_combined, xgb_model)
    df_ranked_dest = pd.concat(model_itinerary.values(), ignore_index=True)

    schedule_response = {"detail_schedules": []}
    # 일정 적용 실행
    final_schedule = apply_itinerary_pattern(df_ranked_dest, df_final_base, user_info)

    for day_key, user_schedule in sorted(final_schedule.items(), key=lambda x: int(x[0].split('_')[1])):
        day_num = int(day_key.split('_')[1])

        detail_schedule = {
            "day": day_num,
            "routes": []
        }
        
        # 각 일정(루트)을 순서대로 RouteResponse에 맞게 변환
        for order_number, route_item in enumerate(user_schedule, start=1):
            destination = {
                "id": route_item.get("dest_id"),
                "type": route_item.get("type"),
                "kr_name": route_item.get("name"),
                "title": route_item.get("title"),
                "title_img": route_item.get("title_img"),
            }
            
            route = {
                "order_number": order_number,
                "destination": destination
            }
            
            detail_schedule["routes"].append(route)
        
        schedule_response["detail_schedules"].append(detail_schedule)

    return schedule_response
