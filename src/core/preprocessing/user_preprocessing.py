import json
import os
import ast
import pandas as pd

# Call json
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_dir = os.path.join(base_dir, "data")
file_path = os.path.join(data_dir, "exuser.json")

with open(file_path, "r", encoding="utf-8") as file:
    user_data = json.load(file)

# Mapping
region_mapping = {
    "tokyo" : 1,
    "osaka" : 2,
    "fukuoka" : 3,
    "sapporo" : 4
}

# travel_duration Skip : daytrip, 1n2d, 2n3d, .....
# parsing column to night, day

transport_mapping = {
    "public" : "대중교통",
    "rental" : "렌터카",
    "taxi" : "택시/차량",
    "walk" : "도보",
    "bike" : "자전거/오토바이",
    "flexible": "상황에 따라 유연"
}

companion_mapping = {
    "friend" : "친구",
    "couple" : "연인",
    "spouse" : "배우자",
    "kid" : "아이",
    "parents" : "부모님",
    "other" : "기타",
}

travel_style_mapping = {
    "experience" : "체험/액티비티",
    "sns" : "SNS 핫플레이스",
    "nature" : "자연",
    "famous" : "유명 관광지는 필수",
    "healing" : "여유롭게 힐링",
    "culture" : "문화/예술/역사",
    "shopping" : "쇼핑은 열정적",
    "food" : "먹방",
}

# 일정의 개수를 지정. relaxed 5, tight 7
schedule_style_mapping = {
    "tight" : 7,
    "relaxed" : 5
}

# Multiple Elements
def map_values(value, mapping):
    if isinstance(value, list):
        mapped_values = [mapping.get(v, v) for v in value]
        return mapped_values if len(mapped_values) > 1 else mapped_values[0]
    return mapping.get(value, value)

# Apply Mapping
for user in user_data:
    user["region"] = region_mapping.get(user["region"], "알 수 없음")
    user["transport"] = map_values(user.get("transport", []), transport_mapping)
    user["travel_style"] = map_values(user.get("travel_style", []), travel_style_mapping)
    user["companion"] = map_values(user.get("companion", []), companion_mapping)
    user["schedule_style"] = schedule_style_mapping.get(user["schedule_style"], "알 수 없음")

user_df = pd.DataFrame(user_data)

# 리스트형 데이터 변환 함수
def convert_to_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith("[") else value
    except (SyntaxError, ValueError):
        return value

user_df["transport"] = user_df["transport"].apply(convert_to_list)
user_df["travel_style"] = user_df["travel_style"].apply(convert_to_list)
user_df["companion"] = user_df["companion"].apply(convert_to_list)

# Transport One-Hot Encoding
def encode_transport(value):
    if isinstance(value, list):  # 다중 선택 처리
        return 0 if any(v in ["렌터카", "택시/차량"] for v in value) else 1
    return 0 if value in ["렌터카", "택시/차량"] else 1

user_df["transport_encoded"] = user_df["transport"].apply(encode_transport)

# Parsing Travel Duration for night , day
def split_travel_duration(duration):
    try:
        if isinstance(duration, str) and 'n' in duration and 'd' in duration:
            night, day = duration.split('n')[0], duration.split('n')[1].replace('d', '')
            return int(night), int(day)
    except ValueError:
        return None, None
    return None, None

user_df[["night", "day"]] = user_df["travel_duration"].apply(lambda x: pd.Series(split_travel_duration(x)))

user_cleaned_filepath = os.path.join(data_dir, "exuser_cleaned.csv")

user_df.to_csv(user_cleaned_filepath, index=False, encoding="utf-8-sig")


