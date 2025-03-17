import pandas as pd
import os
import json

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

# Save processed data
cleaned_file_path = os.path.join(data_dir, "destinations_cleaned.csv")
destinations_df.to_csv(cleaned_file_path, encoding="utf-8-sig")

