import os
import pandas as pd
from src.db.connection import engine
from shapely.wkb import loads


def parse_coordinate_wkb(coord):
    try:
        geom = loads(coord)
        return geom.y, geom.x  # (latitude, longitude)
    except Exception as e:
        print("WKB 파싱 오류:", e)
        return None, None


def parse_coordinate(coordinate):
    if isinstance(coordinate, bytes):
        lat, lng = parse_coordinate_wkb(coordinate)
        return pd.Series([lat, lng])
    elif isinstance(coordinate, str):
        if coordinate.startswith("POINT(") and coordinate.endswith(")"):
            inner = coordinate[6:-1].strip()
            parts = inner.split()
            if len(parts) == 2:
                try:
                    lng = float(parts[0])
                    lat = float(parts[1])
                    return pd.Series([lat, lng])
                except ValueError:
                    return pd.Series([None, None])
        return pd.Series([None, None])
    else:
        return pd.Series([None, None])


def export_destinations_to_csv():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "destinations.csv")

    # ST_AsText를 사용하여 coordinate 값을 WKT 문자열로 변환해서 가져옵니다.
    query = "SELECT *, ST_AsText(coordinate) AS wkt_coordinate FROM destinations"
    df = pd.read_sql(query, engine)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"총 {len(df)} 건의 데이터를 '{file_path}'로 내보냈습니다.")


if __name__ == "__main__":
    export_destinations_to_csv()
