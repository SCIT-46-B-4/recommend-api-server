import pandas as pd
import re

def preprocess_destinations(csv_path: str, output_path: str):
    """
    지정된 CSV 파일을 불러와 전처리를 수행한 뒤, 전처리된 데이터를 output_path에 저장합니다.
    """
    # 1. CSV 불러오기
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # 인코딩은 상황에 맞게 조정

    # 2. 컬럼명, 데이터 샘플 확인 (디버깅 용)
    print("=== 원본 데이터 정보 ===")
    print(df.info())
    print(df.head())

    # 3. 결측치(Missing Value) 처리
    # 예: 결측값을 특정 값으로 대체하거나, 행을 제거하는 방법
    # 여기서는 address, contact 등 문자열 컬럼을 공백("")으로 대체하는 예시
    string_columns = ['address', 'contact', 'homepage', 'how_to_go', 'available_time']
    for col in string_columns:
        if col in df.columns:
            df[col].fillna("", inplace=True)

    # 숫자형 컬럼(score, latitude, longitude 등)에 대해 결측값이 있으면 0 또는 평균값으로 대체
    numeric_columns = ['score', 'latitude', 'longitude']
    for col in numeric_columns:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # 4. 중복 제거 (id 기준 중복 제거 예시)
    if 'id' in df.columns:
        df.drop_duplicates(subset='id', keep='first', inplace=True)

    # 5. 타입 변환 (예: score, latitude, longitude를 float형으로 변환)
    # 데이터 상황에 맞게 오류가 나면 'coerce'로 처리 후 다시 fillna
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 6. 텍스트 정규화 (예: content, feature 등)
    # - HTML 태그, 이모지, 특수 문자 제거
    # - 필요하다면 한글/영어만 남기기 등
    if 'content' in df.columns:
        df['content'] = df['content'].apply(clean_text)

    # 7. 불필요한 열 제거 (예: 임시로 생성된 컬럼 등)
    # df.drop(columns=['temp_col'], errors='ignore', inplace=True)

    # 8. 전처리된 데이터 확인
    print("=== 전처리 후 데이터 정보 ===")
    print(df.info())
    print(df.head())

    # 9. 전처리된 CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"전처리된 데이터가 '{output_path}'에 저장되었습니다.")

def clean_text(text: str) -> str:
    """
    텍스트 컬럼에 대한 간단한 정규화 함수.
    예: HTML 태그, 특수 문자 제거 등
    """
    if not isinstance(text, str):
        return ""
    # HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)
    # 특수 문자 제거 (한글, 영어, 숫자, 공백만 남기기 예시)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    # 다중 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    # 예시 사용
    input_csv = "destinations.csv"  # 원본 CSV 경로
    output_csv = "destinations_cleaned.csv"  # 전처리 후 저장할 CSV 경로
    preprocess_destinations(input_csv, output_csv)
