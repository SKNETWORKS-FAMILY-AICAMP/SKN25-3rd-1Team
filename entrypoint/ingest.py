import os
from src.pipelines.ingestion_pipeline import ingest_faq_data #ingest_selfrepair_data

if __name__ == "__main__":
    TARGET_FAQ_PATH = "./data/raw/faq_data_v4.csv"
    TARGET_SELFREPAIR_FOLDER = r"C:\SKN25-3rd-1Team\data\raw\self-repair"
    
    print("데이터 적재 파이프라인 실행을 시작합니다...\n")

    # 1. FAQ 데이터 파이프라인 실행
    if os.path.exists(TARGET_FAQ_PATH):
        ingest_faq_data(TARGET_FAQ_PATH)
    else:
        print(f"오류: '{TARGET_FAQ_PATH}' 파일을 찾을 수 없습니다.")


    
    # if os.path.exists(TARGET_SELFREPAIR_FOLDER):
    #     ingest_selfrepair_data(TARGET_SELFREPAIR_FOLDER)
    # else:
    #     print(f"오류: '{TARGET_SELFREPAIR_FOLDER}' 폴더를 찾을 수 없습니다.")