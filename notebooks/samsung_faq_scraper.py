import requests
import pandas as pd
import time
import random
from bs4 import BeautifulSoup

# ── 요청 헤더 설정 ──────────────────────────────────────────────────────────────
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Referer": "https://www.samsungsvc.co.kr/",
    "Origin": "https://www.samsungsvc.co.kr",
    "Accept": "application/json, text/plain, */*",
    "Cookie": (
        "ssvcSvcJWT=SDP+eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJzdmMiLCJqdGkiOiItXHUwMDEwXHUwMDE2XHUwMDEwXHUwMDEwJFx1MDAwRSsiLCJhdWQiOiIxMjEuMTM0LjQ2LjI0LCAxNC4wLjExMy45NCwgMTAzLjIzLjEwMC4xMjEiLCJpc3MiOiJJLU9OIiwiaWF0IjoxNzc0OTE0ODIyLCJleHAiOjMyNDcyMTExNjAwfQ.8q8y6zn8yU2TAknbihVnLNqPAomPRNMNC1koqCwIIl5qjWD6sp2QdV4H1N_qbYGDlUMdbGto10rpNbHurhEnCQ; "
        "ssvcReSvcJWT=SDP+eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJzdmMiLCJqdGkiOiIwYTA1MGMzZC1iMjFmLTRhNWMtOGQwYS1kNTk5ZDM2Y2M5N2YiLCJhdWQiOiIxMjEuMTM0LjQ2LjI0LCAxNC4wLjExMy45NCwgMTAzLjIzLjEwMC4xMjEiLCJpc3MiOiJJLU9OIiwiaWF0IjoxNzc0OTE0ODIyLCJleHAiOjMyNDcyMTExNjAwfQ.8d-6nhXuTAvwfwKIHs9VNIBdD-L-ogkHnPYOqVvsfSQ5ZIoZf39m_V5SNXop2gj6cvVywXqx-cNEYZ02T8Ex4A; "
    )
}

# ── 필터 설정 ───────────────────────────────────────────────────────────────────
# 수집 대상 카테고리 코드 (스마트폰)
ALLOWED_CODES = ['10414']

# 수집 제외 키워드 (태블릿, 노트북, 가전 등)
BANNED_KEYWORDS = [
    '태블릿', 'Tablet', 'Tab ', '갤럭시 탭', '갤럭시탭',
    '갤럭시 북', '갤럭시북', '노트북', 'Windows', 'PC',
    '에어컨', '냉장고', '세탁기', '건조기', 'TV'
]

# 삼성 서비스 FAQ 전체 항목 수
TOTAL_SAMSUNG_FAQ = 9151


def is_smartphone(item_json: dict) -> bool:
    """스마트폰 카테고리에 해당하는 항목인지 검증하는 함수.
    
    - 제목 또는 분류 레이블에 제외 키워드가 포함되면 False 반환
    - ALLOWED_CODES에 해당하는 카테고리 코드가 있어야 True 반환
    """
    title = item_json.get('title', '')
    menu_list = item_json.get('menuId', [])

    # 제목에 제외 키워드가 있으면 스킵
    if any(k.lower() in title.lower() for k in BANNED_KEYWORDS):
        return False

    has_target_code = False
    for menu in menu_list:
        label = str(menu.get('label', ''))
        # 분류 레이블에 제외 키워드가 있으면 스킵
        if any(k.lower() in label.lower() for k in BANNED_KEYWORDS):
            return False
        if str(menu.get('value')) in ALLOWED_CODES:
            has_target_code = True

    return has_target_code


def fetch_faq_detail(session: requests.Session, faq_seq: str) -> dict | None:
    """FAQ 상세 내용을 API에서 가져오는 함수."""
    url = f"https://api.samsungsvc.co.kr/svc/hpFaqApi/faqServiceRead"
    params = {"faqSeq": faq_seq, "_siteId": "ssvc"}
    res = session.get(url, params=params, headers=headers, timeout=20)
    if res.status_code == 200:
        return res.json().get('item', {})
    return None


def parse_faq_item(item_data: dict) -> dict:
    """FAQ 상세 데이터를 저장 형식으로 변환하는 함수."""
    content_html = item_data.get('hpFaqContent', '')
    clean_text = BeautifulSoup(str(content_html), "lxml").get_text(separator=" ").strip()
    return {
        "ID": item_data.get('faqSeq'),
        "제목": item_data.get('title'),
        "본문": clean_text,
        "분류": [m.get('label') for m in item_data.get('menuId', [])],
        "조회수": item_data.get('viewCnt')
    }


def start_exhaustive_search():
    """정렬 순서를 바꿔가며 전체 FAQ를 순회하고 스마트폰 항목만 수집하는 메인 함수."""
    session = requests.Session()
    dataset = []
    processed_ids = set()
    total_checked = 0

    # 여러 정렬 기준으로 최대한 많은 항목을 수집
    sort_orders = ["DATA_DESC", "VIEW_DESC", "DATA_ASC", "VIEW_ASC"]

    print(f"--- [전수 조사 시작] 총 {TOTAL_SAMSUNG_FAQ:,}건 대상 ---")

    for order in sort_orders:
        page = 1

        while total_checked < TOTAL_SAMSUNG_FAQ:
            list_url = "https://api.samsungsvc.co.kr/svc/hpFaqApi/faqServiceList"
            params = {
                "category": "10410",
                "product": "10414",
                "page": page,
                "rows": 10,
                "order": order,
                "_siteId": "ssvc"
            }

            try:
                res = session.get(list_url, params=params, headers=headers, timeout=20)
                items = res.json().get('items', [])

                # 더 이상 항목이 없으면 다음 정렬로 전환
                if not items:
                    break

                new_items_on_page = 0

                for item in items:
                    faq_seq = item.get('faqSeq')
                    if not faq_seq or faq_seq in processed_ids:
                        continue

                    new_items_on_page += 1
                    processed_ids.add(faq_seq)
                    total_checked += 1

                    item_data = fetch_faq_detail(session, faq_seq)
                    if item_data and is_smartphone(item_data):
                        dataset.append(parse_faq_item(item_data))

                    progress = (total_checked / TOTAL_SAMSUNG_FAQ) * 100
                    print(
                        f"  진행: {total_checked}/{TOTAL_SAMSUNG_FAQ} ({progress:.1f}%)"
                        f" | 수집된 스마트폰 FAQ: {len(dataset)}건",
                        end='\r'
                    )

                    time.sleep(random.uniform(0.05, 0.15))

                # 페이지 전체가 중복이면 무한 루프 방지를 위해 탈출
                if new_items_on_page == 0:
                    print(f"\n[알림] '{order}' 정렬에서 새로운 데이터 없음 → 다음 정렬로 전환")
                    break

                page += 1

            except Exception as e:
                print(f"\n[오류 발생] {e} → 5초 후 재시도")
                time.sleep(5)
                continue

        if total_checked >= TOTAL_SAMSUNG_FAQ:
            break

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print(f"\n\n[완료] 수집 가능한 모든 항목 조사 완료")
    print(f"최종 수집된 스마트폰 FAQ: {len(dataset)}건")

    if dataset:
        output_file = f"삼성_스마트폰_FAQ_{len(dataset)}건.xlsx"
        pd.DataFrame(dataset).to_excel(output_file, index=False)
        print(f"파일 저장 완료: '{output_file}'")


if __name__ == "__main__":
    start_exhaustive_search()
