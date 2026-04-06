import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

sys.path.append(os.getcwd())

from eval.evaluator import evaluate_sample
from src.graph import rag_app


# =========================
# 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Q_v1.json"
RESULT_DIR = BASE_DIR / "results"

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


# =========================
# 데이터 로딩
# =========================
def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 단건 평가
# =========================
def evaluate_question(item: dict, idx: int) -> dict:
    question = item["question"]
    selected_device = item.get("selected_device", "선택하지 않음")
    category = item.get("category", "")

    result = rag_app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "selected_device": selected_device,
        },
        config={
            "configurable": {
                "thread_id": f"eval-{idx}"
            }
        }
    )

    sample = result.get("eval_data")

    if not sample:
        return {
            "id": item.get("id", idx + 1),
            "category": category,
            "question": question,
            "selected_device": selected_device,
            "answer": "",
            "answer_relevancy_score": None,
            "answer_relevancy_reason": "eval_data 없음",
            "faithfulness_score": None,
            "faithfulness_reason": "eval_data 없음",
            "result_type": "평가 제외",
        }

    scores = evaluate_sample(sample)

    return {
        "id": item.get("id", idx + 1),
        "category": category,
        "question": question,
        "selected_device": selected_device,
        "answer": sample.get("answer", ""),
        "answer_relevancy_score": scores.get("answer_relevancy_score"),
        "answer_relevancy_reason": scores.get("answer_relevancy_reason"),
        "faithfulness_score": scores.get("faithfulness_score"),
        "faithfulness_reason": scores.get("faithfulness_reason"),
        "result_type": "평가 완료",
    }


# =========================
# 결과 저장
# =========================
def save_results(df: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "deepeval_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")


# =========================
# 실행
# =========================
def run() -> None:
    questions = load_questions(DATASET_PATH)
    print(f"질문 개수: {len(questions)}")

    results = [evaluate_question(item, idx) for idx, item in enumerate(questions)]

    df = pd.DataFrame(results)
    print("\n===== 평가 결과 =====")
    print(df[["id", "category", "question", "answer_relevancy_score", "faithfulness_score", "result_type"]])

    save_results(df)


if __name__ == "__main__":
    run()