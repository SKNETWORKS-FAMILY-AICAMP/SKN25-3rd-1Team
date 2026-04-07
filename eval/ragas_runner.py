import json
import os
import sys
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm import tqdm
from datasets import Dataset

sys.path.append(os.getcwd())

from src.graph import rag_app
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall


# =========================
# 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
QUESTION_PATH = BASE_DIR / "dataset" / "Q_v1.json"
REFERENCE_PATH = BASE_DIR / "dataset" / "ragas_reference_v1.json"
RESULT_DIR = BASE_DIR / "results"

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


# =========================
# 데이터 로드
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reference_map(reference_rows):
    return {row["id"]: row for row in reference_rows}


# =========================
# 그래프 저장
# =========================
def save_charts(score_df: pd.DataFrame, timestamp: str):
    chart_paths = []

    metric_cols = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
    avg_scores = score_df[metric_cols].mean(numeric_only=True)

    avg_chart_path = RESULT_DIR / f"{timestamp}_ragas_chart_avg_scores.png"
    plt.figure(figsize=(9, 5))
    bars = plt.bar(avg_scores.index, avg_scores.values)
    plt.ylim(0, 1.05)
    plt.title("RAGAS Average Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=20)

    for bar, value in zip(bars, avg_scores.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(avg_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    chart_paths.append(avg_chart_path)

    return chart_paths


# =========================
# 결과 저장
# =========================
def save_outputs(raw_df: pd.DataFrame, score_df: pd.DataFrame, timestamp: str):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv_path = RESULT_DIR / f"{timestamp}_ragas_results_raw.csv"
    score_csv_path = RESULT_DIR / f"{timestamp}_ragas_scores.csv"
    xlsx_path = RESULT_DIR / f"{timestamp}_ragas_report.xlsx"

    raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    score_df.to_csv(score_csv_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([
        {"metric": "전체 질문 수", "value": len(raw_df)},
        {"metric": "평가 성공 수", "value": int((raw_df["status"] == "success").sum())},
        {"metric": "평가 제외 수", "value": int((raw_df["status"] == "skipped").sum())},
        {"metric": "평균 answer_relevancy", "value": round(score_df["answer_relevancy"].mean(), 3) if not score_df.empty else None},
        {"metric": "평균 faithfulness", "value": round(score_df["faithfulness"].mean(), 3) if not score_df.empty else None},
        {"metric": "평균 context_precision", "value": round(score_df["context_precision"].mean(), 3) if not score_df.empty else None},
        {"metric": "평균 context_recall", "value": round(score_df["context_recall"].mean(), 3) if not score_df.empty else None},
    ])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_results", index=False)
        score_df.to_excel(writer, sheet_name="ragas_scores", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    chart_paths = save_charts(score_df, timestamp)

    print("\n저장 완료")
    print(raw_csv_path)
    print(score_csv_path)
    print(xlsx_path)
    for chart_path in chart_paths:
        print(chart_path)


# =========================
# 실행
# =========================
def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_json(QUESTION_PATH)
    references = load_json(REFERENCE_PATH)
    reference_map = build_reference_map(references)

    # 랜덤 샘플링
    # questions = random.sample(questions, 30)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"질문 개수: {len(questions)}")

    raw_rows = []

    for idx, item in enumerate(tqdm(questions, desc="RAGAS Evaluating")):
        question = item["question"]
        category = item.get("category", "")
        selected_device = item.get("selected_device", "선택하지 않음")
        qid = item.get("id", idx + 1)

        try:
            result = rag_app.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "selected_device": selected_device,
                },
                config={
                    "configurable": {
                        "thread_id": f"ragas-{idx}"
                    }
                }
            )

            sample = result.get("eval_data")
            reference = reference_map.get(qid, {})
            ground_truth = reference.get("ground_truth", "")

            if not sample or not ground_truth:
                raw_rows.append({
                    "id": qid,
                    "category": category,
                    "question": question,
                    "selected_device": selected_device,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": ground_truth,
                    "status": "skipped",
                    "error_message": "eval_data 또는 ground_truth 없음",
                })
                continue

            raw_rows.append({
                "id": qid,
                "category": category,
                "question": question,
                "selected_device": selected_device,
                "answer": sample.get("answer", ""),
                "contexts": sample.get("contexts", []),
                "ground_truth": ground_truth,
                "status": "success",
                "error_message": "",
            })

        except Exception as e:
            raw_rows.append({
                "id": qid,
                "category": category,
                "question": question,
                "selected_device": selected_device,
                "answer": "",
                "contexts": [],
                "ground_truth": "",
                "status": "error",
                "error_message": f"{type(e).__name__}: {str(e)}",
            })

    raw_df = pd.DataFrame(raw_rows)

    eval_rows = raw_df[raw_df["status"] == "success"].copy()

    if eval_rows.empty:
        print("평가 가능한 데이터가 없습니다.")
        save_outputs(raw_df, pd.DataFrame(), timestamp)
        return

    dataset = Dataset.from_dict({
        "question": eval_rows["question"].tolist(),
        "answer": eval_rows["answer"].tolist(),
        "contexts": eval_rows["contexts"].tolist(),
        "ground_truth": eval_rows["ground_truth"].tolist(),
    })

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
    )

    score_df = ragas_result.to_pandas()

    # 원본 question/id 붙여서 보기 좋게 정리
    score_df.insert(0, "id", eval_rows["id"].tolist())
    score_df.insert(1, "category", eval_rows["category"].tolist())
    score_df.insert(2, "question", eval_rows["question"].tolist())

    print("\n===== RAGAS 결과 미리보기 =====")
    print(score_df.to_string(index=False))

    save_outputs(raw_df, score_df, timestamp)


if __name__ == "__main__":
    run()