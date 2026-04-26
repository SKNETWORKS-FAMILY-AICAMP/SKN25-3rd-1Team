import json
import os
import sys
import traceback
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm import tqdm

sys.path.append(os.getcwd())

from eval.evaluator import evaluate_sample
from src.graph import rag_app


# =========================
# 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Q_v1.json"
RESULT_DIR = BASE_DIR / "results"

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# 병렬 worker 수
MAX_WORKERS = 3


# =========================
# 데이터 로드
# =========================
def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 결과 분류
# =========================
def classify_result(row: dict) -> str:
    ar = row.get("answer_relevancy_score")
    faith = row.get("faithfulness_score")
    answer = (row.get("answer") or "").strip()

    safe_keywords = [
        "찾을 수 없습니다",
        "확인할 수 없습니다",
        "정확한 안내가 어렵습니다",
        "서비스센터",
        "점검을 권장",
        "방문하시기 바랍니다",
        "추가 점검이 필요합니다",
    ]

    is_safe = any(keyword in answer for keyword in safe_keywords)

    if faith is not None and faith < 0.7:
        return "위험 응답"
    if ar is not None and ar < 0.7 and is_safe:
        return "안전 응답"
    if ar is not None and ar < 0.7:
        return "관련성 낮음"
    return "양호"


# =========================
# 단일 질문 실행
# =========================
def process_one(item: dict, idx: int) -> dict:
    question = item["question"]
    category = item.get("category", "")
    selected_device = item.get("selected_device", "선택하지 않음")

    base_row = {
        "id": item.get("id", idx + 1),
        "category": category,
        "question": question,
        "selected_device": selected_device,
        "answer": "",
        "answer_relevancy_score": None,
        "answer_relevancy_reason": "",
        "faithfulness_score": None,
        "faithfulness_reason": "",
        "result_type": "",
        "status": "success",
        "error_message": "",
    }

    try:
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
            base_row["result_type"] = "평가 제외"
            base_row["status"] = "skipped"
            base_row["error_message"] = "eval_data 없음"
            return base_row

        scores = evaluate_sample(sample)

        base_row["answer"] = sample.get("answer", "")
        base_row["answer_relevancy_score"] = scores.get("answer_relevancy_score")
        base_row["answer_relevancy_reason"] = scores.get("answer_relevancy_reason", "")
        base_row["faithfulness_score"] = scores.get("faithfulness_score")
        base_row["faithfulness_reason"] = scores.get("faithfulness_reason", "")
        base_row["result_type"] = classify_result(base_row)

        return base_row

    except Exception as e:
        base_row["result_type"] = "실행 오류"
        base_row["status"] = "error"
        base_row["error_message"] = f"{type(e).__name__}: {str(e)}"
        return base_row


# =========================
# 그래프 저장
# =========================
def save_charts(df: pd.DataFrame, timestamp: str) -> list[Path]:
    chart_paths = []

    score_df = df[df["status"] == "success"].copy()

    # 1) 평균 점수 그래프
    avg_ar = score_df["answer_relevancy_score"].dropna().mean()
    avg_faith = score_df["faithfulness_score"].dropna().mean()

    avg_chart_path = RESULT_DIR / f"{timestamp}_chart_avg_scores.png"
    plt.figure(figsize=(8, 5))
    labels = ["Answer Relevancy", "Faithfulness"]
    values = [
        0 if pd.isna(avg_ar) else avg_ar,
        0 if pd.isna(avg_faith) else avg_faith,
    ]
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.title("Average Scores")
    plt.ylabel("Score")

    for bar, value in zip(bars, values):
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

    # 2) 결과 분포 그래프
    type_chart_path = RESULT_DIR / f"{timestamp}_chart_result_type.png"
    counts = df["result_type"].value_counts()

    plt.figure(figsize=(9, 5))
    bars = plt.bar(counts.index.astype(str), counts.values)
    plt.title("Result Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=20)

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            f"{value}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(type_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    chart_paths.append(type_chart_path)

    # 3) 카테고리별 평균 relevancy 그래프
    category_chart_path = RESULT_DIR / f"{timestamp}_chart_category_scores.png"
    category_df = (
        score_df.groupby("category", dropna=False)["answer_relevancy_score"]
        .mean()
        .reset_index()
        .sort_values(by="answer_relevancy_score", ascending=False)
    )

    if not category_df.empty:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(category_df["category"].astype(str), category_df["answer_relevancy_score"])
        plt.ylim(0, 1.05)
        plt.title("Category Scores")
        plt.ylabel("Average Answer Relevancy")
        plt.xticks(rotation=30, ha="right")

        for bar, value in zip(bars, category_df["answer_relevancy_score"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(category_chart_path, dpi=200, bbox_inches="tight")
        plt.close()
        chart_paths.append(category_chart_path)

    return chart_paths


# =========================
# 결과 저장
# =========================
def save_outputs(df: pd.DataFrame, timestamp: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv_path = RESULT_DIR / f"{timestamp}_deepeval_results_raw.csv"
    xlsx_path = RESULT_DIR / f"{timestamp}_deepeval_report.xlsx"
    summary_csv_path = RESULT_DIR / f"{timestamp}_deepeval_summary.csv"

    for col in ["answer_relevancy_score", "faithfulness_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    summary_df = pd.DataFrame([
        {"metric": "전체 질문 수", "value": len(df)},
        {"metric": "성공 수", "value": int((df["status"] == "success").sum())},
        {"metric": "평가 제외 수", "value": int((df["status"] == "skipped").sum())},
        {"metric": "오류 수", "value": int((df["status"] == "error").sum())},
        {
            "metric": "평균 Answer Relevancy",
            "value": round(df["answer_relevancy_score"].dropna().mean(), 3)
            if df["answer_relevancy_score"].notna().any() else None
        },
        {
            "metric": "평균 Faithfulness",
            "value": round(df["faithfulness_score"].dropna().mean(), 3)
            if df["faithfulness_score"].notna().any() else None
        },
        {"metric": "양호 수", "value": int((df["result_type"] == "양호").sum())},
        {"metric": "안전 응답 수", "value": int((df["result_type"] == "안전 응답").sum())},
        {"metric": "관련성 낮음 수", "value": int((df["result_type"] == "관련성 낮음").sum())},
        {"metric": "위험 응답 수", "value": int((df["result_type"] == "위험 응답").sum())},
    ])

    category_summary_df = (
        df[df["status"] == "success"]
        .groupby("category", dropna=False)
        .agg(
            question_count=("id", "count"),
            avg_answer_relevancy=("answer_relevancy_score", "mean"),
            avg_faithfulness=("faithfulness_score", "mean"),
        )
        .reset_index()
    )

    if not category_summary_df.empty:
        category_summary_df["avg_answer_relevancy"] = category_summary_df["avg_answer_relevancy"].round(3)
        category_summary_df["avg_faithfulness"] = category_summary_df["avg_faithfulness"].round(3)

    df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        category_summary_df.to_excel(writer, sheet_name="category_summary", index=False)

    chart_paths = save_charts(df, timestamp)

    print("\n저장 완료")
    print(raw_csv_path)
    print(summary_csv_path)
    print(xlsx_path)
    for chart_path in chart_paths:
        print(chart_path)


# =========================
# 실행
# =========================
def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions(DATASET_PATH)
    questions = random.sample(questions, 30)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"질문 개수: {len(questions)}")

    results = []

    for idx, item in enumerate(tqdm(questions, desc="Evaluating")):
        try:
            result = process_one(item, idx)
            results.append(result)
        except Exception as e:
            results.append({
                "id": item.get("id", idx + 1),
                "category": item.get("category", ""),
                "question": item.get("question", ""),
                "selected_device": item.get("selected_device", "선택하지 않음"),
                "answer": "",
                "answer_relevancy_score": None,
                "answer_relevancy_reason": "",
                "faithfulness_score": None,
                "faithfulness_reason": "",
                "result_type": "실행 오류",
                "status": "error",
                "error_message": f"{type(e).__name__}: {str(e)}",
            })

    df = pd.DataFrame(results)

    preview_cols = [
        "id",
        "category",
        "question",
        "answer_relevancy_score",
        "faithfulness_score",
        "result_type",
        "status",
    ]
    print("\n===== 평가 결과 미리보기 =====")
    print(df[preview_cols].to_string(index=False))

    save_outputs(df, timestamp)


if __name__ == "__main__":
    run()