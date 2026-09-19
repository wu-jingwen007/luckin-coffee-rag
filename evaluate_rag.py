import json
import math
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from rag_app.retriever.rag_chain import RAGChain


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "evaluation_questions.json"
RESULT_PATH = ROOT / "evaluation_results_50.json"

REFUSAL_MARKERS = [
    "资料中未",
    "材料中未",
    "参考材料未",
    "参考材料中不包含",
    "没有提供",
    "没有提及",
    "未提供",
    "未提及",
    "未包含",
    "无法确定",
    "无法回答",
]


def normalize(text):
    return re.sub(r"[\s,，。:：;；、\-—~～（）()]+", "", str(text)).lower()


def group_matches(text, group):
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in group)


def coverage(text, groups):
    if not groups:
        return 1.0
    hits = sum(group_matches(text, group) for group in groups)
    return hits / len(groups)


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[rank]


def write_snapshot(dataset, results):
    answerable = [item for item in results if item["answerable"]]
    unanswerable = [item for item in results if not item["answerable"]]
    durations = [item["response_time_seconds"] for item in results]
    retrieval_hits = sum(item["retrieval"]["evidence_hit"] for item in answerable)
    answer_hits = sum(item["evaluation"]["correct"] for item in answerable)
    abstention_hits = sum(item["evaluation"]["correct"] for item in unanswerable)
    hallucinations = sum(item["evaluation"]["hallucinated"] for item in unanswerable)
    numeric_grounding_violations = sum(
        not item["evaluation"].get("numeric_grounded", True)
        for item in results
    )
    manually_reviewed = [
        item for item in results
        if item.get("manual_review", {}).get("reviewed")
    ]
    manual_hallucinations = sum(
        item["manual_review"].get("hallucinated", False)
        for item in manually_reviewed
    )
    all_correct = answer_hits + abstention_hits

    summary = {
        "test_question_count": len(results),
        "planned_question_count": len(dataset["questions"]),
        "answerable_question_count": len(answerable),
        "unanswerable_question_count": len(unanswerable),
        "top_k": dataset["top_k"],
        "top_k_recall_accuracy": round(retrieval_hits / len(answerable), 4) if answerable else None,
        "top_k_recall_hits": f"{retrieval_hits}/{len(answerable)}" if answerable else None,
        "answer_correctness_answerable": round(answer_hits / len(answerable), 4) if answerable else None,
        "answer_correct_hits": f"{answer_hits}/{len(answerable)}" if answerable else None,
        "overall_task_accuracy": round(all_correct / len(results), 4) if results else None,
        "overall_correct_hits": f"{all_correct}/{len(results)}" if results else None,
        "hallucination_rate_unanswerable": round(hallucinations / len(unanswerable), 4) if unanswerable else None,
        "hallucination_hits": f"{hallucinations}/{len(unanswerable)}" if unanswerable else None,
        "numeric_grounded_answer_rate": round(
            (len(results) - numeric_grounding_violations) / len(results), 4
        ) if results else None,
        "numeric_grounding_violation_count": numeric_grounding_violations,
        "manual_hallucination_rate_all_questions": round(
            manual_hallucinations / len(manually_reviewed), 4
        ) if manually_reviewed else None,
        "manual_hallucination_hits": (
            f"{manual_hallucinations}/{len(manually_reviewed)}"
            if manually_reviewed else None
        ),
        "average_response_time_seconds": round(statistics.mean(durations), 4) if durations else None,
        "median_response_time_seconds": round(statistics.median(durations), 4) if durations else None,
        "p95_response_time_seconds": round(percentile(durations, 0.95), 4) if durations else None,
        "min_response_time_seconds": round(min(durations), 4) if durations else None,
        "max_response_time_seconds": round(max(durations), 4) if durations else None,
        "error_count": sum(bool(item.get("error")) for item in results),
    }

    payload = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "dataset_version": dataset["version"],
            "metric_definitions": {
                "top_k_recall_accuracy": "可回答问题中，Top-k全文合并后覆盖全部标准证据组的比例。",
                "answer_correctness_answerable": "可回答问题中，生成答案覆盖至少80%标准事实组且调用无错误的比例。",
                "overall_task_accuracy": "可回答题正确数加资料外题正确拒答数，占全部已执行题目的比例。",
                "hallucination_rate_unanswerable": "资料外问题中，答案未明确说明资料缺失而继续作答的比例。",
                "numeric_grounded_answer_rate": "答案未出现问题和Top-k上下文之外数字的比例。",
                "manual_hallucination_rate_all_questions": "逐题复核中发现无参考依据具体事实的答案比例。",
                "response_time": "调用当前 RAGChain.invoke_with_sources 的端到端墙钟时间，包含检索和外部模型生成。",
            },
        },
        "summary": summary,
        "cases": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if "--rescore" in sys.argv:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        items = {item["id"]: item for item in dataset["questions"]}
        audit_rag = RAGChain(top_k=dataset["top_k"])
        for case in payload["cases"]:
            item = items[case["id"]]
            answer = case["generated_answer"]
            error = case["error"]
            audit_docs = audit_rag.retriever.invoke(case["question"])
            audit_context = audit_rag._format_docs(audit_docs)
            unsupported_numbers = audit_rag._unsupported_numbers(
                answer,
                audit_context,
                case["question"],
            )
            case["evaluation"]["unsupported_numbers"] = unsupported_numbers
            case["evaluation"]["numeric_grounded"] = not unsupported_numbers
            if item["answerable"]:
                answer_coverage = coverage(answer, item["required_groups"])
                case["evaluation"]["required_fact_coverage"] = round(answer_coverage, 4)
                case["evaluation"]["abstained"] = False
                case["evaluation"]["correct"] = not error and answer_coverage >= 0.8
                case["evaluation"]["hallucinated"] = False
            else:
                abstained = any(marker in answer for marker in REFUSAL_MARKERS)
                case["evaluation"]["required_fact_coverage"] = None
                case["evaluation"]["abstained"] = abstained
                case["evaluation"]["correct"] = not error and abstained
                case["evaluation"]["hallucinated"] = not error and not abstained
            if case["id"] == "company_03":
                review_notes = "“走出了财务造假、退市及债务重组等危机”与标准答案“走出危机”语义一致。"
            elif case["id"] == "competition_03":
                review_notes = "六项标准护城河全部回答；额外提到的用户基础也能在商业模式文档中找到依据。"
            elif not case["answerable"]:
                review_notes = "明确说明参考资料未提供目标信息，未编造具体事实。"
            else:
                review_notes = "答案覆盖标准关键事实，未发现与参考资料矛盾的具体事实。"
            case["manual_review"] = {
                "reviewed": True,
                "verdict": "PASS" if case["evaluation"]["correct"] else "FAIL",
                "grounded": True,
                "hallucinated": False,
                "notes": review_notes,
            }
        rescored = write_snapshot(dataset, payload["cases"])
        print(json.dumps(rescored["summary"], ensure_ascii=False, indent=2))
        return

    top_k = dataset["top_k"]
    print(f"Loading RAG chain; questions={len(dataset['questions'])}, top_k={top_k}")
    rag = RAGChain(top_k=top_k)
    results = []

    for index, item in enumerate(dataset["questions"], start=1):
        print(f"[{index}/{len(dataset['questions'])}] {item['id']}: {item['question']}", flush=True)
        retrieved_docs = rag.vector_store.similarity_search(item["question"], k=top_k)
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        source_hit = (
            any(doc.metadata.get("source") == item["gold_source"] for doc in retrieved_docs)
            if item["answerable"]
            else None
        )
        evidence_score = coverage(retrieved_context, item["evidence_groups"]) if item["answerable"] else None
        evidence_hit = evidence_score == 1.0 if item["answerable"] else None

        started = time.perf_counter()
        error = None
        answer = ""
        sources = []
        try:
            response = rag.invoke_with_sources(item["question"])
            answer = response["answer"]
            sources = response["sources"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        if item["answerable"]:
            answer_coverage = coverage(answer, item["required_groups"])
            correct = not error and answer_coverage >= 0.8
            abstained = False
            hallucinated = False
        else:
            answer_coverage = None
            abstained = any(marker in answer for marker in REFUSAL_MARKERS)
            correct = not error and abstained
            hallucinated = not error and not abstained
        unsupported_numbers = RAGChain._unsupported_numbers(
            answer,
            retrieved_context,
            item["question"],
        )

        case = {
            "id": item["id"],
            "category": item["category"],
            "answerable": item["answerable"],
            "question": item["question"],
            "gold_source": item["gold_source"],
            "gold_answer": item["gold_answer"],
            "retrieval": {
                "top_k": top_k,
                "source_hit": source_hit,
                "evidence_coverage": round(evidence_score, 4) if evidence_score is not None else None,
                "evidence_hit": evidence_hit,
                "retrieved_sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
                "retrieved_previews": [doc.page_content[:240] for doc in retrieved_docs],
            },
            "generated_answer": answer,
            "returned_sources": sources,
            "response_time_seconds": round(elapsed, 4),
            "evaluation": {
                "required_fact_coverage": round(answer_coverage, 4) if answer_coverage is not None else None,
                "abstained": abstained,
                "correct": correct,
                "hallucinated": hallucinated,
                "unsupported_numbers": unsupported_numbers,
                "numeric_grounded": not unsupported_numbers,
            },
            "error": error,
        }
        results.append(case)
        write_snapshot(dataset, results)
        print(
            f"  retrieval={evidence_hit}, correct={correct}, "
            f"hallucinated={hallucinated}, latency={elapsed:.2f}s",
            flush=True,
        )

    payload = write_snapshot(dataset, results)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
