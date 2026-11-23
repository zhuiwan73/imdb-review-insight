"""Compare keyword, vector, and hybrid retrieval on the same labeled questions."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

from evaluate import metric_mean, read_cases
from review_rag import ReviewIndex


PROJECT_DIR = Path(__file__).resolve().parent


def measure_search(index, movie_id, question, top_k, expected):
    started = perf_counter()
    hits = index.search(movie_id, question, top_k)
    elapsed_ms = (perf_counter() - started) * 1000

    review_ids = [hit["review_id"] for hit in hits]
    if len(review_ids) > top_k or len(set(review_ids)) != len(review_ids):
        raise ValueError("Expected at most top_k distinct reviews per search")
    matched = [review_id for review_id in review_ids if review_id in expected]
    first_rank = next(
        (rank for rank, review_id in enumerate(review_ids, 1) if review_id in expected), None
    )
    return {
        "retrieved_review_ids": review_ids,
        "retrieved_chunk_ids": [hit["chunk_id"] for hit in hits],
        "matched_review_ids": matched,
        "first_relevant_rank": first_rank,
        "recall": len(matched) / len(expected) if expected else None,
        "reciprocal_rank": (1 / first_rank if first_rank else 0.0) if expected else None,
        "latency_ms": round(elapsed_ms, 4),
    }


def summarize_results(results, top_k):
    scored = [result for result in results if result["recall"] is not None]
    latencies = [result["latency_ms"] for result in results]
    return {
        f"recall_at_{top_k}": metric_mean([result["recall"] for result in scored]),
        f"mrr_at_{top_k}": metric_mean([result["reciprocal_rank"] for result in scored]),
        "questions_with_relevant_hit": sum(bool(result["matched_review_ids"]) for result in scored),
        "retrieval_p50_ms": round(median(latencies), 3),
        "retrieval_p95_ms": round(
            quantiles(latencies, n=100, method="inclusive")[94] if len(latencies) > 1 else latencies[0],
            3,
        ),
    }


def compare_indexes(indexes, cases, top_k):
    # Exercise each retrieval path before starting timing. No corpus build or LLM calls are timed.
    _, warm_movie, warm_question, _ = cases[0]
    for index in indexes.values():
        index.search(warm_movie, warm_question, top_k)

    case_reports = []
    for case_number, (qid, movie_id, question, expected) in enumerate(cases):
        # Rotate method order to limit warm-start and CPU-load bias across the three methods.
        names = list(indexes)
        offset = case_number % len(names)
        results = {
            name: measure_search(indexes[name], movie_id, question, top_k, expected)
            for name in names[offset:] + names[:offset]
        }
        winners = []
        if expected:
            best = max((result["recall"], result["reciprocal_rank"]) for result in results.values())
            winners = [
                name for name in indexes
                if (results[name]["recall"], results[name]["reciprocal_rank"]) == best
            ]
        case_reports.append(
            {
                "qid": qid,
                "movie_id": movie_id,
                "question": question,
                "relevant_review_ids": sorted(expected),
                "best_methods": winners,
                "results": {name: results[name] for name in indexes},
            }
        )

    summaries = {}
    for name in indexes:
        results = [case["results"][name] for case in case_reports]
        summary = summarize_results(results, top_k)
        if name != "bm25":
            paired = [
                (case["results"][name]["recall"], case["results"]["bm25"]["recall"])
                for case in case_reports if case["relevant_review_ids"]
            ]
            summary["recall_vs_bm25"] = {
                "improved": sum(value > baseline for value, baseline in paired),
                "unchanged": sum(value == baseline for value, baseline in paired),
                "worsened": sum(value < baseline for value, baseline in paired),
            }
        summaries[name] = summary
    return summaries, case_reports


def compare(corpus_path, questions_path, model_dir, top_k):
    from embedding import MODEL_ID, MODEL_REVISION, MiniLMEncoder
    from retrieval import DenseIndex, HybridIndex

    cases = read_cases(corpus_path, questions_path)
    started = perf_counter()
    bm25 = ReviewIndex.from_jsonl(corpus_path)
    bm25_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    encoder = MiniLMEncoder(model_dir)
    model_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    dense = DenseIndex(bm25, encoder)
    corpus_encoding_ms = (perf_counter() - started) * 1000

    indexes = {"bm25": bm25, "dense": dense, "hybrid": HybridIndex(bm25, dense)}
    summaries, case_reports = compare_indexes(indexes, cases, top_k)
    answerable = sum(bool(expected) for _, _, _, expected in cases)
    movies = bm25.movies()
    return {
        "scope": (
            "Distinct review IDs, same movie and source chunks, same top-k. Recall and MRR only "
            "use questions with judged relevant evidence. Rankings do not assess answer quality or abstention."
        ),
        "latency_scope": (
            "Warmed single-query retrieval includes query encoding and ranking. "
            "Corpus preparation and generation are excluded. Timings are local to this run."
        ),
        "top_k": top_k,
        "corpus": corpus_path.name,
        "questions": questions_path.name,
        "counts": {
            "movies": len(movies),
            "reviews": sum(movie["review_count"] for movie in movies),
            "questions": len(cases),
            "answerable": answerable,
            "unsupported": len(cases) - answerable,
        },
        "environment": {"system": platform.system(), "python": platform.python_version(), "device": "CPU"},
        "embedding": {"model": MODEL_ID, "reference_revision": MODEL_REVISION},
        "preparation_ms": {
            "bm25_build": round(bm25_ms, 3),
            "encoder_load": round(model_ms, 3),
            "corpus_encoding_and_index": round(corpus_encoding_ms, 3),
        },
        "methods": summaries,
        "cases": case_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT_DIR / "data" / "imdb_local.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_DIR / "data" / "imdb_eval.jsonl")
    parser.add_argument("--model-dir", type=Path, help="Directory containing local MiniLM model files")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum distinct reviews retrieved per method")
    parser.add_argument("--out", type=Path, help="Save the full report as JSON")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.out and args.out.resolve() in {args.corpus.resolve(), args.questions.resolve()}:
        parser.error("--out must be different from the corpus and questions files")
    try:
        report = compare(args.corpus, args.questions, args.model_dir, args.top_k)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
