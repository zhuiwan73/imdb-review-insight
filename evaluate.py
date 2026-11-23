"""Measure review-level retrieval and answerability on labeled questions."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from review_rag import ReviewIndex, answer_extractive


PROJECT_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            location = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{location}: expected a JSON object")
            yield location, row


def required_text(row, key, location):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{location}: {key} must be nonempty without surrounding whitespace")
    return value


def read_cases(corpus_path, questions_path):
    reviews_by_movie = {}
    for location, review in read_jsonl(corpus_path):
        movie = required_text(review, "movie_id", location)
        review_id = required_text(review, "review_id", location)
        movie_reviews = reviews_by_movie.setdefault(movie, set())
        if review_id in movie_reviews:
            raise ValueError(f"{location}: duplicate review_id {review_id!r} for {movie!r}")
        movie_reviews.add(review_id)

    cases = []
    known_questions = set()
    for location, case in read_jsonl(questions_path):
        qid = required_text(case, "qid", location)
        movie_id = required_text(case, "movie_id", location)
        question = required_text(case, "question", location)
        if qid in known_questions:
            raise ValueError(f"{location}: duplicate qid {qid!r}")
        known_questions.add(qid)
        if movie_id not in reviews_by_movie:
            raise ValueError(f"{location}: unknown movie_id {movie_id!r}")
        expected = case.get("relevant_review_ids")
        if not isinstance(expected, list) or any(
            not isinstance(review_id, str) or not review_id.strip() for review_id in expected
        ):
            raise ValueError(f"{location}: relevant_review_ids must be a list of review IDs")
        if len(expected) != len(set(expected)):
            raise ValueError(f"{location}: duplicate relevance judgment")
        unknown = set(expected) - reviews_by_movie[movie_id]
        if unknown:
            raise ValueError(f"{location}: review IDs outside the selected movie: {sorted(unknown)}")
        cases.append((qid, movie_id, question, set(expected)))
    if not cases:
        raise ValueError(f"{questions_path}: no evaluation questions")
    return cases


def metric_mean(values):
    return round(fmean(values), 6) if values else None


def evaluate(corpus_path, question_path, top_k):
    cases = read_cases(corpus_path, question_path)
    index = ReviewIndex.from_jsonl(corpus_path)
    answerable_count = sum(bool(expected) for _, _, _, expected in cases)
    unsupported_count = len(cases) - answerable_count
    recall_scores = []
    reciprocal_ranks = []
    case_reports = []
    true_answers = 0
    true_abstentions = 0

    for qid, movie_id, question, expected in cases:
        hits = index.search(movie_id, question, top_k)
        retrieved = list(dict.fromkeys(hit["review_id"] for hit in hits))
        matched = expected.intersection(retrieved)
        response = answer_extractive(question, hits)
        status = response["status"]
        if expected:
            recall_scores.append(len(matched) / len(expected))
            first_rank = next(
                (rank for rank, review_id in enumerate(retrieved, 1) if review_id in expected),
                None,
            )
            reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
            true_answers += status == "answered"
        else:
            true_abstentions += status == "insufficient_evidence"
        case_reports.append(
            {
                "qid": qid,
                "movie_id": movie_id,
                "question": question,
                "relevant_review_ids": sorted(expected),
                "retrieved_review_ids": retrieved,
                "matched_review_ids": sorted(matched),
                "cited_review_ids": list(
                    dict.fromkeys(citation["review_id"] for citation in response["citations"])
                ),
                "status": status,
            }
        )

    return {
        "retriever": "bm25",
        "answer_mode": "extractive",
        "top_k": top_k,
        "corpus": str(corpus_path),
        "questions": str(question_path),
        "scope": (
            "Top-k results are scored by distinct review ID. Retrieval averages "
            "use answerable questions only. Answerability metrics "
            "evaluate whether a response was given, not whether its claims were supported."
        ),
        "counts": {
            "questions": len(cases),
            "answerable": answerable_count,
            "unsupported": unsupported_count,
            "answered_answerable": true_answers,
            "abstained_unsupported": true_abstentions,
        },
        "metrics": {
            f"recall_at_{top_k}": metric_mean(recall_scores),
            f"mrr_at_{top_k}": metric_mean(reciprocal_ranks),
            "answerable_response_rate": (
                round(true_answers / answerable_count, 6) if answerable_count else None
            ),
            "unsupported_abstention_rate": (
                round(true_abstentions / unsupported_count, 6) if unsupported_count else None
            ),
        },
        "cases": case_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT_DIR / "data" / "demo_reviews.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_DIR / "data" / "demo_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=5, help="maximum number of review results")
    parser.add_argument("--out", type=Path, help="path for a report with per-question details")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        report = evaluate(args.corpus, args.questions, args.top_k)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
