"""Compare retrieval coverage and oracle evidence on the frozen IMDb questions."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

from evidence_gate import answer_with_evidence_gate
from evaluate_generation import write_report
from review_rag import ReviewIndex, answer_with_ollama


# Verified passages for the two legacy questions with no labeled review in Top 5.
LATE_PASSAGES = {"imdb-03": ["test-pos-9833:2"],
                 "imdb-45": ["test-neg-417:3", "test-pos-1627:1"]}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def ranks(hits, key, evidence):
    return [n for n, hit in enumerate(hits, 1) if hit[key] in evidence]


def audit(data, stem, questions, key, expected):
    index = ReviewIndex.from_jsonl(data / f"{stem}.jsonl")
    records = []
    for q in read(data / questions):
        evidence = q[key]
        if not evidence:
            continue
        qid = q["qid"]
        movie, question = q["movie_id"], q["question"]
        top20 = index.search(movie, question, 20)
        raw20 = index.search(movie, question, 20, distinct_reviews=False)
        top5 = top20[:5]
        if [hit["chunk_id"] for hit in top5] != expected[qid]:
            raise ValueError(f"Frozen Top 5 mismatch for {qid}")
        field = "chunk_id" if key == "supporting_chunk_ids" else "review_id"
        review_ranks = ranks(top20, field, evidence)
        row = {
            "qid": qid, "movie_id": movie, "question": question, "reference_ids": evidence,
            "distinct_ranks": review_ranks, "raw_passage_ranks": ranks(raw20, field, evidence),
            "location": ("top5" if any(n <= 5 for n in review_ranks) else
                         "rank6_to_20" if review_ranks else "outside_top20"),
            "top5_ids": [hit["chunk_id"] for hit in top5],
        }
        records.append((row, top5))
    return records


def run_model(url, model, question, hits, gated):
    entry = answer_with_evidence_gate if gated else answer_with_ollama
    start = perf_counter()
    result = entry(model, question, hits, base_url=url)
    return {
        "status": result["status"], "reason": (result.get("generation") or {}).get("abstention_reason"),
        "findings": result.get("findings", []), "gate": result.get("gate"),
        "latency_ms": round((perf_counter() - start) * 1000, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    data = args.data
    old = json.loads((data / "imdb_gate_frozen.json").read_text(encoding="utf-8"))
    new = json.loads((data / "imdb_gate_holdout.json").read_text(encoding="utf-8"))
    frozen = audit(data, "imdb_local", "imdb_eval.jsonl", "relevant_review_ids",
                   {row["qid"]: row["candidate_chunk_ids"] for row in old["cases"]})
    exact = audit(data, "imdb_evidence_local", "imdb_evidence_holdout.jsonl", "supporting_chunk_ids",
                  {row["qid"]: row["candidate_chunk_ids"] for row in new["cases"]})
    previous = {row["qid"]: row["methods"]["gated"]["status"] for row in new["cases"]}
    old_index = ReviewIndex.from_jsonl(data / "imdb_local.jsonl")
    late = []
    for row, top5 in frozen:
        ids = LATE_PASSAGES.get(row["qid"])
        if not ids:
            continue
        top20 = old_index.search(row["movie_id"], row["question"], 20)
        gold = [hit for hit in top20 if hit["chunk_id"] in ids]
        if len(gold) != len(ids):
            raise ValueError(f"Verified late passages missing for {row['qid']}")
        row["verified_late_passage_ranks"] = dict(
            (hit["chunk_id"], n) for n, hit in enumerate(top20, 1) if hit["chunk_id"] in ids
        )
        late.append((row, top5, gold))
    report = {"review_level": [row for row, _ in frozen], "passage_level": [row for row, _ in exact]}
    write_report(args.out, report)
    if args.retrieval_only:
        return
    with urlopen(args.ollama_url + "/api/version", timeout=5) as response:
        report["ollama_version"] = json.load(response)["version"]
    with urlopen(args.ollama_url + "/api/tags", timeout=5) as response:
        tags = json.load(response)["models"]
    report["model"] = next(item for item in tags if item["name"] == args.model)
    failures = [(row, top5, [hit for hit in top5 if hit["chunk_id"] in row["reference_ids"]])
                for row, top5 in exact if previous[row["qid"]] == "insufficient_evidence"]
    for row, top5, gold in failures + late:
        row["conditions"] = {}
        for name, hits in (("top5", top5), ("oracle", gold)):
            for gated in (False, True):
                suffix = "gated" if gated else "direct"
                row["conditions"][f"{name}_{suffix}"] = run_model(
                    args.ollama_url, args.model, row["question"], hits, gated,
                )
        write_report(args.out, report)
        print(row["qid"], {key: value["status"] for key, value in row["conditions"].items()}, flush=True)


if __name__ == "__main__":
    main()
