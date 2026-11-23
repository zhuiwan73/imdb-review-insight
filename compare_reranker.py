"""Re-rank a fixed BM25 Top 20 with a pretrained cross-encoder."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

from diagnose_evidence import LATE_PASSAGES, read
from evaluate_generation import write_report
from review_rag import ReviewIndex


MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"


def metrics(cases, key):
    recalls, reciprocal = [], []
    hits = 0
    for case in cases:
        relevant = set(case["reference_ids"])
        ranks = [n for n, value in enumerate(case[key][:5], 1) if value in relevant]
        recalls.append(len(ranks) / len(relevant))
        reciprocal.append(1 / ranks[0] if ranks else 0)
        hits += bool(ranks)
    return {"questions": len(cases), "hit_at_5": hits,
            "recall_at_5": sum(recalls) / len(cases), "mrr_at_5": sum(reciprocal) / len(cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, help="Optional project-local Transformers dependencies")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with (args.model_dir / "model.safetensors").open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != SHA256:
            parser.error("The pretrained weight hash does not match.")
    if args.runtime:
        sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True,
    ).eval()
    report = {"model": MODEL, "weights_sha256": SHA256, "candidate_pool": "BM25 best chunk per review, up to 20"}
    splits = (("review_level", "imdb_local", "imdb_eval", "relevant_review_ids", "review_id"),
              ("passage_level", "imdb_evidence_local", "imdb_evidence_holdout", "supporting_chunk_ids", "chunk_id"))
    for name, corpus, questions, gold_key, id_key in splits:
        index = ReviewIndex.from_jsonl(args.data / f"{corpus}.jsonl")
        cases = []
        for q in read(args.data / f"{questions}.jsonl"):
            if not q[gold_key]:
                continue
            hits = index.search(q["movie_id"], q["question"], top_k=20)
            if hits:
                tokens = tokenizer([q["question"]] * len(hits), [hit["text"] for hit in hits],
                                   padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.inference_mode():
                    scores = model(**tokens).logits.flatten().tolist()
                ranking = [hit for _, hit in sorted(zip(scores, hits),
                           key=lambda pair: (-pair[0], pair[1]["chunk_id"]))]
            else:
                ranking = []
            cases.append({
                "qid": q["qid"], "reference_ids": q[gold_key],
                "bm25": [hit[id_key] for hit in hits], "reranked": [hit[id_key] for hit in ranking],
                "verified_late_ids_in_reranked_top5": [
                    hit["chunk_id"] for hit in ranking[:5] if hit["chunk_id"] in LATE_PASSAGES.get(q["qid"], ())
                ],
            })
        report[name] = {"bm25": metrics(cases, "bm25"), "reranked": metrics(cases, "reranked"), "cases": cases}
    write_report(args.out, report)
    print(json.dumps({k: {n: x for n, x in v.items() if n != "cases"}
                      for k, v in report.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
