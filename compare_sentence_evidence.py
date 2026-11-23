"""Compare three evidence policies on frozen, film-disjoint BM25 inputs."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from compare_evidence_gate import measure
from diagnose_evidence import read
from evidence_gate import answer_with_evidence_gate
from evaluate_generation import write_report
from review_rag import ReviewIndex, answer_with_ollama
from sentence_evidence import answer_with_sentence_evidence


METHODS = {
    "direct": answer_with_ollama,
    "gate": answer_with_evidence_gate,
    "sentence_ids": answer_with_sentence_evidence,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args()
    url = args.ollama_url.rstrip("/")
    with urlopen(url + "/api/version", timeout=5) as response:
        version = json.load(response)["version"]
    with urlopen(url + "/api/tags", timeout=5) as response:
        model = next(item for item in json.load(response)["models"] if item["name"] == args.model)
    index = ReviewIndex.from_jsonl(args.corpus)
    report = {"ollama_version": version, "model": args.model, "digest": model["digest"], "cases": []}
    methods = list(METHODS)
    for n, row in enumerate(read(args.questions)):
        hits = index.search(row["movie_id"], row["question"])
        if [hit["chunk_id"] for hit in hits] != row["candidate_chunk_ids"]:
            raise ValueError(f"Fixed candidates differ for {row['qid']}.")
        order = n % len(methods)
        row["results"] = {name: measure(METHODS[name], {"question": row["question"], "candidates": hits},
                           args.model, url) for name in methods[order:] + methods[:order]}
        report["cases"].append(row)
        write_report(args.out, report)
        print(row["qid"], {name: value["status"] for name, value in row["results"].items()}, flush=True)
    return int(any(result["status"] == "error" for case in report["cases"] for result in case["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
