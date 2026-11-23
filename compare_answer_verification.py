"""Paired comparison using exactly the same sentence selection for both arms."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from compare_evidence_gate import measure
from diagnose_evidence import read
from evaluate_generation import write_report
from review_rag import ReviewIndex
from sentence_evidence import answer_with_sentence_evidence
from verify_sentence_answer import verify_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args()
    if args.out.resolve() in {args.corpus.resolve(), args.questions.resolve()}:
        parser.error("Output must differ from input files.")
    url = args.ollama_url.rstrip("/")
    with urlopen(url + "/api/version", timeout=5) as response:
        version = json.load(response)["version"]
    with urlopen(url + "/api/tags", timeout=5) as response:
        models = json.load(response)["models"]
    model = next(item for item in models if item["name"] == args.model)
    index = ReviewIndex.from_jsonl(args.corpus)
    report = {"ollama_version": version, "model": args.model, "digest": model["digest"],
              "questions_sha256": sha256(args.questions.read_bytes()).hexdigest(),
              "corpus_sha256": sha256(args.corpus.read_bytes()).hexdigest(),
              "verifier_sha256": sha256(Path(__file__).with_name("verify_sentence_answer.py").read_bytes()).hexdigest(),
              "cases": []}
    for row in read(args.questions):
        hits = index.search(row["movie_id"], row["question"])
        if [hit["chunk_id"] for hit in hits] != row["candidate_chunk_ids"]:
            raise ValueError(f"Fixed candidates differ for {row['qid']}.")
        case = {"question": row["question"], "candidates": hits}
        selected = measure(answer_with_sentence_evidence, case, args.model, url)
        if selected["status"] == "error":
            verified = {"status": "error", "reason": "selection_failed"}
        else:
            cited = set(selected["citation_ids"])
            captured = {"status": selected["status"], "findings": selected["findings"],
                        "citations": [hit for hit in hits if hit["chunk_id"] in cited],
                        "generation": {"abstention_reason": selected["reason"]}, "gate": None}

            def verify(model, question, _hits, *, base_url):
                return verify_selected(model, question, captured, base_url=base_url)

            verified = measure(verify, case, args.model, url)
        row["results"] = {"selected": selected, "verified": verified}
        report["cases"].append(row)
        write_report(args.out, report)
        print(row["qid"], selected["status"], verified["status"], verified.get("reason"), flush=True)
    return int(any(x["status"] == "error" for row in report["cases"] for x in row["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
