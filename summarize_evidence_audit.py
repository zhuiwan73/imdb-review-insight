"""Check and summarize a claim-level audit of the held-out comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
METHODS = ("direct", "gated")
ANSWERS = ("full", "partial", "wrong", "abstain", "error")
CLAIMS = ("supported", "partial", "unsupported")


def summarize(report, judgments):
    cases = {case["qid"]: case for case in report["cases"]}
    rows = {row["qid"]: row for row in judgments}
    if len(rows) != len(judgments) or len(cases) != len(report["cases"]) or rows.keys() != cases.keys():
        raise ValueError("The audit must contain one unique judgment for each case.")
    summary = {}
    for method in METHODS:
        answers = {context: Counter() for context in ("full", "partial", "none")}
        claims = Counter()
        for qid, case in cases.items():
            review = rows[qid]["methods"][method]
            answer = review["answer"]
            support = review["claims"]
            if answer not in ANSWERS or not isinstance(support, list) or any(x not in CLAIMS for x in support):
                raise ValueError(f"Invalid audit label for {qid}/{method}.")
            if len(support) != len(case["methods"][method].get("findings", [])):
                raise ValueError(f"Finding count does not match the evaluation for {qid}/{method}.")
            answers[case["expected_context"]][answer] += 1
            claims.update(support)
        total = sum(claims.values())
        summary[method] = {
            "answers": {context: {label: ratings[label] for label in ANSWERS}
                        for context, ratings in answers.items()},
            "claims": {label: claims[label] for label in CLAIMS},
            "fully_supported_claim_rate": round(claims["supported"] / total, 6) if total else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DATA / "imdb_gate_holdout.json")
    parser.add_argument("--audit", type=Path, default=DATA / "imdb_gate_holdout_audit.jsonl")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    with args.audit.open(encoding="utf-8-sig") as stream:
        judgments = [json.loads(line) for line in stream if line.strip()]
    print(json.dumps(summarize(report, judgments), indent=2))


if __name__ == "__main__":
    main()
