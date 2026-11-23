"""Aggregate reviewed question judgments and claim-level citation support."""

import argparse
from collections import Counter
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
CONTEXT_LABELS = ("full", "partial", "none")
ANSWER_LABELS = ("full", "partial", "wrong", "abstain", "error")
SUPPORT_LABELS = ("supported", "partial", "unsupported")


def counts(counter, labels):
    return {label: counter[label] for label in labels}


def summarize(report, judgments):
    cases = {case["qid"]: case for case in report["cases"]}
    reviewed = {row["qid"]: row for row in judgments}
    if (len(reviewed) != len(judgments) or len(cases) != len(report["cases"])
            or cases.keys() != reviewed.keys()):
        raise ValueError("The audit must contain exactly one judgment for each reported question.")

    answerable = Counter()
    unsupported = Counter()
    answerable_context = Counter()
    answerable_abstention_context = Counter()
    support = Counter()
    for qid, case in cases.items():
        audit = reviewed[qid]
        context = audit["context"]
        answer = audit["answer"]
        claims = audit["claims"]
        if context not in CONTEXT_LABELS or answer not in ANSWER_LABELS:
            raise ValueError(f"Invalid context or answer label for {qid}.")
        if not isinstance(claims, list) or len(claims) != len(case.get("findings", ())):
            raise ValueError(f"The number of finding judgments differs from the model output for {qid}.")
        if any(label not in SUPPORT_LABELS for label in claims):
            raise ValueError(f"Invalid citation-support label for {qid}.")

        support.update(claims)
        if case["relevant_review_ids"]:
            answerable[answer] += 1
            answerable_context[context] += 1
            if answer == "abstain":
                answerable_abstention_context[context] += 1
        else:
            unsupported[answer] += 1

    total_claims = sum(support.values())
    return {
        "questions_reviewed": len(cases),
        "answerable_answers": counts(answerable, ANSWER_LABELS),
        "unsupported_answers": counts(unsupported, ANSWER_LABELS),
        "answerable_context_coverage": counts(answerable_context, CONTEXT_LABELS),
        "answerable_abstentions_by_context": counts(answerable_abstention_context, CONTEXT_LABELS),
        "findings": counts(support, SUPPORT_LABELS),
        "fully_supported_finding_rate": round(support["supported"] / total_claims, 6) if total_claims else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT / "data" / "imdb_generation.json")
    parser.add_argument("--audit", type=Path, default=PROJECT / "data" / "imdb_generation_audit.jsonl")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8-sig"))
        with args.audit.open(encoding="utf-8-sig") as stream:
            judgments = [json.loads(line) for line in stream if line.strip()]
        summary = summarize(report, judgments)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
