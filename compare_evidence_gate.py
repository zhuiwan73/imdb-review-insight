"""Compare direct and evidence-gated answers on the same fixed BM25 Top 5."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter, sleep

import evidence_gate
import review_rag
from evaluate_generation import (
    DATA, append_row, build_cases, load_checkpoint, locked_checkpoint, make_config, retryable, write_report,
)


METHODS = {"direct": review_rag.answer_with_ollama, "gated": evidence_gate.answer_with_evidence_gate}


def read_labels(path):
    labels = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["qid"]
            context = item.get("expected_context", item.get("context"))
            if qid in labels or context not in ("full", "partial", "none"):
                raise ValueError(f"Invalid or duplicate context judgment for {qid}.")
            labels[qid] = context, item.get("supporting_chunk_ids")
    return labels


def measure(answer, case, model, url):
    started = perf_counter()
    for attempt in range(2):
        try:
            result = answer(model, case["question"], case["candidates"], base_url=url)
            break
        except review_rag.ModelResponseError as error:
            if attempt == 0 and retryable(error):
                sleep(1)
                continue
            return {"status": "error", "reason": str(error),
                    "latency_ms": round(1000 * (perf_counter() - started), 3)}
    generation = result.get("generation") or {}
    return {
        "status": result["status"], "reason": generation.get("abstention_reason"),
        "findings": result.get("findings", []), "gate": result.get("gate"),
        "citation_ids": [item["chunk_id"] for item in result["citations"]],
        "latency_ms": round(1000 * (perf_counter() - started), 3),
    }


def summarize(records):
    totals = Counter(row["expected_context"] for row in records)
    methods = {}
    for name in METHODS:
        groups = {context: Counter() for context in ("full", "partial", "none")}
        reasons = Counter()
        latencies = []
        for row in records:
            arm = row["methods"][name]
            status = arm["status"]
            # An unusable model output is tracked as an error, separately from abstention.
            if arm["reason"] == "invalid_findings":
                status = "error"
            groups[row["expected_context"]][status] += 1
            if arm["reason"] is not None:
                reasons[arm["reason"]] += 1
            if status != "error":
                latencies.append(arm["latency_ms"])
        methods[name] = {
            "by_context": {context: {status: groups[context][status]
                           for status in ("answered", "insufficient_evidence", "error")}
                           for context in groups},
            "reasons": dict(reasons),
            "p50_ms": round(median(latencies), 3) if latencies else None,
            "p95_ms": (round(quantiles(latencies, n=100, method="inclusive")[94], 3)
                       if len(latencies) > 1 else latencies[0] if latencies else None),
        }
    return {"questions": len(records), "contexts": {context: totals[context]
            for context in ("full", "partial", "none")}, "methods": methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Context judgments, defaulting to the questions file")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--ollama-version", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels_path = args.labels or args.questions
    checkpoint = args.out.with_suffix(".checkpoint.jsonl")
    if len({p.resolve() for p in (args.corpus, args.questions, args.out, checkpoint)}) != 4:
        parser.error("Input, output, and checkpoint files must be distinct.")
    if labels_path.resolve() in {args.out.resolve(), checkpoint.resolve()}:
        parser.error("Context labels must not be used as an output path.")
    config = {
        **make_config(args.corpus, args.questions, args.model, args.ollama_url, args.model_digest),
        "gate_sha256": sha256(Path(evidence_gate.__file__).read_bytes()).hexdigest(),
        "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(), "ollama_version": args.ollama_version,
    }
    cases = list(build_cases(args.corpus, args.questions))
    labels = read_labels(labels_path)
    if {case["qid"] for case in cases} != labels.keys():
        parser.error("Every question must have exactly one context judgment.")
    for case in cases:
        context, support = labels[case["qid"]]
        if support is not None:
            if (not isinstance(support, list) or not set(support) <= set(case["candidate_chunk_ids"])
                    or (context == "full" and not support) or (context == "none" and support)):
                parser.error(f"Evidence IDs disagree with the fixed BM25 Top 5 for {case['qid']}.")

    with locked_checkpoint(checkpoint) as stream:
        records = load_checkpoint(stream, config)
        for index, case in enumerate(cases):
            qid = case["qid"]
            saved = records.get(qid, {}).get("methods", {})
            methods = {}
            order = tuple(METHODS) if index % 2 == 0 else tuple(reversed(METHODS))
            for name in order:
                if name in saved and saved[name]["status"] != "error":
                    methods[name] = saved[name]
                else:
                    methods[name] = measure(METHODS[name], case, args.model, args.ollama_url)
            if methods != saved:
                row = {
                    "kind": "case", **{key: value for key, value in case.items() if key != "candidates"},
                    "expected_context": labels[qid][0], "supporting_chunk_ids": labels[qid][1],
                    "methods": methods,
                }
                records[qid] = row
                append_row(stream, row)
            print(f"[{index + 1}/{len(cases)}] {qid} "
                  + " ".join(f"{name}={methods[name]['status']}" for name in METHODS), flush=True)
    ordered = [records[case["qid"]] for case in cases]
    summary = summarize(ordered)
    write_report(args.out, {**config, **summary, "cases": ordered})
    print(json.dumps(summary, indent=2))
    return int(any(item["error"] for method in summary["methods"].values()
                   for item in method["by_context"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
