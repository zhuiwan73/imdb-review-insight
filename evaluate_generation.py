"""Evaluate local Ollama answers using fixed, movie-scoped BM25 evidence."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median, quantiles
from tempfile import NamedTemporaryFile
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import review_rag
from evaluate import read_cases


DATA = Path(__file__).resolve().parent / "data"
TOP_K = 5


def make_config(corpus, questions, model, ollama_url, model_digest):
    return {
        "schema_version": 1,
        "retriever": "bm25",
        "top_k": TOP_K,
        "model": model,
        "model_digest": model_digest,
        "ollama_url": ollama_url,
        "corpus": corpus.name,
        "corpus_sha256": sha256(corpus.read_bytes()).hexdigest(),
        "questions": questions.name,
        "questions_sha256": sha256(questions.read_bytes()).hexdigest(),
        "implementation_sha256": sha256(Path(review_rag.__file__).read_bytes()).hexdigest(),
    }


@contextmanager
def locked_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Cannot lock checkpoint {path}. Another run may be using it.") from error
        yield stream


def append_row(stream, row):
    stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    stream.flush()
    os.fsync(stream.fileno())


def load_checkpoint(stream, config):
    stream.seek(0)
    data = stream.read()
    if data and not data.endswith(b"\n"):
        # A hard interruption can leave the final write incomplete. Keep only committed rows.
        end = data.rfind(b"\n") + 1
        stream.seek(end)
        stream.truncate()
        data = data[:end]

    header = {"kind": "run", "config": config}
    if not data:
        append_row(stream, header)
        return {}
    rows = {}
    for number, line in enumerate(data.split(b"\n")[:-1], 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid checkpoint row {number}.") from error
        if number == 1:
            if row != header:
                raise ValueError("Checkpoint settings changed. Use a new output or checkpoint path.")
            continue
        if not isinstance(row, dict) or row.get("kind") != "case" or not isinstance(row.get("qid"), str):
            raise ValueError(f"Invalid checkpoint case at row {number}.")
        rows[row["qid"]] = row
    return rows


def retryable(error):
    cause = error.__cause__
    if isinstance(cause, HTTPError):
        return cause.code in (408, 429, 500, 502, 503, 504)
    return isinstance(cause, (URLError, TimeoutError))


def run_case(case, model, ollama_url, retries):
    row = {"kind": "case", **case}
    started = perf_counter()
    for attempt in range(1, retries + 2):
        try:
            response = review_rag.answer_with_ollama(
                model, case["question"], case["candidates"], base_url=ollama_url,
            )
        except review_rag.ModelResponseError as error:
            if attempt <= retries and retryable(error):
                sleep(min(attempt, 3))
                continue
            row.update(outcome="error", status="error", error=str(error))
            break

        status = response["status"]
        generation = response.get("generation") or {}
        reason = generation.get("abstention_reason")
        if status == "answered":
            outcome = "answered"
        elif reason in ("model_rejected", "no_candidates"):
            outcome = reason
        else:
            outcome = "invalid_model_output"

        citation_ids = [citation["chunk_id"] for citation in response["citations"]]
        citation_review_ids = [citation["review_id"] for citation in response["citations"]]
        raw_ids = generation.get("raw_citation_ids")
        candidate_ids = {hit["chunk_id"] for hit in case["candidates"]}
        row.update(
            status=status, outcome=outcome, answer=response["answer"], error=None,
            findings=response.get("findings", []), generation=generation,
            cited_chunk_ids=citation_ids, cited_review_ids=list(dict.fromkeys(citation_review_ids)),
            emitted_citation_count=len(citation_ids),
            invalid_emitted_citation_count=sum(identifier not in candidate_ids for identifier in citation_ids),
            raw_citation_count=len(raw_ids) if isinstance(raw_ids, list) else None,
            invalid_raw_citation_count=(
                sum(not isinstance(identifier, str) or identifier not in candidate_ids for identifier in raw_ids)
                if isinstance(raw_ids, list) else None
            ),
        )
        break
    row["attempts"] = attempt
    row["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return row


def build_cases(corpus, questions):
    index = review_rag.ReviewIndex.from_jsonl(corpus)
    for qid, movie_id, question, expected in read_cases(corpus, questions):
        hits = index.search(movie_id, question, TOP_K)
        retrieved = [hit["review_id"] for hit in hits]
        yield {
            "qid": qid, "movie_id": movie_id, "question": question, "candidates": hits,
            "relevant_review_ids": sorted(expected),
            "retrieved_review_ids": retrieved, "candidate_chunk_ids": [hit["chunk_id"] for hit in hits],
            "matched_review_ids": sorted(expected.intersection(retrieved)),
        }


def rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def summarize(config, cases, rows):
    records = [rows[case["qid"]] for case in cases if case["qid"] in rows]
    answerable = [case for case in cases if case["relevant_review_ids"]]
    unsupported = len(cases) - len(answerable)
    with_answer = [row for row in records if row["relevant_review_ids"]]
    without_answer = [row for row in records if not row["relevant_review_ids"]]
    answered_answerable = sum(row["outcome"] == "answered" for row in with_answer)
    answered_unsupported = sum(row["outcome"] == "answered" for row in without_answer)
    abstained_answerable = sum(row["outcome"] in ("model_rejected", "no_candidates") for row in with_answer)
    model_rejected_unsupported = sum(row["outcome"] == "model_rejected" for row in without_answer)
    no_candidates_unsupported = sum(row["outcome"] == "no_candidates" for row in without_answer)
    abstained_unsupported = model_rejected_unsupported + no_candidates_unsupported
    retrieved_hits = sum(bool(case["matched_review_ids"]) for case in answerable)
    citation_hits = sum(
        bool(set(row["relevant_review_ids"]).intersection(row.get("cited_review_ids", [])))
        for row in with_answer
    )
    emitted = sum(row.get("emitted_citation_count", 0) for row in records)
    invalid_emitted = sum(row.get("invalid_emitted_citation_count", 0) for row in records)
    raw_observed = [row for row in records if row.get("raw_citation_count") is not None]
    raw = sum(row["raw_citation_count"] for row in raw_observed)
    invalid_raw = sum(row["invalid_raw_citation_count"] for row in raw_observed)
    latencies = [
        row["latency_ms"] for row in records
        if row.get("generation", {}).get("made_model_call") and row["outcome"] != "error"
    ]
    return {
        **config,
        "scope": (
            "Response and abstention rates use the full selected labeled group as the denominator. "
            "Invalid findings and request/parsing errors are reported separately. Review-ID overlap and "
            "citation-ID validity do not establish semantic support; that requires reviewing cited excerpts."
        ),
        "latency_scope": (
            "Model-call latency measures the elapsed end-to-end answer call, including any transport retries. "
            "Rows resolved without a model call and rows with request/parsing errors are excluded."
        ),
        "complete": len(records) == len(cases) and all(row["outcome"] != "error" for row in records),
        "counts": {
            "questions": len(cases), "answerable": len(answerable), "unsupported": unsupported,
            "pending": len(cases) - len(records), "error": sum(row["outcome"] == "error" for row in records),
            "invalid_model_output": sum(row["outcome"] == "invalid_model_output" for row in records),
            "answered_answerable": answered_answerable, "abstained_answerable": abstained_answerable,
            "answered_unsupported": answered_unsupported, "abstained_unsupported": abstained_unsupported,
            "model_rejected_unsupported": model_rejected_unsupported,
            "no_candidates_unsupported": no_candidates_unsupported,
            "answerable_with_known_evidence_retrieved": retrieved_hits,
            "answerable_with_known_review_cited": citation_hits,
            "model_calls_with_response": len(latencies),
            "raw_citation_ids": raw, "invalid_raw_citation_ids": invalid_raw,
            "emitted_citation_ids": emitted, "invalid_emitted_citation_ids": invalid_emitted,
        },
        "metrics": {
            "answerable_response_rate": rate(answered_answerable, len(answerable)),
            "answerable_abstention_rate": rate(abstained_answerable, len(answerable)),
            "unsupported_abstention_rate": rate(abstained_unsupported, unsupported),
            "unsupported_response_rate": rate(answered_unsupported, unsupported),
            "answerable_known_evidence_hit_rate": rate(retrieved_hits, len(answerable)),
            "answerable_known_review_citation_rate": rate(citation_hits, len(answerable)),
            "candidate_review_recall_at_5": (
                round(sum(len(case["matched_review_ids"]) / len(case["relevant_review_ids"])
                          for case in answerable) / len(answerable), 6) if answerable else None
            ),
            "emitted_citation_id_validity_rate": rate(emitted - invalid_emitted, emitted),
            "raw_citation_id_validity_rate": rate(raw - invalid_raw, raw),
            "model_call_p50_ms": round(median(latencies), 3) if latencies else None,
            "model_call_p95_ms": (
                round(quantiles(latencies, n=100, method="inclusive")[94], 3)
                if len(latencies) > 1 else latencies[0] if latencies else None
            ),
        },
        # Checkpoint retains excerpts for manual review. The report refers to them by IDs.
        "cases": [{key: value for key, value in row.items() if key not in ("kind", "candidates")}
                  for row in records],
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA / "imdb_local.jsonl")
    parser.add_argument("--questions", type=Path, default=DATA / "imdb_eval.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", help="Optional Ollama model digest for checkpoint reproducibility")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=DATA / "imdb_generation.json")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the .checkpoint.jsonl sidecar of --out")
    parser.add_argument("--limit", type=int, help="Only process the first N questions, for a smoke run")
    parser.add_argument("--retries", type=int, default=1, help="Extra attempts for transient service errors")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.retries < 0:
        parser.error("--limit must be positive and --retries nonnegative.")
    url = urlsplit(args.ollama_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.path.rstrip("/") or url.query or url.fragment:
        parser.error("--ollama-url must be a base URL such as http://127.0.0.1:11434.")
    ollama_url = args.ollama_url.rstrip("/")
    checkpoint = args.checkpoint or args.out.with_suffix(".checkpoint.jsonl")
    paths = (args.corpus, args.questions, args.out, checkpoint)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Corpus, questions, report, and checkpoint paths must be distinct.")
    interrupted = False
    try:
        config = make_config(args.corpus, args.questions, args.model, ollama_url, args.model_digest)
        cases = list(build_cases(args.corpus, args.questions))[:args.limit]
        with locked_checkpoint(checkpoint) as stream:
            rows = load_checkpoint(stream, config)
            try:
                for position, case in enumerate(cases, 1):
                    qid = case["qid"]
                    if qid in rows and rows[qid]["outcome"] != "error":
                        continue
                    row = run_case(case, args.model, ollama_url, args.retries)
                    append_row(stream, row)
                    rows[qid] = row
                    print(f"[{position}/{len(cases)}] {qid}: {row['outcome']} ({row['latency_ms']:.0f} ms)", flush=True)
            except KeyboardInterrupt:
                interrupted = True
            report = summarize(config, cases, rows)
            write_report(args.out, report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    return 130 if interrupted else 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
