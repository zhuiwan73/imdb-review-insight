"""Run the local review explorer and JSON API."""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from embedding import DEFAULT_MODEL_DIR
from evidence_gate import answer_with_evidence_gate
from review_rag import ModelResponseError, ReviewIndex, answer_extractive, answer_with_ollama


PROJECT = Path(__file__).resolve().parent
RETRIEVER_LABELS = {
    "bm25": "关键词 · BM25",
    "dense": "语义 · MiniLM",
    "hybrid": "混合 · BM25 + MiniLM",
}


class ReviewHandler(BaseHTTPRequestHandler):
    def send_content(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_content(status, "application/json; charset=utf-8", body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_content(200, "text/html; charset=utf-8", self.server.page)
        elif path == "/api/movies":
            self.send_json(200, self.server.index.movies())
        elif path == "/api/retrievers":
            self.send_json(200, {
                "items": [{"id": name, "label": RETRIEVER_LABELS[name]} for name in self.server.retrievers],
                "selected": self.server.default_retriever,
            })
        elif path == "/favicon.ico":
            self.send_content(204, "image/x-icon", b"")
        else:
            self.send_json(404, {"error": "页面不存在。"})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/ask":
            self.send_json(404, {"error": "接口不存在。"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "请求应使用 JSON。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 8192:
                raise ValueError("请求为空或超过大小限制。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求应为 JSON 对象。")
            movie_id, question = payload.get("movie_id"), payload.get("question")
            if not isinstance(movie_id, str) or not movie_id:
                raise ValueError("请选择电影。")
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 300:
                raise ValueError("问题长度应为 1 到 300 个字符。")
            question = question.strip()
            method = payload.get("retriever", self.server.default_retriever)
            if not isinstance(method, str) or method not in self.server.retrievers:
                raise ValueError("请选择当前服务支持的检索方式。")
            hits = self.server.retrievers[method].search(movie_id, question)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "请求 JSON 无法读取。"})
            return
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except KeyError:
            self.send_json(404, {"error": "评论库中没有这部电影。"})
            return

        try:
            if self.server.model:
                answer = answer_with_evidence_gate if self.server.evidence_gate else answer_with_ollama
                result = answer(self.server.model, question, hits, base_url=self.server.ollama_url)
            else:
                result = answer_extractive(question, hits)
        except ModelResponseError as exc:
            self.send_json(502, {"error": str(exc)})
            return
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT / "data" / "demo_reviews.jsonl")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ollama-model", help="Name of a model available in local Ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Base URL of the Ollama server")
    parser.add_argument("--evidence-gate", action="store_true",
                        help="Check exact source quotes before generating an Ollama answer")
    parser.add_argument("--open-browser", action="store_true", help="Open the local page after startup")
    parser.add_argument("--semantic", type=Path, nargs="?", const=DEFAULT_MODEL_DIR,
                        help="Enable dense and hybrid retrieval, optionally with a local model directory")
    args = parser.parse_args()
    if args.evidence_gate and not args.ollama_model:
        parser.error("--evidence-gate requires --ollama-model")
    try:
        index = ReviewIndex.from_jsonl(args.corpus)
        page = (PROJECT / "static" / "index.html").read_bytes()
        retrievers = {"bm25": index}
        if args.semantic is not None:
            from embedding import MiniLMEncoder

            print("Preparing the semantic index. Please wait...", flush=True)
            encoder = MiniLMEncoder(args.semantic)
            from retrieval import DenseIndex, HybridIndex

            dense = DenseIndex(index, encoder)
            retrievers.update(dense=dense, hybrid=HybridIndex(index, dense))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        if args.semantic is not None:
            print("Semantic retrieval could not start. You can still use Start-IMDb.cmd for BM25.", file=sys.stderr)
        parser.exit(1, f"{exc}\n")
    server.index = index
    server.retrievers = retrievers
    server.default_retriever = "bm25"
    server.page = page
    server.model = args.ollama_model
    server.ollama_url = args.ollama_url
    server.evidence_gate = args.evidence_gate
    mode = f"Ollama ({server.model})" if server.model else "keyword retrieval"
    with server:
        address = f"http://127.0.0.1:{server.server_port}"
        print(f"{address}  |  {mode}", flush=True)
        if args.open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
