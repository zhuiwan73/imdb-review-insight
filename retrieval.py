"""Movie-scoped dense retrieval and reciprocal rank fusion."""

from collections import defaultdict
from threading import Lock

import numpy as np


def rank_hits(hits, top_k):
    return sorted(hits, key=lambda hit: (-hit["score"], hit["chunk_id"]))[:top_k]


class DenseIndex:
    """Score the same passages as BM25 using exact cosine similarity."""

    def __init__(self, base, encoder):
        self._base = base
        self._encoder = encoder
        self._encoder_lock = Lock()
        self._passages = []
        self._movie_spans = {}
        for movie in base.movies():
            movie_id = movie["movie_id"]
            start = len(self._passages)
            self._passages.extend(base.passages(movie_id))
            self._movie_spans[movie_id] = slice(start, len(self._passages))

        self._vectors = None
        if self._passages:
            self._vectors = self._encode([passage["text"] for passage in self._passages])

    def _encode(self, texts):
        # Some tokenizer implementations mutate their internal padding settings.
        with self._encoder_lock:
            vectors = np.asarray(self._encoder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not vectors.shape[1]:
            raise ValueError("嵌入模型没有为每条文本返回一个向量。")
        if not np.isfinite(vectors).all():
            raise ValueError("嵌入模型返回了无效的向量。")
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(lengths, 1e-12)

    def movies(self):
        return self._base.movies()

    def search(self, movie_id, question, top_k=5):
        span = self._movie_spans[movie_id]
        if top_k < 1:
            raise ValueError("top_k 必须大于零。")
        if span.start == span.stop or not question.strip():
            return []

        query = self._encode([question])[0]
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError("查询向量和影评向量的维度不一致。")
        if not np.any(query):
            return []
        scores = np.clip(self._vectors[span] @ query, -1.0, 1.0).tolist()

        best = {}
        for passage, score in zip(self._passages[span], scores):
            review_id = passage["review_id"]
            previous = best.get(review_id)
            if previous is None or (-score, passage["chunk_id"]) < (-previous["score"], previous["chunk_id"]):
                best[review_id] = {**passage, "score": score}
        return rank_hits(best.values(), top_k)


class HybridIndex:
    """Fuse full same-movie BM25 and dense rankings at review level."""

    RRF_OFFSET = 60

    def __init__(self, base, dense):
        self._base = base
        self._dense = dense
        self._review_counts = {movie["movie_id"]: movie["review_count"] for movie in base.movies()}

    def movies(self):
        return self._base.movies()

    def search(self, movie_id, question, top_k=5):
        depth = self._review_counts[movie_id]
        if top_k < 1:
            raise ValueError("top_k 必须大于零。")

        scores = defaultdict(float)
        citation = {}
        for index in (self._base, self._dense):
            for rank, hit in enumerate(index.search(movie_id, question, depth), start=1):
                review_id = hit["review_id"]
                scores[review_id] += 1 / (self.RRF_OFFSET + rank)
                # Keep the original chunk from the higher-ranked source. On equal
                # ranks, the chunk ID gives a deterministic tie without mixing scores.
                candidate = (rank, hit["chunk_id"])
                if review_id not in citation or candidate < citation[review_id][0]:
                    citation[review_id] = (candidate, hit)

        hits = ({**citation[review_id][1], "score": score} for review_id, score in scores.items())
        return rank_hits(hits, top_k)
