"""retriever.py —— 四种检索策略与 RRF 融合。

模式：
  vector     只走向量
  bm25       只走稀疏
  rrf        向量 + BM25 走 RRF 融合
  rrf_rerank  RRF 后再过 bge-reranker 精排

四条口径（总纲 7.5 / 7.9 / 7.11）：
  1. 每一路设阈值，低于阈值丢弃；
  2. BM25 全零时提前返回，防止阈值失效；
  3. RRF 只看名次，k 是衰减速度不是准确率旋钮；
  4. 重排只对少量候选做，超时要能降级（失败时退回 RRF 结果）。
"""


def rrf(result_lists, k=60, top_n=None):
    """Reciprocal Rank Fusion：只按名次加权合并，绕开两路分数不可比的问题。"""
    score = {}
    for rs in result_lists:
        for rank, (doc_id, _s) in enumerate(rs, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(score.items(), key=lambda x: -x[1])
    return ordered[:top_n] if top_n else ordered


class HybridRetriever:
    def __init__(self, vector_index, bm25_index, reranker=None):
        self.vi = vector_index
        self.bi = bm25_index
        self.reranker = reranker

    @classmethod
    def with_indexes(cls, blocks, reranker=None):
        """一次性建双索引（学习/实验阶段用，几十秒级）。先 reset 保证 collection 与 blocks 一一对应。"""
        from .index import BM25Index, VectorIndex

        vi = VectorIndex()
        vi.reset()
        vi.add(blocks)
        bi = BM25Index(
            [b["block_id"] for b in blocks],
            [b["text"] for b in blocks],
        )
        return cls(vi, bi, reranker)

    def _vector(self, q, k, min_score=None):
        return self.vi.search(q, k, min_score)

    def _bm25(self, q, k, min_ratio=0.0):
        return self.bi.search(q, k, min_ratio)

    def _rerank(self, q, cands, top_n):
        """精排：候选不足或模型异常时降级为原顺序。"""
        if self.reranker is None or len(cands) <= top_n:
            return cands[:top_n]
        try:
            pairs = [[q, self.bi.texts[self.bi.block_ids.index(bid)]] for bid, _ in cands]
            scores = self.reranker.compute_score(pairs)
            scored = sorted(
                zip([c[0] for c in cands], [c[1] for c in cands], scores),
                key=lambda x: -float(x[2]),
            )
            return [(bid, float(s)) for bid, _orig, s in scored[:top_n]]
        except Exception:
            return cands[:top_n]

    def search(
        self,
        q,
        mode="rrf",
        top_k=5,
        vector_k=20,
        bm25_k=20,
        rrf_k=60,
        bm25_min_ratio=0.15,
        vector_min=None,
        rerank_top=20,
        vector_q=None,
    ):
        """统一入口，返回 list[dict]：block_id / score / rank / source。

        vector_q：可选，替代 q 作为向量路的查询文本（HyDE 假设文档用）。
        BM25 与重排仍用原始 q——假设文档是散文体，词面匹配与相关性判定都需要真问题。
        """
        vq = vector_q or q
        if mode == "vector":
            hits = self._vector(vq, top_k, vector_min)
            return self._fmt(hits, "vector")
        if mode == "bm25":
            hits = self._bm25(q, top_k, bm25_min_ratio)
            return self._fmt(hits, "bm25")
        if mode == "rrf":
            fused = rrf(
                [self._vector(vq, vector_k, vector_min), self._bm25(q, bm25_k, bm25_min_ratio)],
                k=rrf_k,
                top_n=top_k,
            )
            return self._fmt(fused, "rrf")
        if mode == "rrf_rerank":
            fused = rrf(
                [self._vector(vq, vector_k, vector_min), self._bm25(q, bm25_k, bm25_min_ratio)],
                k=rrf_k,
                top_n=rerank_top,
            )
            final = self._rerank(q, fused, top_k)
            return self._fmt(final, "rrf+rerank")
        raise ValueError(f"unknown mode: {mode}")

    @staticmethod
    def _fmt(hits, source):
        return [
            {"block_id": bid, "score": round(float(s), 6), "rank": i + 1, "source": source}
            for i, (bid, s) in enumerate(hits)
        ]
