"""verify_clean.py —— 在干净 470 索引上复现主实验的 vector/rrf 组。

对照基准：
  主实验 runs/20261114-012122-bench：vector all R@10=0.125, rrf all R@10=0.300
  chunk 重跑 500 组：               vector all R@10=0.500, rrf all R@10=0.508
两次必须一致（同 blocks、同 queries、同 top_k）——不一致说明有索引污染或路径差异。
"""

import json

from src import corpus, metrics
from src.bench import load_queries
from src.retriever import HybridRetriever


def main():
    blocks = corpus.load_blocks()["blocks"]
    queries = load_queries()
    print(f"[verify] blocks={len(blocks)} queries={len(queries)}")

    r = HybridRetriever.with_indexes(blocks)
    print(f"[verify] collection count={r.vi.count()}")

    for mode in ("vector", "bm25", "rrf"):
        per = []
        for q in queries:
            hits = r.search(q["text"], mode=mode, top_k=10)
            per.append({"type": q["type"], "gold": q["gold"], "results": {mode: hits}, "lat_ms": {mode: 0.0}})
        s = metrics.summarize(per, (3, 5, 10))
        for row in s:
            if row["type"] == "all":
                print(
                    f"[verify] {mode:7s} R@3={row['recall@3']:.3f} R@5={row['recall@5']:.3f} "
                    f"R@10={row['recall@10']:.3f} MRR={row['mrr']:.3f} nDCG={row['ndcg']:.3f}"
                )


if __name__ == "__main__":
    main()
