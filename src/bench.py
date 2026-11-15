"""bench.py —— 四组对照主实验。

固定项（总纲 7.10）：同一语料、同一分块、同一嵌入、同一测试集、同一 top-k 口径。
四组：vector / bm25 / rrf / rrf_rerank。
指标：Recall@3/5/10、MRR、nDCG、时延 p50，并按 精确词 / 语义改写 分组。
"""

import json
import time
from datetime import datetime
from pathlib import Path

from . import config, corpus, metrics
from .retriever import HybridRetriever

MODES = ("vector", "bm25", "rrf", "rrf_rerank")
# HyDE 伪模式：向量路换假设文档、BM25/重排仍用原 query
HYDE_MODES = ("hyde_rrf", "hyde_rerank")
K_LIST = (3, 5, 10)


def load_queries(path=None):
    path = Path(path) if path else config.QUERIES_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def run_bench(blocks, queries, modes=MODES, reranker=None, out_dir=None, top_k=10, hyde=False):
    """跑一轮对照，返回 (summary, per_query, meta)。

    hyde=True 时追加 hyde_rrf / hyde_rerank 两组：向量路查询换成
    LLM 生成的假设文档（src/hyde.py，按 query 缓存），BM25 与重排不变。
    """
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(config.RUNS_DIR) / f"{ts}-bench"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[bench] 建索引中……")
    retr = HybridRetriever.with_indexes(blocks, reranker=reranker)
    print(f"[bench] 索引就绪：{retr.vi.count()} 块")

    hyde_map = {}
    if hyde:
        from . import hyde as hyde_mod

        hyde_map = hyde_mod.bench_hyde_passages([q["text"] for q in queries], workers=8)

    all_modes = list(modes) + (list(HYDE_MODES) if hyde else [])
    per_query = []
    for i, q in enumerate(queries, 1):
        row = {
            "id": q["id"],
            "type": q.get("type", "unknown"),
            "text": q["text"],
            "gold": q["gold"],
            "results": {},
            "lat_ms": {},
        }
        if hyde:
            row["hyde_passage"] = hyde_map.get(q["text"])
        for mode in all_modes:
            t0 = time.perf_counter()
            try:
                if mode in HYDE_MODES:
                    base = "rrf" if mode == "hyde_rrf" else "rrf_rerank"
                    hits = retr.search(
                        q["text"], mode=base, top_k=top_k,
                        vector_q=hyde_map.get(q["text"]),
                    )
                else:
                    hits = retr.search(q["text"], mode=mode, top_k=top_k)
                row["results"][mode] = hits
            except Exception as e:  # 单组失败不炸整批
                row["results"][mode] = None
                row.setdefault("errors", {})[mode] = repr(e)
            row["lat_ms"][mode] = round((time.perf_counter() - t0) * 1000, 1)
        per_query.append(row)
        print(f"[bench] {i}/{len(queries)} {q['id']}: " + "  ".join(
            f"{m}@{len(row['results'][m]) if row['results'][m] else 0}" for m in all_modes
        ))

    summary = metrics.summarize(per_query, K_LIST)
    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "modes": list(modes),
        "k_list": list(K_LIST),
        "top_k": top_k,
        "blocks": len(blocks),
        "queries": len(queries),
        "hyde": hyde,
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL if reranker else None,
        "corpus": corpus.corpus_manifest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "per_query.json").write_text(
        json.dumps(per_query, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[bench] 结果写入 {out_dir}")
    return summary, per_query, meta


def print_summary(summary):
    print()
    print(f"{'类型':<8}{'模式':<14}{'n':>3}  " + "  ".join(f"{k:>10}" for k in K_LIST) + f"  {'MRR':>7}{'nDCG':>8}{'p50ms':>8}")
    for r in summary:
        vals = "  ".join(
            (f"{r[f'recall@{k}']:.3f}" if r[f'recall@{k}'] is not None else "-").rjust(10)
            for k in K_LIST
        )
        print(f"{r['type']:<8}{r['mode']:<14}{r['n']:>3}  {vals}  {r['mrr']:>7.3f}{r['ndcg']:>8.3f}{r['lat_p50_ms']:>8.1f}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="四组对照主实验")
    parser.add_argument("--blocks", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true", help="跳过重排组")
    parser.add_argument("--hyde", action="store_true",
                        help="追加 hyde_rrf / hyde_rerank 两组（LLM 生成假设文档走向量路）")
    args = parser.parse_args(argv)

    blocks = corpus.load_blocks(args.blocks)["blocks"] if args.blocks else corpus.load_blocks()["blocks"]
    queries = load_queries(args.queries)
    modes = tuple(m for m in MODES if not (m == "rrf_rerank" and args.no_rerank))
    reranker = None if args.no_rerank else config.build_reranker()
    summary, per_query, meta = run_bench(blocks, queries, modes=modes, reranker=reranker,
                                       out_dir=args.out, top_k=args.top_k, hyde=args.hyde)
    print_summary(summary)


if __name__ == "__main__":
    main()
