"""probes.py —— 工程探针：把"经验口径"变成可测数字。

六组探针（每组只动一个变量）：
  tokenize   中文 BM25 分词方式：jieba+领域词 / jieba / 单字
  threshold  BM25 相对阈值对 RRF 融合的影响（0 / 0.1 / 0.2 / 0.3）
  rrf_k      RRF 衰减常数 k（10 / 30 / 60 / 100 / 1000）
  chunk      块大小对 Recall 的影响（300 / 500 / 800，重建索引）
  candidates 粗排候选数对最终 Recall 与时延的影响（10 / 20 / 50）
  embedding  中文专用 embedding 对照（miniLM / bge-small-zh / m3e-small，重建索引）

判据来自总纲 7.11 六个坑：不分词、不补词、乱调 k、阈值失效。
embedding 探针单独用 --only embedding 跑（耗时长，不进 run_all）。
"""

import json
import time
from datetime import datetime
from pathlib import Path

from . import config, corpus, metrics
from .bench import load_queries
from .index import BM25Index, DOMAIN_WORDS, VectorIndex
from .retriever import HybridRetriever, rrf


def _load():
    blocks = corpus.load_blocks()["blocks"]
    queries = load_queries()
    return blocks, queries


def _new_retriever(blocks, bm25_kw=None, reranker=None):
    from .index import VectorIndex

    vi = VectorIndex()
    vi.reset()
    vi.add(blocks)
    bi = BM25Index(
        [b["block_id"] for b in blocks],
        [b["text"] for b in blocks],
        **(bm25_kw or {}),
    )
    return HybridRetriever(vi, bi, reranker=reranker)


def _eval_mode(retr, queries, mode, top_k=10):
    per = []
    for q in queries:
        t0 = time.perf_counter()
        hits = retr.search(q["text"], mode=mode, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        per.append({"type": q.get("type", "unknown"), "gold": q["gold"], "results": {mode: hits}, "lat_ms": {mode: lat}})
    return metrics.summarize(per, (3, 5, 10))


def probe_tokenize(queries, blocks, out_dir):
    """中文分词方式对 BM25 与混合检索的影响。

    只重建 BM25 索引，向量索引全程共用一份：同一份块的两个索引重建存在
    top-20 边界漂移（见 scripts/check_index_determinism.py），若每组都重建
    向量索引，分词差异会与索引噪声混淆。
    """
    variants = {
        "jieba+领域词": {"dict_words": DOMAIN_WORDS, "use_jieba": True},
        "jieba": {"dict_words": [], "use_jieba": True},
        "单字": {"dict_words": None, "use_jieba": False},
    }
    vi = VectorIndex()
    vi.reset()
    vi.add(blocks)
    ids = [b["block_id"] for b in blocks]
    texts = [b["text"] for b in blocks]
    results = {}
    for name, kw in variants.items():
        bi = BM25Index(ids, texts, **kw)
        retr = HybridRetriever(vi, bi)
        rows_b = _eval_mode(retr, queries, "bm25")
        rows_r = _eval_mode(retr, queries, "rrf")
        results[name] = {
            "bm25": [r for r in rows_b if r["type"] == "all"],
            "rrf": [r for r in rows_r if r["type"] == "all"],
        }
        print(f"[probe:tokenize] {name}: " + "  ".join(
            f"{r['mode']} R@10={r['recall@10']:.3f} p50={r['lat_p50_ms']:.1f}ms" for r in rows_b + rows_r
        ))
    _dump(out_dir, "probe_tokenize.json", results)
    return results


def probe_threshold(queries, blocks, out_dir):
    """BM25 相对阈值对 RRF 融合的影响。"""
    retr = _new_retriever(blocks)
    results = {}
    for ratio in (0.0, 0.1, 0.2, 0.3):
        per = []
        cands = []
        for q in queries:
            t0 = time.perf_counter()
            v = retr._vector(q["text"], 20)
            b = retr._bm25(q["text"], 20, min_ratio=ratio)
            hits = retr._fmt(rrf([v, b], top_n=10), "rrf")
            lat = (time.perf_counter() - t0) * 1000
            per.append({"type": q.get("type", "unknown"), "gold": q["gold"], "results": {"rrf": hits}, "lat_ms": {"rrf": lat}})
            cands.append((len(v), len(b)))
        s = metrics.summarize(per, (3, 5, 10))
        avg_v = sum(c[0] for c in cands) / len(cands)
        avg_b = sum(c[1] for c in cands) / len(cands)
        results[str(ratio)] = {
            "summary": [r for r in s if r["type"] == "all"],
            "avg_vector_cands": round(avg_v, 1),
            "avg_bm25_cands": round(avg_b, 1),
        }
        print(f"[probe:threshold] min_ratio={ratio}: avg_cands v={avg_v:.1f} b={avg_b:.1f} " + "  ".join(
            f"R@10={r['recall@10']:.3f} p50={r['lat_p50_ms']:.1f}ms" for r in s if r["type"] == "all"
        ))
    _dump(out_dir, "probe_threshold.json", results)
    return results


def probe_rrf_k(queries, blocks, out_dir):
    """RRF 衰减常数 k 的影响（只看名次）。"""
    retr = _new_retriever(blocks)
    results = {}
    for k in (10, 30, 60, 100, 1000):
        per = []
        for q in queries:
            t0 = time.perf_counter()
            hits = retr.search(q["text"], mode="rrf", top_k=10, rrf_k=k)
            lat = (time.perf_counter() - t0) * 1000
            per.append({"type": q.get("type", "unknown"), "gold": q["gold"], "results": {"rrf": hits}, "lat_ms": {"rrf": lat}})
        s = metrics.summarize(per, (3, 5, 10))
        results[str(k)] = [r for r in s if r["type"] == "all"]
        print(f"[probe:rrf_k] k={k}: " + "  ".join(
            f"R@10={r['recall@10']:.3f} p50={r['lat_p50_ms']:.1f}ms" for r in s if r["type"] == "all"
        ))
    _dump(out_dir, "probe_rrf_k.json", results)
    return results


def probe_chunk(queries, out_dir):
    """块大小对 Recall 的影响（重建索引）。"""
    results = {}
    for size in (300, 500, 800):
        blocks = corpus.build_blocks(chunk_size=size)
        retr = _new_retriever(blocks)
        rows_v = _eval_mode(retr, queries, "vector")
        rows_r = _eval_mode(retr, queries, "rrf")
        results[str(size)] = {
            "blocks": len(blocks),
            "vector": [r for r in rows_v if r["type"] == "all"],
            "rrf": [r for r in rows_r if r["type"] == "all"],
        }
        print(f"[probe:chunk] size={size}: blocks={len(blocks)} " + "  ".join(
            f"{r['mode']} R@10={r['recall@10']:.3f}" for r in rows_v + rows_r
        ))
    _dump(out_dir, "probe_chunk.json", results)
    return results


def probe_candidates(queries, blocks, out_dir):
    """粗排候选数对 Recall 与时延的影响。"""
    retr = _new_retriever(blocks, reranker=config.build_reranker())
    results = {}
    for k in (10, 20, 50):
        per = []
        for q in queries:
            t0 = time.perf_counter()
            hits = retr.search(q["text"], mode="rrf_rerank", top_k=10, vector_k=k, bm25_k=k)
            lat = (time.perf_counter() - t0) * 1000
            per.append({"type": q.get("type", "unknown"), "gold": q["gold"], "results": {"rrf_rerank": hits}, "lat_ms": {"rrf_rerank": lat}})
        s = metrics.summarize(per, (3, 5, 10))
        results[str(k)] = [r for r in s if r["type"] == "all"]
        print(f"[probe:candidates] k={k}: " + "  ".join(
            f"R@10={r['recall@10']:.3f} p50={r['lat_p50_ms']:.1f}ms" for r in s if r["type"] == "all"
        ))
    _dump(out_dir, "probe_candidates.json", results)
    return results


# bge-small-zh 官方口径：检索任务给 query 加指令，文档侧不加（MiniLM 无此用法）。
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# (显示名, 模型 repo, 查询侧前缀)。minilm 为同会话重跑基线，与主实验归档数字互验。
EMBED_SPECS = {
    "minilm": (config.EMBED_MODEL, None),
    "bge-small-zh": ("BAAI/bge-small-zh-v1.5", BGE_ZH_QUERY_PREFIX),
    "m3e-small": ("moka-ai/m3e-small", None),
}

# bm25 与嵌入无关，主实验已给；这里只看换嵌入会变的三个模式。
EMBED_PROBE_MODES = ("vector", "rrf", "rrf_rerank")


def probe_embedding(queries, blocks, out_dir, tags=None):
    """中文专用 embedding 对照（本实验后续探针）。

    换嵌入必然重建向量索引，故每个模型一个独立 collection（corpus-embed-<tag>）；
    BM25 索引、重排器与嵌入无关，全程共用一份。查询侧前缀只对 bge-small-zh 生效。
    对照口径与主实验一致：同一语料、同一分块、同一测试集、同一 top-k。
    """
    tags = list(tags or EMBED_SPECS)
    reranker = config.build_reranker()
    bi = BM25Index([b["block_id"] for b in blocks], [b["text"] for b in blocks])
    results = {"specs": {t: {"model": EMBED_SPECS[t][0], "query_prefix": EMBED_SPECS[t][1]} for t in tags}}
    per_query = {}

    for tag in tags:
        repo, prefix = EMBED_SPECS[tag]
        fn = config.make_embedding_function(repo, prefix)
        vi = VectorIndex(collection=f"corpus-embed-{tag}", embedding_function=fn)
        vi.reset()
        vi.add(blocks)
        print(f"[probe:embedding] {tag} ({repo}) 索引就绪：{vi.count()} 块，重建耗时见日志")

        retr = HybridRetriever(vi, bi, reranker=reranker)
        results[tag] = {"summary": {}}
        per_query[tag] = {}
        for mode in EMBED_PROBE_MODES:
            per = []
            for q in queries:
                t0 = time.perf_counter()
                hits = retr.search(q["text"], mode=mode, top_k=10)
                lat = (time.perf_counter() - t0) * 1000
                per.append({
                    "id": q["id"], "type": q.get("type", "unknown"),
                    "gold": q["gold"], "results": {mode: hits}, "lat_ms": {mode: lat},
                })
            s = metrics.summarize(per, (3, 5, 10))
            results[tag]["summary"][mode] = s
            per_query[tag][mode] = per
            print(f"[probe:embedding] {tag} {mode}: " + "  ".join(
                f"R@10={r['recall@10']:.3f}({r['type']})" for r in s
            ))

    _dump(out_dir, "probe_embedding.json", results)
    _dump(out_dir, "probe_embedding_per_query.json", per_query)
    return results


def _dump(out_dir, name, obj):
    path = Path(out_dir) / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def run_all(out_dir=None):
    blocks, queries = _load()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir) if out_dir else Path(config.RUNS_DIR) / f"{ts}-probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probes] 语料 {len(blocks)} 块，问题 {len(queries)} 条，输出 {out_dir}")

    probe_tokenize(queries, blocks, out_dir)
    probe_threshold(queries, blocks, out_dir)
    probe_rrf_k(queries, blocks, out_dir)
    probe_candidates(queries, blocks, out_dir)
    probe_chunk(queries, out_dir)
    print(f"[probes] 完成，结果在 {out_dir}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="工程探针")
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="tokenize|threshold|rrf_k|chunk|candidates|embedding")
    args = parser.parse_args(argv)

    if not args.only:
        run_all(args.out)
    else:
        blocks, queries = _load()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else Path(config.RUNS_DIR) / f"{ts}-probe-{args.only}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fn = {
            "tokenize": lambda: probe_tokenize(queries, blocks, out_dir),
            "threshold": lambda: probe_threshold(queries, blocks, out_dir),
            "rrf_k": lambda: probe_rrf_k(queries, blocks, out_dir),
            "chunk": lambda: probe_chunk(queries, out_dir),
            "candidates": lambda: probe_candidates(queries, blocks, out_dir),
            "embedding": lambda: probe_embedding(queries, blocks, out_dir),
        }[args.only]
        fn()


if __name__ == "__main__":
    main()
