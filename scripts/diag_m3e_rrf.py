"""定向诊断 m3e-small / rrf 与归档的差异来源。

三份数据对照：
  A. 持久化 collection（9:22 首跑写入）上的 rrf 逐条结果
  B. 同进程 reset + 重建后的 rrf 逐条结果
  C. 归档 probe_embedding.json 的 m3e-small rrf 汇总
  D. 重放 probe_embedding_per_query.json 的 m3e-small rrf 逐条结果

只读不写产出目录；重建会覆盖 corpus-embed-m3e-small，但数据可再生（与首跑同路径）。
用法：python scripts/diag_m3e_rrf.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, corpus, metrics  # noqa: E402
from src.bench import load_queries  # noqa: E402
from src.index import BM25Index, VectorIndex  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402

OUT = ROOT / "runs" / "20261114-embed-probe"
TAG = "m3e-small"
REPO, PREFIX = "moka-ai/m3e-small", None
MODES = ("vector", "rrf")


def run_all(vi, bi, queries):
    retr = HybridRetriever(vi, bi)
    out = {}
    for mode in MODES:
        rows = []
        for q in queries:
            hits = retr.search(q["text"], mode=mode, top_k=10)
            rows.append({"id": q["id"], "type": q.get("type", "unknown"),
                         "gold": q["gold"], "results": {mode: hits}})
        out[mode] = rows
    return out


def main():
    blocks = corpus.load_blocks()["blocks"]
    queries = load_queries()
    bi = BM25Index([b["block_id"] for b in blocks], [b["text"] for b in blocks])
    fn = config.make_embedding_function(REPO, PREFIX)

    # A: 持久化
    vi = VectorIndex(collection=f"corpus-embed-{TAG}", embedding_function=fn)
    assert vi.count() == len(blocks), vi.count()
    print(f"[A] 持久化 collection: {vi.count()} 块")
    a_rows = run_all(vi, bi, queries)

    # B: 重建
    vi.reset()
    vi.add(blocks)
    print(f"[B] 重建后 collection: {vi.count()} 块")
    b_rows = run_all(vi, bi, queries)

    # D: 重放逐条
    replay = json.loads((OUT / "probe_embedding_per_query.json").read_text(encoding="utf-8"))[TAG]

    def summarize(rows):
        return metrics.summarize(rows, (3, 5, 10))

    archive = json.loads((OUT / "probe_embedding.json").read_text(encoding="utf-8"))[TAG]["summary"]

    for mode in MODES:
        sa = summarize(a_rows[mode])
        sb = summarize(b_rows[mode])
        all_a = [r for r in sa if r["type"] == "all"][0]
        all_b = [r for r in sb if r["type"] == "all"][0]
        all_c = [r for r in archive[mode] if r["type"] == "all"][0]
        print(f"\n== {mode} ==")
        print(f"[A 持久化] R@3={all_a['recall@3']:.4f} R@5={all_a['recall@5']:.4f} R@10={all_a['recall@10']:.4f}")
        print(f"[B 重建  ] R@3={all_b['recall@3']:.4f} R@5={all_b['recall@5']:.4f} R@10={all_b['recall@10']:.4f}")
        print(f"[C 归档  ] R@3={all_c['recall@3']:.4f} R@5={all_c['recall@5']:.4f} R@10={all_c['recall@10']:.4f}")

    # 逐条对比 A vs B vs D，打印差异查询
    diffs = []
    for i, q in enumerate(queries):
        a_hits = a_rows["rrf"][i]["results"]["rrf"]
        b_hits = b_rows["rrf"][i]["results"]["rrf"]
        d_hits = replay["rrf"][i]["results"]["rrf"]
        a_ids = [h["block_id"] for h in a_hits]
        b_ids = [h["block_id"] for h in b_hits]
        d_ids = [h["block_id"] for h in d_hits]
        if a_ids != b_ids or a_ids != d_ids:
            golds = set(q["gold"])
            ga = len(golds & set(a_ids))
            gb = len(golds & set(b_ids))
            gd = len(golds & set(d_ids))
            diffs.append((q["id"], q["type"], len(golds), ga, gb, gd, a_ids, b_ids, d_ids))
    print(f"\n[diff] A/B/D 任不一致的 query 数：{len(diffs)} / {len(queries)}")
    for d in diffs:
        qid, qtype, ng, ga, gb, gd, a_ids, b_ids, d_ids = d
        print(f"  {qid} ({qtype}, gold={ng}): A={ga} B={gb} D={gd}")
        print(f"    A: {a_ids}")
        print(f"    B: {b_ids}")
        print(f"    D: {d_ids}")


if __name__ == "__main__":
    main()
