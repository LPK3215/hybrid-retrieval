"""重放 embedding 对照探针：复用 corpus-embed-* 已建索引，只重跑查询侧。

probe_embedding 探针首跑时为省时间只落盘了 semantic 逐条；本脚本用同一套
HybridRetriever 代码路径补齐全量逐条（20 条 query × 3 模型 × 3 模式），
产出 probe_embedding_per_query.json，不重建索引、不覆盖现有产出。

用法：python scripts/replay_embed_probe_per_query.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, corpus, probes
from src.bench import load_queries
from src.index import BM25Index, VectorIndex
from src.retriever import HybridRetriever


def main():
    out_dir = ROOT / "runs" / "20261114-embed-probe"
    blocks = corpus.load_blocks()["blocks"]
    queries = load_queries()
    reranker = config.build_reranker()
    bi = BM25Index([b["block_id"] for b in blocks], [b["text"] for b in blocks])

    per_query = {}
    for tag, (repo, prefix) in probes.EMBED_SPECS.items():
        fn = config.make_embedding_function(repo, prefix)
        vi = VectorIndex(collection=f"corpus-embed-{tag}", embedding_function=fn)
        assert vi.count() == len(blocks), f"{tag} 索引块数 {vi.count()} != {len(blocks)}"
        retr = HybridRetriever(vi, bi, reranker=reranker)
        per_query[tag] = {}
        for mode in probes.EMBED_PROBE_MODES:
            rows = []
            for q in queries:
                t0 = time.perf_counter()
                hits = retr.search(q["text"], mode=mode, top_k=10)
                lat = (time.perf_counter() - t0) * 1000
                rows.append({
                    "id": q["id"], "type": q.get("type", "unknown"),
                    "gold": q["gold"], "results": {mode: hits}, "lat_ms": {mode: lat},
                })
            per_query[tag][mode] = rows
            print(f"[replay] {tag} {mode}: {len(rows)} 条")
    (out_dir / "probe_embedding_per_query.json").write_text(
        json.dumps(per_query, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[replay] 写入 {out_dir / 'probe_embedding_per_query.json'}")


if __name__ == "__main__":
    main()
