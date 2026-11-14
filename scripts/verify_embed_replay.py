"""核对 embedding 对照重放产出：逐条结构与汇总一致性。

- probe_embedding_per_query.json 每 tag × mode 应有 20 条；
- 用 src.metrics.summarize 重算的 recall/mrr/ndcg 应与 probe_embedding.json
  归档 summary 完全一致（同索引、同模型，确定性；时延不比对）；
- 归档 summary 是首跑产物，逐条文件是重放产物，两者互验。

用法：python scripts/verify_embed_replay.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402

OUT = ROOT / "runs" / "20261114-embed-probe"
ARCHIVE = OUT / "probe_embedding.json"
REPLAY = OUT / "probe_embedding_per_query.json"


def main():
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))

    ok = True
    print(f"{'tag':14s} {'mode':12s} {'recall@3':>16s} {'recall@5':>16s} {'recall@10':>16s} {'p50(ms)':>10s}")
    for tag in ("minilm", "bge-small-zh", "m3e-small"):
        for mode in ("vector", "rrf", "rrf_rerank"):
            rows = replay[tag][mode]
            assert len(rows) == 20, f"{tag}/{mode} 条数 {len(rows)} != 20"
            s = metrics.summarize(rows, (3, 5, 10))
            cur = {r["type"]: r for r in s}
            old = {r["type"]: r for r in archive[tag]["summary"][mode]}

            for qtype in ("exact", "semantic", "all"):
                c, o = cur[qtype], old[qtype]
                for key in ("recall@3", "recall@5", "recall@10", "mrr", "ndcg"):
                    diff = abs((c[key] or 0) - (o[key] or 0))
                    if diff > 1e-9:
                        ok = False
                        print(f"  !! {tag}/{mode}/{qtype} {key}: replay={c[key]} archive={o[key]} diff={diff}")
            a = cur["all"]
            print(f"{tag:14s} {mode:12s} {a['recall@3']:>16.3f} {a['recall@5']:>16.3f} {a['recall@10']:>16.3f} {a['lat_p50_ms']:>10.1f}")

    print("struct: 20 条/tag/mode 全通过")
    print("summary: " + ("与归档完全一致 (recall/mrr/ndcg)" if ok else "存在差异，见上方 !! 行"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
