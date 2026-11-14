"""embedding 对照的判读输出：汇总表 + 语义题逐条命中。

数据源：runs/20261114-embed-probe/probe_embedding_per_query.json（重放会话全量逐条）。
用法：python scripts/analyze_embed_semantic.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402

REPLAY = ROOT / "runs" / "20261114-embed-probe" / "probe_embedding_per_query.json"


def main():
    pq = json.loads(REPLAY.read_text(encoding="utf-8"))
    tags = ("minilm", "bge-small-zh", "m3e-small")
    modes = ("vector", "rrf", "rrf_rerank")

    print("== 汇总（all 口径）==")
    print(f"{'tag':14s} {'mode':12s} {'exR@10':>7s} {'semR@10':>8s} {'allR@10':>8s} {'allMRR':>7s} {'p50':>7s}")
    for tag in tags:
        for mode in modes:
            s = metrics.summarize(pq[tag][mode], (3, 5, 10))
            row = {r["type"]: r for r in s}
            print(f"{tag:14s} {mode:12s} "
                  f"{row['exact']['recall@10']:>7.3f} {row['semantic']['recall@10']:>8.3f} "
                  f"{row['all']['recall@10']:>8.3f} {row['all']['mrr']:>7.3f} {row['all']['lat_p50_ms']:>7.1f}")

    print("\n== 语义题逐条（top10 命中 gold 数 / gold 总数）==")
    header = " ".join(f"{t[:9]:>10s}" for t in tags for _ in modes)
    for want in ("semantic", "exact"):
        print(f"-- {want} --")
        print(f"{'qid':5s} {'golds':>5s} " + header)
        for qi in range(len(pq["minilm"]["rrf_rerank"])):
            q = pq["minilm"]["rrf_rerank"][qi]
            if q["type"] != want:
                continue
            golds = set(q["gold"])
            cells = []
            for tag in tags:
                for mode in modes:
                    hits = pq[tag][mode][qi]["results"][mode]
                    h = len(golds & {x["block_id"] for x in hits})
                    cells.append(f"{h}/{len(golds):<3d}".rjust(10))
            print(f"{q['id']:5s} {len(golds):>5d} " + " ".join(cells))


if __name__ == "__main__":
    main()
