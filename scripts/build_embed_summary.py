"""从重放逐条数据重算并覆写 probe_embedding.json 汇总。

背景：首跑（2026-11-14 09:22）为省时间只落汇总，逐条缺失；重放会话
（scripts/replay_embed_probe_per_query.py）产出全量逐条后，以逐条为准重算汇总，
保证 probe_embedding.json 与 probe_embedding_per_query.json 出自同一次测量。
已知差异：m3e-small 的 rrf 受 HNSW 近似索引跨次波动影响（±0.02），
首跑汇总与重放相差 1~2 个边界块，已在实验文档记录。

用法：python scripts/build_embed_summary.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import metrics, probes  # noqa: E402

OUT = ROOT / "runs" / "20261114-embed-probe"


def main():
    per_query = json.loads((OUT / "probe_embedding_per_query.json").read_text(encoding="utf-8"))

    results = {"specs": {t: {"model": probes.EMBED_SPECS[t][0], "query_prefix": probes.EMBED_SPECS[t][1]}
                         for t in probes.EMBED_SPECS}}
    for tag in probes.EMBED_SPECS:
        assert tag in per_query, tag
        results[tag] = {"summary": {}}
        for mode in probes.EMBED_PROBE_MODES:
            rows = per_query[tag][mode]
            assert len(rows) == 20, (tag, mode, len(rows))
            results[tag]["summary"][mode] = metrics.summarize(rows, (3, 5, 10))

    (OUT / "probe_embedding.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[summary] 覆写 {OUT / 'probe_embedding.json'}（{len(probes.EMBED_SPECS)} 模型 × "
          f"{len(probes.EMBED_PROBE_MODES)} 模式 × 20 条，全部逐条重算）")


if __name__ == "__main__":
    main()
