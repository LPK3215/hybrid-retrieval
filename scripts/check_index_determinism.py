"""check_index_determinism.py —— 同一份块连续重建向量索引，检查结果是否完全一致。

背景：tokenize 探针里两个分词完全相同的变体跑出的 RRF 指标不同（BM25 相同，
差异只能来自向量路）。本脚本在单进程内连建两次索引，对 20 条 query 比
top-20 block_id 集合与顺序，量化「重建漂移」。

用法：python scripts/check_index_determinism.py [重复次数=2]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import corpus  # noqa: E402
from src.index import VectorIndex  # noqa: E402


def snapshot(blocks, queries, k=20):
    vi = VectorIndex()
    vi.reset()
    vi.add(blocks)
    return {q["id"]: [b for b, _s in vi.search(q["text"], k)] for q in queries}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    blocks = corpus.load_blocks()["blocks"]
    queries = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))["queries"]
    print(f"[check] {len(blocks)} 块 / {len(queries)} 条 query / 重复 {reps} 次")

    snaps = [snapshot(blocks, queries) for _ in range(reps)]
    base = snaps[0]
    for i, snap in enumerate(snaps[1:], start=2):
        diff_q = []
        for qid, ids in base.items():
            if snap[qid] != ids:
                same_set = set(snap[qid]) == set(ids)
                diff_q.append((qid, same_set))
        print(f"[check] 第 1 次 vs 第 {i} 次：{len(diff_q)}/{len(base)} 条 query 的 top-20 不同"
              + (f"（{sum(1 for _, s in diff_q if s)} 条仅顺序不同）" if diff_q else ""))
        for qid, same_set in diff_q[:5]:
            print(f"         {qid}: 集合相同={same_set}")


if __name__ == "__main__":
    main()
