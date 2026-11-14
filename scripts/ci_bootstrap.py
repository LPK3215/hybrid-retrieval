"""ci_bootstrap.py —— 对 bench 的 per_query.json 做 bootstrap 置信区间。

用法：
  python scripts/ci_bootstrap.py runs/<bench目录> [--iters 10000] [--seed 42]

对每个 (type, mode) 组合，对 recall@3/5/10、MRR、nDCG 做有放回重采样，
报告均值与 95% 百分位区间。回答的问题：「20 条小样本上的比例结论，
换成 50 条后置信区间收窄多少、结论方向是否还成立」。
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics  # noqa: E402


def per_query_scores(per_query, k_list=(3, 5, 10)):
    """把 per_query 展成 {(type, mode): {metric: [per-query values]}}。"""
    out = {}
    for q in per_query:
        for mode, hits in (q.get("results") or {}).items():
            if hits is None:
                continue
            for qtype in (q["type"], "all"):
                if qtype != "all" and qtype != q["type"]:
                    continue
                key = (qtype, mode)
                e = out.setdefault(key, {f"recall@{k}": [] for k in k_list} | {"mrr": [], "ndcg": []})
                for k in k_list:
                    r = metrics.recall_at_k(q["gold"], hits, k)
                    if r is not None:
                        e[f"recall@{k}"].append(r)
                e["mrr"].append(metrics.mrr_at_k(q["gold"], hits, len(hits)))
                e["ndcg"].append(metrics.ndcg_at_k(q["gold"], hits, len(hits)))
    return out


def boot_ci(values, iters, rng):
    n = len(values)
    if n == 0:
        return None
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return sum(values) / n, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_query = json.loads((Path(args.run_dir) / "per_query.json").read_text(encoding="utf-8"))
    scores = per_query_scores(per_query)
    rng = random.Random(args.seed)

    print(f"{'类型':<9}{'模式':<14}{'指标':<10}{'n':>4}  {'均值':>7}  {'95% CI':>17}")
    for (qtype, mode), e in sorted(scores.items()):
        for metric_name, vals in e.items():
            r = boot_ci(vals, args.iters, rng)
            if r is None:
                continue
            mean, lo, hi = r
            print(f"{qtype:<9}{mode:<14}{metric_name:<10}{len(vals):>4}  {mean:>7.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
