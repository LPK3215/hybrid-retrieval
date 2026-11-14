"""metrics.py —— 检索评测指标。

Recall@k：gold 中被召回的比例（macro over queries）
MRR@k：首个命中 gold 的倒数排名均值
nDCG@k：二值相关性（gold 命中=1）的折损累计增益
"""

import math


def recall_at_k(golds, hits, k):
    golds = set(golds)
    if not golds:
        return None
    top = [h["block_id"] for h in hits[:k]]
    return len(golds & set(top)) / len(golds)


def mrr_at_k(golds, hits, k):
    for i, h in enumerate(hits[:k]):
        if h["block_id"] in set(golds):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(golds, hits, k):
    golds = set(golds)
    dcg = 0.0
    for i, h in enumerate(hits[:k]):
        rel = 1.0 if h["block_id"] in golds else 0.0
        dcg += rel / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(golds), k)))
    return dcg / ideal if ideal else 0.0


def summarize(per_query, k_list=(3, 5, 10)):
    """把每问结果汇总成分策略、分类型的指标表。

    per_query: [{"type": exact|semantic, "results": {mode: [hit dicts] | None}, ...}]
    """
    out = {}
    for q in per_query:
        for mode, hits in (q.get("results") or {}).items():
            if hits is None:
                continue
            key = (q["type"], mode)
            entry = out.setdefault(key, {"n": 0, "recall": {k: [] for k in k_list}, "mrr": [], "ndcg": [], "lat_ms": []})
            entry["n"] += 1
            for k in k_list:
                r = recall_at_k(q["gold"], hits, k)
                if r is not None:
                    entry["recall"][k].append(r)
            entry["mrr"].append(mrr_at_k(q["gold"], hits, len(hits)))
            entry["ndcg"].append(ndcg_at_k(q["gold"], hits, len(hits)))
            entry["lat_ms"].append(q.get("lat_ms", {}).get(mode, 0.0))

    table = []
    for (qtype, mode), e in sorted(out.items()):
        row = {"type": qtype, "mode": mode, "n": e["n"]}
        for k in k_list:
            v = e["recall"][k]
            row[f"recall@{k}"] = sum(v) / len(v) if v else None
        row["mrr"] = sum(e["mrr"]) / len(e["mrr"]) if e["mrr"] else None
        row["ndcg"] = sum(e["ndcg"]) / len(e["ndcg"]) if e["ndcg"] else None
        row["lat_p50_ms"] = sorted(e["lat_ms"])[len(e["lat_ms"]) // 2] if e["lat_ms"] else None
        table.append(row)

    # 全体合并（不分类型）
    allq = {}
    for q in per_query:
        for mode, hits in (q.get("results") or {}).items():
            if hits is None:
                continue
            e = allq.setdefault(mode, {"n": 0, "recall": {k: [] for k in k_list}, "mrr": [], "ndcg": [], "lat_ms": []})
            e["n"] += 1
            for k in k_list:
                r = recall_at_k(q["gold"], hits, k)
                if r is not None:
                    e["recall"][k].append(r)
            e["mrr"].append(mrr_at_k(q["gold"], hits, len(hits)))
            e["ndcg"].append(ndcg_at_k(q["gold"], hits, len(hits)))
            e["lat_ms"].append(q.get("lat_ms", {}).get(mode, 0.0))
    for mode, e in sorted(allq.items()):
        row = {"type": "all", "mode": mode, "n": e["n"]}
        for k in k_list:
            v = e["recall"][k]
            row[f"recall@{k}"] = sum(v) / len(v) if v else None
        row["mrr"] = sum(e["mrr"]) / len(e["mrr"]) if e["mrr"] else None
        row["ndcg"] = sum(e["ndcg"]) / len(e["ndcg"]) if e["ndcg"] else None
        row["lat_p50_ms"] = sorted(e["lat_ms"])[len(e["lat_ms"]) // 2] if e["lat_ms"] else None
        table.append(row)
    return table
