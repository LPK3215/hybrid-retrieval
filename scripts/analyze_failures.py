"""analyze_failures.py —— 主实验失败案例分析（一次性分析脚本）。

用法：
  python scripts/analyze_failures.py [bench 目录名]
  默认取 runs/ 下最新的 *-bench 目录。

输出 runs/analysis_failures.md：每条 query 在四模式下的 gold 命中数、
各模式零命中清单、semantic 类逐题诊断、exact 类 BM25/RRF 名次对照、gold 块规模。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BLOCKS = ROOT / "data" / "blocks.json"

MODES = ("vector", "bm25", "rrf", "rrf_rerank")


def pick_bench() -> Path:
    if len(sys.argv) > 1:
        p = RUNS / sys.argv[1]
        if not p.exists():
            raise SystemExit(f"bench 目录不存在：{p}")
        return p
    benches = sorted(d for d in RUNS.iterdir() if d.is_dir() and d.name.endswith("-bench"))
    if not benches:
        raise SystemExit(f"runs/ 下没有 *-bench 目录：{RUNS}")
    return benches[-1]


def main():
    bench = pick_bench()
    per = json.loads((bench / "per_query.json").read_text(encoding="utf-8"))
    blocks = {b["block_id"]: b for b in json.loads(BLOCKS.read_text(encoding="utf-8"))["blocks"]}

    lines = []
    lines.append(f"# 主实验逐条分析（{bench.name}）\n")

    # 1. 总览表
    lines.append("## 一、逐条命中总览\n")
    lines.append("| id | 类型 | gold 数 | vector | bm25 | rrf | rrf_rerank |")
    lines.append("|---|---|---|---|---|---|---|")
    for q in per:
        hits = {}
        for m in MODES:
            res = q["results"].get(m) or []
            ids = [h["block_id"] for h in res]
            hits[m] = sum(1 for g in q["gold"] if g in ids)
        lines.append(
            f"| {q['id']} | {q['type']} | {len(q['gold'])} | "
            f"{hits['vector']}/{len(q['gold'])} | {hits['bm25']}/{len(q['gold'])} | "
            f"{hits['rrf']}/{len(q['gold'])} | {hits['rrf_rerank']}/{len(q['gold'])} |"
        )

    # 2. 零命中明细
    lines.append("\n## 二、各模式零命中 query 清单\n")
    for m in MODES:
        zero = [q for q in per if not any(g in [h["block_id"] for h in (q["results"].get(m) or [])] for g in q["gold"])]
        lines.append(f"- **{m}**：{len(zero)} 条零命中 —— " + "、".join(q["id"] for q in zero))

    # 3. 每类指标里最差的模式（头部 vs 深部）
    lines.append("\n## 三、exact 类：BM25 的 gold 排名 vs RRF/重排 的名次\n")
    for q in per:
        if q["type"] != "exact":
            continue
        b_ids = [h["block_id"] for h in (q["results"].get("bm25") or [])]
        r_ids = [h["block_id"] for h in (q["results"].get("rrf") or [])]
        b_hit_rank = {g: b_ids.index(g) + 1 for g in q["gold"] if g in b_ids}
        r_hit_rank = {g: r_ids.index(g) + 1 for g in q["gold"] if g in r_ids}
        if b_hit_rank or r_hit_rank:
            lines.append(
                f"- {q['id']}：BM25 rank={b_hit_rank}，RRF rank={r_hit_rank}"
            )

    # 4. semantic 类逐题诊断
    lines.append("\n## 四、semantic 类逐题诊断（各模式 gold 命中与 vector top10）\n")
    for q in per:
        if q["type"] != "semantic":
            continue
        lines.append(f"\n### {q['id']}：{q['text']}")
        lines.append(f"- gold：{', '.join(q['gold'])}")
        for m in MODES:
            ids = [h["block_id"] for h in (q["results"].get(m) or [])]
            hit = [g for g in q["gold"] if g in ids]
            lines.append(f"- {m}：命中 {len(hit)}/{len(q['gold'])}" + (f"，命中 {hit}" if hit else ""))
        for g in q["gold"]:
            b = blocks.get(g)
            if b:
                lines.append(f"  - gold[{g}] = {b['doc_id']} | {b['section'][:50]}")
                lines.append(f"    - 原文前 150 字：{b['text'][:150].replace(chr(10), ' ')}")

    # 5. gold 块规模
    lines.append("\n## 五、gold 块规模\n")
    sizes = [len(blocks[g]["text"]) for q in per for g in q["gold"] if g in blocks]
    if sizes:
        lines.append(f"- 共 {len(sizes)} 个 gold 引用（含重复），长度 min={min(sizes)} max={max(sizes)} 平均={sum(sizes)//len(sizes)}")

    out = RUNS / "analysis_failures.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] {bench.name} -> {out}")


if __name__ == "__main__":
    main()
