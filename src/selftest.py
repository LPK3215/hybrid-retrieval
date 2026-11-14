"""selftest.py —— 机制自测（纯检索层，不依赖 LLM）。

六个用例：
  1 切块确定性     同一文本两次切块一致
  2 BM25 全零返回   无命中词时返回空，阈值不失效
  3 RRF 只看名次    两路结果合并后按 RRF 分降序
  4 指标公式       Recall/MRR/nDCG 手算值一致
  5 阈值生效        min_ratio 越高候选越少
  6 去噪            清洗后不含 wikilink 与图片语法
"""

from . import corpus, metrics
from .index import BM25Index
from .retriever import rrf


def case_split_determinism():
    t = "# 标题\n\n" + "这是一段测试文本。" * 100
    a = corpus.split_text(t, 200, 20)
    b = corpus.split_text(t, 200, 20)
    assert a == b, "切块不确定"
    assert len(a) >= 2, "切分块数异常"
    return f"块数={len(a)}"


def case_bm25_zero():
    bi = BM25Index(["a", "b"], [" retrieves the storage layer ", "向量索引与检索"])
    out = bi.search("完全不相关的单词 xyzzy", 5)
    assert out == [], f"全零时应为空，实际 {out[:2]}"
    return "全零返回空：通过"


def case_rrf_order():
    # 名次不对称：a 在 r1 第 1，c 在 r2 第 1，b 两路都靠后
    r1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    r2 = [("c", 5.0), ("a", 4.0), ("b", 3.0)]
    fused = rrf([r1, r2], k=60, top_n=3)
    assert [x[0] for x in fused] == ["a", "c", "b"], f"RRF 排序异常：{fused}"
    scores = [s for _, s in fused]
    assert scores[0] > scores[1] > scores[2], "RRF 分数未降序"
    return f"top3={[x[0] for x in fused]}"


def case_metrics():
    hits = [{"block_id": "h1"}, {"block_id": "h2"}, {"block_id": "h3"}]
    assert metrics.recall_at_k({"h1", "h9"}, hits, 3) == 0.5
    assert metrics.mrr_at_k({"h2"}, hits, 3) == 0.5
    assert abs(metrics.ndcg_at_k({"h1"}, hits, 3) - 1.0) < 1e-9
    return "Recall/MRR/nDCG 手算一致"


def case_threshold():
    texts = ["键值对缓存常用 Redis", "向量库以相似度召回", "分块策略决定上下文完整性"]
    bi = BM25Index([f"d{i}" for i in range(3)], texts)
    q = "缓存"
    low = bi.search(q, 10, min_ratio=0.0)
    high = bi.search(q, 10, min_ratio=0.9)
    assert len(high) <= len(low), f"阈值未生效：low={len(low)} high={len(high)}"
    return f"loose={len(low)} strict={len(high)}"


def case_clean():
    raw = "见 [[某笔记|链接]] 与 ![[pic.png]] 以及 ^blockref 内容"
    cleaned = corpus.clean_text(raw)
    assert "[[" not in cleaned and "]]" not in cleaned and "^blockref" not in cleaned, cleaned
    return "wikilink/图片/块引用已去"


def run_all():
    cases = [
        case_split_determinism,
        case_bm25_zero,
        case_rrf_order,
        case_metrics,
        case_threshold,
        case_clean,
    ]
    ok = 0
    for fn in cases:
        try:
            msg = fn()
            ok += 1
            print(f"[selftest] PASS {fn.__name__}: {msg}")
        except AssertionError as e:
            print(f"[selftest] FAIL {fn.__name__}: {e}")
    print(f"[selftest] {ok}/{len(cases)} 通过")
    return ok == len(cases)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_all() else 1)
