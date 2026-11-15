"""hyde.py —— HyDE（Hypothetical Document Embeddings）查询侧改写。

思路：不用原始 query 的 embedding 去检索，而是先让 LLM 写一段
「假设性答案文档」（与语料同体裁的技术段落），用它的 embedding 检索。
对「用词失配」类语义题有效——query 与文档用词差异大时，
假设文档把问题翻译成了文档侧的语言。

只作用于向量路：BM25 与重排仍用原始 query
（HyDE 文本是散文体，喂给词面匹配和相关性判官都是噪声）。
生成结果按 query 缓存到 data/hyde_cache.json，重复跑 bench 不重复调 LLM。
"""

import json
import time
from pathlib import Path

from . import config

PROMPT = """请根据下面这个问题，写一段假设性的技术文档段落来回答它。

要求：
- 直接写答案内容本身，像文档里的一节正文，200~400 字
- 使用具体的技术术语、参数名、产品名，不要泛泛而谈
- 不要写「假设」「可能」「本文将」这类元叙述
- 不要回答问题编号，不要加标题

问题：{question}

段落："""

_cache = None


def _cache_path():
    return config.DATA_DIR / "hyde_cache.json"


def _load_cache():
    global _cache
    if _cache is None:
        p = _cache_path()
        _cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache


def _client():
    import openai

    config.load_env()
    import os

    return openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "") or "none",
        timeout=120,
    )


def hypothetical_passage(question: str, use_cache=True):
    """返回 HyDE 假设文档。use_cache=False 强制重新生成。"""
    cache = _load_cache() if use_cache else {}
    if question in cache:
        return cache[question]

    conf_model = None
    config.load_env()
    import os

    conf_model = os.getenv("OPENAI_MODEL", "")
    resp = _client().chat.completions.create(
        model=conf_model,
        messages=[{"role": "user", "content": PROMPT.format(question=question)}],
        temperature=0.7,
        max_tokens=800,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("HyDE 生成为空")

    if use_cache:
        cache[question] = text
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return text


def bench_hyde_passages(questions, workers=1):
    """预生成一批 query 的 HyDE 文档，返回 {question: passage}。

    workers>1 时用线程池并发（LLM 调用是 IO 等待，并发安全）。
    """
    out = {}
    t0 = time.time()
    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(hypothetical_passage, q): q for q in questions}
            for fut, q in futs.items():
                try:
                    out[q] = fut.result()
                except Exception as e:  # noqa: BLE001
                    out[q] = None
                    print(f"[hyde] 生成失败 {q[:30]}: {e}")
    else:
        for q in questions:
            try:
                out[q] = hypothetical_passage(q)
            except Exception as e:  # noqa: BLE001
                out[q] = None
                print(f"[hyde] 生成失败 {q[:30]}: {e}")
    print(f"[hyde] {len(out)}/{len(questions)} 条假设文档，{time.time() - t0:.1f}s")
    return out
