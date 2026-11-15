"""contextual.py —— Contextual Retrieval（文档侧结构补丁）。

思路：切块后每个 chunk 单独嵌入时丢了「它在文档哪一节、承上启下讲什么」
的结构信息，跨文档同主题的块因此在向量空间里几乎重合。给每个块前置一段
LLM 生成的定位上下文再嵌入，把结构信息补回向量。

只作用于向量索引的文档侧文本：BM25、重排、query 全部不变。
上下文按 block_id 缓存到 data/ctx_cache.json，重复跑不重复调 LLM。
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

PROMPT = """下面是技术文档《{doc}》中「{section}」一节里的一段内容（第 {idx} 段）。

请用一句话（不超过 60 字）说明这段内容在文档中的位置和作用：
它在讲什么主题、与前后段落是什么关系（展开/举例/对比/总结/承上启下）。
只输出这一句话，不要加解释、不要加引号。

段落内容：
{text}
"""

_cache = None


def _cache_path():
    return config.DATA_DIR / "ctx_cache.json"


def _load_cache():
    global _cache
    if _cache is None:
        p = _cache_path()
        _cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache


def _save_cache():
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_load_cache(), ensure_ascii=False, indent=1), encoding="utf-8")


_client_obj = None


def _client():
    global _client_obj
    if _client_obj is None:
        import openai

        config.load_env()
        _client_obj = openai.OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "") or "none",
            timeout=120,
        )
    return _client_obj


def block_context(block, use_cache=True):
    """为单个 block 生成定位上下文句。"""
    cache = _load_cache() if use_cache else {}
    bid = block["block_id"]
    if bid in cache:
        return cache[bid]
    doc = Path(block.get("doc_path", "")).stem or block.get("doc_id", "")
    resp = _client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", ""),
        messages=[{"role": "user", "content": PROMPT.format(
            doc=doc, section=block.get("section", ""),
            idx=bid.rsplit("-", 1)[-1], text=block["text"][:800])}],
        temperature=0.0,
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip().strip('"「」')
    if not text:
        raise RuntimeError(f"context 生成为空: {bid}")
    if use_cache:
        cache[bid] = text
        _save_cache()
    return text


def build_contexts(blocks, workers=8, use_cache=True):
    """批量生成全部块的上下文，返回 {block_id: ctx}。"""
    cache = _load_cache() if use_cache else {}
    todo = [b for b in blocks if b["block_id"] not in cache]
    print(f"[ctx] 缓存 {len(cache)} 条，待生成 {len(todo)} 条", flush=True)
    t0 = time.time()
    done = [0]

    def work(b):
        try:
            ctx = block_context(b, use_cache=use_cache)
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"[ctx] {done[0]}/{len(todo)} …", flush=True)
            return b["block_id"], ctx
        except Exception as e:  # noqa: BLE001
            print(f"[ctx] 失败 {b['block_id']}: {e}", flush=True)
            return b["block_id"], None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for bid, ctx in ex.map(work, todo):
            if ctx:
                cache[bid] = ctx
    _save_cache()
    out = {b["block_id"]: cache.get(b["block_id"], "") for b in blocks}
    missing = sum(1 for v in out.values() if not v)
    print(f"[ctx] 完成 {len(out) - missing}/{len(blocks)}，缺 {missing}，{time.time() - t0:.0f}s", flush=True)
    return out


def contextualized_blocks(blocks, contexts):
    """返回文本被前置上下文替换的块副本（block_id 不变）。"""
    return [
        dict(b, text=f"[上下文：{contexts.get(b['block_id'], '')}]\n{b['text']}")
        for b in blocks
    ]
