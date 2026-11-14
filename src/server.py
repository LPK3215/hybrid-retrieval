"""server.py —— D04 混合检索工具网关。

端点：
  GET  /                  → 前端单页（static/index.html）
  GET  /api/health        → 索引状态与模型信息
  GET  /api/queries       → 标注测试集（供前端匹配 gold、实时算指标）
  POST /api/search        → {"q","mode","top_k"} → 命中块 + 时延
  GET  /api/runs          → runs/ 下最近一次 bench summary
  GET  /api/blocks        → 块抽样（doc_id / section / 前 200 字）

模式：vector / bm25 / rrf / rrf_rerank。重排模型首次请求时懒加载。
"""

import json
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, corpus
from .index import BM25Index, VectorIndex
from .retriever import HybridRetriever

app = FastAPI(title="D04 混合检索工具网关", version="1.0.0", docs_url=None, redoc_url=None)

_retr = None
_reranker = None


def get_retriever() -> HybridRetriever:
    global _retr
    if _retr is None:
        vi = VectorIndex()
        try:
            bi = BM25Index.load()
        except FileNotFoundError:
            blocks = corpus.load_blocks()["blocks"]
            bi = BM25Index([b["block_id"] for b in blocks], [b["text"] for b in blocks])
        _retr = HybridRetriever(vi, bi)
    return _retr


def load_blocks_map():
    return {b["block_id"]: b for b in corpus.load_blocks()["blocks"]}


@app.get("/api/health")
def health():
    r = get_retriever()
    return {
        "status": "ok",
        "blocks": r.vi.count(),
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/queries")
def queries():
    return json.loads(config.QUERIES_PATH.read_text(encoding="utf-8"))


@app.get("/api/blocks")
def blocks(doc_id: str = "", limit: int = 20):
    all_blocks = corpus.load_blocks()["blocks"]
    picked = [b for b in all_blocks if not doc_id or b["doc_id"] == doc_id]
    return {
        "total": len(all_blocks),
        "doc_ids": sorted({b["doc_id"] for b in all_blocks}),
        "blocks": [
            {
                "block_id": b["block_id"],
                "doc_id": b["doc_id"],
                "section": b["section"],
                "text": b["text"][:200],
            }
            for b in picked[:limit]
        ],
    }


@app.post("/api/search")
def search(payload: dict):
    q = (payload.get("q") or "").strip()
    if not q:
        return {"error": "empty query"}
    mode = payload.get("mode", "rrf")
    if mode not in ("vector", "bm25", "rrf", "rrf_rerank"):
        return {"error": f"unknown mode: {mode}"}
    top_k = max(1, min(int(payload.get("top_k", 5)), 50))

    r = get_retriever()
    global _reranker
    if mode == "rrf_rerank":
        if _reranker is None:
            _reranker = config.build_reranker()
        r.reranker = _reranker
    elif r.reranker is not None:
        r.reranker = None  # 单路/融合模式不挂重排，避免无效加载

    t0 = time.perf_counter()
    hits = r.search(q, mode=mode, top_k=top_k)
    lat_ms = round((time.perf_counter() - t0) * 1000, 1)

    bmap = load_blocks_map()
    out = []
    for h in hits:
        b = bmap.get(h["block_id"], {})
        out.append(
            {
                "block_id": h["block_id"],
                "rank": h["rank"],
                "score": h["score"],
                "source": h["source"],
                "doc_id": b.get("doc_id", ""),
                "section": b.get("section", ""),
                "text": b.get("text", "")[:420],
            }
        )
    return {"q": q, "mode": mode, "top_k": top_k, "lat_ms": lat_ms, "results": out}


@app.get("/api/runs")
def runs():
    """汇总 runs/ 下最近一次 bench 的 summary（供前端展示基线数字）。"""
    runs_dir = config.RUNS_DIR
    out = []
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir(), reverse=True):
            s = d / "summary.json"
            if s.exists():
                out.append({"dir": d.name, "summary": json.loads(s.read_text(encoding="utf-8"))})
    return {"runs": out[:5]}


@app.get("/")
def index_page():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
