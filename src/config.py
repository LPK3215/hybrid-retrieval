"""config.py —— 路径、环境变量、离线嵌入与重排模型的装配。

所有模型只在本地 HuggingFace 缓存里找（HF_HUB_OFFLINE=1），实验零网络、可复现。
换嵌入或重排模型只改本文件常量。
"""

import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
BLOCKS_PATH = DATA_DIR / "blocks.json"
QUERIES_PATH = DATA_DIR / "queries.json"
RUNS_DIR = ROOT / "runs"
STATIC_DIR = ROOT / "static"

# 本地嵌入模型：中文友好的多语 MiniLM（缓存里已有）。
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 本地重排模型：bge-reranker-base（缓存里已有，CPU 可跑）。
RERANK_MODEL = "BAAI/bge-reranker-base"


def load_env(path=None):
    """从项目 .env 读配置；已存在的环境变量不覆盖。"""
    if path is None:
        path = ROOT / ".env"
    path = Path(path)
    if not path.exists():
        return None
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return path


def _local_snapshot_dir(repo_id, required_files):
    """取本地缓存的快照目录；缓存缺失时回退 repo id（可能联网）。

    required_files 支持多个候选锚点文件：有的模型仓不带
    config_sentence_transformers.json（如 m3e-small），退到 config.json。
    """
    if isinstance(required_files, str):
        required_files = [required_files]
    try:
        from huggingface_hub import try_to_load_from_cache

        for name in required_files:
            p = try_to_load_from_cache(repo_id, name)
            if p and os.path.exists(p):
                return os.path.dirname(p)
    except Exception:
        pass
    return repo_id


_embedders = {}


def make_embedder(model=None, query_prefix=None):
    """按 repo_id 懒加载嵌入模型，同 (model, query_prefix) 只加载一次。

    返回 (SentenceTransformer, query_prefix)：query 前缀供查询侧选用
    （bge-small-zh 按官方口径检索任务给 query 加指令，文档侧不加）。
    """
    repo = model or EMBED_MODEL
    key = (repo, query_prefix or "")
    if key not in _embedders:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(
            _local_snapshot_dir(
                repo,
                ["config_sentence_transformers.json", "config.json"],
            ),
            local_files_only=True,
        )
        _embedders[key] = (m, query_prefix or "")
    return _embedders[key]


def _encode(model, texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def build_embedder():
    """默认嵌入（config.EMBED_MODEL，无 query 前缀），兼容旧调用。"""
    m, _prefix = make_embedder()
    return lambda texts: _encode(m, texts)


def make_embedding_function(model=None, query_prefix=None):
    """适配 chromadb 的 embedding function 接口；query 前缀只加在查询侧。"""
    m, prefix = make_embedder(model, query_prefix)

    class _LocalEmbeddingFunction:
        def embed_documents(self, input):
            return [e.tolist() for e in _encode(m, input)]

        def embed_query(self, input):
            texts = [prefix + t for t in input] if prefix else input
            return [e.tolist() for e in _encode(m, texts)]

        def __call__(self, input):
            return self.embed_documents(input)

        def name(self):
            return model or EMBED_MODEL

    return _LocalEmbeddingFunction()


def build_embedding_function():
    """默认 embedding function（config.EMBED_MODEL，无 query 前缀）。"""
    return make_embedding_function()


_reranker = None


def build_reranker():
    """懒加载本地重排模型（bge-reranker-base，CPU）。"""
    global _reranker
    if _reranker is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from FlagEmbedding import FlagReranker

        path = _local_snapshot_dir(RERANK_MODEL, "config.json")
        _reranker = FlagReranker(path, use_cpu=True)
    return _reranker


def ensure_dirs():
    for d in (DATA_DIR, RUNS_DIR, STATIC_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
