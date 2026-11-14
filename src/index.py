"""index.py —— 双索引：Chroma 向量索引 + BM25 稀疏索引。

向量路：本地多语 MiniLM + chromadb（持久化到 data/chroma）。
稀疏路：jieba 分词 + rank_bm25 BM25Okapi（pickle 持久化 token 化结果，重建只要毫秒）。
"""

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from . import config

# 领域专有词：中文 BM25 不分词或不带词典时，业务黑话命中不了。
DOMAIN_WORDS = [
    "混合检索", "重排", "召回率", "向量库", "倒排索引", "检查点", "幂等",
    "多租户", "语义检索", "相关性", "检索增强", "知识库", "嵌入模型",
    "上下文", "提示词", "智能体", "工作流", "容错", "降级", "熔断",
    "命中率", "召回", "融合", "阈值", "分块", "索引", "语料",
]


def tokenize(text, dict_words=None, use_jieba=True):
    """中文分词。use_jieba=False 时退化为单字切分（对照实验用）。"""
    if not use_jieba:
        return list(text)
    for w in dict_words or []:
        jieba.add_word(w)
    return [t for t in jieba.lcut(text) if t.strip()]


class VectorIndex:
    def __init__(self, chroma_dir=None, collection="corpus", embedding_function=None):
        import chromadb

        self.chroma_dir = str(chroma_dir or config.CHROMA_DIR)
        self.collection_name = collection
        self._embedding_function = (
            embedding_function or config.build_embedding_function()
        )
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        """删除并重建 collection。

        必须显式调用：Chroma 对已存在 ID 的 add 是覆盖语义（probes 重建索引
        时复用同名 collection 会把不同尺寸的块混进同一索引），任何重建索引的
        入口都应先 reset，保证 collection 与 blocks.json 一一对应。
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, blocks):
        ids = [b["block_id"] for b in blocks]
        docs = [b["text"] for b in blocks]
        metas = [
            {"doc_id": b["doc_id"], "doc_path": b.get("doc_path", ""), "section": b.get("section", "")}
            for b in blocks
        ]
        self.collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def count(self):
        return self.collection.count()

    def search(self, q, k, min_score=None):
        """返回 [(block_id, score)]，score = 1 - cosine 距离，越大越相关。"""
        res = self.collection.query(query_texts=[q], n_results=min(k, self.count()))
        out = []
        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [1.0] * len(ids)
        for bid, d in zip(ids, dists):
            score = 1.0 - float(d)
            if min_score is not None and score < min_score:
                continue
            out.append((bid, score))
        return out

    def size_mb(self):
        total = 0
        for p in Path(self.chroma_dir).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / 1024 / 1024


class BM25Index:
    def __init__(self, block_ids=None, texts=None, dict_words=None, use_jieba=True):
        self.block_ids = list(block_ids or [])
        self.texts = list(texts or [])
        self.dict_words = dict_words
        self.use_jieba = use_jieba
        self.bm25 = None
        if self.block_ids:
            self._fit()

    def _fit(self):
        self.tokens = [tokenize(t, self.dict_words, self.use_jieba) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)

    def add(self, block_ids, texts):
        self.block_ids.extend(block_ids)
        self.texts.extend(texts)
        self._fit()

    def search(self, q, k, min_ratio=0.0):
        """返回 [(block_id, score)]。min_ratio=0 表示只用 top-k，不设相对阈值。"""
        if self.bm25 is None or not self.block_ids:
            return []
        scores = self.bm25.get_scores(tokenize(q, self.dict_words, self.use_jieba))
        best = float(scores.max()) if len(scores) else 0.0
        if best <= 0:
            return []  # 全零必须提前返回，否则阈值失效
        limit = best * min_ratio if min_ratio > 0 else 0.0
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] < limit:
                break
            out.append((self.block_ids[i], float(scores[i])))
            if len(out) >= k:
                break
        return out

    def save(self, path=None):
        path = Path(path) if path else config.DATA_DIR / "bm25.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "block_ids": self.block_ids,
                    "texts": self.texts,
                    "dict_words": self.dict_words,
                    "use_jieba": self.use_jieba,
                },
                f,
            )

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else config.DATA_DIR / "bm25.pkl"
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["block_ids"], d["texts"], d.get("dict_words"), d.get("use_jieba", True))
        return obj

    def size_mb(self):
        path = Path(config.DATA_DIR) / "bm25.pkl"
        return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0
