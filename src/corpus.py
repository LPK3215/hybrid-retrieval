"""corpus.py —— 读真实语料、清洗、切块，产出 blocks.json。

语料来自笔记库的真实技术文档（见 data/corpus/README 记录来源）。
清洗只去 Obsidian 语法与图片，不改写正文；切块按标题层级 + 长度上限。
"""

import json
import re
from pathlib import Path

from . import config

# 切块默认参数（可被 bench/probes 覆盖）
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

_WIKILINK = re.compile(r"!?\[\[[^\]]+\]\]")
_BLOCKREF = re.compile(r"\^[A-Za-z0-9_-]+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"<[^>]+>")
_HRULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.M)
_BLANK_RUN = re.compile(r"\n{3,}")

# 代码块围栏：切块时先换成占位符，避免块内 # 注释被当标题、块内空行被当段落边界
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_CODE_PLACEHOLDER = "\x00CODE{}\x00"

MIN_CHUNK_CHARS = 40  # 短于该长度的块对检索是噪声（标题碎片、表格行）


def clean_text(text: str) -> str:
    """去 Obsidian 语法与图片，压缩多余空行，保留 markdown 结构。"""
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _IMAGE.sub("", text)
    text = re.sub(r"!?\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = _BLOCKREF.sub("", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)  # 引用前缀
    text = _HRULE.sub("", text)  # 分隔线
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def _sections(lines):
    """按 markdown 标题切出 (section_path, text) 列表。"""
    out = []
    path = []
    buf = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if buf:
                out.append((" / ".join(path) or "(前言)", "".join(buf)))
                buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            path = path[: level - 1] + [title]
        else:
            buf.append(line)
    if buf:
        out.append((" / ".join(path) or "(前言)", "".join(buf)))
    return [o for o in out if o[1].strip()]


def split_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """标题感知切分：先按标题分节，节内按长度上限切块，允许 overlap 字重叠。

    - 代码块先换成占位符：块内 # 不被当标题、块内空行不拆块；
    - 段落超过 size 时按滑动窗口硬切，保证没有超长块；
    - 短于 MIN_CHUNK_CHARS 的碎片直接丢弃（标题碎片、表格行）。
    """
    code_blocks = []

    def _stash(m):
        idx = len(code_blocks)
        code_blocks.append(m.group(0))
        return _CODE_PLACEHOLDER.format(idx)

    protected = _CODE_FENCE.sub(_stash, text)

    blocks = []
    for section, body in _sections(protected.splitlines(keepends=True)):
        buf = ""
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if not buf:
                buf = para
            elif len(buf) + 2 + len(para) <= size:
                buf = buf + "\n\n" + para
            else:
                blocks.append((section, buf))
                tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
                buf = (tail + "\n\n" + para).strip() if tail else para
        if buf:
            blocks.append((section, buf))

    def _restore(s):
        return re.sub(r"\x00CODE(\d+)\x00", lambda m: code_blocks[int(m.group(1))], s)

    final = []
    for section, chunk in blocks:
        chunk = _restore(chunk)
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        if len(chunk) <= size:
            final.append((section, chunk))
            continue
        start = 0
        step = max(size - overlap, 1)
        while start < len(chunk):
            final.append((section, chunk[start:start + size]))
            start += step
    return [b for b in final if b[1].strip()]


def load_corpus(corpus_dir=None):
    """读语料目录，返回清洗后的 (doc_id, doc_path, text) 列表。"""
    d = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    docs = []
    for p in sorted(d.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        docs.append((p.stem, str(p.relative_to(d.parent.parent)), clean_text(raw)))
    return docs


def build_blocks(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP, corpus_dir=None):
    """产出去重后的块列表：每块含 block_id / doc_id / doc_path / section / text。"""
    docs = load_corpus(corpus_dir)
    blocks = []
    seen = set()
    for doc_id, doc_path, text in docs:
        for i, (section, chunk) in enumerate(split_text(text, chunk_size, overlap)):
            if chunk in seen:
                continue
            seen.add(chunk)
            blocks.append(
                {
                    "block_id": f"{doc_id}-{i:04d}",
                    "doc_id": doc_id,
                    "doc_path": doc_path,
                    "section": section,
                    "text": chunk,
                }
            )
    return blocks


def save_blocks(blocks, path=None):
    path = Path(path) if path else config.BLOCKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "overlap": DEFAULT_OVERLAP,
                "count": len(blocks),
                "blocks": blocks,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def load_blocks(path=None):
    path = Path(path) if path else config.BLOCKS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_manifest(corpus_dir=None):
    """语料清单：文件名 -> 字符数，写进实验 meta。"""
    docs = load_corpus(corpus_dir)
    return [
        {"doc_id": d[0], "path": d[1], "chars": len(d[2])} for d in docs
    ]
