"""cli.py —— 统一命令行入口。

用法：
  python -m src.cli build-index        切块 + 建双索引 + 存盘
  python -m src.cli selftest           机制自测（不依赖 LLM）
  python -m src.cli bench              四组对照主实验
  python -m src.cli probes             工程探针（--only tokenize|threshold|rrf_k|chunk|candidates|embedding）
  python -m src.cli demo "问题"        命令行演示：三种检索并排
  python -m src.cli serve              启动 FastAPI 前端
"""

import json
import sys

from . import config, corpus


def cmd_build_index(args):
    blocks = corpus.build_blocks()
    corpus.save_blocks(blocks)
    print(f"[build-index] 语料 {len(blocks)} 块 -> {config.BLOCKS_PATH}")

    from .index import BM25Index, VectorIndex

    vi = VectorIndex()
    vi.reset()
    vi.add(blocks)
    bi = BM25Index([b["block_id"] for b in blocks], [b["text"] for b in blocks])
    bi.save()
    print(f"[build-index] 向量 {vi.count()} 块，索引 {vi.size_mb():.1f} MB；BM25 {len(bi.block_ids)} 块，{bi.size_mb():.1f} MB")
    return 0


def cmd_selftest(args):
    from .selftest import run_all

    return 0 if run_all() else 1


def cmd_bench(args):
    from .bench import main as bench_main

    return bench_main(args)


def cmd_probes(args):
    from .probes import main as probes_main

    return probes_main(args)


def cmd_demo(args):
    q = " ".join(args) if args else "混合检索怎么融合两路结果？"
    from .index import BM25Index, VectorIndex
    from .retriever import HybridRetriever

    vi = VectorIndex()
    if vi.count() == 0:
        print("索引为空，请先 build-index")
        return 1
    bi = BM25Index.load()
    retr = HybridRetriever(vi, bi)
    blocks = {b["block_id"]: b for b in corpus.load_blocks()["blocks"]}
    for mode in ("vector", "bm25", "rrf"):
        hits = retr.search(q, mode=mode, top_k=5)
        print(f"\n=== {mode} ===")
        for h in hits:
            b = blocks[h["block_id"]]
            print(f"  #{h['rank']} [{h['score']:.4f}] {b['doc_id']} | {b['section'][:40]}")
            print(f"      {b['text'][:80].replace(chr(10), ' ')}…")
    return 0


def cmd_serve(args):
    import uvicorn

    uvicorn.run("src.server:app", host="127.0.0.1", port=8622, reload=False)
    return 0


COMMANDS = {
    "build-index": cmd_build_index,
    "selftest": cmd_selftest,
    "bench": cmd_bench,
    "probes": cmd_probes,
    "demo": cmd_demo,
    "serve": cmd_serve,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config.ensure_dirs()
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    return COMMANDS[cmd](rest) or 0


if __name__ == "__main__":
    sys.exit(main())
