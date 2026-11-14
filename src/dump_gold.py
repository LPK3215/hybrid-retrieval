"""导出候选 gold 块全文到文件，供人工标注核对。"""

from . import config
from .show_blocks import show

BATCH1 = [
    "rag-total-0040", "rag-total-0041", "rag-total-0047", "rag-total-0051",
    "rag-total-0054", "rag-total-0055",
    "chroma-0049", "chroma-0017",
    "flow-control-0033", "flow-control-0047",
    "qdrant-0009", "qdrant-0010",
    "rag-total-0044", "rag-total-0081",
    "rag-total-0060", "chroma-0065",
    "flow-control-0031", "flow-control-0024", "flow-control-0025", "flow-control-0051",
    "rag-total-0095", "retrieval-paradigm-0014",
    "chroma-0022", "chroma-0092",
    "retrieval-paradigm-0015", "rag-total-0098",
]

BATCH2 = [
    "chroma-0000", "chroma-0004", "chroma-0104",
    "rag-total-0058", "rag-total-0063", "kb-rag-0032",
    "kb-rag-0026", "kb-rag-0036", "rag-total-0029",
    "kb-rag-0034", "retrieval-paradigm-0016",
    "kb-rag-0006", "kb-rag-0022", "rag-total-0023",
    "kb-rag-0010", "kb-rag-0031", "rag-total-0045",
    "kb-rag-0028", "rag-total-0052", "kb-rag-0027",
]


def main():
    import io

    for name, ids in (("batch1", BATCH1), ("batch2", BATCH2)):
        buf = io.StringIO()
        import contextlib

        with contextlib.redirect_stdout(buf):
            show(ids)
        (config.DATA_DIR / f"gold_{name}.txt").write_text(buf.getvalue(), encoding="utf-8")
        print(f"written gold_{name}.txt")


if __name__ == "__main__":
    main()
