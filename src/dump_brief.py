"""dump_brief.py —— 导出块摘要（block_id / section / 前 150 字），供标注测试集时人工核对。"""

import json
from pathlib import Path

from . import config


def main():
    d = json.loads(config.BLOCKS_PATH.read_text(encoding="utf-8"))
    lines = []
    for b in d["blocks"]:
        text = " ".join(b["text"].split())
        lines.append(f"{b['block_id']} | {b['section'][:44]} | {text[:150]}")
    out = config.DATA_DIR / "block_brief.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {len(lines)} -> {out}")


if __name__ == "__main__":
    main()
