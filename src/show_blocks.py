"""show_blocks.py —— 按 block_id 打印块全文，供人工标注 gold 时核对。"""

import json
import sys

from . import config


def show(block_ids):
    d = json.loads(config.BLOCKS_PATH.read_text(encoding="utf-8"))
    by_id = {b["block_id"]: b for b in d["blocks"]}
    for bid in block_ids:
        b = by_id.get(bid)
        if b is None:
            print(f"### {bid} NOT FOUND")
            continue
        print(f"### {bid} | {b['section']}")
        print(b["text"])
        print()


if __name__ == "__main__":
    show(sys.argv[1:])
