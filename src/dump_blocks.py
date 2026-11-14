"""临时工具：导出块清单文本，供人工标注测试集。用后删除。"""

import json

from . import config

d = json.loads((config.BLOCKS_PATH).read_text(encoding="utf-8"))
out = []
for x in d["blocks"]:
    t = x["text"].replace("\n", " ¶ ")
    out.append(f"{x['block_id']} [{x['doc_id']}] {x['section'][:46]} :: {t[:130]}")
(config.DATA_DIR / "block_dump.txt").write_text("\n".join(out), encoding="utf-8")
print(f"dumped {len(out)} blocks -> data/block_dump.txt")
