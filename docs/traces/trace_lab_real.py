# -*- coding: utf-8 -*-
"""追踪实验②：真实数据上追踪一篇文档的一生（全程只读）

抓 data/raw/wiki 的第一篇真实文档，沿着七层链路查它每一站的踪迹：
  raw 行 → first_seen 时钟台账 → changelog 治理史 → clean/dedup/quality/mix/release 各表
  → 根 manifest 的逐阶段指纹链 → runs.jsonl 近期运行
"""
import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parent.parent.parent  # docs/traces/x.py → 仓库根
sys.path.insert(0, str(ROOT))  # 按路径执行时仓库根不在 sys.path，先引导


def find_in(layer: str, paths, doc_id: str):
    """在某一层的 part 文件里找 doc_id，返回行 dict 或 None。"""
    for p in paths:
        t = ds.dataset(p, format="parquet").to_table(
            filter=ds.field("doc_id") == doc_id)
        if t.num_rows:
            return t.to_pylist()[0]
    return None


# ---------------------------------------------------------------- 1) 出生：raw 层第一行
raw_part = ROOT / "data" / "raw" / "wiki" / "ingest" / "part-00000.parquet"
doc = next(pq.ParquetFile(raw_part).iter_batches(batch_size=1)).to_pylist()[0]
doc_id = doc["doc_id"]
title = (doc.get("source_metadata") or {}).get("page_title")
print("=" * 72)
print(f"[追踪对象] doc_id={doc_id[:24]}…  title={title!r}")
print(f"           char_count={doc['char_count']}  license={doc['license']}  "
      f"first_seen_at={doc['first_seen_at']}")

# ---------------------------------------------------------------- 2) 时钟台账
print("\n[first_seen 台账] （时钟不进行数据行，落在这本账里）")
for f in sorted((ROOT / "data" / "meta" / "first_seen").glob("*.json")):
    ledger = json.loads(f.read_text(encoding="utf-8"))
    if doc_id in ledger:
        print(f"    {f.name}: {doc_id[:20]}… → {ledger[doc_id]}")

# ---------------------------------------------------------------- 3) 治理史（changelog）
print("\n[changelog 治理史] （这篇文档在数据变更账上的全部记录）")
from common.changelog import query_by_doc
history = query_by_doc(doc_id)
if not history:
    print("    （零记录——它一路放行，没被任何阶段动过；这也是信息）")
for rec in history:
    print(f"    {rec.get('changed_at', '')} {rec['stage']:<9} {rec['action']:<10} "
          f"{rec['reason']:<16} run={rec.get('run_id', '')[:24]}")

# ---------------------------------------------------------------- 4) 七层逐站查踪
print("\n[七层逐站]")
stations = [
    ("clean", [ROOT / "data" / "clean" / "source=wiki" / "part-00000.parquet"], None),
    ("dedup", sorted((ROOT / "data" / "dedup" / "source=wiki").glob("part-*.parquet")), None),
    ("quality", [ROOT / "data" / "quality" / "source=wiki" / "part-00000.parquet"],
     lambda r: {k: r[k] for k in r if k.startswith("quality_")}),
    ("mix", sorted((ROOT / "data" / "mix" / "candidates").glob("part-*.parquet")),
     lambda r: {"token_count": r.get("token_count"), "mix_batch_id": r.get("mix_batch_id")}),
    ("release", sorted((ROOT / "data" / "release").glob("v*/part-*.parquet")),
     lambda r: {"进入发布集": True}),
]
for name, paths, pick in stations:
    hit = find_in(name, paths, doc_id)
    if hit is None:
        print(f"    {name:<9} ✗ 不在此层（被该阶段或上游移除/未采样）")
    else:
        extra = f" {pick(hit)}" if pick else ""
        print(f"    {name:<9} ✓{extra}")

# ---------------------------------------------------------------- 5) 指纹链（根 manifest）
print("\n[根 manifest.json 指纹链] （每一阶段的输入指纹+产物哈希——信任的链条）")
m = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
for stage in [k for k in ("ingest", "clean", "dedup", "quality", "mix", "release", "metrics", "downstream") if k in m]:
    e = m[stage]
    print(f"    {stage:<11} input_fp={e.get('input_fingerprint', '?')[:20]}…  "
          f"产物文件数={len(e.get('output_hashes', {}))}")

# ---------------------------------------------------------------- 6) runs 近期运行
print("\n[runs.jsonl 最近 4 条事件] （每次运行的生死簿）")
events = [json.loads(l) for l in (ROOT / "data" / "meta" / "runs.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
for e in events[-4:]:
    print(f"    {e.get('event'):<7} {e.get('run_id', '')[:30]:<32} "
          f"{e.get('status', ''):<8} {e.get('note', '')}")
