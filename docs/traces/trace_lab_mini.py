# -*- coding: utf-8 -*-
"""追踪实验①：沙箱里的 clean 阶段全机制观察（跑完自清理，不留仓库残留）

观察点（全部运行时包装，零源码改动）：
  P1 clean.base.apply_rules   —— 规则链逐行判定
  P2 沙箱 manifest.json       —— 指纹 + 产物哈希（缓存的三条件）
  P3 沙箱 runs.jsonl          —— run 生命周期（executed → cached → force 重算）
  P4 沙箱 changelog.jsonl     —— 两段式提交后的数据变更账
  P5 force 重算哈希对比       —— 字节级幂等的直接证据
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parent.parent.parent  # docs/traces/x.py → 仓库根
sys.path.insert(0, str(ROOT))  # 按路径执行时仓库根不在 sys.path，先引导
TMP = Path(tempfile.mkdtemp(prefix="lotus-trace-"))
RAW = ROOT / "data" / "raw" / "trace_src"

print("=" * 72)
print(f"[沙箱] {TMP}")

# ---------------------------------------------------------------- P1 探针
import clean.base as base

_orig_apply = base.apply_rules


def traced_apply(row, rules):
    final, changes = _orig_apply(row, rules)
    if final is None:
        verdict = f"drop({changes[-1]['reason']})"
    elif changes:
        verdict = f"transform({changes[-1]['action']}/{changes[-1]['reason']})"
    else:
        verdict = "keep"
    print(f"    [P1 规则链] {row['source_id']:<10} cjk={row['cjk_char_count']:>5} → {verdict}")
    return final, changes


base.apply_rules = traced_apply

# ---------------------------------------------------------------- 迷你 raw（7 行，各行有明确剧本）
from common import ids
from common.textstats import text_stats
from ingest.spine_schema import SPINE_PARQUET_SCHEMA


def row(sid, content, **over):
    s = text_stats(content)
    r = {"doc_id": ids.spine_doc_id("trace_src", sid), "schema_version": "0.3.2",
         "source": "trace_src", "source_id": sid, "url": None, "canonical_url": None,
         "content": content, "content_hash": ids.content_hash(content),
         "content_format": "plain_text", "raw_uri": None,
         "original_encoding": None, "language": "zh",
         "char_count": s["char_count"], "cjk_char_count": s["cjk_char_count"],
         "utf8_byte_count": s["utf8_byte_count"], "source_timestamp": None,
         "timestamp_source": "none", "first_seen_at": "2026-11-06T00:00:00+00:00",
         "genre": "wiki", "license": "CC0", "license_status": "permissive",
         "commercial_use": "allowed",
         "source_metadata": {"page_title": None, "publisher": None, "section": None}}
    r.update(over)
    return r


GOOD = ("这是一篇结构完整的演示正文。第一段陈述事实与定义。\n"
        "第二段展开背景与沿革，内容充实有条理。\n"
        "第三段给出结论并附参考指向。") * 3
MIXED = "这是一点中文。This sentence is mostly English words dragging the ratio down. " * 20

rows = [
    row("good-1", GOOD),                                       # 剧本: keep
    row("dup-1", GOOD),                                        # 剧本: keep（clean 不去重——阶段分工）
    row("short-1", "太短"),                                     # 剧本: drop(too_short)
    row("latin-1", MIXED),                                     # 剧本: drop(lang_mismatch)
    row("moji-1", ("正文内容。" + "\ufffd" * 80) * 20),          # 剧本: drop(mojibake)
    row("pii-1", "登记信息：张三，身份证号 11010519491231002X，电话回访。" + GOOD[:200]),
    # 剧本: transform(redact/pii) —— 唯一不走 drop 的删除类动作
]

part = RAW / "ingest" / "part-00000.parquet"
part.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows, schema=SPINE_PARQUET_SCHEMA), part)
sha = hashlib.sha256(part.read_bytes()).hexdigest()
(RAW / "manifest.json").write_text(json.dumps(
    {"ingest": {"input_fingerprint": "trace-demo", "output_hashes": {
        "part-00000.parquet": sha}}}), encoding="utf-8")
print(f"[布置] 迷你 raw 就绪: 7 行 → data/raw/trace_src/ingest/part-00000.parquet")

# ---------------------------------------------------------------- 隔离账本
import clean.run as crun
import common.changelog as changelog_mod

changelog_mod.CHANGELOG_PATH = TMP / "changelog.jsonl"          # 全局账本 → 沙箱
crun.STAGING_CHANGELOG = TMP / "changelog.staging.jsonl"        # 两段式 part → 沙箱

# ---------------------------------------------------------------- 跑三遍：executed → cached → force
def show_runs():
    events = [json.loads(l) for l in (TMP / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for e in events[-2:]:
        print(f"    [P3 runs] {e['event']:<7} {e['run_id'][:26]:<28} "
              f"{e.get('status', ''):<8} note={e.get('note', '')}")


print("\n" + "=" * 72)
print("[第 1 遍] 首跑（预期: 全部规则实时判定 + transform 真执行）")
r1 = crun.run(sources=("trace_src",), output_root=TMP / "out", runs_path=TMP / "runs.jsonl")
show_runs()

print("\n[第 2 遍] 同命令重跑（预期: cached=True, 规则链零执行, 账本零新增）")
n_before = len((TMP / "changelog.jsonl").read_text(encoding="utf-8").splitlines())
r2 = crun.run(sources=("trace_src",), output_root=TMP / "out", runs_path=TMP / "runs.jsonl")
n_after = len((TMP / "changelog.jsonl").read_text(encoding="utf-8").splitlines())
show_runs()
print(f"    [P4 账本] changelog 行数 {n_before} → {n_after}（两段式+dedupe：重跑不喂胖账本）")

print("\n[第 3 遍] force=True 强制重算（预期: 重算但产物字节一致）")
r3 = crun.run(sources=("trace_src",), output_root=TMP / "out", runs_path=TMP / "runs.jsonl",
              force=True)
h1 = sorted(r1["output_hashes"].items())
h3 = sorted(r3["output_hashes"].items())
print(f"    [P5 幂等] 首跑 vs force 重算哈希逐文件相等: {h1 == h3}")

# ---------------------------------------------------------------- 产物与账本解剖
print("\n" + "=" * 72)
print("[P2 沙箱 manifest.json]（缓存三条件中的两件：指纹 + 产物哈希）")
m = json.loads((TMP / "out" / "manifest.json").read_text(encoding="utf-8"))
for stage, e in m.items():
    print(f"    {stage}: input_fingerprint={e['input_fingerprint'][:24]}…  "
          f"output_hashes={ {k: v[:16] + '…' for k, v in e['output_hashes'].items()} }")

print("\n[P4 沙箱 changelog.jsonl 全文（每条数据去留的账）")
for line in (TMP / "changelog.jsonl").read_text(encoding="utf-8").splitlines():
    rec = json.loads(line)
    print(f"    {rec['doc_id'][:14]}… {rec['action']:<10} {rec['reason']:<14} "
          f"run={rec['run_id'][:20]}… detail={(rec.get('detail') or '')[:48]}")

print(f"\n[暂存区] 两段式 part 文件已随 commit 删除: "
      f"{not (TMP / 'changelog.staging.jsonl').exists()}")

out_part = TMP / "out" / "clean" / "source=trace_src" / "part-00000.parquet"
# 直读文件而非 dataset API：source= 目录名会被 hive 分区推断吃掉，与行内 source 列打架
kept = pq.ParquetFile(out_part).read().to_pylist()
print(f"\n[产物] clean 表幸存 {len(kept)} 行: {[r['source_id'] for r in kept]}")
pii_row = next(r for r in kept if r["source_id"] == "pii-1")
print(f"    pii-1 脱敏后 content 前 60 字: {pii_row['content'][:60]!r}")
print(f"    pii-1 脱敏后 content_hash == ids.content_hash(新 content): "
      f"{pii_row['content_hash'] == ids.content_hash(pii_row['content'])}")

# ---------------------------------------------------------------- 清理
shutil.rmtree(RAW)
print(f"\n[清理] 已移除临时源 data/raw/trace_src（沙箱 {TMP} 留给你翻看）")
