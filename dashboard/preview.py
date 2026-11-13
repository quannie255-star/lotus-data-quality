"""dashboard/preview.py — 旋钮「秒级预览」引擎（控制台专用，只读不落盘）

给外行人「拧一下旋钮立刻看到效果」的能力：从 raw 层抽 N 行样本，用「当前旋钮值
（可覆盖）」跑 clean 规则链 + quality 打分，返回真实删除率/放行率，秒级出结果。

设计纪律（与全量 pipeline 的边界）：
- 只读：绝不写任何产物、绝不碰 changelog、绝不碰 runs.jsonl——预览不是一次 run。
- 不污染全局缓存：clean.base.policy() 与 quality.engine.policy() 都是 lru_cache，
  预览直接构造「覆盖后的 policy dict」传给纯函数，绕开缓存，不改文件、不清缓存。
- 确定性：抽样用固定 seed，同旋钮同输入 → 同预览数字（可复现，与项目宪法一致）。
- 预览只跑 clean 规则 + quality 打分，跳过 dedup 的 MinHash（那才是 27–75 分钟的
  大头）——预览回答的是「这条规则会删多少 / 放行多少」，不是「去重后剩多少」。

用法（纯函数，可单测）：
    from dashboard.preview import run_preview
    result = run_preview(overrides={"clean": {"too_short": {"min_cjk_chars": 20}}},
                         n_sample=10000)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"

# 预览开放的旋钮：{类别: {旋钮键: 配置路径}}。只暴露「外行人能看懂」的阈值。
# 内部技术参数（num_perm/bands/fffd 密度）不进预览，避免误改。
KNOB_SPEC = {
    "clean": {
        "min_cjk_chars": ("too_short", "min_cjk_chars", "过短门槛（正文字数下限）"),
        "min_han_ratio": ("lang_ratio", "min_han_ratio", "中文占比下限（0~1）"),
        "max_link_density": ("boilerplate", "max_link_density", "链接密度上限（0~1）"),
    },
    "dedup": {
        "verify_threshold": ("verify_threshold", "verify_threshold", "近重复相似度阈值（0~1）"),
    },
    "quality": {
        "min_total_score": ("gate", "min_total_score", "门禁放行总分线（0~1）"),
        "min_dim_score": ("gate", "min_dim_score", "单维分数下限（0~1）"),
    },
}


def _load_cfg(name: str) -> dict:
    return json.loads((CONFIGS / f"{name}_policy.json").read_text(encoding="utf-8"))


def _deep_get(cfg: dict, path: tuple[str, ...]) -> dict:
    """沿 path 取到「叶子所在的父容器」，返回 (容器dict, 叶子key)。"""
    node = cfg
    for key in path[:-1]:
        node = node[key]
    return node, path[-1]


def read_knobs() -> dict:
    """读当前三类旋钮的现值 → {类别: {旋钮键: 值}}。"""
    out: dict = {}
    for cat, knobs in KNOB_SPEC.items():
        cfg = _load_cfg(cat)
        out[cat] = {}
        for knob, (container, leaf, _desc) in knobs.items():
            if cat == "dedup":
                out[cat][knob] = cfg.get(container, {}).get(leaf) if isinstance(
                    cfg.get(container), dict) else cfg.get(leaf)
            else:
                node, key = _deep_get(cfg, (container, leaf))
                out[cat][knob] = node[key]
    return out


def apply_overrides(cfg: dict, overrides: dict, category: str) -> dict:
    """把旋钮覆盖写进一份 config 的深拷贝（不碰原文件）。overrides = {旋钮键: 新值}。"""
    import copy
    new = copy.deepcopy(cfg)
    for knob, value in (overrides or {}).items():
        spec = KNOB_SPEC.get(category, {}).get(knob)
        if spec is None:
            continue
        container, leaf, _desc = spec
        if category == "dedup":
            new[leaf] = value
        else:
            node, key = _deep_get(new, (container, leaf))
            node[key] = value
    return new


def _sample_raw_rows(n: int, seed: int = 2026) -> Iterator[dict]:
    """从 raw 层确定性抽 n 行（跨源按顺序轮转读，够 n 停）。"""
    raw_dirs = sorted((DATA / "raw").glob("*/ingest"))
    yielded = 0
    for d in raw_dirs:
        for part in sorted(d.glob("part-*.parquet")):
            for batch in pq.ParquetFile(part).iter_batches(batch_size=1000):
                rows = batch.to_pylist()
                for r in rows:
                    if yielded < n:
                        yield r
                        yielded += 1
                    else:
                        return
                if yielded >= n:
                    return


def run_preview(overrides: dict | None = None, n_sample: int = 10000,
                seed: int = 2026) -> dict:
    """跑预览。返回统计 dict，供控制台展示。

    overrides: {类别: {旋钮键: 新值}}，None = 用当前配置。
    返回: {n_sample, clean: {kept, removed, by_rule, pii}, quality: {pass, hold, remove}}
    """
    from clean import rules as rule_fns
    from clean.base import apply_rules
    from quality.engine import score_row, gate as quality_gate

    clean_cfg = apply_overrides(_load_cfg("clean"), (overrides or {}).get("clean"), "clean")
    qual_cfg = apply_overrides(_load_cfg("quality"), (overrides or {}).get("quality"), "quality")

    # 按 rules_order 装配规则（规则函数本体无状态，直接复用）
    rule_loaded = []
    for name in clean_cfg["rules_order"]:
        fn = getattr(rule_fns, name, None)
        if fn is None:
            continue
        rule_loaded.append((name, fn))

    clean_stats = {"kept": 0, "removed": 0, "by_rule": {}, "pii": 0}
    qual_stats = {"pass": 0, "hold": 0, "remove": 0}
    qual_weights = qual_cfg["weights"]

    for row in _sample_raw_rows(n_sample, seed):
        # ---- clean 规则链（直接传覆盖后的 cfg，绕开全局 policy 缓存）----
        final, changes = apply_rules_with_cfg(row, rule_loaded, clean_cfg)
        if final is None:
            clean_stats["removed"] += 1
            rule = changes[-1]["rule"] if changes else "?"
            clean_stats["by_rule"][rule] = clean_stats["by_rule"].get(rule, 0) + 1
            continue
        clean_stats["kept"] += 1
        for ch in changes:
            if ch.get("action") == "redact":
                clean_stats["pii"] += 1

        # ---- quality 打分 + 门禁（用覆盖后的 weights/gate）----
        scores = _score_with_cfg(final, qual_weights, qual_cfg)
        verdict, _reason = quality_gate_with_cfg(scores, qual_cfg["gate"])
        qual_stats[verdict] = qual_stats.get(verdict, 0) + 1

    return {
        "n_sample": n_sample,
        "clean": clean_stats,
        "quality": qual_stats,
    }


def apply_rules_with_cfg(row: dict, rules: list, clean_cfg: dict) -> tuple:
    """apply_rules 的预览版：cfg 从参数传入，不读全局 policy()。"""
    changes = []
    for name, fn in rules:
        verdict = fn(row, clean_cfg[name])
        kind = verdict[0]
        if kind == "keep":
            continue
        if kind == "drop":
            changes.append({"rule": name, "reason": verdict[1], "action": "remove"})
            return None, changes
        if kind == "transform":
            _, new_row, reason, action, meta = verdict
            changes.append({"rule": name, "reason": reason, "action": action,
                            "meta": meta or {}})
            row = new_row
            continue
        raise ValueError(f"规则 {name} 返回非法判定 {verdict!r}")
    return row, changes


def _score_with_cfg(row: dict, weights: dict, qual_cfg: dict) -> dict:
    """quality 打分预览版：weights/gate 从参数传入，不读全局 policy()。

    注意：uniqueness 需要 dedup 签名（跨阶段复用），预览里没有 → 用 1.0 占位并标注；
    non_pollution 需要 benchmark 索引，未就绪时该维=1.0（与全量行为一致）。
    """
    from quality import engine as qe

    dims = {
        "quality_completeness": qe.score_completeness(row),
        "quality_uniqueness": qe.score_uniqueness(row, None),  # 预览无签名，uniqueness 退化为 1.0
        "quality_consistency": qe.score_consistency(row),
        "quality_non_pollution": qe.score_non_pollution(row, None),  # 无 benchmark 索引 → 1.0
        "quality_length": qe.score_length(row),
    }
    total = sum(dims[k] * weights[k.split("_", 1)[1]] for k in dims)
    return {**dims, "quality_score": round(total, 4)}


def quality_gate_with_cfg(scores: dict, gate_cfg: dict) -> tuple:
    """门禁预览版：gate 配置从参数传入。"""
    band_lo, band_hi = gate_cfg["quarantine_band"]
    if scores["quality_score"] < band_lo or any(
            scores[k] < gate_cfg["min_dim_score"] for k in scores
            if k.startswith("quality_") and k != "quality_score"):
        return "remove", "low_quality"
    if scores["quality_score"] < gate_cfg["min_total_score"] or scores["quality_score"] < band_hi:
        return "hold", "quarantine_borderline"
    return "pass", None
