"""dashboard/preview.py 与 dashboard/knobs.py 的测试（控制台旋钮）

锁住三个关键行为：
1. 旋钮写回有范围校验（越界拦截）；
2. 预览只读——不写 configs、不污染全局 policy 缓存；
3. 预览删除率方向正确（门槛调高 → 删更多）。
"""

import json
from pathlib import Path

import pytest

from dashboard import knobs
from dashboard.preview import run_preview, read_knobs, apply_overrides, _load_cfg

ROOT = Path(__file__).resolve().parent.parent

requires_raw = pytest.mark.skipif(
    not any((ROOT / "data" / "raw").glob("*/ingest")),
    reason="raw 层产物缺失（data/ 不进仓库；先跑 ingest）")


def test_knob_bounds_enforced():
    """越界值必须被拦，且不落盘。"""
    # min_han_ratio 必须在 [0,1]
    with pytest.raises(ValueError):
        knobs.write_knob("clean", "min_han_ratio", 5.0)
    with pytest.raises(ValueError):
        knobs.write_knob("clean", "min_han_ratio", -0.1)
    # min_cjk_chars 必须在 [1,1000]
    with pytest.raises(ValueError):
        knobs.write_knob("clean", "min_cjk_chars", 0)
    # 未知旋钮
    with pytest.raises(ValueError):
        knobs.write_knob("clean", "nonexistent_knob", 1)


def test_knob_write_roundtrip(monkeypatch):
    """写回 → 读回一致；且不破坏 config 其他字段。"""
    orig = _load_cfg("clean")
    knobs.write_knob("clean", "min_cjk_chars", 99)
    assert read_knobs()["clean"]["min_cjk_chars"] == 99
    # 其他字段原样
    assert _load_cfg("clean")["too_short"]["reason"] == "too_short"
    # 恢复
    knobs.write_knob("clean", "min_cjk_chars", orig["too_short"]["min_cjk_chars"])
    assert read_knobs()["clean"]["min_cjk_chars"] == orig["too_short"]["min_cjk_chars"]


def test_preview_is_read_only():
    """预览不改 config 文件（只读约束）。"""
    before = _load_cfg("clean")
    run_preview(overrides={"clean": {"min_cjk_chars": 500}}, n_sample=50)
    after = _load_cfg("clean")
    assert before == after, "预览不应修改 configs/*.json"


@requires_raw
def test_preview_higher_threshold_removes_more():
    """门槛调高 → 删除更多（方向正确性，用真实 raw 样本）。"""
    r_base = run_preview(overrides=None, n_sample=2000)
    r_tight = run_preview(overrides={"clean": {"min_cjk_chars": 50}}, n_sample=2000)
    assert r_tight["clean"]["removed"] >= r_base["clean"]["removed"], (
        "门槛调高应删除不减少")


def test_apply_overrides_does_not_mutate_original():
    """apply_overrides 深拷贝，不改原 dict。"""
    cfg = _load_cfg("clean")
    orig_min = cfg["too_short"]["min_cjk_chars"]
    new = apply_overrides(cfg, {"min_cjk_chars": 999}, "clean")
    assert new["too_short"]["min_cjk_chars"] == 999
    assert cfg["too_short"]["min_cjk_chars"] == orig_min, "原 dict 不应被改"
