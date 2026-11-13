"""dashboard/autotune.py 的测试（带约束的旋钮自动搜索）

锁住四个关键行为：
1. 候选 #0 恒为基线，预算（总评估次数）不被突破；
2. 约束真的在拦：高平均分但保留率不达标的候选不能当选——
   把 `feasible` 判定写成恒真（或去掉 min_keep 比较）本测试会变红；
3. 可行性/目标值全部未达标时，best=None 且不编造推荐值；
4. 同 seed 同网格 → 同一份结果（确定性，可复现）。

另有一条真实数据冒烟（raw 缺失时跳过）：搜索不写 configs，结果可与基线对照。
"""

import json
from pathlib import Path

import pytest

from dashboard import autotune
from dashboard.preview import _load_cfg, read_knobs

ROOT = Path(__file__).resolve().parent.parent

requires_raw = pytest.mark.skipif(
    not any((ROOT / "data" / "raw").glob("*/ingest")),
    reason="raw 层产物缺失（data/ 不进仓库；先跑 ingest）")


def _fake_evaluate(high_quality_threshold: int = 4):
    """门槛高于阈值 → 少放行但平均分高；否则多放行但平均分低。"""
    def evaluate(overrides):
        t = overrides.get("clean", {}).get("min_cjk_chars", 1)
        if t >= high_quality_threshold:
            return {"n": 100, "pass": 30, "score_sum": 27.0}   # 0.90 / 0.30
        return {"n": 100, "pass": 80, "score_sum": 60.0}       # 0.75 / 0.80
    return evaluate


def test_baseline_first_and_budget_respected():
    """候选 #0 是基线；评估次数不超过预算。"""
    calls = []

    def evaluate(overrides):
        calls.append(overrides)
        return {"n": 100, "pass": 60, "score_sum": 45.0}

    result = autotune.search(
        evaluate, grid={("clean", "min_cjk_chars"): [1, 2, 3, 4, 5]}, budget=4)
    assert len(result["history"]) == 4 == len(calls)
    assert result["history"][0]["baseline"] is True
    assert result["history"][0]["overrides"] == {}
    assert result["best"]["index"] == 0  # 全部同分 → 基线（保留率并列时先到者胜）


def test_constraint_blocks_high_quality_low_keep():
    """约束真的在拦：高平均分但保留率 0.30 < 0.5 的候选不能当选。

    变异证据：把 search() 里的 `keep_rate >= min_keep` 改成恒真，
    这里的最优会变成候选 #?（threshold>=4，mean 0.90），断言当场变红。
    """
    result = autotune.search(
        _fake_evaluate(), grid={("clean", "min_cjk_chars"): [1, 4, 5]},
        budget=10, min_keep=0.5)
    best = result["best"]
    assert best is not None
    assert best["keep_rate"] >= 0.5
    assert best["mean_quality"] == 0.75
    assert best["overrides"].get("clean", {}).get("min_cjk_chars", 1) < 4


def test_no_feasible_candidate_returns_none():
    """全部候选不可行 → best=None，不编造推荐值。"""
    def evaluate(overrides):
        return {"n": 100, "pass": 10, "score_sum": 9.0}  # 保留率 0.10

    result = autotune.search(
        evaluate, grid={("clean", "min_cjk_chars"): [10, 20]},
        budget=5, min_keep=0.5)
    assert result["best"] is None
    assert result["improvement"]["mean_quality"] is None
    assert all(not rec["feasible"] for rec in result["history"])


def test_deterministic_same_seed():
    """同 seed 同网格 → 完全相同的候选顺序与结果。"""
    grid = {("clean", "min_cjk_chars"): [1, 2, 3, 4, 5, 6, 7]}
    r1 = autotune.search(_fake_evaluate(), grid=grid, budget=5, seed=7)
    r2 = autotune.search(_fake_evaluate(), grid=grid, budget=5, seed=7)
    assert json.dumps(r1, ensure_ascii=False, sort_keys=True) == \
        json.dumps(r2, ensure_ascii=False, sort_keys=True)
    r3 = autotune.search(_fake_evaluate(), grid=grid, budget=5, seed=8)
    assert [r["overrides"] for r in r1["history"]] != [r["overrides"] for r in r3["history"]]


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        autotune.search(_fake_evaluate(), budget=1)
    with pytest.raises(ValueError):
        autotune.search(_fake_evaluate(), budget=2, min_keep=1.5)


@requires_raw
def test_preview_wiring_and_read_only():
    """真实 raw 上端到端：结果含基线、样本数一致；不写 configs。"""
    before = {name: _load_cfg(name) for name in ("clean", "quality")}
    result = autotune.search(
        lambda ov: autotune.evaluate_preview(ov, n_sample=300, seed=2026),
        grid={("quality", "min_total_score"): [0.5]}, budget=2, min_keep=0.0)
    after = {name: _load_cfg(name) for name in ("clean", "quality")}
    assert before == after, "自动搜索不应修改 configs/*.json"
    assert result["history"][0]["n"] > 0
    assert result["history"][0]["n"] == result["history"][1]["n"], "同一候选集必须读同一份样本"
    assert read_knobs()["quality"]["min_total_score"] == before["quality"]["gate"]["min_total_score"]
