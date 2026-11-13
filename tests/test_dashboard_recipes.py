"""dashboard 三档配方 + 报告导出测试

覆盖：
- 配方文件合法性（三档全过校验）+ 变异证据（未知键/越界值必须红）
- diff/apply 的行为（apply 不落真配置——猴补丁捕获）
- 报告 HTML：四个数据块注入渲染、缺失块占位不编数、HTML 转义（XSS 门禁）

注：本文件注入的 comparison/ppl/preview 数据均为**合成 fixture**（测试专用，
非真实产物）；真实数字只来自 data/ 下的阶段产物文件。
"""

import pytest

from dashboard import recipes as recipes_mod
from dashboard.report import build_report_html


# ---- 配方校验门禁 ----

def test_all_recipes_valid():
    """生产配方文件三档全部合法（键已知、值在 KNOB_BOUNDS 内）。"""
    rs = recipes_mod.load_recipes()
    assert set(rs) == {"clean_first", "diverse_first", "dedup_first"}
    for name, r in rs.items():
        validated = recipes_mod.validate_recipe(r)
        assert validated, name
        # 三档都必须覆盖全部三类旋钮（写回后配置完整，不留半套）
        assert set(validated) == {"clean", "dedup", "quality"}, name


def test_recipe_rejects_unknown_knob():
    """变异：配方含未知旋钮键 → ValueError（门禁红）。"""
    bad = {"knobs": {"clean": {"not_a_knob": 1}}}
    with pytest.raises(ValueError, match="未知旋钮"):
        recipes_mod.validate_recipe(bad)
    bad_cat = {"knobs": {"nope": {"min_cjk_chars": 1}}}
    with pytest.raises(ValueError, match="未知旋钮类别"):
        recipes_mod.validate_recipe(bad_cat)


def test_recipe_rejects_out_of_bounds():
    """变异：越界值（min_han_ratio=1.5 / 负数）→ ValueError（门禁红）。"""
    with pytest.raises(ValueError):
        recipes_mod.validate_recipe({"knobs": {"clean": {"min_han_ratio": 1.5}}})
    with pytest.raises(ValueError):
        recipes_mod.validate_recipe({"knobs": {"quality": {"min_total_score": -0.1}}})


def test_recipe_missing_knobs_field():
    with pytest.raises(ValueError, match="knobs"):
        recipes_mod.validate_recipe({"label": "x"})


def test_diff_with_current_covers_all_knobs():
    """diff 表覆盖配方全部旋钮，含 当前/配方/变化 三列语义。"""
    rows = recipes_mod.diff_with_current("clean_first")
    assert len(rows) == 6  # clean 3 + dedup 1 + quality 2
    assert {r["类别"] for r in rows} == {"clean", "dedup", "quality"}
    for r in rows:
        assert r["变化"] == "不变" or "→" in r["变化"]


def test_apply_recipe_passes_validated_knobs(monkeypatch):
    """apply = validate + write_knobs（捕获写入值，不碰真配置）。"""
    captured = {}

    def fake_write(overrides):
        captured.update(overrides)
        return sum(len(v) for v in overrides.values())

    monkeypatch.setattr(recipes_mod.knobs, "write_knobs", fake_write)
    n = recipes_mod.apply_recipe("dedup_first")
    assert n == 6
    assert captured["dedup"]["verify_threshold"] == 0.6
    assert isinstance(captured["clean"]["min_cjk_chars"], int)  # int 旋钮不漂移成 float


def test_apply_recipe_invalid_writes_nothing(monkeypatch):
    """非法配方 → ValueError 且 write_knobs 根本不被调用（全有或全无）。"""
    calls = []
    monkeypatch.setattr(recipes_mod.knobs, "write_knobs", lambda o: calls.append(o))
    monkeypatch.setattr(recipes_mod, "load_recipes",
                        lambda: {"bad": {"knobs": {"clean": {"min_han_ratio": 9}}}})
    with pytest.raises(ValueError):
        recipes_mod.apply_recipe("bad")
    assert calls == []


# ---- 报告渲染 ----

_SYNTH_COMP = {"before": "v0-raw", "after": "v1-release",
               "metrics": {"exact_dup_rate": {"before": 0.1, "after": 0.0,
                                              "delta": -0.1, "verdict": "改善"}}}
_SYNTH_PPL = {"engine": "char-5gram", "eval_n": 600, "train_chars_cap": 1500000,
              "arms": {"raw": {"ppl": 3278.0, "ci95": [1, 2], "trained_chars": 100},
                       "clean": {"skipped": "上游未就绪"}}}
_SYNTH_PREVIEWS = {"干净优先": {"n_sample": 100, "clean": {"removed": 30},
                                "quality": {"pass": 50, "remove": 20}}}
_SYNTH_ATTR = [{"源": "wiki", "阶段": "clean", "原因": "too_short", "条数": 42}]


def test_report_renders_all_sections():
    doc = build_report_html(_SYNTH_COMP, _SYNTH_PPL, _SYNTH_PREVIEWS, _SYNTH_ATTR)
    assert "exact_dup_rate" in doc and "改善" in doc
    assert "comparison.json" in doc                # 数字来源标注
    assert "3278" in doc and "上游未就绪" in doc    # ppl 臂 + skipped 臂
    assert "抽样口径" in doc and "干净优先" in doc  # 配方预览显式口径
    assert "too_short" in doc and "42" in doc      # 账本摘要


def test_report_missing_blocks_show_placeholder():
    """缺失数据块 → 占位提示 + 产出命令，不编数、不静默省略。"""
    doc = build_report_html(None, None, None, None)
    assert doc.count("missing") >= 4
    assert "python -m metrics.run" in doc
    assert "python -m downstream.run" in doc


def test_report_escapes_html():
    """XSS 门禁：注入的键/值含 HTML 时必须转义（报告可能被浏览器直接打开）。"""
    evil = {"before": "<script>alert(1)</script>", "after": "v1",
            "metrics": {"<img src=x onerror=alert(2)>": {"before": 0, "after": 0,
                                                         "delta": 0, "verdict": "改善"}}}
    doc = build_report_html(evil, None, None, None)
    assert "<script>" not in doc
    assert "<img src=x" not in doc
    assert "&lt;script&gt;" in doc
