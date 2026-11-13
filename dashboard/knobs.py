"""dashboard/knobs.py — 旋钮读写模块（控制台专用）

把「拧旋钮」这件事封装成两个纯函数：
- read_knobs(): 读当前三类旋钮值
- write_knob(category, knob, value): 把新值写回 configs/*_policy.json（带范围校验）

写回是「改标准不动代码」的落点——因为项目「配置进指纹」，改了配置后全量重算
会自动识别（无需手动 force）。本模块只负责安全地改 JSON，不触发重算。

范围校验（防外行人误改出非法值）由 KNOB_SPEC 里的 bounds 定义。
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.preview import KNOB_SPEC, _deep_get, _load_cfg

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# 每个旋钮的合法范围：(min, max)。None 表示不限。写回前校验，越界抛 ValueError。
KNOB_BOUNDS = {
    "clean": {
        "min_cjk_chars": (1, 1000),
        "min_han_ratio": (0.0, 1.0),
        "max_link_density": (0.0, 1.0),
    },
    "dedup": {
        "verify_threshold": (0.0, 1.0),
    },
    "quality": {
        "min_total_score": (0.0, 1.0),
        "min_dim_score": (0.0, 1.0),
    },
}

# 整数旋钮：写回时保持 int 类型（避免 10 → 10.0 的格式漂移污染 git diff / 指纹）。
INT_KNOBS = {"clean": {"min_cjk_chars"}}


def read_knobs() -> dict:
    """读当前旋钮值 → {类别: {旋钮键: 值}}（复用 preview 的读逻辑）。"""
    from dashboard.preview import read_knobs as _read
    return _read()


def knob_specs() -> dict:
    """返回旋钮的完整规格（含描述和范围），供 UI 渲染。"""
    out = {}
    for cat, knobs in KNOB_SPEC.items():
        out[cat] = {}
        for knob, (container, leaf, desc) in knobs.items():
            lo, hi = KNOB_BOUNDS.get(cat, {}).get(knob, (None, None))
            out[cat][knob] = {"desc": desc, "min": lo, "max": hi}
    return out


def validate_value(category: str, knob: str, value) -> float:
    """校验并归一化旋钮值，越界抛 ValueError。整型旋钮返回 int，其余返回 float。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"旋钮 {knob} 的值 {value!r} 不是数字")
    lo, hi = KNOB_BOUNDS.get(category, {}).get(knob, (None, None))
    if lo is not None and v < lo:
        raise ValueError(f"{knob} 不能小于 {lo}（给了 {v}）")
    if hi is not None and v > hi:
        raise ValueError(f"{knob} 不能大于 {hi}（给了 {v}）")
    # 整型旋钮保持 int（避免 10 → 10.0 污染配置字节）
    if knob in INT_KNOBS.get(category, {}):
        return int(v)
    return v


def write_knob(category: str, knob: str, value) -> None:
    """把旋钮新值写回 configs/{category}_policy.json。越界抛 ValueError。

    **只精准替换目标字段的字节，其余字节原样保留**——因为项目「配置进指纹」用文件
    字节做锚（snapshot_configs 按 read_bytes 拷贝），整个文件 json.dumps 重写会引入
    格式漂移（0.30→0.3、数组展开、10→10.0），污染指纹触发无意义重算。写前原子。
    """
    import re

    if category not in KNOB_SPEC or knob not in KNOB_SPEC[category]:
        raise ValueError(f"未知旋钮: {category}.{knob}")

    v = validate_value(category, knob, value)
    spec = KNOB_SPEC[category][knob]
    container, leaf, _desc = spec

    path = CONFIGS / f"{category}_policy.json"
    text = path.read_text(encoding="utf-8")

    # 精准定位 `"leaf": <旧值>`，只替换值部分（保留键名、冒号、缩进、逗号、尾随注释）
    # 目标 leaf 在文件中唯一（每个旋钮只出现一次作为键）
    pattern = re.compile(r'("' + re.escape(leaf) + r'"\s*:\s*)([^,\n}\]]+)')
    new_value = str(v) if isinstance(v, int) else repr(v) if isinstance(v, float) else json.dumps(v)
    # 浮点用 repr 保最短表示（0.3 而非 0.30）；但保持与 Python float 一致的合法 JSON
    new_text, n_sub = pattern.subn(lambda m: m.group(1) + new_value, text, count=1)

    if n_sub != 1:
        raise ValueError(f"定位 {category}.{knob}（键 {leaf}）失败——配置结构可能已变")

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)  # 原子替换，避免半截写


def write_knobs(overrides: dict) -> int:
    """批量写回多类旋钮。返回成功写入的旋钮数。越界抛 ValueError（不部分提交）。"""
    # 先全部校验，再统一写（避免写一半失败留下脏配置）
    validated = {}
    for cat, knobs in (overrides or {}).items():
        for knob, value in knobs.items():
            validated[(cat, knob)] = validate_value(cat, knob, value)
    for (cat, knob), value in validated.items():
        write_knob(cat, knob, value)
    return len(validated)
