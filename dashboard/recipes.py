"""dashboard/recipes.py — 三档配方（干净/多样/去重优先）加载、校验、应用

配方 = 一组旋钮预设值（configs/recipes.json），与控制台旋钮同一套 KNOB_SPEC 键。
- load_recipes(): 读配方文件
- validate_recipe(): 每个旋钮键必须已知、值必须在 KNOB_BOUNDS 内——**门禁能红**：
  配方文件被改坏（未知键/越界值）时抛 ValueError，不允许写回生产配置
- diff_with_current(): 配方 vs 当前生产值的逐旋钮差异（UI 展示代价用）
- apply_recipe(): 校验通过后批量写回（复用 knobs.write_knobs 的原子精准替换）

配方数值已定稿（2026-11-13 owner 采纳建议值）——每档理由见 configs/recipes.json
的 rationale 字段与顶层 note（三档只在同一批旋钮上朝三个正交方向偏移，一档一变量）。
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard import knobs
from dashboard.preview import KNOB_SPEC, read_knobs

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
RECIPES_PATH = CONFIGS / "recipes.json"


def load_recipes() -> dict:
    """读三档配方。文件缺失/JSON 坏 → 抛异常（fail-fast，配方是显式契约）。"""
    doc = json.loads(RECIPES_PATH.read_text(encoding="utf-8"))
    recipes = doc.get("recipes")
    if not recipes:
        raise ValueError(f"{RECIPES_PATH} 缺 recipes 字段或为空")
    return recipes


def validate_recipe(recipe: dict) -> dict:
    """校验配方：旋钮键必须都在 KNOB_SPEC 内，值必须过 KNOB_BOUNDS。

    返回 {类别: {旋钮键: 归一化值}}（int 旋钮保持 int）。任一非法 → ValueError，
    且不返回部分结果（全有或全无，与 write_knobs 同纪律）。
    """
    out: dict = {}
    knob_map = recipe.get("knobs")
    if not knob_map:
        raise ValueError("配方缺 knobs 字段")
    for cat, kv in knob_map.items():
        if cat not in KNOB_SPEC:
            raise ValueError(f"配方含未知旋钮类别: {cat}")
        for knob, value in kv.items():
            if knob not in KNOB_SPEC[cat]:
                raise ValueError(f"配方含未知旋钮: {cat}.{knob}")
            out.setdefault(cat, {})[knob] = knobs.validate_value(cat, knob, value)
    return out


def diff_with_current(name: str) -> list:
    """配方 vs 当前生产值 → [{类别, 旋钮, 当前, 配方, 变化}]（UI 表格直用）。"""
    recipe = load_recipes()[name]
    validated = validate_recipe(recipe)
    current = read_knobs()
    rows = []
    for cat, kv in validated.items():
        for knob, new in kv.items():
            cur = current[cat][knob]
            rows.append({"类别": cat, "旋钮": knob, "当前": cur, "配方": new,
                         "变化": "不变" if cur == new else f"{cur} → {new}"})
    return rows


def apply_recipe(name: str) -> int:
    """校验 → 写回生产配置。返回写入旋钮数。非法配方 ValueError（不写任何文件）。"""
    recipe = load_recipes()[name]
    validated = validate_recipe(recipe)
    return knobs.write_knobs(validated)
