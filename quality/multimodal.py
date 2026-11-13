"""quality/multimodal.py — 多模态联合打分与联合门禁（M11）

分工纪律：各模态各管各的维度分（文本五维归 quality/engine，图像结构维归 image 插件），
本模块只做**模态层权衡**——按 configs/multimodal_policy.json 的 modality_weights 把各模态
quality_score 加权成联合分，再过联合门禁。

联合门禁语义（短板木桶律）：配对样本任何一侧烂，整对烂——
  任一模态 quality_score < min_modality_score → remove(low_quality)
  联合分 < min_joint_score                  → remove(low_quality)
hold（隔离待复核）语义由配对完整性规则承担（孤儿对 hold，不静默删），不在联合层重复。
"""

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def policy() -> dict:
    return json.loads((ROOT / "configs" / "multimodal_policy.json").read_text(encoding="utf-8"))


def joint_score(per_modality: dict, weights: dict | None = None) -> float | None:
    """各模态 quality_score 按权重加权 → 联合分。

    参数: per_modality — {"text": {"quality_score": 0.8, ...}, "image": {...}}
          weights — 缺省读 policy().joint_gate.modality_weights
    返回: 加权联合分（round 4）；任一模态缺分/分值为 None → None（空读数不假装）
    异常: ValueError — 权重和≠1、权重键与模态键不一致（旋钮装错位 fail-fast，不默认 0）
    幂等: 纯函数
    """
    w = dict(policy()["joint_gate"]["modality_weights"]) if weights is None else dict(weights)
    if set(w) != set(per_modality):
        raise ValueError(f"模态权重键 {sorted(w)} 与模态分键 {sorted(per_modality)} 不一致")
    total_w = sum(w.values())
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(f"模态权重和须为 1，当前 {total_w}")
    score, seen = 0.0, []
    for m, s in per_modality.items():
        v = (s or {}).get("quality_score")
        if v is None:
            return None
        score += v * w[m]
        seen.append(f"{m}={v}")
    return round(score, 4)


def joint_gate(per_modality: dict, gate_policy: dict | None = None) -> tuple:
    """联合门禁三分语义的 remove/pass 两分（hold 归配对完整性规则）。

    返回: (verdict, reason)；verdict ∈ pass/remove
    幂等: 纯函数；联合分 None（某模态无分）→ remove(no_score)，不静默放行
    """
    g = gate_policy or policy()["joint_gate"]
    js = joint_score(per_modality, g["modality_weights"])
    if js is None:
        return "remove", "no_score"
    for m, s in per_modality.items():
        if (s or {}).get("quality_score", 1.0) < g["min_modality_score"]:
            return "remove", f"low_quality_modality:{m}"
    if js < g["min_joint_score"]:
        return "remove", "low_quality"
    return "pass", None
