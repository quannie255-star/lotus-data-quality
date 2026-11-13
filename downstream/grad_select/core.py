"""downstream/grad_select/core.py — P11 基础版：线性模型梯度打分与 top-k 筛选（纯 Python 串行）

课程口径（并行计算课程项目选题说明 Version3 · P11 附录）：
- 样本 = (x, y)，固定参数 w，损失 L = 1/2 (wᵀx − y)²，逐样本梯度 g = (wᵀx − y)·x
- 参考平均梯度 ḡref = (1/M) Σ g′ⱼ
- 候选分数 sᵢ = gᵢᵀḡref / (‖gᵢ‖₂·‖ḡref‖₂)，**分母为零时分数记 0**
- 选 k = ⌈0.05·N⌉ 条（默认比例 0.05），按分数从高到低，**同分按样本编号升序**
- 计算期间不更新模型（w 固定）

本文件是「正确性基准实现」：零第三方依赖、逐样本串行、FP64 精度计算
（输入数据按课程要求为 FP32 值，见 datagen.py）。并行版（torch_kernel.py）
的正确性以本实现为参照。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def sample_gradient(w: list[float], x: list[float], y: float) -> list[float]:
    """单样本梯度 g = (wᵀx − y)·x（解析式，不经自动微分——课程附录公式）。"""
    dot = math.fsum(wi * xi for wi, xi in zip(w, x))
    r = dot - y
    return [r * xi for xi in x]


def mean_gradient(w: list[float], samples: list[tuple[list[float], float]]) -> list[float]:
    """参考集平均梯度 ḡref = (1/M) Σ g′ⱼ（串行累加后除 M）。"""
    if not samples:
        raise ValueError("参考集为空")
    d = len(w)
    acc = [0.0] * d
    for x, y in samples:
        g = sample_gradient(w, x, y)
        for i in range(d):
            acc[i] += g[i]
    m = len(samples)
    return [a / m for a in acc]


def cosine_score(g: list[float], g_ref: list[float]) -> float:
    """余弦相似度；任一方 L2 范数为零 → 分数记 0（课程附录约定）。"""
    num = math.fsum(a * b for a, b in zip(g, g_ref))
    ng = math.sqrt(math.fsum(a * a for a in g))
    nr = math.sqrt(math.fsum(b * b for b in g_ref))
    if ng == 0.0 or nr == 0.0:
        return 0.0
    return num / (ng * nr)


@dataclass(frozen=True)
class Selection:
    """筛选结果：全部分数 + 选中编号（升序）+ k。课程要求「输出全部分数及选中编号」。"""
    scores: list[float]
    selected: list[int]
    k: int


def select_topk(w: list[float],
                candidates: list[tuple[list[float], float]],
                reference: list[tuple[list[float], float]],
                ratio: float = 0.05,
                k: int | None = None) -> Selection:
    """串行基础版：打分全部候选 → 按 (−分数, 编号) 排序 → 选前 k 条。

    k 默认 ⌈ratio·N⌉（课程口径 5%）；同分按编号升序（排序键显式含编号，不依赖稳定性）。
    """
    if not candidates:
        raise ValueError("候选集为空")
    g_ref = mean_gradient(w, reference)
    scores = [cosine_score(sample_gradient(w, x, y), g_ref) for x, y in candidates]
    n = len(scores)
    k_eff = k if k is not None else math.ceil(ratio * n)
    k_eff = max(0, min(k_eff, n))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return Selection(scores=scores, selected=sorted(order[:k_eff]), k=k_eff)
