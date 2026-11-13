"""lora_finetune 模型层口径测试（不需要下载大模型，只测纯逻辑）

核心口径：select_topk_by_score 必须与课程基础版 grad_select.core.select_topk
**逐位同口径**（k=⌈0.05N⌉、降序、同分按编号升序）——这是「内核只写一次，
两处落地」的防漂移门禁。
"""

import math

import pytest

torch = pytest.importorskip("torch")

from downstream.grad_select import core
from downstream.lora_finetune.model import cosine_score_torch, select_topk_by_score


@pytest.mark.parametrize("n", [1, 4, 41, 100, 2396])
def test_k_ceiling_same_as_course(n):
    scores = [0.5] * n
    assert len(select_topk_by_score(scores)) == math.ceil(0.05 * n)


def test_tie_break_by_index_same_as_course():
    scores = [0.3] * 100
    assert select_topk_by_score(scores) == list(range(5))


def test_order_matches_core_on_random_scores():
    """跨实现防漂移：构造单位向量样本使 core 的余弦分数 = 给定分数，
    两实现的选中集合必须完全一致。

    构造：w=(0,0) 时 g=(0−y)·x，取 y=-1 → g=x；参考集单样本 (1,0),y=-1 → ḡref=(1,0)。
    单位向量 x=(s, √(1-s²)) 的 cos(x, ḡref) = s（FP64 下 1e-12 级误差，不制造近平分）。
    """
    import random

    rng = random.Random(7)
    scores = [round(rng.uniform(-0.999, 0.999), 3) for _ in range(200)]
    cand = [([s, math.sqrt(1 - s * s)], -1.0) for s in scores]
    ref = [([1.0, 0.0], -1.0)]
    core_sel = core.select_topk([0.0, 0.0], cand, ref).selected
    lora_sel = select_topk_by_score(scores)
    assert lora_sel == core_sel


def test_cosine_zero_norm_convention():
    """零范数梯度 → 分数 0（与 core.cosine_score 同约定）。"""
    g_ref = torch.tensor([1.0, 0.0])
    assert cosine_score_torch(torch.zeros(2), g_ref, float(g_ref.norm())) == 0.0
    assert cosine_score_torch(g_ref, torch.zeros(2), 0.0) == 0.0


def test_cosine_values_match_core():
    """非退化向量的余弦值与 core.cosine_score 在 FP32 容差内一致。"""
    g = torch.tensor([0.3, -0.7, 1.2])
    g_ref = torch.tensor([-0.5, 0.2, 0.9])
    got = cosine_score_torch(g, g_ref, float(g_ref.norm()))
    want = core.cosine_score(g.tolist(), g_ref.tolist())
    assert got == pytest.approx(want, abs=1e-6)
