"""downstream/grad_select/torch_kernel.py — P11 并行版：PyTorch 向量化打分与筛选

与 core.py 同一套课程口径（g=(wᵀx−y)x、ḡref 平均、余弦分数、零分母记 0、
k=⌈0.05N⌉、同分按编号）。三种计算形态，用于回答「并行/批处理到底赚在哪」：

- select_loop  逐样本循环（torch 向量运算但 Python 级 O(N) 迭代）——批处理的反面对照
- select_batch 整批一次算出 (N,d) 梯度矩阵——最大化向量化，峰值显存/内存 ~4Nd 字节
- select_chunk 分块批处理，逐块打分后立即释放梯度块——课程附录「分块计算并及时
  释放梯度」的实现，内存有上界，结果与 select_batch 逐位一致

并行性来源：候选样本间独立（embarrassingly parallel）；参考梯度求平均是一次
归约（torch 内部完成，注意其同步语义）。整批平均梯度**不能**替代逐样本梯度
（课程附录明示），故 (N,d) 矩阵按行保留逐样本梯度。
"""

from __future__ import annotations

import math

import torch

from downstream.grad_select.core import Selection


def _to_tensors(w, candidates, reference, device):
    X = torch.tensor([x for x, _ in candidates], dtype=torch.float32, device=device)
    y = torch.tensor([y for _, y in candidates], dtype=torch.float32, device=device)
    Xr = torch.tensor([x for x, _ in reference], dtype=torch.float32, device=device)
    yr = torch.tensor([y for _, y in reference], dtype=torch.float32, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    return X, y, Xr, yr, wt


def _ref_mean_gradient(Xr, yr, wt) -> torch.Tensor:
    """ḡref = mean_j (wᵀx′ⱼ − y′ⱼ)·x′ⱼ —— (M,d) 逐样本梯度按行平均（归约）。"""
    Gr = (Xr @ wt - yr).unsqueeze(1) * Xr
    return Gr.mean(dim=0)


def _scores_from_grads(G, g_ref, g_ref_norm) -> torch.Tensor:
    """逐行余弦分数；任一方范数为零 → 0（与 core.cosine_score 同约定）。"""
    num = G @ g_ref
    ng = G.norm(dim=1)
    denom = ng * g_ref_norm
    return torch.where(denom > 0, num / torch.where(denom > 0, denom, torch.ones_like(denom)),
                       torch.zeros_like(num))


def _finish(scores: torch.Tensor, ratio: float, k: int | None) -> Selection:
    n = scores.numel()
    k_eff = k if k is not None else math.ceil(ratio * n)
    k_eff = max(0, min(k_eff, n))
    # stable 排序保证同分按编号升序（与 core 的 (−分数, 编号) 排序键等价）
    order = torch.argsort(-scores, stable=True)[:k_eff]
    sel = sorted(order.tolist())
    return Selection(scores=scores.tolist(), selected=sel, k=k_eff)


def select_loop(w, candidates, reference, ratio=0.05, k=None, device="cpu") -> Selection:
    """对照形态：torch 向量运算 + Python 逐样本循环（不做批处理）。"""
    X, y, Xr, yr, wt = _to_tensors(w, candidates, reference, device)
    g_ref = _ref_mean_gradient(Xr, yr, wt)
    g_ref_norm = g_ref.norm()
    scores = torch.empty(len(candidates), dtype=torch.float32, device=device)
    for i in range(len(candidates)):
        gi = (torch.dot(X[i], wt) - y[i]) * X[i]
        scores[i] = _scores_from_grads(gi.unsqueeze(0), g_ref, g_ref_norm)[0]
    return _finish(scores, ratio, k)


def select_batch(w, candidates, reference, ratio=0.05, k=None, device="cpu") -> Selection:
    """整批形态：一次算出 (N,d) 梯度矩阵并打分。峰值内存 ~4Nd 字节（FP32）。"""
    X, y, Xr, yr, wt = _to_tensors(w, candidates, reference, device)
    g_ref = _ref_mean_gradient(Xr, yr, wt)
    G = (X @ wt - y).unsqueeze(1) * X
    scores = _scores_from_grads(G, g_ref, g_ref.norm())
    del G
    return _finish(scores, ratio, k)


def select_chunk(w, candidates, reference, ratio=0.05, k=None, device="cpu",
                 chunk_rows: int = 4096) -> Selection:
    """分块形态：每 chunk_rows 行一块，打分后立即释放该块梯度（内存有上界）。

    N 不整除 chunk_rows 时尾块照算（课程边界用例）。结果与 select_batch 逐位一致：
    每块内的运算序列相同，且块间无交叉归约。
    """
    X, y, Xr, yr, wt = _to_tensors(w, candidates, reference, device)
    g_ref = _ref_mean_gradient(Xr, yr, wt)
    g_ref_norm = g_ref.norm()
    n = X.shape[0]
    scores = torch.empty(n, dtype=torch.float32, device=device)
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        Xc, yc = X[lo:hi], y[lo:hi]
        G = (Xc @ wt - yc).unsqueeze(1) * Xc
        scores[lo:hi] = _scores_from_grads(G, g_ref, g_ref_norm)
        del G
    return _finish(scores, ratio, k)


VARIANTS = {"loop": select_loop, "batch": select_batch, "chunk": select_chunk}
