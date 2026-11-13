"""grad_select 测试 — P11 课程口径的正确性门禁

覆盖（课程 PDF 明列的调试与边界用例）：
1. 官方调试用例：w=(0,0)、参考 x=(1,0),y=-1 → 分数 1/0/-1/0、选中编号 0
2. 全部零梯度 → 分数全 0，按编号选前 k
3. 同分 → 按编号升序选
4. N 不整除块大小 → 分块结果与整批逐位一致
5. k=⌈0.05N⌉ 向上取整口径
6. torch 三形态 vs 串行基准：分数在 FP32 容差内、选中编号完全一致
7. 变异测试（门禁能红）：见 test_mutation_* —— 故意破坏约定必须被断言抓住
"""

import math

import pytest

from downstream.grad_select import core, datagen


def test_pdf_debug_case():
    """课程附录调试用例：分数应为 1、0、-1、0，选中编号为 0。"""
    w = [0.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]          # g′ = (0−(−1))·(1,0) = (1,0)
    cand = [([1.0, 0.0], -1.0),         # g = (1,0)   → cos = 1
            ([0.0, 1.0], -1.0),         # g = (0,1)   → cos = 0
            ([-1.0, 0.0], -1.0),        # g = (-1,0)  → cos = -1
            ([0.0, 0.0], 0.0)]          # g = (0,0)   → 零范数 → 0
    r = core.select_topk(w, cand, ref)
    assert r.scores == pytest.approx([1.0, 0.0, -1.0, 0.0])
    assert r.k == 1                      # ⌈0.05·4⌉ = 1
    assert r.selected == [0]


def test_all_zero_gradients_selects_by_index():
    """全部零梯度：分数全 0（零分母约定），同分按编号 → 选前 k 个编号。"""
    w = [1.0, 0.0]
    ref = [([0.0, 0.0], 0.0)]           # 参考梯度也是零向量
    cand = [([0.0, 0.0], 0.0)] * 10
    r = core.select_topk(w, cand, ref)
    assert r.scores == [0.0] * 10
    assert r.k == 1                      # ⌈0.05·10⌉ = 1
    assert r.selected == [0]


def test_ties_broken_by_index():
    """同分样本按编号升序选取（排序键显式含编号，不依赖排序稳定性）。"""
    w = [1.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]
    same = ([1.0, 0.0], -1.0)
    cand = [same] * 100
    r = core.select_topk(w, cand, ref)
    assert r.k == 5                      # ⌈0.05·100⌉ = 5
    assert r.selected == [0, 1, 2, 3, 4]
    assert all(s == pytest.approx(1.0) for s in r.scores)


def test_k_ceiling():
    """k = ⌈0.05·N⌉ 向上取整：N=41 → 3；N=10 → 1；N=100 → 5。"""
    w = [1.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]
    for n, expect_k in ((41, 3), (10, 1), (100, 5), (1, 1)):
        cand = [([1.0, 0.0], -1.0)] * n
        assert core.select_topk(w, cand, ref).k == expect_k == math.ceil(0.05 * n)


def test_empty_inputs_rejected():
    with pytest.raises(ValueError):
        core.select_topk([0.0], [], [([1.0], 1.0)])
    with pytest.raises(ValueError):
        core.mean_gradient([0.0], [])


def test_datagen_fp32_and_reproducible():
    """数据生成：同种子逐位一致；值确实是 FP32 可表示的。"""
    a = datagen.gen_dataset(64, 8, seed=42)
    b = datagen.gen_dataset(64, 8, seed=42)
    assert a.X == b.X and a.y == b.y
    for row in a.X:
        for v in row:
            assert datagen.to_fp32(v) == v
            assert -1.0 <= v < 1.0


# ---- torch 并行形态（无 torch 环境只跳过本节，不影响串行用例）----

try:
    import torch  # noqa: F401
    from downstream.grad_select import torch_kernel as tk
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 未安装")


@requires_torch
@pytest.mark.parametrize("variant", ["loop", "batch", "chunk"])
def test_torch_variants_match_serial(variant):
    """随机数据上三种 torch 形态与串行基准：分数在 FP32 容差内、选中编号完全一致。"""
    fn = tk.VARIANTS[variant]
    for (n, d) in ((257, 5), (1000, 32)):   # 257 故意不整除常见块大小
        w = datagen.gen_param(d, seed=777 + d)
        cand = datagen.gen_dataset(n, d, seed=101).as_pairs()
        ref = datagen.gen_dataset(64, d, seed=102).as_pairs()
        base = core.select_topk(w, cand, ref)
        got = fn(w, cand, ref, chunk_rows=100) if variant == "chunk" else fn(w, cand, ref)
        assert got.selected == base.selected
        assert got.k == base.k
        assert max(abs(a - b) for a, b in zip(got.scores, base.scores)) < 1e-5


@requires_torch
def test_chunk_equals_batch_bitwise():
    """分块与整批在 float32 下逐位一致（块内运算序列相同、块间无交叉归约）。"""
    w = datagen.gen_param(32, seed=7)
    cand = datagen.gen_dataset(1001, 32, seed=8).as_pairs()   # 1001 % 100 = 1 尾块
    ref = datagen.gen_dataset(64, 32, seed=9).as_pairs()
    a = tk.select_batch(w, cand, ref)
    b = tk.select_chunk(w, cand, ref, chunk_rows=100)
    assert a.scores == b.scores
    assert a.selected == b.selected


@requires_torch
def test_torch_pdf_debug_case():
    """课程调试用例在并行形态下同样成立。"""
    w = [0.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]
    cand = [([1.0, 0.0], -1.0), ([0.0, 1.0], -1.0), ([-1.0, 0.0], -1.0), ([0.0, 0.0], 0.0)]
    for fn in tk.VARIANTS.values():
        r = fn(w, cand, ref)
        assert r.scores == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-6)
        assert r.selected == [0]


# ---- 变异测试：门禁能红的证据 ----

def test_mutation_zero_denominator_convention():
    """变异：若把「零分母记 0」约定破坏（裸除），零梯度样本会抛 ZeroDivisionError
    （Python float 0/0 是异常不是 nan——这正是该约定必须显式实现的原因）。

    这里不改产品代码，而是用猴补丁模拟破坏，证明断言真的在拦；改回 → 绿。
    """
    orig = core.cosine_score

    def broken(g, g_ref):
        num = math.fsum(a * b for a, b in zip(g, g_ref))
        ng = math.sqrt(math.fsum(a * a for a in g))
        nr = math.sqrt(math.fsum(b * b for b in g_ref))
        return num / (ng * nr)   # 去掉零分母保护

    w = [0.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]
    cand = [([1.0, 0.0], -1.0), ([0.0, 0.0], 0.0)]
    try:
        core.cosine_score = broken
        with pytest.raises(ZeroDivisionError):   # 破坏后必须炸（红）
            core.select_topk(w, cand, ref)
    finally:
        core.cosine_score = orig
    r = core.select_topk(w, cand, ref)
    assert r.scores == [1.0, 0.0]                # 改回 → 绿


def test_mutation_tie_break_order():
    """变异：若排序键丢掉编号（只按 −分数），同分选取依赖实现细节；本测试钉死
    「同分按编号升序」与「分数高者优先」两条口径。"""
    w = [1.0, 0.0]
    ref = [([1.0, 0.0], -1.0)]
    cand = [([1.0, 0.0], -1.0)] * 100
    good = core.select_topk(w, cand, ref)
    assert good.k == 5 and good.selected == [0, 1, 2, 3, 4]
    # 分数不同的构造：0 与 2 同分且最高（cos=1），1 与参考梯度正交（cos=0）
    # 注意余弦对梯度模长不敏感（尺度不变），构造低分样本必须改方向而非改模长
    cand2 = [([1.0, 0.0], -1.0), ([0.0, 1.0], -1.0), ([1.0, 0.0], -1.0)]
    good2 = core.select_topk(w, cand2, ref, k=2)
    assert good2.scores == pytest.approx([1.0, 0.0, 1.0])
    assert good2.selected == [0, 2]
