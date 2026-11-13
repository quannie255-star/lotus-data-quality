"""lora_finetune 数据层测试（真实语料，不依赖 torch）

覆盖：三套数据集构建、口径参数（M=64、参考/测试分割）、防泄漏剔除、确定性。
数据源为**真实治理产物**（data/release、data/clean），非合成 fixture。
"""

import hashlib

import pytest

from downstream.lora_finetune import data as data_mod

ROOT = data_mod.ROOT
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "release").exists() or not (ROOT / "data" / "clean").exists(),
    reason="真实治理产物缺失（data/ 不进仓库；先跑 release/clean 阶段）")


@pytest.fixture(scope="module")
def ds():
    """全量构建一次（实测秒级），模块内共享。"""
    return data_mod.build()


def test_dataset_shapes(ds):
    """候选=release 全量（减去重叠），参考 M=64，测试=其余评测行。"""
    assert len(ds.reference) == 64
    assert len(ds.test) == 536
    assert len(ds.candidates) > 1000          # release v0.1 实际 2402 篇（含重叠剔除）
    assert ds.excluded_overlap >= 0


def test_no_leak_between_candidates_and_eval(ds):
    """防泄漏硬校验：任何候选文档的内容哈希都不得命中参考/测试行。"""
    eval_hashes = {hashlib.sha256(t.encode()).hexdigest()
                   for t in ds.reference + ds.test}
    for doc in ds.candidates:
        assert hashlib.sha256(doc.encode()).hexdigest() not in eval_hashes


def test_reference_test_disjoint(ds):
    """参考集与测试集不相交（课程口径：参考集与最终测试集分开）。"""
    assert not (set(ds.reference) & set(ds.test))


def test_deterministic_rebuild():
    """两次构建逐条一致（数据构建无随机性——抽样种子冻结在 _eval_lines）。"""
    a = data_mod.build(candidate_limit=30)
    b = data_mod.build(candidate_limit=30)
    assert a.candidates == b.candidates
    assert a.reference == b.reference
    assert a.test == b.test


def test_sources_annotated(ds):
    """数据来源标注进结构（报告可溯源：每个数字指得到文件）。"""
    assert "release" in ds.sources["candidates"]
    assert "seed=2026" in ds.sources["reference+test"]


def test_eval_seed_guard():
    """评测集种子纪律：改 seed 直接 AssertionError（种子一经发布永不改）。"""
    with pytest.raises(AssertionError):
        data_mod._eval_lines(seed=1234)
