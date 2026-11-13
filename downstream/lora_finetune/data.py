"""downstream/lora_finetune/data.py — 因果链终证的三套数据集构建

课程 PDF「模型实验定义」口径：
- 候选集 D_cand：待筛选样本（Lotus release 层语料 = 治理产物）
- 参考集 D_ref：代表目标任务（M 条，与测试集**分开**）
- 独立测试集：评价微调效果，绝不参与打分与训练

数据来源与防泄漏：
- 候选集读 data/release/v0.1/**/part-*.parquet 的 content 列（真实治理产物）
- 参考/测试沿用 downstream ppl 消融的冻结评测集口径（clean 层、EVAL_SEED=2026、
  600 行）：前 M=64 行做参考集，其余做测试集——与既有 −4.3% 消融同源可比
- **泄漏防护**：参考/测试行若与任一候选文档内容哈希相同，该候选被剔除并计数
  （release ⊂ clean 分层链路，理论上可能包含评测行；剔除数进报告，不静默）
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass(frozen=True)
class Datasets:
    candidates: list[str]
    reference: list[str]
    test: list[str]
    excluded_overlap: int = 0
    sources: dict = field(default_factory=dict)


def _release_docs(limit: int | None = None) -> list[str]:
    import pyarrow.parquet as pq

    docs: list[str] = []
    for part in sorted((DATA / "release").glob("**/part-*.parquet"), key=str):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=256, columns=["content"]):
            docs.extend(v for v in batch.column("content").to_pylist() if v)
        if limit and len(docs) >= limit:
            return docs[:limit]
    return docs


def _eval_lines(n: int = 600, seed: int = 2026) -> list[str]:
    """复用 downstream.run 的冻结评测集口径（clean 层、同种子同行序）。"""
    from downstream.run import EVAL_SEED, _lines_from

    assert seed == EVAL_SEED, "评测集种子一经发布永不改（与 doc_id 同纪律）"
    parts = sorted((DATA / "clean").glob("**/part-*.parquet"), key=str)
    return _lines_from(parts, n=n, seed=seed)


def build(m_ref: int = 64, candidate_limit: int | None = None) -> Datasets:
    """构建三套数据集。m_ref 与 P11 课程口径一致（M=64）。"""
    lines = _eval_lines()
    if len(lines) < m_ref + 10:
        raise RuntimeError(f"clean 层评测行不足（{len(lines)}）——先跑 clean 阶段")
    reference, test = lines[:m_ref], lines[m_ref:]

    eval_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in lines}
    candidates, excluded = [], 0
    for doc in _release_docs(candidate_limit):
        if hashlib.sha256(doc.encode("utf-8")).hexdigest() in eval_hashes:
            excluded += 1
            continue
        candidates.append(doc)
    if not candidates:
        raise RuntimeError("release 层无候选文档——先跑 release 阶段")

    return Datasets(
        candidates=candidates, reference=reference, test=test,
        excluded_overlap=excluded,
        sources={"candidates": "data/release/v0.1/**/part-*.parquet (content)",
                 "reference+test": "data/clean/**/part-*.parquet, seed=2026, 前64行=参考/其余=测试"})
