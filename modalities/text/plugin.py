"""modalities/text/plugin.py — 文本模态插件（M11）：绑定既有文本实现

设计决策（M11-D1）：既有文本实现（clean/rules+pii、dedup/engine、quality/engine）
**原地不动**，本插件以适配器方式绑定——物理迁移会破坏「向后兼容」红线（golden record、
134 项测试、已发布数据行都钉死原路径），且插件层本就只需稳定的绑定点。
生产链路仍走原阶段编排（clean/run.py 等）；插件层供多模态调度器按统一接口消费文本能力。
"""

from dataclasses import dataclass

from common import ids
from common.textstats import text_stats
from modalities.base import Cleaner, Deduper, QualityScorer, Sample


@dataclass
class TextSample(Sample):
    """文本样本：模态载荷 = content（格式声明 content_format）。"""

    content: str = ""
    content_format: str = "plain_text"
    language: str = "zh"

    def to_row(self) -> dict:
        """导出为 clean/quality 规则可直接消费的行 dict（补齐三计数与哈希）。"""
        s = text_stats(self.content)
        return {
            "doc_id": self.doc_id, "source": self.source, "source_id": self.source_id,
            "url": self.url, "content": self.content, "content_format": self.content_format,
            "content_hash": ids.content_hash(self.content),
            "char_count": s["char_count"], "cjk_char_count": s["cjk_char_count"],
            "utf8_byte_count": s["utf8_byte_count"], "genre": self.genre,
            "license": self.license,
        }


def from_row(sample: TextSample, row: dict) -> TextSample:
    """行 dict 回填样本（clean transform 后的重建；身份字段以行为准——content 可能已脱敏）。"""
    return TextSample(doc_id=row["doc_id"], source=row["source"],
                      source_id=row["source_id"], modality="text", url=row.get("url"),
                      license=row.get("license", sample.license),
                      genre=row.get("genre", sample.genre),
                      content=row["content"],
                      content_format=row.get("content_format", sample.content_format),
                      language=sample.language)


class TextCleaner(Cleaner):
    """文本清洗器：直连 clean/base.apply_rules（六规则 + PII，阈值全在 clean_policy）。"""

    def clean(self, sample: Sample) -> tuple:
        from clean.base import apply_rules, load_rules

        row, changes = apply_rules(sample.to_row(), load_rules())
        if row is None:  # drop 短路；reason 取最后一条变更（即触发删除的规则）
            return ("drop", changes[-1]["reason"])
        if changes:      # transform（目前仅 PII redact）；审计载荷全量转交
            last = changes[-1]
            return ("transform", from_row(sample, row), last["reason"],
                    last["action"], {"changes": changes})
        return ("keep",)


class TextQualityScorer(QualityScorer):
    """文本打分器：直连 quality/engine.score_row（五维 + 加权总分）。"""

    def score(self, sample: Sample) -> dict:
        from quality.engine import score_row

        return score_row(sample.to_row())


class TextDeduper(Deduper):
    """文本去重器：直连 dedup/engine.dedup_pass（exact + MinHashLSH 近重）。

    簇代表/相似度/cluster_id 口径与 dedup 阶段账本完全一致——插件不另立标准。
    """

    def dedup(self, samples: list) -> dict:
        from dedup.engine import dedup_pass

        rows = []
        for s in samples:
            row = s.to_row()
            row["normalized_hash"] = ids.normalized_content_hash(s.content)
            rows.append(row)
        report = dedup_pass(rows)
        by_id = {s.doc_id: s for s in samples}
        return {
            "keep": [by_id[r["doc_id"]] for r in report["keep"]],
            "removed": [{**r, "sample": by_id[r["row"]["doc_id"]]} for r in report["removed"]],
            "clusters": report["clusters"],
        }
