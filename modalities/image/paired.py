"""modalities/image/paired.py — 图文配对模态插件 v0（M11，规则版）

范围声明（叙事诚实）：v0 = 规则版——配对完整性清洗（孤儿 hold 不静默删）、图像结构
检查、字节级精确去重（图+文全同 → pair_exact_dup；同图不同文 → image_exact_dup，
按源信任序折叠）、联合门禁（文本侧直连五维打分 + 图像侧结构分，权重在 multimodal_policy）。
**v1 增量**：CLIP 跨模态对齐分接入（clip_align.py，config 开关 clip.enabled——开启后
per_modality 增加 cross_modal 第三键，图文不匹配可被联合门禁拦截）。
**未实装不假装**：感知哈希近重（dedup/api.PerceptualHashEngine 预留）、
多模态 PII（图内文字/人脸）均为路线图 🚧 项。

身份纪律：image_bytes_hash（图像字节 sha256）是图像侧内容身份，参与去重不参与
doc_id 键材料（宪法第 3 条）；缺图像哈希的样本不参与去重（未检测 ≠ 空 ≠ 有值）。
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common import ids
from dedup.api import ExactKeyEngine
from modalities.base import Cleaner, Deduper, QualityScorer, Sample
from quality.multimodal import joint_gate, joint_score

ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def mm_policy() -> dict:
    return json.loads((ROOT / "configs" / "multimodal_policy.json").read_text(encoding="utf-8"))


@dataclass
class ImageTextSample(Sample):
    """图文配对样本：caption（文本侧载荷）+ 图像引用与字节身份（图像侧载荷）。

    image_bytes_len 是已知的字节数（采集时账面值）；不解码图像——感知级检查属预留项。
    """

    modality: str = "image_text"
    caption: str = ""
    image_uri: str | None = None
    image_bytes_hash: str | None = None
    image_bytes_len: int = 0
    image_format: str | None = None


class PairedCleaner(Cleaner):
    """配对完整性清洗：孤儿对按配置 hold（默认，隔离待复核）或 drop；其余结构性缺陷 drop。"""

    def clean(self, sample: Sample) -> tuple:
        rules = mm_policy()["image_text"]["pair_rules"]
        orphan = sample.image_bytes_hash is None or not sample.caption.strip()
        if orphan:
            return ("drop" if rules["orphan_policy"] == "drop" else "hold"), "orphan_pair"
        if len(sample.caption) < rules["min_caption_chars"]:
            return ("drop", "caption_too_short")
        if sample.image_bytes_len < rules["min_image_bytes"]:
            return ("drop", "image_bytes_too_small")
        return ("keep",)


class PairedQualityScorer(QualityScorer):
    """联合打分：文本侧 = 五维打分器直连（空 caption 直接 0 分，不喂垃圾给打分器）；
    图像侧 = v0 结构分；跨模态侧 = CLIP 对齐分（clip.enabled 开关，关闭时读数 None
    不假装）。联合分与联合门禁走 quality/multimodal（权重是配置旋钮：双模态用
    modality_weights，三模态用 modality_weights_cross_modal——和都必须=1）。

    clip_scorer 参数仅供测试注入替身；生产路径按配置 lazy 单例（clip_align.get_scorer）。
    """

    def __init__(self, clip_scorer=None):
        self._clip_injected = clip_scorer

    def _clip(self):
        cfg = mm_policy()["image_text"].get("clip", {})
        if not cfg.get("enabled"):
            return None
        if self._clip_injected is not None:
            return self._clip_injected
        from modalities.image.clip_align import get_scorer
        return get_scorer(cfg)

    def score(self, sample: Sample) -> dict:
        caption_score = self._caption_score(sample)
        img_score = 1.0 if (sample.image_bytes_hash is not None
                            and sample.image_bytes_len >=
                            mm_policy()["image_text"]["pair_rules"]["min_image_bytes"]) else 0.0
        per_modality = {"text": {"quality_score": caption_score},
                        "image": {"quality_score": img_score}}
        gate_cfg = mm_policy()["joint_gate"]
        weights_key = "modality_weights"

        clip = self._clip()
        align = None
        if clip is not None:
            align = clip.score(sample.image_uri, sample.caption)
            per_modality["cross_modal"] = {"quality_score": align}
            weights_key = "modality_weights_cross_modal"

        weights = gate_cfg[weights_key]
        verdict, reason = joint_gate(per_modality, {**gate_cfg, "modality_weights": weights})
        return {
            "pair_caption_quality": caption_score,
            "pair_image_structural": img_score,
            "pair_cross_modal_alignment": align,   # clip 关闭=未实装读数 None；开启但坏图=None → no_score
            "quality_score": joint_score(per_modality, weights),
            "modality_scores": per_modality,
            "joint_verdict": verdict,
            "joint_reason": reason,
        }

    @staticmethod
    def _caption_score(sample: ImageTextSample) -> float:
        if not sample.caption.strip():
            return 0.0
        from quality.engine import score_row

        row = {"content": sample.caption, "char_count": len(sample.caption), "genre": "default"}
        return score_row(row)["quality_score"]


class PairedDeduper(Deduper):
    """图文去重 v0：图像字节 sha256 精确分组（ExactKeyEngine，模态无关实现）。

    簇代表源信任序复用 dedup_policy.canonical_policy（事实单一来源，不在
    multimodal_policy 重复登记）；同簇内文亦同 → pair_exact_dup，同图不同文 →
    image_exact_dup。缺图像哈希的样本键为唯一占位（未检测 ≠ 空 ≠ 有值——孤儿不互相折叠）。
    clusters 报告观察到的全部图像重复簇；removed 只含按 policy 实际执行的折叠
    （image_exact_dup_policy=keep 时只折叠完全相同对）。
    """

    def __init__(self):
        from dedup.engine import policy as dedup_policy

        self._engine = ExactKeyEngine(
            source_trust=dedup_policy()["canonical_policy"]["source_trust"])

    def dedup(self, samples: list) -> dict:
        items = [{"doc_id": s.doc_id, "source": s.source,
                  "key": s.image_bytes_hash or f"no-image:{s.doc_id}",
                  "size": len(s.caption),
                  "caption_hash": ids.normalized_content_hash(s.caption)}
                 for s in samples]
        report = self._engine.run(items)
        by_id = {i["doc_id"]: i for i in items}
        samples_by_id = {s.doc_id: s for s in samples}
        collapse = mm_policy()["image_text"]["dedup"]["image_exact_dup_policy"]
        removed, kept_samples = [], set()
        for r in report["removed"]:
            same_caption = (r["item"]["caption_hash"]
                            == by_id[r["ref_doc_id"]]["caption_hash"])
            if collapse == "keep_pairs_only" and not same_caption:
                kept_samples.add(r["item"]["doc_id"])
                continue
            reason = "pair_exact_dup" if same_caption else "image_exact_dup"
            removed.append({**r, "reason": reason,
                            "sample": samples_by_id[r["item"]["doc_id"]]})
        removed_ids = {r["item"]["doc_id"] for r in removed}
        keep = [s for s in samples
                if s.doc_id not in removed_ids or s.doc_id in kept_samples]
        return {"keep": keep, "removed": removed, "clusters": report["clusters"]}
