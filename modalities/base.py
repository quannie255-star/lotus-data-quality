"""modalities/base.py — 模态插件接口规范（M11）

核心层（审计账本/配方/幂等缓存/门禁调度/版本发布/运行时编排）只认本文件的接口，
不 import 任何具体模态插件；新增模态 = 新增 modalities/<m>/ 插件包 + register()，
核心层零改动——宪法第 2 条「阶段间只通过表耦合」在模态维度的推广。

边界纪律：
- 本文件只依赖 stdlib；模态专属依赖（图像解码器/音频指纹库/深度学习框架）一律进插件包，
  核心层永不背负（架构 §7.1）；
- Sample 身份字段沿用 spine 语义：doc_id 确定性、modality/modality_pair_id 为追加字段，
  不进 doc_id 键材料（宪法第 3 条「键材料永不扩展」）；
- 判定协议与 clean/base.py 同构，另增 hold（隔离待复核——坏数据不静默的一等公民表达）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# 数据级模态枚举（schema/multimodal_schema.json 的 enum 与此对齐——事实单一来源在 schema，
# 此处常量仅做 import 期 fail-fast 校验）。
# 插件目录是能力维度（text/image/audio/video 四个插件包）；modality 字段值是数据维度
# （image_text = image 插件提供的图文配对样本）。
MODALITY_VALUES = ("text", "image", "image_text", "audio", "video")


@dataclass
class Sample:
    """统一样本基类：spine 身份字段模态无关，模态载荷由子类追加。

    doc_id 由调用方按 common/ids.py 规则计算（确定性身份不归插件管）；
    modality_pair_id 标识同一样本内跨模态成员的配对关系（如同一对 图-文 共享一个
    pair_id），与 doc_id 正交。
    """

    doc_id: str
    source: str
    source_id: str
    modality: str = "text"
    url: str | None = None
    modality_pair_id: str | None = None
    license: str = "unknown"
    genre: str = "unknown"

    def __post_init__(self):
        if self.modality not in MODALITY_VALUES:
            raise ValueError(f"未知模态 {self.modality!r}；合法值: {MODALITY_VALUES}")

    def to_dict(self) -> dict:
        """导出为行 dict（metrics/账本侧的交换格式）。"""
        from dataclasses import asdict
        return asdict(self)


class Collector(ABC):
    """模态采集器：外部源 → 样本流。ingest 的模态化入口（骨架，各插件按需实装）。"""

    @abstractmethod
    def collect(self, **kwargs):
        """产出 Sample 迭代器；下载级幂等策略由实现自定（参照 ingest/wiki.py 形状）。"""


class Cleaner(ABC):
    """模态清洗器：单样本 → 判定。

    判定协议（与 clean/base.py 同构 + hold）：
      ("keep",)                                          保留
      ("drop", reason)                                   删除（changelog action=remove）
      ("hold", reason)                                   隔离待复核（action=quarantine）
      ("transform", new_sample, reason, action, meta)    改写并携带审计载荷
    """

    @abstractmethod
    def clean(self, sample: Sample) -> tuple:
        """幂等要求：纯函数，同输入同判定。"""


class Deduper(ABC):
    """模态去重器：样本流 → keep/removed/clusters。

    removed 条目必带 {sample, reason, ref_doc_id, similarity, cluster_id}——
    与 dedup 阶段账本口径一致（谁压掉谁、相似度多少，宪法第 5 条）。
    """

    @abstractmethod
    def dedup(self, samples: list) -> dict:
        """返回 {"keep": [...], "removed": [...], "clusters": {cluster_id: [doc_id...]}}。
        幂等要求：全字典序 tiebreak，同输入同输出。"""


class QualityScorer(ABC):
    """模态质量打分器：单样本 → 维度分 dict（全部 0~1，越高越好）。

    键名约定 "<modality_prefix>_<dim>"，总分键 "quality_score"；未实装的维度
    值为 None（空读数不假装有数——指标卡同款纪律）。
    """

    @abstractmethod
    def score(self, sample: Sample) -> dict:
        """幂等要求：纯函数。"""
