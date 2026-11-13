"""modalities/image/ — 图像模态插件（🚧 v1：规则版 + CLIP 跨模态对齐）

import 即注册（modalities.register("image", ...)）。v1 覆盖配对清洗/联合门禁/字节级
精确去重 + CLIP 图文对齐打分（clip_align.py，config 开关 clip.enabled——关闭时
对齐读数 None 不假装，v0 行为不变）；感知哈希为预留接口。范围声明见 paired.py 模块头。
"""

from modalities import register
from modalities.image.paired import (  # noqa: F401
    ImageTextSample, PairedCleaner, PairedDeduper, PairedQualityScorer,
)

register("image", cleaner=PairedCleaner(), deduper=PairedDeduper(),
         scorer=PairedQualityScorer())
