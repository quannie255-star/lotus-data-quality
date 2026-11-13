"""modalities/text/ — 文本模态插件（✅ 已绑定既有实现）

import 即注册（modalities.register("text", ...)）。组件直连既有模块，行为与文本
生产链路零差异；边界与决策见 ARCHITECTURE.md §7 与 DEVLOG M11-D1。
"""

from modalities import register
from modalities.text.plugin import (
    TextCleaner, TextDeduper, TextQualityScorer, TextSample, from_row,  # noqa: F401
)

register("text", cleaner=TextCleaner(), deduper=TextDeduper(), scorer=TextQualityScorer())
