"""modalities/audio/ — 音频模态插件（⬜ 预留位，未实装）

接口规范已就位（modalities/base.py），采集/清洗/去重/打分组件未实装：
modalities.get("audio") 显式抛 PluginNotRegisteredError——预留不等于假装可用。
实装时按 ARCHITECTURE.md §7.4 步骤走；音频指纹去重接口已在 dedup/api.py 预留。
"""
