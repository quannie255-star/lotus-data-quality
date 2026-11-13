"""modalities/video/ — 视频模态插件（⬜ 预留位，未实装）

接口规范已就位（modalities/base.py），组件未实装：modalities.get("video") 显式抛
PluginNotRegisteredError。视频样本天然是「音+像+文(字幕)」多载荷复合体，实装时应
复用 image/audio 两个插件的组件按载荷组合，而不是另写一套（ARCHITECTURE.md §7.4）。
"""
