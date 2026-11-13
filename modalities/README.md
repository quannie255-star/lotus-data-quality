# modalities/ — 模态插件层

分层与边界：**核心层（common/quality/release/metrics/mix/dags）模态无关，插件层模态相关**。
完整规范见 `ARCHITECTURE.md` §7，配置与开发指南见 `MANUAL.md` § 多模态插件配置指南。

| 插件 | 状态 | 说明 |
|---|---|---|
| `text/` | ✅ 已绑定 | 适配器直连既有文本实现（clean/rules+pii · dedup/engine · quality/engine），行为与文本生产链路零差异 |
| `image/` | 🚧 v1 规则 + CLIP 对齐 | 图文配对（`ImageTextSample`）：孤儿 hold、联合门禁（权重在 `configs/multimodal_policy.json`）、字节级精确去重；**新增** `clip_align.py` CLIP 对齐打分（`clip.enabled` 开关：开启后 cross_modal 第三模态进联合门禁，关闭时读数 null 不假装）；感知哈希为预留接口 |
| `audio/` | ⬜ 预留 | 未实装，`get("audio")` 显式报错 |
| `video/` | ⬜ 预留 | 未实装；实装时按载荷复用 image/audio 插件组件 |

快速上手：

```python
import modalities
modalities.load_all()                      # 注册 text + image
cleaner = modalities.get("image", "cleaner")
cleaner.clean(sample)                      # ("keep",) / ("drop", reason) / ("hold", reason) / ("transform", ...)
```

纪律：插件可按需引入模态依赖（解码器/深度学习框架），核心层永不背负；
未实装的读数是 `None`、未实装的接口调用显式报错——不静默、不假装。
