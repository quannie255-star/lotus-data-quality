# schema

职责：行级数据契约（JSON Schema）。代码与配置服从这里，不反向。

| 文件 | 管什么 |
|---|---|
| `unified_schema.json` | spine 0.3.2：身份字段、policies、stage_extensions（被 golden record 钉死） |
| `multimodal_schema.json` | 多模态样本：`modality` 枚举、图文配对载荷 |
| `dedup_signature_schema.json` | 去重签名表 |
| `removal_log_schema.json` | 变更账记录 |
| `text_normalization.schema.json` | 规范化配置 |

测试：`pytest tests/test_registry.py -q`
