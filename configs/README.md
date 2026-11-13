# configs

职责：全部可调参数的唯一来源——阈值、权重、规则、配比都在这里，代码中不写事实字面量。

| 配置 | 管什么 |
|---|---|
| `clean_policy.json` | 六条清洗规则的阈值 |
| `dedup_policy.json` | 精确/近重复去重参数 |
| `quality_policy.json` | 五维权重与门禁下限 |
| `mix_policy.json` | 分层策略与 token 预算 |
| `source_registry.json` / `ingest_policy.json` / `pdf_sources.json` | 源登记、license 策略、PDF 源清单 |
| `text_normalization.json` / `url_canonicalization.json` | 文本与 URL 规范化 |
| `reason_registry.json` | 合法删除原因枚举（唯一来源） |
| `recipes.json` | 干净优先 / 多样优先 / 去重优先三档配方 |
| `multimodal_policy.json` | 多模态联合权重、`clip.enabled` 开关 |

改配置会进入阶段指纹并触发重算；旋钮反查见 `MANUAL.md`。
