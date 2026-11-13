# quality

职责：五维质量打分与门禁三分（放行 / 删除 / 隔离），并产出 gate report。

- 入口：`python -m quality.run`
- 引擎：`engine.py`（`score_row` / `gate` / 污染 n-gram 索引）
- 多模态：`multimodal.py`（联合加权分 + 短板木桶律门禁）
- 输入 → 输出：`data/dedup/source={s}/` → `data/quality/source={s}/` + `gate_report.json`
- 配置：`configs/quality_policy.json`
- 测试：`pytest tests/test_quality.py -q`
