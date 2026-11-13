# clean

职责：把 raw 的 spine 行流按六条规则清洗，并把每次去留写入变更账。

- 入口：`python -m clean.run`（`run.py`）
- 规则：`rules.py`（样板/语言/过短等六条）；PII 在 `pii.py`（GB11643 + Luhn 校验后脱敏）
- 判定协议：`base.py` 的 `apply_rules` / `iter_clean`，keep / drop / hold / transform 四态
- 输入 → 输出：`data/raw/{source}/ingest/` → `data/clean/source={s}/part-*.parquet`
- 阈值：全部在 `configs/clean_policy.json`；改配置会进入指纹并触发重算
- 测试：`pytest tests/test_clean.py tests/test_pii.py -q`
