# common

职责：全仓共用的身份、审计、幂等与日志底座。各阶段只通过它读写账本，不各自造一套。

| 文件 | 职责 |
|---|---|
| `ids.py` | `doc_id` / `content_hash` / `normalized_content_hash`（确定性身份） |
| `runs.py` | run 生命周期事件（start/finish，失败也留痕） |
| `changelog.py` | 变更账校验、两段式追加、按 doc/reason 查询 |
| `idempotent.py` | 输入指纹 + 缓存命中三条件（指纹 ∧ 目录 ∧ 逐文件验货） |
| `stage.py` | WP-01 六件套：`configs_digest` / `snapshot_configs` / `latest_outputs` / `StagedChangeLog` / `run_pipeline` |
| `first_seen.py` | `doc_id` → 首见时间台账（pending→commit） |
| `textstats.py` | 字符 / CJK / 文本统计 |
| `log.py` | stderr 日志 + stdout 结构化输出双通道 |

契约全文见 `docs/wp-01-stage-protocol.md`。

测试：`pytest tests/test_ids.py tests/test_runs.py tests/test_changelog.py tests/test_idempotent.py tests/test_first_seen.py -q`
