# stream

职责：Kafka 流式入口，把新文档写入 `corpus-ingest` topic。

- 文件：`producer.py`（`build_messages` / `produce` / `consume_peek`）
- 语义：`doc_id` 确定性生成，重放消息会被下游 dedup 折叠——至少一次投递不产生重复数据
- 状态：**已端到端验证（2026-11-13）**——20 条确定性消息（`build_messages(n=20, seed=7)`，
  payload sha256 `c1310ec4…`）生产到 `corpus-ingest`，从头部消费 3 条，
  doc_id 与上一轮重放完全一致（`sim-7-00000` → `4a8be7cc…`）。
  证据：`data/metrics/orchestration_m9.json`

复现（宿主机 bash；payload 由 Python 生成，producer/consumer 在 Kafka 容器内执行）：

```bash
python -c "from stream.producer import build_messages; print('\n'.join(build_messages(20, 7)))" > /tmp/msgs.jsonl
MSYS_NO_PATHCONV=1 docker compose exec -T kafka \
  /opt/bitnami/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 --topic corpus-ingest < /tmp/msgs.jsonl
MSYS_NO_PATHCONV=1 docker compose exec -T kafka \
  /opt/bitnami/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic corpus-ingest \
  --from-beginning --max-messages 3 --timeout-ms 8000
```

`MSYS_NO_PATHCONV=1` 是 Git Bash 专有：不加会把容器内 `/opt/bitnami/...` 改写成 Windows 路径。
Spark Structured Streaming 消费作业（kafka→raw 分区）仍未做，见 `producer.py` 文件头。
