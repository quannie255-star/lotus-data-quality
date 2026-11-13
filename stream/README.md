# stream

职责：Kafka 流式入口，把新文档写入 `corpus-ingest` topic。

- 文件：`producer.py`（`build_messages` / `produce`）
- 语义：`doc_id` 确定性生成，重放消息会被下游 dedup 折叠——至少一次投递不产生重复数据
- 状态：代码就绪，端到端验收未做（见 `ARCHITECTURE.md` §6）
