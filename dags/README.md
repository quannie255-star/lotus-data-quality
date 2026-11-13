# dags

职责：Airflow DAG 定义（四源扇入 → clean → dedup → quality → mix → release → downstream）。

- 文件：`corpus_pipeline.py`
- 状态：DAG 已定义，尚未挂载到 Airflow 容器（激活步骤见 `DEVELOPMENT_LOG.md` M9）
- 手工等价入口：各阶段 `python -m {clean,dedup,quality,mix,release}.run`
