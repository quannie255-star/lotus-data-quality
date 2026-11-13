# dags

职责：Airflow DAG 定义（四源扇入 → clean → dedup → quality → mix → release → downstream）。

- 文件：`corpus_pipeline.py`
- 状态：**已挂载并验证（2026-11-13）**——`airflow dags list` 解析出 `corpus_pipeline`
  （10 个任务），`airflow dags show` 的 dot 输出逐边核对拓扑；`airflow tasks test` 跑
  `clean` 返回 SUCCESS（exit 0）且命中 `run_stage` 幂等缓存。证据：
  `data/metrics/orchestration_m9.json`
- 激活：`docker compose up -d --build airflow`（镜像装流水线运行时依赖，numpy 钉 1.26.4；
  仓库挂载到 `/opt/airflow/lotus` 并注入 `PYTHONPATH`，任务 cwd 即仓库根）
- 手工等价入口：各阶段 `python -m {clean,dedup,quality,mix,release}.run`

## 验证命令（宿主机 bash）

```bash
docker compose exec -T airflow airflow dags list
docker compose exec -T airflow airflow dags show corpus_pipeline
MSYS_NO_PATHCONV=1 docker compose exec -T airflow \
  airflow tasks test corpus_pipeline clean 2026-11-13
```

Airflow 2.10 没有 `airflow dags list-tasks`，列任务用 `airflow tasks list corpus_pipeline`。
