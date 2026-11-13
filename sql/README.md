# sql

职责：Hive / Spark 外部表 DDL 与验收查询。

| 文件 | 用途 |
|---|---|
| `raw_docs.sql` | raw 层外部表（`PARTITIONED BY source`，文件系统分区发现）+ 行数验收 |
| `multimodal_extensions.sql` | 多模态追加列 DDL（`modality` / `modality_pair_id`） |

前置：Docker 栈已启动、parquet 已同步到 MinIO；容器与端口见 `ARCHITECTURE.md` §2。
