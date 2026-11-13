# Lotus（濯莲）架构说明（ARCHITECTURE）

> 本文是 **as-built** 文档：只描述当前已存在的代码与行为；规划与未做项集中在 §6。
> 完整设计决策在 `docs/wp-01-stage-protocol.md`（契约）与 `DEVELOPMENT_LOG.md`
> （决策日志，编号 M{n}-D{k} 被代码注释引用）。

## 1. 数据流（当前实态）

```
四源（适配器统一映射为 spine 行）
  wiki  : HF 20231101.zh，230,792 篇真实维基
  oscar : mock jsonl（真实源接入后替换适配器）
  news  : 本地 jsonl（同上）
  pdf   : URL/本地 PDF → PyMuPDF 文本层 + 原件字节归档
        │  run_stage（配置进指纹 + 逐文件验货 + run 审计）
        ▼
data/raw/{source}/ingest/part-*.parquet
        │  latest_outputs 上游验货
        ▼
clean    六规则清洗 + PII 真脱敏          → data/clean/source={s}/
dedup    精确 + MinHash 近重 + 并查集聚簇 → data/dedup/source={s}/ + signatures + report
quality  五维打分 + 门禁三分              → data/quality/source={s}/ + gate_report
mix      体裁/源分层 + token 预算抽样     → data/mix/candidates/
release  license 执法 + semver + 溯源卡   → data/release/v0.1/
        │
        ├──► metrics     治理前后指标卡（v0-raw vs v1-release）
        └──► downstream  四臂 ppl（raw/clean/dedup/release）+ 梯度筛选 + LoRA 终证

查询面（可选）：本地 parquet ──mc cp──► MinIO s3a://raw/docs/
                ──► Spark 外部表 raw_docs（PARTITIONED BY source，文件系统分区发现）

横切（被所有阶段消费）：
  data/meta/runs.jsonl        run 生命周期事件（start/finish，失败也留痕）
  data/meta/changelog.jsonl   每条数据的去留账（原因/引用/相似度）
  data/meta/first_seen/*.json doc_id → 首见时间台账（pending→commit）
  各阶段 manifest.json        输入指纹 + 产物哈希（缓存命中的验货依据）
```

## 2. 容器栈（docker-compose，7 服务）

| 服务 | 镜像 | 状态/角色 | 关键定制 |
|---|---|---|---|
| postgres | postgres:16-alpine | healthy；metastore+airflow 两个逻辑库 | init.sql 建 airflow 库 |
| minio | bitnamilegacy/minio:2025.5.24 | healthy；数据本体 | MINIO_DEFAULT_BUCKETS 自动建 warehouse/logs；数据目录 /bitnami/data/minio |
| metastore | apache/hive:4.0.0 | Up；表目录（thrift:9083） | SERVICE_OPTS 传 jdo + s3a；外挂 PG 驱动 + hadoop-aws 3.3.6 |
| spark-master/worker | corpus/spark:3.5.3-s3a（自建） | Up；计算 | Dockerfile 补 hadoop-aws 3.3.4 + sdk-bundle 并 chmod；conf 无密钥（env 注入凭据） |
| airflow | apache/airflow:2.10.4 | Up；调度（DAG 未挂，M9） | LocalExecutor；DAG/日志挂卷 |
| kafka | bitnamilegacy/kafka:3.7.1 | Up（未验收，M9） | KRaft 单节点无 ZooKeeper |

网络：compose 默认网络 `corpus-net`，服务名即 DNS。发布端口仅 5 个（9000/9001/7077/8080/8081）。

## 3. 模块清单（as-built）

| 模块 | 输入 | 输出 | 关键函数 |
|---|---|---|---|
| `common/ids.py` | source+source_id / 文本 / 文本+config | 确定性 ID 与哈希 | `spine_doc_id` `content_hash` `normalized_content_hash` |
| `common/runs.py` | stage 名 | run_id / 事件流 | `start` `finish` `read_events` `query_by_run` |
| `common/changelog.py` | 变更记录 dict | 校验后的 JSONL 追加 | `validate` `append_change(dedupe)` `query_by_doc` `count_by_reason` |
| `common/idempotent.py` | transform+输入 | 产物目录+审计 | `run_stage` `fingerprint`（缓存命中 = 指纹∧目录∧逐文件验货） |
| `common/first_seen.py` | doc_id | 首见时间 | `FirstSeenLedger.resolve/commit` |
| `common/textstats.py` | str | 三计数 | `text_stats` `cjk_char_count` |
| `common/log.py` | LOG_LEVEL | stderr 日志 | `setup` `get_logger` |
| `ingest/wiki.py` | HF 分片 parquet | spine parquet | `download` `iter_docs` `map_row` `run` |
| `ingest/oscar.py` | mock jsonl（N1 后换真实源） | spine parquet + quarantine parquet | `write_mock_shards` `classify_row` `run`（护栏 0.05） |
| `ingest/sampling.py` | 分片清单 | 确定性抽样 + manifest 快照 | `sample_shards` `write_sample_manifest` |
| `ingest/pdf.py` | 源清单（URL/本地 PDF） | spine parquet + 归档件 + quarantine | `fetch_bytes` `archive_raw` `run` |
| `ingest/news.py` | 本地 jsonl（N2 后换真实语料） | spine parquet | `iter_docs` `run`（fail-fast 策略） |
| `parse/pdf_extractor.py` | PDF 文件 | ExtractResult(text/pages/warnings) | `extract_pdf`（PyMuPDF 文本层） |
| `common/stage.py` | 配置/上游目录/变更记录 | 指纹·快照·验货·两段式账本 | `configs_digest` `snapshot_configs` `latest_outputs` `StagedChangeLog` `run_pipeline` |
| `clean/base.py`+`rules.py` | spine 行流 | 保留行流 + stats + 变更记录 | `apply_rules` `iter_clean`（六规则，阈值全在 clean_policy） |
| `clean/run.py` | data/raw/{src}/ingest | data/clean/source={s}/part-*.parquet + changelog | `run`（四源聚合，验货上游） |
| `dedup/engine.py`+`run.py` | data/clean | data/dedup/source={s} + signatures.parquet + report.json | `dedup_pass`（exact+MinHashLSH+并查集，跨源矩阵在 report） |
| `quality/engine.py`+`run.py` | data/dedup | data/quality/source={s}（+五维分数列）+ gate_report.json | `score_row` `gate` `build_ngram_index` |
| `mix/run.py` | data/quality | data/mix/candidates（+token_count/mix_batch_id） | `stratified_sample` `token_count_estimate` |
| `release/run.py` | data/mix | data/release/v{semver}/（part + provenance_card.yaml） | `next_version` `build_provenance_card`（license 执法） |
| `downstream/ppl.py`+`run.py` | 四层语料 | data/downstream/report.json（四臂 ppl + CI） | `CharNgramLM` `bootstrap_ci`（N7：纯 py 5-gram） |
| `stream/producer.py` | — | Kafka topic corpus-ingest | `build_messages` `produce`（doc_id 确定→重放被 dedup 折叠） |
| `dags/corpus_pipeline.py` | — | Airflow DAG 定义 | 四源扇入→clean→…→downstream（激活条件见 DEVLOG M9） |
| `dashboard/app.py` | data/ 本地镜像 | Streamlit 四视图 | `funnel` `removal_attribution` `quality_distribution` `runs_audit` |
| `clean/pii.py` | 文本 | 命中/脱敏 | `detect` `redact`（GB11643 + Luhn 校验） |
| `sql/raw_docs.sql` | — | 外部表 DDL + 验收查询 | — |
| `modalities/base.py` | 模态插件注册 | 插件绑定 | `Sample` `Cleaner` `Deduper` `QualityScorer` `register` `get` |
| `modalities/text/` | 绑定既有文本实现（clean/dedup/quality） | 插件适配器 | `TextCleaner` `TextDedupEngine` `TextQualityScorer` |
| `modalities/image/` | 🚧 v1 规则版 + CLIP 对齐 | 图文配对样本（v0 规则版：配对清洗/联合门禁/字节级精确去重）+ `clip_align.py` 对齐打分与 `cross_modal` 第三模态门禁 | `ImageTextSample` `PairedCleaner` `PairedDeduper` `PairedQualityScorer` |
| `modalities/audio/`、`modalities/video/` | 预留位（未实装） | — | — |
| `quality/multimodal.py` | 各模态分数 dict | 联合加权分 + 联合门禁 | `joint_score` `joint_gate` |
| `dedup/api.py` | 样本流 | 去重引擎抽象 + 注册表 | `DedupEngine` `register_engine` `dedup_samples`（感知哈希/音频指纹为预留接口） |
| `metrics/multimodal.py` | 配对样本行 | 多模态指标（配对完整性/跨模态重复率） | `pair_metrics` `compute_multimodal_card` |
| `smoke/m1.py` | — | 五步 PASS/FAIL | — |

## 4. 阶段间契约（当前实态 + WP-01 目标态）

**已实装**：ingest 的 run() 形状 = download → run_stage(指纹/验货/审计) → 本地 parquet →
人工 mc cp 上 MinIO。数据行 = spine 字段子集（未检测字段不出现，见 wiki.py 模块头三分法）。

**契约目标态**（WP-01，M4 起实装，此处只列差异）：表路径语法（`data/{zone}/{stage}/source={s}/`）、
`configs_digest` 并入指纹、`StagedChangeLog` 两段式提交、`latest_outputs` 上游验货、
CLI 退出码 0/1/2/3。完整契约与五个决策记录（D1-D5）见 `docs/wp-01-stage-protocol.md`。

## 5. 关键决策索引（详见 DEVELOPMENT_LOG M1 与 wp-01）

| 决策 | 一句话 | 证伪方法 |
|---|---|---|
| M1-D1 时钟不进数据行 | first_seen 走台账，force 重算字节一致 | `test_forced_rerun_is_byte_identical` |
| M1-D2 缓存要验货 | 命中三条件含逐文件哈希对账 | 篡改产物测试 |
| M1-D3 run 即审计 | 失败也留 finish(failed) 痕迹 | 集成测试事件序列断言 |
| M1-D4 未检测≠空 | 字段三态：不出现/null/有值 | golden record |
| M1-D5 事实单一来源 | registry 字面量禁入代码 | 源码 grep 元测试 |
| M1-D6 日志双通道 | stderr 日志 / stdout 结构化 | — |
| wp-01 D1-D5 | content 替换语义/配置指纹/changelog 两段式/分区策略/薄驱动 | 各附测试，见 wp-01 §4 |

## 6. 路线图位

| 状态 | 内容 |
|---|---|
| ✅ 已完成 | 四源采集（wiki 为真实源；oscar/news 为 mock 适配器）· clean · dedup · quality · mix · release · 四臂 ppl · 指标卡 · dashboard（六 Tab + 三档配方 + 报告导出）· 多模态插件层文本绑定 · CLIP 对齐内核与评测标定 · P11 梯度筛选 · LoRA 因果链终证（文本）· 带约束旋钮自动搜索（`dashboard/autotune.py`） |
| 🚧 建设中 | 中文真实图文语料与阈值重标定（`clip.enabled` 暂为 false）· 图文配对链路接入七层编排 · Kafka/Airflow 激活（代码与容器就绪，DAG 未挂载） |
| ⬜ 未做 | 多模态场景的因果链终证 · 感知哈希/音频指纹 · 音视频插件实装 |

模块级契约与实验协议：`docs/blueprint.md` §2/§3/§4；工作包队列：`docs/backlog.md`。

---

## 7. 多模态分层架构（M11 起：核心层模态无关 + 插件层模态相关）

> 定位：把「治理底座」与「模态实现」解耦。底座只认插件接口，不 import 任何具体模态；
> 新增模态 = 新增 `modalities/<m>/` 插件包并注册，核心层零改动——这是宪法第 2 条
> 「阶段间只通过表耦合，不通过代码调用耦合」在模态维度的推广。
> 状态：**文本已绑定、图文 v1 已落地**；音视频为预留位，多模态链路尚未接入七层阶段编排。

### 7.1 分层边界

| 层 | 内容 | 边界纪律 |
|---|---|---|
| **核心层（模态无关）** | `common/`（ids/runs/changelog/idempotent/stage/first_seen 审计与幂等账本）· `quality/` 门禁调度 · `release/` 版本发布 · `metrics/` 指标卡框架 · `mix/` 配比 · `dags/`+`stream/` 编排 | 禁止 import `modalities/` 下任何具体模态插件；不引入模态专属依赖（解码器/深度学习框架一律在插件层） |
| **插件层（模态相关）** | `modalities/text|image|audio|video/` 各自的采集器、清洗器、去重器、质量打分器 | 依赖核心层接口；模态专属依赖按需引入并在插件 README 声明 |

### 7.2 插件接口规范（`modalities/base.py`，as-built）

- `Sample`：统一样本基类（dataclass）。spine 身份字段（`doc_id/source/source_id/url/...`）
  与模态无关，`modality`、`modality_pair_id` 为**追加字段**（不进 doc_id 键材料——宪法第 3 条）；
  模态载荷由子类扩展：`TextSample.content`、`ImageTextSample.caption + image_uri + image_bytes_hash`。
- 四个抽象基类：`Collector`（源 → 样本流）、`Cleaner`（样本流 → 判定 keep/drop/hold/transform，
  判定协议与 `clean/base.py` 同构）、`Deduper`（样本流 → keep/removed/clusters，removed 必带
  reason/ref/similarity——账本口径不变）、`QualityScorer`（样本 → 维度分 + 总分，0~1）。
- 注册表：`modalities.register(modality, ...)` / `modalities.get(modality)`；插件包 import 即注册，
  未注册模态 fail-fast（坏数据不静默的插件版）。
- 多模态联合门禁：`quality/multimodal.py` 按配置加权各模态分数（权重旋钮在
  `configs/multimodal_policy.json`），任一模态破下限 → 联合 remove；文本五维权重仍归
  `configs/quality_policy.json`——各模态各管各的维度，联合层只管权衡。
- 跨模态去重接口：`dedup/api.py` 定义 `DedupEngine` 抽象 + 注册表；文本 exact/MinHash 近重为
  默认实现（适配既有 `dedup/engine.py`），图像字节级精确哈希已实装（纯 stdlib），
  感知哈希/音频指纹为**预留接口**（未实装，调用即显式报错，不假装）。

### 7.3 现状一览（叙事诚实）

| 插件 | 状态 | 内容 |
|---|---|---|
| `modalities/text/` | ✅ 已绑定 | 适配器绑定既有文本实现（`clean/rules+pii`、`dedup/engine`、`quality/engine`）；文本链路 100% 走原路径，行为零变化 |
| `modalities/image/` | 🚧 v1 规则版 + CLIP 对齐 | 图文配对样本（`ImageTextSample`）：配对完整性清洗（孤儿 hold 不静默删）、联合门禁、字节级精确去重（按源信任序选簇代表）。**v1 增量**：`clip_align.py` CLIP 图文对齐打分（CN-CLIP/OpenAI CLIP 双后端自适应，torch 仅在本层），`clip.enabled` 开启后 `per_modality` 增 cross_modal 第三模态键进联合门禁（权重 `modality_weights_cross_modal`），图文不匹配 → remove(low_quality_modality:cross_modal)、坏图 → remove(no_score)；关闭时对齐分 = null（空读数不假装，v0 行为不变）。评测与阈值标定在 `clip_bench/`（DL 大作业本体，标定产物回填 clip.cos_lo/cos_hi）。感知哈希仍为预留接口 |
| `modalities/audio/`、`modalities/video/` | ⬜ 预留 | 目录与接口位就绪，未实装 |

schema 边界：多模态字段定义在 `schema/multimodal_schema.json`（含 `modality` 枚举、
`modality_pair_id`、图文载荷字段）；`unified_schema.json` 保持 0.3.2 不动——golden record
与已发布数据行钉死该版本，spine 升级（追加 `modality` 列）留待首个多模态真实源落地时执行，
避免无谓的连锁重算（决策记录：DEVLOG M11-D2）。

### 7.4 新增模态的标准步骤

1. `schema/multimodal_schema.json` 追加该模态的样本定义（载荷字段 + 校验规则）；
2. 新建 `modalities/<m>/` 插件包：`Sample` 子类 + 四个基类实现 + `register(...)`；
3. `configs/multimodal_policy.json` 追加该模态旋钮节（阈值/权重全部进配置，零字面量）；
4. `tests/test_modalities.py` 追加接口冒烟测试与链路集成测试；
5. 七层链路编排（clean/dedup/quality/mix/release 的 run 壳）不动——模态差异全部收在插件层，
   阶段产物仍走 `data/{zone}/{stage}/source={s}/` 表路径语法与审计账本（WP-01 §1 不变，
   `modality` 是列不是分区，无宪法豁免）。
