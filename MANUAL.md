# Lotus（濯莲）操作手册（MANUAL）

> 你想改什么 → 查反查表 → 去那个文件那个参数改。改完「要重跑哪些阶段」列告诉你影响面。
> 通用规则：**改了任何进入数据或配置的东西，下游阶段必须 force 重算**（M4 前指纹不含配置，
> 属已知缺陷，见 DEVELOPMENT_LOG M1 遗留）。

## 反查表：我想改 X → 去哪改

| 我想改… | 位置 | 当前值 | 影响范围 | 改后要重跑 |
|---|---|---|---|---|
| **数据源（加一个源）** | `configs/source_registry.json`（登记 license/genre/format/id 策略）+ 新建 `ingest/<source>.py` 适配器（照 `ingest/wiki.py` 形状）+ `docs/source_metadata_key_registry.md`（登记 map 键） | 4 源已登记（wiki/oscar/news/pdf），适配器实装 1 个 | 新增源；已有源不受影响 | 新适配器自己的 ingest；下游全链路（新源汇入） |
| 换 wiki 的 dump 日期/分片数 | `ingest/wiki.py` 顶部 `DUMP_DATE` / `N_SHARDS` / `SHARD_FILENAME` | 20231101 / 6 | doc 的 source_timestamp 与数据量 | `python -m ingest.wiki --force 不存在时删 data/raw/wiki/` → 重新上传 MinIO → S5 |
| **清洗规则**（增删改/阈值） | `configs/clean_policy.json`（各规则节）+ `clean/rules.py`（纯函数）+ reason 码登记 `configs/reason_registry.json`；顺序改 `rules_order` | min_cjk=10 / han_ratio=0.25 / fffd≤2‰ / 链接≤0.30 / 短行≤0.80 | clean 及全部下游 | `python -m clean.run --force`（M4 起配置已进指纹，**无需 --force 也会重算**） |
| **去重阈值** | `configs/dedup_policy.json` | num_perm=128 / bands=16 / verify=0.8 / char 3-gram | dedup 层及下游 | 改后重跑 dedup 即自动重算（配置进指纹）；W4 网格实验回填 |
| **质量门禁阈值** | `configs/quality_policy.json`（weights/dims/gate 三节） | min_total=0.55 / 隔离带 0.45-0.55 / 维度下限 0.30 | quality/mix/release | 重跑 quality 即重算 |
| **抽样比例/配比** | `configs/mix_policy.json` | 预算 2M token / wiki .6 news .2 oscar .15 pdf .05 / seed 42 | mix 输出构成 | 重跑 mix；改 tokenizer 需 force |
| **输出路径（本地产物）** | `ingest/wiki.py` 的 `OUTPUT_DIR`（L91） | `data/raw/wiki/` | 该阶段产物位置；MinIO 上传目标同步改 | 仅影响新跑；已有产物不迁移 |
| **输出路径（MinIO/表）** | 上传命令里的目标前缀 + `sql/raw_docs.sql` 的 `LOCATION` | `s3a://raw/docs/` | 外部表位置 | 重跑 S5（DROP/CREATE 自动） |
| **日志级别** | 环境变量 `LOG_LEVEL`（`common/log.py` 消费） | `INFO` | 全部入口的 stderr 日志量 | 无需重跑，即时生效：`LOG_LEVEL=DEBUG python -m ingest.wiki` |
| 文本规范化规则 | `configs/text_normalization.json`（改任意规则**必须**递增 `version`） | v1.0.0（NFKC+t2s 折叠等，见 decisions 节） | normalized_content_hash 口径 = 去重判定 | M5 起生效；改规则 → dedup 及下游 force |
| OSCAR 坏行护栏/语言默认 | `configs/ingest_policy.json` 的 `oscar` 节 | 护栏 0.05；语言默认 zh | oscar 的 quarantine 率与 fail-fast 阈值 | `python -m ingest.oscar --force` |
| 抽样种子/份数 | `configs/ingest_policy.json` 的 `sample` 节 | seed=42, k=2 | 大源首跑样本构成 | 重跑 ingest；manifest 落 data/meta/samples/ |
| OSCAR mock 源 | `ingest/oscar.py` 的 `MOCK_DIR`/`write_mock_shards` | data/mock/oscar，4×50 行 | mock 数据形态 | N1 实测后整个 mock 节删除 |
| PDF 源清单 | `configs/pdf_sources.json` 的 `files` | 空（测试用生成件） | pdf 源的文档集合 | `python -m ingest.pdf` 前先录 URL/路径 |
| PDF 扫描页阈值 | `parse/pdf_extractor.py` 的 `SCANNED_PAGE_CHAR_THRESHOLD` | 10 | 扫描页误报率 | 改后 force 重跑 pdf ingest |
| PDF 坏文件护栏 | `configs/ingest_policy.json` 的 `pdf.bad_file_guardrail` | 0.34 | pdf run fail-fast 阈值 | force 重跑 |
| news 输入文件 | `ingest/news.py` 的 `DATA_FILE` | data/mock/news/news.jsonl（30 行 mock） | news 数据形态 | N2 换真实语料时改 download 部分 |
| wiki 的 license/genre 等声明 | `configs/source_registry.json` 的 `sources.wiki` | CC BY-SA 4.0 / wiki | 新 ingest 的行；旧行不追溯 | force 重跑 ingest（golden record 测试会同步要求更新） |
| 单条记录的 schema 字段 | `schema/unified_schema.json`（spine/stage_extensions）+ `ingest/wiki.py` 的 `PARQUET_SCHEMA` + golden 测试 `tests/test_ingest_wiki.py` | v0.3.2 | 全管道契约 | schema_version 递增 + 全链路重算 |
| **多模态联合门禁权重/配对规则** | `configs/multimodal_policy.json`（joint_gate 与 image_text 两节） | text .6 / image .4（clip 关）；三模态 .5/.2/.3（clip 开）；孤儿 hold；caption≥5 字；图≥1024B | 图文配对样本的门禁与清洗；文本五维不受影响 | 重跑对应模态阶段即自动重算（配置进指纹） |
| **CLIP 跨模态对齐门禁** | `configs/multimodal_policy.json` 的 `image_text.clip` 节 | enabled=true；cos_lo=0.361/cos_hi=0.4942（中文在域标定回填）；模型 data/models/cn-clip-vit-base-patch16 | 开启后图文不匹配 → remove(low_quality_modality:cross_modal)；标定与评测见 `clip_bench/` | 切换 enabled = 配置变更 → 多模态阶段自动重算 |
| 多模态样本 schema | `schema/multimodal_schema.json`（sample_base + 各模态扩展；unified_schema 不动） | v0.1.0 | 插件层样本契约 | 追加字段只增不改；升 spine 前先过 golden |
| 容器栈（服务/端口/凭据） | `docker-compose.yml`；凭据 `.env`（不进 git） | 7 服务 | 基础设施层 | `docker compose up -d` 重建受影响服务 |
| Metastore 的 S3A/DB 连接 | `docker-compose.yml` metastore 服务的 `SERVICE_OPTS` | 见文件内注释 | metastore | `docker compose up -d metastore --force-recreate` |

## 微调点索引（总表）

> 「后续里程碑将新增」不是 TBD——是本手册随里程碑增行的既定节奏（每个里程碑交付新增行）。

| 微调点 | 位置 | 当前值 | 影响范围 | 调整建议 | 关联阶段 |
|---|---|---|---|---|---|
| wiki 分片数 | `python -m ingest.wiki N` 或 `ingest/wiki.py run(n_shards=)` | 1 | 数据量 | 验收用 1；W3 去重实验扩到 6 | ingest |
| 抽样种子/份数 | M2 起 `ingest/sampling.py`（WP-01 契约已定） | 未实装 | OSCAR 首跑样本 | 确定性抽样，种子进 git | ingest |
| 规范化 version | `configs/text_normalization.json` 的 `version` | 1.0.0 | 哈希口径兼容性 | 只递增不回退；改动写 decisions 理由 | clean/dedup |
| PII 掩码策略 | `clean/pii.py` 的 `MASK_SPANS` | id(3,4)/bank(4,4)/phone(3,4) | 脱敏产出 | 头尾保留位数权衡泄漏 vs 可读性 | clean |
| 批处理大小 | `ingest/wiki.py` 的 `BATCH_SIZE` | 10000 | 内存占用 | OOM 才降；不影响产物字节 | 全阶段 |
| pytest 范围 | `pytest.ini` | 全 tests/ | 验证面 | — | 测试 |
| 日志级别 | `LOG_LEVEL` 环境变量 | INFO | 日志量 | DEBUG 排障，INFO 日常 | 全阶段 |
| 强制重算 | 各阶段 `run(force=True)`；CLI `--force` | False | 缓存旁路 | 改配置后必须（M4 前） | 全阶段 |
| runs/changelog 台账路径 | `common/runs.py`/`changelog.py` 顶部常量 | `data/meta/` | 审计账本位置 | 不要动，看板依赖 | 全阶段 |
| 清洗规则阈值 | M4 起 `configs/`（clean） | 未实装 | clean 删除率 | 阈值扫描实验 E3 定 | clean |
| 去重参数 | M5 起 `configs/dedup_policy.json` | 未实装 | 误删/漏重曲线 | 网格实验 E1 定 | dedup |
| 质量权重/门禁 | M6 起 `configs/quality_weights.json` | 未实装 | 门禁通过率 | kappa 审计校准 | quality |
| 配比与 token 预算 | M7 起 `configs/mix_policy.json` | 未实装 | 发布集构成 | 按下游 ppl 反馈调 | mix |

## 常见操作配方

**改了配置/代码后完整重跑**（M4 前：“指纹不含配置”缺陷下的标准姿势）：

```bash
python - <<'PY'
from ingest import wiki
r = wiki.run(force=True)
print(r["cached"], r["rows"])
PY
# 然后重新上传 MinIO + python smoke/m1.py
```

**看某条文档的完整治理史**（M4 起 changangelog 有数据后）：

```bash
python - <<'PY'
from common.changelog import query_by_doc
for r in query_by_doc("<doc_id>"): print(r)
PY
```

**看跑了哪些 run**：`data/meta/runs.jsonl` 直接读，或
`python -c "from common import runs; [print(e) for e in runs.read_events()]"`

## 多模态插件配置指南（M11 起）

多模态治理的旋钮全部在 **`configs/multimodal_policy.json`**，分两节：

| 旋钮 | 位置 | 当前值 | 语义 |
|---|---|---|---|
| `joint_gate.modality_weights` | text 0.6 / image 0.4 | clip 关闭时的双模态权重（和必须=1，改错自动 fail-fast） |
| `joint_gate.modality_weights_cross_modal` | text 0.5 / image 0.2 / cross_modal 0.3 | clip 开启时的三模态权重（和必须=1）已定稿 2026-11-13：文本为主体、对齐为新增关键判别力、图像结构分信息量最低；先上线攒账本再校准 |
| `joint_gate.min_modality_score` | 0.30 | 短板木桶律：任一模态破此线 → 联合 remove |
| `joint_gate.min_joint_score` | 0.55 | 联合总分下限 |
| `image_text.clip.enabled` | true | CLIP 对齐门禁总开关：开启后打分器加载（缺 torch/模型 fail-fast 抛错，不静默降级） |
| `image_text.clip.model_path` | data/models/cn-clip-vit-base-patch16 | 模型目录（config.json architectures 自适应 CN-CLIP / OpenAI CLIP） |
| `image_text.clip.cos_lo` / `cos_hi` | 0.0 / 0.4（占位） | 余弦→[0,1] 归一化区间；正式值来自 `clip_bench` 标定（负样本 p95 / 正样本 p50），产物在 `data/clip/calibration_*.json` |
| `image_text.pair_rules.orphan_policy` | hold | 缺图或缺文的对：hold（隔离待复核）/ drop；**不许静默放行** |
| `image_text.pair_rules.min_caption_chars` | 5 | caption 过短 → drop(caption_too_short) |
| `image_text.pair_rules.min_image_bytes` | 1024 | 图像字节过小 → drop(image_bytes_too_small) |
| `image_text.dedup.image_exact_dup_policy` | collapse_by_source_trust | 同图不同文是否折叠；`keep_pairs_only` = 只折叠图+文全同的对 |

边界纪律（改旋钮前必读）：

- **文本五维阈值不在这里**——仍在 `configs/quality_policy.json`（各模态各管各的维度，multimodal 只管模态层权衡）；
- **簇代表源信任序不在这里**——仍在 `configs/dedup_policy.json` 的 `canonical_policy`（事实单一来源）；
- 改任何旋钮 = 改产物：配置进指纹，下游阶段自动重算（WP-01 D2 语义对多模态同样成立）；
- metastore 侧 `modality` / `modality_pair_id` 追加列见 `sql/multimodal_extensions.sql`（只追加，历史行读出 NULL = 未检测，三态语义）。

快速试拧（不落任何数据，纯函数层）：

```bash
python - <<'PY'
import modalities
from modalities.image import ImageTextSample
modalities.load_all()
cleaner = modalities.get("image", "cleaner")
s = ImageTextSample(doc_id="d1", source="wiki", source_id="1",
                    caption="一张测试用配对文本。", image_uri="s3://x/1.png",
                    image_bytes_hash="a" * 64, image_bytes_len=2048)
print(cleaner.clean(s))          # ('keep',)
PY
```

## 新增模态开发规范（M11 起）

按 `ARCHITECTURE.md` §7.4 的五步走，红线与自检：

1. **schema 先行**：在 `schema/multimodal_schema.json` 追加该模态样本定义（载荷字段 + 校验），**不改 unified_schema**（0.3.2 被 golden record 钉死；升 spine 留待首个真实多模态源）；
2. **插件包**：新建 `modalities/<m>/`，实现 `Sample` 子类 + 所需基类（`Collector`/`Cleaner`/`Deduper`/`QualityScorer`，见 `modalities/base.py` 模块头的判定协议），包 `__init__` 里 `register(...)`；
3. **旋钮进配置**：阈值/权重一律登记 `configs/multimodal_policy.json`，插件代码零字面量（宪法第 6 条，有元测试风格先例）；
4. **测试随行**：`tests/test_modalities.py` 追加接口冒烟 + 链路集成；预留能力（未实装的打分/去重）必须显式 `None`/`NotImplementedError` 并有测试钉住「不可用是可见的」；
5. **核心层零改动**：`tests/test_modalities.py::test_core_layer_never_imports_modality_plugins` 守卫这条边界——核心层 import 插件即测试变红。

纪律要点：模态专属依赖（图像解码器/音频指纹库/深度学习框架）只进插件包；未实装的读数是 `None`、未实装的接口调用显式报错——**不静默、不假装**。
