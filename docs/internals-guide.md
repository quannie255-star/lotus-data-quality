# Lotus（濯莲）内部机理指南（Internals Guide）

> 定位：**as-built 的内部实现讲解**——数据流转、核心算法、模块依赖、状态管理四个维度逐层拆解，
> 面向「要在这个仓库上独立开发与维护的人」。与其它文档的分工：`ARCHITECTURE.md` 是模块清单、
> `docs/wp-01-stage-protocol.md` 是阶段契约、本文回答「为什么与怎么运转」。
> 配套可重跑实验：`docs/traces/`（沙箱机制实验 + 真实数据单文档追踪），§8 有实验实录。
> 维护规则：本文描述的是代码实态，任何机制变更（尤其 common/）须同步本文。

---

## 0. 心智模型：账本驱动的确定性数据工厂

整个项目只做一件事：把脏语料治理成可发布的训练集。但它用一个不寻常的方式做——**每一步计算
都被三个问题约束：你是谁（身份）、你动了什么（账本）、要不要重算（指纹）**。

理解本项目的钥匙：它同时维护两个平行世界——

- **数据面**：一张张 parquet 表（每阶段一张，只增不改上一张）；
- **账面**：三本账 + 一份指纹（`runs.jsonl` / `changelog.jsonl` / `first_seen/` / 各目录 `manifest.json`）。

所有「酷炫能力」（可复现、可审计、改配置自动重算）不是功能，而是这两个世界之间四条纪律的
自然推论。记住这一点，下面每个模块都会变得显然。

## 1. 全景数据流

```
                     ┌──────────────── 数据面（parquet 表）────────────────┐
 HF wiki ──ingest──► data/raw/{源}/           （原始层，永不变）
 mock oscar ──┐      data/clean/source={s}/    （六规则清洗 + PII 脱敏）
 news/pdf ────┤  ──► data/dedup/source={s}/    （精确+MinHash 近重，+signatures+report）
              │      data/quality/source={s}/  （五维分+门禁，+gate_report）
              │      data/mix/candidates/      （token 预算分层采样）
              │      data/release/v0.1/        （+provenance_card.yaml，license 执法）
              │      data/downstream/          （四臂 ppl 报告）
              └──────────────────────────────────────────────────────────┘
                     ┌──────────────── 账面（横切，所有阶段共用）────────────┐
                     data/meta/runs.jsonl        每次 run 的 start/finish
                     data/meta/changelog.jsonl   每条数据的每次去留
                     data/meta/first_seen/*.json doc_id → 首见时间
                     data/{目录}/manifest.json   每张表的输入指纹 + 产物哈希
                     └──────────────────────────────────────────────────────┘
```

空间法则（WP-01 §1）：表路径 `data/{zone}/{stage}/source={s}/part-*.parquet`；分区只按
source；mix 起取消分区（配比取代来源成为主查询维度）。**阶段之间零代码调用**——clean 不知道
dedup 的存在，它只读 dedup 的表。这就是「一表一阶段」。

## 2. 底座六模块（common/）——项目的真正主体

### 2.1 `common/ids.py` — 三个哈希，三种身份

| 函数 | 定义 | 用途 | 关键性质 |
|---|---|---|---|
| `spine_doc_id` | sha256(`source:source_id`)（`ids.py:160`） | 文档身份 | **永不复算**——与内容、URL、时间无关 |
| `content_hash` | sha256(utf8(content))（`ids.py:37`） | 内容指纹 | 零规范化，ingest 早筛 |
| `normalized_content_hash` | 规范化后再 sha256（`ids.py:152`） | 去重指纹 | 跨全半角/简繁/空白形态识同文 |

第三者的规范化流水线是「规则=数据」的典范：七个步骤函数（零宽删除、NFKC、繁简折叠、空白
收尾……）注册在 `_STEP_FUNCS` 表（`ids.py:127`），**执行顺序由 `configs/text_normalization.json`
的 `pipeline_order` 决定**，代码里没有硬编码规则值。改规范化规则必须递增配置 `version`——
去重身份的口径变了。

### 2.2 `common/runs.py` — run 生命周期账

一次阶段执行 = 一个 run。`start()` 写 start 事件并生成 run_id（`stage-UTC时间戳-随机8位`，
`runs.py:36`）；`finish()` 写终态事件。两条硬校验（`runs.py:97`）：没有 start 不许 finish、
不许重复 finish。**失败也必须 finish(failed) 再抛异常**——「没记录」和「成功」在审计上必须
可区分。stage 合法值读自 `reason_registry.json`，代码里不另写一份。

### 2.3 `common/changelog.py` — 数据变更账

每条 remove/redact/quarantine 是一行 JSON，**写入前三重校验**（`changelog.py:72`）：
① jsonschema 结构校验；② 跨字段规则表（`changelog.py:37`，如 `action=redact` 必须带
`pii_types`、`remove/modify` 必须带 `old_hash`——规则写成数据表而非 if 堆叠）；③ reason 码
必须登记在 registry 且 stage 归属匹配。dedupe 键 = `(doc_id, stage, action, reason, old_hash)`，
重跑不喂胖日志。

### 2.4 `common/idempotent.py` — 幂等引擎（全项目最核心的 150 行）

`run_stage()`（`idempotent.py:79`）是每个阶段的执行壳：

```
runs.start(stage)                                 # ① 先记账（哪怕马上命中缓存）
  ├─ 命中缓存？三条件（idempotent.py:121）：
  │    输入指纹相同 ∧ 产物目录在 ∧ 逐文件 sha256 与 manifest 对账通过
  │    → 返回 {cached: True}，transform 零执行
  └─ 未命中：
       建临时目录 .{stage}.tmp-{uuid8}/           # ② 不直接写终态目录
       transform(inputs, work_dir, run_id)        # ③ 业务逻辑全在这里
       rmtree(final_dir) + work_dir.rename(...)   # ④ 原子替换
       逐文件算哈希 → 写 manifest.json（先 .tmp 再 replace）
runs.finish(stage, success, note="cached"|"executed")
任何异常 → finish(failed, note=异常) 后原样上抛
```

两个精髓：**缓存要验货**（`_outputs_intact`，`idempotent.py:62`——防有人手改产物后被当合格品
发出）；**指纹是链式哈希**（`fingerprint()`，`idempotent.py:25`：排序后的 文件名+长度+sha256
逐个喂入 digest，顺序稳定）。

### 2.5 `common/stage.py` — WP-01 六件套

- **`configs_digest` + `snapshot_configs`**（`stage.py:26,43`）：把本阶段配置文件**拷贝**进
  `.clean.configs/` 之类的快照目录，拷贝件并入 `input_paths` 参与指纹。效果：改
  `clean_policy.json` → 指纹变 → 自动重算。修复了 M1 时代缺陷「改配置忘 force = 静默输出
  过期结果」。
- **`latest_outputs`**（`stage.py:67`）：下游读上游表之前先验它的 manifest——逐文件哈希对账，
  不符抛 `UpstreamNotReadyError`。`only=` 参数是 M6 真实事故的补丁：quality 曾把共享
  manifest 里 clean+dedup 两层都读进来，43 万行算了两遍。
- **`StagedChangeLog`**（`stage.py:113`）：**两段式提交**——transform 期间变更记录暂存
  `changelog.staging.jsonl`（part 文件）；只有 run_stage 成功返回后才 `commit(run_id)`
  批量入全局账（一次读全量建 dedupe 键集合，O(n)——替代曾 O(n²) 的逐条扫描）；失败路径
  `discard()`。实现**账实一致**：全局 changelog 里出现的每个 run_id 必属于一次成功 run；
  失败 run 的尸检在 runs.jsonl，在 changelog 里零痕迹。
- **`run_pipeline`**（`stage.py:199`）：刻意做薄的顺序驱动器，不做 DAG——那是 Airflow 的活。

### 2.6 `common/first_seen.py` — 把时钟从数据行里赶出去

`first_seen_at` 要进数据行，但 `now()` 进数据行 = 字节级幂等的死刑。解法是台账：
`resolve(doc_id)` 查历史→查本次 pending→才取 now（`first_seen.py:43`）；`commit()` 在
**transform 成功后**才原子合并落盘。数据没发布，就不该有「首次入库」这回事。

## 3. 一个阶段的完整解剖：clean（所有阶段同构）

`clean/run.py:46` 的 `run()` 是 WP-01 阶段契约的标准实现，五步：

1. **快照配置**：`snapshot_configs([clean_policy.json, reason_registry.json])` → `.clean.configs/`；
2. **定位并验货输入**：对每个有 raw 产物的源 `latest_outputs(data/raw/{源})`（`clean/run.py:64`）；
3. **幂等执行**：`run_stage("clean", transform, inputs+snaps, data/)`（`clean/run.py:92`）。
   transform 闭包内：此刻才创建 `StagedChangeLog`（run_id 此刻才真实存在——账本里不许有
   占位身份）→ `iter_clean` 逐行过规则链 → 幸存行按源写 `source={s}/part-00000.parquet`；
4. **账本收口**：成功 → `slog.commit(run_id)`；失败 → `discard()`；**cached 命中 → 也
   discard**（part 文件内容本就重复，`clean/run.py:100`）；
5. 返回 StageResult。

规则链在 `clean/base.py:apply_rules`：规则是无状态纯函数
`fn(row, cfg) -> ("keep",) | ("drop", reason) | ("transform", new_row, reason, action, meta)`，
装配顺序由配置 `rules_order` 决定。PII 脱敏走 transform 判定，审计载荷（old_hash/new_hash/
pii_types）走 meta 侧信道——不污染表的封闭 schema。

`quality/run.py` 是第二个变奏：输出表 = spine + 8 个分数列（`QUALITY_EXT_SCHEMA`，
`quality/run.py:27`）；`gate_report.json` 本身也是产物（参与指纹）；hold 行写 `quarantine` +
`review_status=pending_review`（隔离是一等公民，可复核可放回）。看懂这两个 run，就看懂了
全部七个阶段——mix/release/downstream 只是 transform 业务不同，骨架一致。

## 4. 核心算法层

**去重**（`dedup/engine.py:85` `dedup_pass`，两级）：
① exact——按 `normalized_hash` 分组，同组非簇代表直接删（similarity=1.0）；
② near——只在 exact 幸存者上建 MinHashLSH（128 排列 × 16 带，隐式阈值 ≈0.74）产生**候选对**，
候选对再算**真实 Jaccard ≥ 0.8** 才算数——LSH 是召回手段不是判决；
③ 并查集成簇（`UnionFind`，合并方向按字节序，与插入顺序无关）；簇代表 = 源信任序 → 内容长
降序 → doc_id 升序（`_pick_canonical`，全确定性）。

**质量五维**（`quality/engine.py`）：每维一个纯函数，0~1 加权总分，门禁三分。最有故事的一维
是 `score_completeness`（`engine.py:24`）：只统计 ≥15 字符「句式行」的标点结尾率——标题/列表/
引用天然不以标点结尾，把它们算进分母等于「把结构当缺陷」，这曾误杀 27.4% 的维基文档。门禁的
牙齿是维度下限 0.30，不只是总分。

**下游验证协议**（`downstream/ppl.py`）——防两个经典混淆因子：不是「在 A 语料上算 ppl vs 在
B 上算 ppl」（分布不同不可比），而是**用 A 训 LM₁、用 B 训 LM₂，在同一冻结评测集上比 ppl**，
报 bootstrap 95% CI（seed=42 固定）。`CharNgramLM` 是纯 Python 5-gram add-k 平滑
（`ppl.py:22`）；`corpus_text` 抽训练语料时行级洗牌（防只吃到字典序头部）。

**指标卡**（`metrics/card.py:36`）：确定性蓄水池抽样 ≤20k 行（seed=2026）；全量只有
n_docs/total_chars；语义多样性未接入就是 `null`——空读数不假装。

**模态插件层**（`modalities/`，M11）：注册表 + `Sample`/`Cleaner`/`Deduper`/`QualityScorer`
四基类；文本插件是既有引擎的适配器绑定；元测试守卫「核心层永不 import 插件」。

## 5. 状态管理全景

| 状态 | 位置 | 写者 | 生命周期语义 |
|---|---|---|---|
| 数据本体 | `data/{zone}/{stage}/source=s/*.parquet` | 阶段 transform（原子替换） | 不可变；上游表永不改 |
| 指纹+产物哈希 | 各目录 `manifest.json` | run_stage | 命中缓存的三条件之一 |
| 运行审计 | `data/meta/runs.jsonl` | runs.start/finish | 追加式，失败也有尸检 |
| 数据变更审计 | `data/meta/changelog.jsonl` | StagedChangeLog.commit | **只有成功 run 能写** |
| 时钟 | `data/meta/first_seen/*.json` | FirstSeenLedger.commit | transform 成功才落盘 |
| 暂存区 | `changelog.staging.jsonl`、`.stage.tmp-*/` | transform 运行中 | 崩溃即废，零全局副作用 |
| 配置快照 | `.{stage}.configs/` | snapshot_configs | 兼指纹锚与 provenance |
| 注册表 | `configs/*.json` | 人（唯一事实源） | 代码零字面量，元测试守卫 |

**崩溃恢复语义**（设计上最漂亮的部分）：任何一步崩——产物目录未动（work_dir 还没
rename）、全局账本零新增（part 被 discard 或根本没 commit）、runs.jsonl 有失败尸检、重试
绝对安全（dedupe 键吸收重复、first_seen pending 未提交）。不需要任何「恢复工具」，重跑同
一条命令就是恢复。

## 6. 模块依赖关系

```
configs/*.json（唯一事实源，lru_cache 只读）
     ▲
common/{ids, runs, changelog, idempotent, stage, first_seen, textstats}
     ▲                                    ▲
各阶段 run.py（clean/dedup/quality/mix/release/downstream）──相互零 import
     ▲                                    ▲
     │ 阶段间只通过 latest_outputs(表) 耦合   │
modalities/*（插件层，单向依赖核心；元测试守卫反向依赖）   dashboard（只读 data/）
```

依赖方向严格单向：configs ← common ← 阶段 ← 插件/看板。分层边界由
`tests/test_modalities.py::test_core_layer_never_imports_modality_plugins` 守卫。

## 7. 与并行计算的交叉点（P11 前置知识）

字节级幂等（宪法第 4 条）要求同样输入两次跑出字节一致的产物——但并行分片-归约天然顺序
不定。**并行化必须做有序归约**：分片按字典序分配、聚合前排序、或用可结合且可交换的确定性
算子。这是「并行计算知识」与「数据工程宪法」的交叉考题：做对了，P11 的 per-example
梯度/损失打分才能直接进 `quality/` 的打分体系；做错了，第一次并行重算就会破坏
`test_forced_rerun_is_byte_identical` 一类的合同测试。

现成可复用设施：`downstream/ppl.py` 的 `CharNgramLM.log2_prob` 就是 per-example loss 的
零依赖起点；四臂 ppl 协议就是「梯度分筛 top/bottom 验证」的对照框架。

## 8. 追踪实验实录（2026-11-05 跑，脚本在 `docs/traces/`）

### 实验①：沙箱里的 clean 阶段全机制（`docs/traces/trace_lab_mini.py`）

7 行构造数据（好文×2/同文对/过短/英文/乱码/含真身份证号）跑三遍同命令：

```
[第 1 遍] 首跑 —— P1 规则链逐行实时判定：
    good-1  cjk= 159 → keep
    dup-1   cjk= 159 → keep          ← clean 不去重：同文对留给 dedup 阶段（阶段分工）
    short-1 cjk=   2 → drop(too_short)
    latin-1 cjk= 120 → drop(lang_mismatch)
    moji-1  cjk=  80 → drop(lang_mismatch)   ← 剧本外的真实发现：乱码行汉字占比先崩，
                                               lang_ratio 在 mojibake 之前拦截——规则顺序即语义
    pii-1   cjk= 173 → transform(redact/pii_redacted)
[第 2 遍] cached=True，规则链零执行；changelog 行数 4 → 4（两段式+dedupe 不喂胖账本）
[第 3 遍] force=True 重算：首跑 vs force 产物哈希逐文件相等 —— 字节级幂等直接证据
[账本] 4 条记录（3 remove + 1 redact）全部带 run_id；暂存 part 已随 commit 删除
[产物] 幸存 3 行；pii-1 脱敏后 '110***********002X' 且 content_hash == ids.content_hash(新 content)
```

### 实验②：真实数据单文档追踪（`docs/traces/trace_lab_real.py`）

抓 `data/raw/wiki` 第一篇真实文档（恰好是维基「数学」词条，9,538 字）沿七层查踪：

```
raw       ✓ doc_id=e8851d05… / CC BY-SA 4.0 / first_seen_at=2026-10-29T14:33:16
台账      ✓ first_seen/wiki.json 记录与行内值一致（时钟台账在起作用）
changelog ○ 零记录——一路放行没被任何阶段动过（零记录也是信息：审计上可区分「没查过」）
clean     ✓
dedup     ✓
quality   ✓ 五维分 {completeness 0.8, 其余 1.0, 总分 0.95}，quality_run_id 可回溯
mix       ✗ 未被 2M token 分层采样选中（发布集只有 2,402 篇）
release   ✗ 同上
manifest  七个阶段各有一条输入指纹（clean→dedup→quality→mix→release→metrics→downstream 信任链）
runs      尾部出现一条只有 start 没有 finish 的 clean 运行——被中断的执行，账本如实暴露
```

两段实验分别回答了「机制怎么转」（沙箱可重放）与「数据去了哪」（真实账本可回溯）。
