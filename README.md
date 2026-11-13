# 濯莲 Lotus

![tests](https://img.shields.io/badge/tests-214%20passed-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**可编程的数据质量控制器 —— 数据清洗不再是写死的脚本，而是你手里的一排旋钮。**

> **Lotus — a programmable data quality controller. Clean data isn't written, it's tuned.**

一行技术定位：**面向大模型训练数据集的可审计、可复现、可权衡数据治理流水线。**

- 📌 **当前稳定支持**：中文文本数据集治理（23 万篇真实文档端到端验证）
- 🚧 **演进方向**：多模态控制器（图文/音视频配对样本清洗、跨模态去重、模态对齐门禁）

---

## 为什么不是又一个清洗脚本

传统数据清洗是写死的规则：想改一个标准要动代码，效果好不好全凭感觉，换一批数据就得重写一遍。濯莲把治理过程变成一台**控制台**——每个决策是一个旋钮，每次结果是一排读数，每次去留有一本账：

- **📊 看得见** —— 质量是仪表读数，不是一句「感觉更干净了」。每篇文档五维打分（完整性/唯一性/一致性/干净度/长度），每个数据版本出一张指标卡。23 万篇实测语料：治理前后六项指标全部改善（如句式完整性 0.726 → 1.000，精确重复率归零），下游困惑度消融 clean 较 raw **−4.3%**（带置信区间）。
- **🎛️ 拧得动** —— 每个治理决策都是一个配置旋钮。清洗阈值、去重参数、门禁权重、配比策略全部在 `configs/*.json`，改标准改配置不改代码；配置指纹进入幂等缓存，改了配置自动重算，绝不静默输出过期结果。
- **🔁 换得起** —— 治理配方可以切换。同一份原始数据，「干净优先 / 多样优先 / 去重优先」三档配方（`configs/recipes.json`）在看板一键应用：先看逐旋钮 diff 与 1 万行抽样的代价预览，再触发全链重算，代价与收益在指标卡上直接对比；并可一键导出自包含 HTML 的「数据质量评估报告」（指标卡 + 四臂消融 + 配方代价 + 账本摘要）。
- **📒 查得回** —— 每条数据的每次去留都入账本。changelog 记录删除原因、引用对象、相似度；run 生命周期全量留痕，两段式提交保证账实一致；字节级幂等，任何历史版本可一键复现。

**脏的进来，纯的出去，旋钮在你手里。**

## 同类工具对标

| 工具 | 定位 | 与 Lotus 的差异 |
|---|---|---|
| [datatrove](https://github.com/huggingface/datatrove) | 大规模语料处理流水线库（过滤/去重/采样的可组合 blocks） | 强在吞吐与规模化；Lotus 补的是**治理视角**：门禁消融、配置指纹幂等、逐条审计账本、下游困惑度验证闭环 |
| [dolma / dolphin 系清洗工具包](https://github.com/allenai/dolma) | 一次性数据清洗流水线（规则标签导向） | 跑一遍出结果，改标准=重跑工程；Lotus 是**可反复调参的控制器**：拧旋钮 → 自动重算 → 前后指标卡对比 → 账本归因 |
| [Argilla](https://github.com/argillaio/argilla) | 数据标注与人工反馈平台 | 人工标注优先；Lotus 是自动化治理与审计，两者互补而非替代（标注数据同样可以作为一条「源」进入治理链路） |
| 常规 ETL（Airflow / dbt 等） | 数据搬运与表变换编排 | 回答「数据到没到、对不对得上」；Lotus 回答「**这批数据质量如何、改一个阈值的代价是什么、结果能否复现**」 |

一句话：Lotus 不和它们比搬运规模，Lotus 的差异位是**可调参、可审计、可权衡、可复现**的数据质量实验平台。

## 文本链路实测结果

七层治理链路已在真实数据上端到端跑通：**多源采集 → 清洗 → 去重 → 质量门禁 → 配比采样 → 版本化发布 → 下游验证**。每个数字可归因、每次运行可复现、每个阈值可配置。

| 层 | 结果 |
|---|---|
| 采集 | 231,314 篇真实多源中文文档 |
| 清洗 | 删 6.1%（样板 8,609 / 非目标语言 3,771 / 过短 1,900），PII 真脱敏 82 条 |
| 去重 | 精确 + 近重复删 0.52%，全部留有「谁压掉谁、相似度多少」的账 |
| 质量门禁 | 五维打分，放行 88.5% |
| 配比发布 | 2M token 预算分层采样，发布 v0.1（license 执法剔除不合规源） |
| 下游验证 | 四臂困惑度消融：clean 较 raw **−4.3%**（带置信区间）；并发现配比采样的分布代价。因果链终证：Qwen3-0.6B LoRA 三臂对照（梯度 top-5% / 随机 5% / 全量），全量 SFT 使测试集 ppl 降 **20%**，梯度筛选优势在强微调区间稳健（减损 **8.6%**），温和区间 5% 选择逊于全量（如实报，见 `downstream/lora_finetune/`） |
| 工程底座 | 7 容器基础设施 + 214 项测试通过（含变异测试/合同测试；未装 torch 时其中 5 项并行臂自动跳过）+ 9 个真实踩坑实录 |

> 以上为**文本链路**实测。核心治理底座（审计账本、幂等缓存、质量门禁调度、版本发布、运行时编排）是**模态无关**的，多模态能力通过插件层扩展——见 `ARCHITECTURE.md` §7 与 `modalities/`。

### 路线图（控制台的四块仪表）

```
✅ 审计闭环     每条数据去留可归因到 run，产物逐文件验货
✅ 旋钮全量开放 全部阈值/规则/配比在配置层，改标准不动代码
✅ 文本链路稳定 23w中文文档端到端验证，困惑度消融完成
✅ 指标卡       治理前后全指标对比看板（六项指标表+图，一键导出评估报告 HTML）
🚧 多模态插件层 接入图文配对样本；跨模态去重、模态对齐校验（CLIP 对齐分已接入，
                config 开关 clip.enabled，评测标定见 clip_bench/）、多模态PII脱敏
✅ 配方引擎     干净/多样/去重三档配方一键切换 + 抽样代价预览（多模态联合配方未做）
⬜ 旋钮自动搜索 给定目标指标，带约束自动搜索最优清洗参数
✅ 因果链终证(文本) Qwen3-0.6B LoRA 三臂对照（梯度 top-5% / 随机 5% / 全量 2396 篇，同锚点同测试集）：
                全量 SFT 使独立测试集 ppl 降 **20%**（治理语料下游收益成立）；梯度筛选优势只在强微调
                （遗忘）区间稳健（选对数据减损 8.6%），温和区间 5% 选择逊于全量约 15%（如实报，不粉饰）。
                诚实结论与原始记录见 downstream/lora_finetune/（多模态场景终证未做）
```

## 快速上手

前置：Windows + Docker Desktop + Python 3.11+。

```bash
git clone https://github.com/quannie255-star/-Lotus.git && cd -Lotus
pip install -r requirements.txt
pip install -r requirements-grad.txt   # 可选：P11 并行臂 + LoRA 因果链终证（torch 镜像装法见该文件头注）

setx HF_ENDPOINT "https://hf-mirror.com"     # 新开终端生效；huggingface-cli login

cp .env.example .env                          # 改两个 replace_me
bash docker/metastore/README.md 中的三条 curl # metastore 外挂 jar（一次性）

docker compose up -d --build                  # 七容器栈（镜像源方案见 dev-lessons #004）
python -m ingest.wiki 1                       # 首个源：23 万篇真实维基文档
python smoke/m1.py                            # 一行冒烟：PASS/FAIL

streamlit run dashboard/app.py                # 看板：指标卡/三档配方/评估报告导出
python -m downstream.grad_select.benchmark    # P11 梯度筛选：串行 vs 并行基准 → JSON
python -m downstream.lora_finetune.run        # 因果链终证：Qwen3-0.6B LoRA 双臂对照（GPU；加 --full-arm 补全量第三臂）
```

## 文档地图

| 想了解 | 去哪 |
|---|---|
| 怎么改任何参数/规则/阈值 | `MANUAL.md`（反查表 + 微调点索引） |
| 数据质量看板 / 三档配方 / 评估报告导出 | `dashboard/README.md` |
| 梯度数据筛选（P11 课程内核）与 LoRA 因果链终证 | `downstream/grad_select/`、`downstream/lora_finetune/`（各含 DESIGN/口径说明/开发日志） |
| 多模态插件怎么配、怎么开发新模态 | `MANUAL.md` § 多模态插件配置指南 / § 新增模态开发规范 |
| 内部实现机理（数据流/账本/幂等/追踪实验） | `docs/internals-guide.md`（配 `docs/traces/` 可重跑实验） |
| 架构与设计决策 | `ARCHITECTURE.md`（§7 多模态分层）、`docs/blueprint.md`、`docs/wp-01-stage-protocol.md` |
| 开发过程与踩坑实录、机制问答 | `DEVELOPMENT_LOG.md`、`docs/dev-lessons.md`（9 个真实坑）、`docs/design-qa.md` |
| 阶段契约与项目宪法 | `docs/wp-01-stage-protocol.md` §0 七条不变量 |

## 项目宪法（七条不变量）

原始数据不可变 · 一表一阶段 · 确定性身份 · 字节级幂等 · 变更可归因 · 事实单一来源 · 坏数据不静默。任何偏离都是 bug，豁免必须显式记录（见 `docs/wp-01-stage-protocol.md`）。

---

License: MIT · 作者 quannie · 项目前期代号 datalens

> **仓库名注记**：GitHub 仓库名为 `-Lotus`（历史命名含前导横杠，不合仓库命名规范），文档内一律以 **Lotus** 指代本项目；clone 命令以实际仓库地址为准。若日后在 GitHub 侧更名，旧地址会自动重定向，不影响既有链接。
