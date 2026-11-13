# clip_bench — 图文跨模态对齐判别（DL 大作业本体 + Lotus 多模态门禁内核）

## 是什么 / 一句话定位

用 CLIP 判断「一张图 + 一段文字」是否语义匹配：输出 0~1 对齐分 + 阈值判定 +
梯度显著图证据。本目录是课程作业本体（数据构建/评测/标定/demo/报告），
对齐分内核在 `modalities/image/clip_align.py`（Lotus 多模态质量门禁同源复用）。

## 快速上手

```bash
# 依赖：pip install -r requirements-grad.txt（torch/transformers；torchvision 装法见该文件头注）
# 模型：CN-CLIP 本地已备 data/models/cn-clip-vit-base-patch16（ModelScope 下载，见开发日志）

# 英文基准评测（Flickr8k：标定集 500 图 / 测试集 1500 图，GPU 数分钟）
python -m clip_bench.run_eval --model data/models/cn-clip-vit-base-patch16 --tag cnclip

# 中文在域评测 + 标定（240 对真实人工中文 caption，CPU 约 1 分钟）
python -m clip_bench.zh_eval --corpus data/clip/zh_pairs --tag zh_children

# Lotus 门禁抽样报告（生产配置；--whatif-clip 用标定值临时开启后预览）
python -m clip_bench.zh_eval --corpus data/clip/zh_pairs --tag zh_children --gate --whatif-clip

# 交互 demo（上传图片 + caption → 对齐分/判定/显著图）
streamlit run clip_bench/demo_app.py

# 测试（指标手算对照 + 数据构建口径，不加载大模型）
python -m pytest tests/test_clip_bench.py tests/test_clip_integration.py tests/test_zh_eval.py -q
```

## 数据（真实数据，铁律 1）

**英文基准**：Flickr8k 8,091 张真实照片 × 5 条人工英文 caption（40,460 对）。
来源 ModelScope `llm416101531/Flickr8k`（Flickr8k.zip，1.1GB），解压于 `data/clip/flickr8k/`。

**中文在域**：`data/clip/zh_pairs/`（不入库，`metadata.jsonl` 记录 `image`/`text` 字段）——
HF `svjack/Chinese_Children_Image_Captioning_Dataset_Split0` 的 240 对真实儿童绘本图 +
人工中文 caption（caption 去空白归一，配对原样来自数据集，非机器翻译）。重下口径见
`开发日志.md` 2026-11-13 条目；语料仅用于阈值标定与门禁抽样，不做公开再分发。

负样本按课程方案自造：跨图随机配对（易负）+ 模型挖掘的最易混淆 caption（硬负）。
标定集/测试集固定 seed=2026 切分且**不相交**——阈值只在标定集定，指标只在测试集报
（中文 240 对：120 标定 / 120 测试，测试集内部再按正/易负/硬负各 1/3 拆 AUC 分口径）。

## 产出物

| 文件 | 内容 |
|---|---|
| `data/clip/eval_report_{tag}.json` | 判别 Acc/P/R/F1/AUC（整体+仅硬负）+ 检索 R@1/5/10 原始记录 |
| `data/clip/calibration_{tag}.json` | 约登阈值 + 正负分布分位数 + cos_lo/cos_hi 建议值（回填 Lotus 配置用） |
| `data/metrics/multimodal_{tag}.json` | 联合门禁抽样报告（verdict 分布 / 对齐分与联合分分位 / 误杀样例） |
| `data/clip/eval_report_zh_children.json`、`calibration_zh_children.json` | 中文在域评测与标定（2026-11-13，240 对真实中文图文） |
| `实验报告-DL大作业.md` | 方法/数据/结果/误差分析/局限（课程提交材料） |

## 与 Lotus 的关系（一份内核，两处落地）

```
modalities/image/clip_align.py   ← 内核：编码/余弦/标定归一/空读数None
        ├─ clip_bench/           ← 课程作业：评测 + 标定 + demo
        └─ modalities/image/paired.py → joint_gate 第三模态 cross_modal
           （configs/multimodal_policy.json 的 clip.enabled 开关，
             开启后图文不匹配样本被联合门禁 remove(low_quality_modality:cross_modal)）
```

标定产物回填 `configs/multimodal_policy.json` 的 `clip.cos_lo/cos_hi`——
Lotus 门禁的对齐分归一化区间来自真实数据标定，不拍脑袋。

**当前状态（2026-11-13）**：`clip.enabled=true`。中文在域标定完成（240 对，120 标定 /
120 测试）：约登阈值 0.401（cos 空间）= align 0.30 门禁线，pos_p95 0.4942 = align 1.0；
`cos_lo=0.361 / cos_hi=0.4942` 已回填。生产路径抽样 120 对真实中文图文 **0 误杀**
（`data/metrics/multimodal_zh_children.json`），测试集 AUC easy 0.981 / all 0.669 /
hard 0.356。决策与局限全文见 `configs/multimodal_policy.json` 的 `enabled_decision`。

## 已知边界（诚实声明）

- CLIP 对齐分是「图文语义匹配」弱信号，不判断「文说得对不对」。
- 低对齐分 ≠ 一定脏（文字图/抽象图不敏感）→ Lotus 侧建议 hold 复核语义由
  pair_rules 承担，联合门禁 remove 只在分数确实破线时发生。
- CN-CLIP 跑英文 caption 是跨语言弱口径（预演已实测：AUC easy 0.984 / hard 0.589）；
  英文正式口径用 OpenAI CLIP ViT-B/32，中文能力验证用 CN-CLIP + 中文 caption 测试。
- 中文标定语料是**儿童绘本短句**（240 对）：hard AUC 0.356 低于随机，说明该领域里
  语义近似对的余弦几乎不可分——门禁只能拦明显错配，不能当「语义正确性」判据；
  生产长文本语料上线后应增量复标。
- 硬负样本由被评模型自己挖掘（存在自证成分）——口径已在报告与 口径说明.md 声明。
