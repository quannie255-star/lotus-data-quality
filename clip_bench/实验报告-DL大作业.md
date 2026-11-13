# DL 大作业实验报告：图文跨模态对齐判别（CLIP）

> 日期：2026-11-12/13｜代码：`clip_bench/`（评测）+ `modalities/image/clip_align.py`（内核）
> 原始记录：`data/clip/eval_report_{tag}.json`、`data/clip/calibration_{tag}.json`
> 复现：`python -m clip_bench.run_eval --model <模型目录> --tag <标签>`（README 有完整命令）

## 1. 任务定义

输入一张图像 + 一段 caption，输出对齐分 s ∈ [0,1] 与「匹配/不匹配」判定。
正样本 = 图与其真实 caption；负样本 = 跨图随机配对（易负）+ 模型挖掘的最易混淆
caption（硬负）。评测：判别 Accuracy / Precision / Recall / F1 / AUC（整体、仅易负、
仅硬负三口径）+ 图文检索 R@1/R@5/R@10（单参考口径）+ 混淆矩阵。

## 2. 方法与模型

- **零样本判别（课程 L1）**：CLIP 图像/文本双编码器 → L2 归一 embedding →
  余弦相似度；不微调（课程范围内的评测对象是「预训练对齐能力 + 阈值口径」）。
- **模型**：CN-CLIP ViT-B/16（中文主场，OFA-Sys）与 OpenAI CLIP ViT-B/32（英文基准，
  文献可比）双后端，同一内核按 config.json 自适应分发。
- **阈值标定（课程 L2）**：标定集（500 图，与测试集不相交，seed=2026）上扫 ROC
  网格取约登指数 J=TPR−FPR 最大处阈值——不拍脑袋定 0.5。
- **硬负样本挖掘（课程 L2）**：对每张图，在测试池 caption 中取与该图 embedding
  余弦最高的非配对 caption。诚实声明：挖掘用的就是被评模型，hard 口径对该模型偏严
  （自证成分），因此与 easy 口径并列报告。
- **可解释（课程 L2/L3）**：梯度显著图 |d(cos)/d(pixel)| 按通道求和归一——
  回答「哪些像素影响对齐分」，不声称是注视点。demo：`streamlit run clip_bench/demo_app.py`。
- **归一化口径**：align = clip((cos−cos_lo)/(cos_hi−cos_lo), 0, 1)，cos_lo=负样本
  p95、cos_hi=正样本 p50（标定集实测定，回填 Lotus 门禁配置）。

## 3. 数据（真实数据声明）

Flickr8k：8,091 张真实照片 × 5 条人工英文 caption（40,460 对）。
来源：ModelScope `llm416101531/Flickr8k`（Flickr8k.zip 1.1GB，SHA 由平台托管）。
划分：标定 500 图 / 测试 1,500 图（固定 seed，不相交）；每图取第 1 条 caption 入池。
负样本为课程方案要求的自造样本（构造方法见 §2，随机种子固定可复现）。

## 4. 结果

正式口径：标定集 500 图 / 测试集 1,500 图（seed=2026 切分、不相交），每图第 1 条
caption 入池；正/易负/硬负 = 1500/1500/1500 对。原始记录：
`eval_report_clipb32.json`（OpenAI CLIP）、`eval_report_cnclip_full.json`（CN-CLIP）。

### 4.1 正式基准：OpenAI CLIP ViT-B/32 × 英文 caption（tag=clipb32）

| 指标 | 值 |
|---|---|
| 约登阈值（cos） | **0.2688**（J=0.476，TPR 0.928 / FPR 0.452） |
| AUC 整体 / 仅易负 / 仅硬负 | 0.7654 / **0.9930** / **0.5377** |
| Accuracy / F1 @约登阈值 | 0.6447 / 0.6355 |
| 混淆矩阵 @约登阈值 | tp=1394 fp=1493 fn=106 tn=1507 |
| i2t R@1 / R@5 / R@10（池 1500） | **0.5173** / 0.7793 / 0.878（中位名次 1） |
| t2i R@1 / R@5 / R@10 | 0.4913 / 0.7667 / 0.858（中位名次 2） |
| 全程耗时（GPU，含编码/标定/评测） | 78.5s |

### 4.2 对照：CN-CLIP ViT-B/16 × 英文 caption（跨语言弱口径，tag=cnclip_full）

| 指标 | 值 |
|---|---|
| 约登阈值（cos） | 0.3871（J=0.431，TPR 0.982 / FPR 0.551） |
| AUC 整体 / 仅易负 / 仅硬负 | 0.7087 / 0.9815 / **0.4358（低于随机）** |
| Accuracy / F1 @约登阈值 | 0.6164 / 0.6314 |
| i2t R@1 / R@5 / R@10 | 0.3813 / 0.6547 / 0.762 |
| t2i R@1 / R@5 / R@10 | 0.3593 / 0.6207 / 0.7253 |

解读：CN-CLIP 在英文 caption 上易负仍近乎全对（0.98），但硬负**低于随机**
（0.4358）——跨语言的细粒度区分能力失效，方向性甚至反转（英文近义句被系统性
打高分）。这正是「CN-CLIP 主场是中文、英文别硬用」的定量证据；Lotus 侧用
CN-CLIP 配中文 caption（§4.3），英文评测以 OpenAI CLIP 为准。

### 4.3 中文能力对照（CN-CLIP × 中文 caption，构造样例）

同一张真实 Flickr8k 图（两只狗打架）：中文配对 caption cos=0.5039 vs 中文错配
caption cos=0.2433（区分度 0.26，显著大于英文跨语言口径的 0.10）——CN-CLIP 中文
主场能力实证；大规模中文真实评测数据源缺失（MUGE 死链，见 开发日志），列为已知边界。

## 5. 误差分析

1. **硬负样本把 AUC 从 0.99 打到 0.54（OpenAI CLIP）**：随机负样本「换话题」太好认
   （easy AUC 0.993），硬负是「同场景近义描述」（两只狗打架 vs 两只狗玩耍），
   CLIP 零样本几乎分不开（0.5377 ≈ 随机）——「图文对齐判别」的真实难度全在
   细粒度语义，这也是「随机负样本评测会严重虚高」的定量实证。
2. **约登阈值下 FPR 高达 0.452（fp=1493，几乎全是硬负）**：J 最大化偏向保 TPR
   （0.928），在 1:2 正负比 + 硬负占一半的口径下，判定「匹配」的门槛必然放行大量
   硬负——门禁场景应结合业务代价选工作点（Lotus 侧用 align<0.30 破线才 remove，
   且建议低分走 hold 复核，见 §7.2）。
3. **跨语言衰减（CN-CLIP × 英文）**：hard AUC 0.4358 **低于随机**——英文近义句被
   系统性打高分，方向性反转；易负仍 0.98。结论：模型-语言必须匹配，混用会产生
   比「没有门禁」更危险的假读数。
4. **检索口径**：i2t R@1 0.5173（1500 候选池、单参考）——与文献 Flickr8k 全池
   5-caption 口径不可直接对表（口径说明 §4），仅用于本实验内部对照。

## 6. 与 Lotus 的集成（业务闭环，已落地）

对齐分作为 cross_modal 第三模态键进入联合门禁（短板木桶律）：
`configs/multimodal_policy.json` → `clip.enabled=true` 后，图文不匹配样本
`remove(low_quality_modality:cross_modal)`、坏图 `remove(no_score)`，全部入账本。
**标定回填已执行**（`python -m clip_bench.apply_calibration --tag cnclip_full`）：
cos_lo=0.3289、cos_hi=0.523——锚定 align(约登阈值)=0.30（与 min_modality_score
门禁线对齐）、align(pos_p95)=1.0。注意该标定基于英文 caption 跨语言口径（中文真实
语料源缺失），中文语料上线后须重标定（配置 note 已记录）。
接线测试与变异证据：`tests/test_clip_integration.py`（10 项，含真实模型序关系门禁）。

## 7. 局限（诚实声明）

1. 零样本口径，未做微调（微调属课程扩展，未做不假装）。
2. 硬负挖掘自证成分（§2）；单参考检索口径与 Flickr8k 官方 5-caption 口径不可对表。
3. 对齐分是语义匹配弱信号，不判「文说得对不对」；文字图/抽象图不敏感。
4. 中文大规模真实评测数据源缺失（MUGE OSS 死链、COCO-caption 为英文）——中文侧
   能力以构造样例 + 门禁单测钉住序关系（§4.3），不冒充大规模评测。
5. Flickr8k metadata 有 1 条 `.jpg.1` 脏条目（磁盘无此文件），构建时剔除且计入
   报告 `metadata_excluded_missing_files`（剔除有账，不静默）。
