# clip_bench · DESIGN

## 一句话定位

图文对齐判别的评测与标定套件：给「对齐分」一个可信的口径（阈值从数据标定来、
指标在分离测试集上报、硬负样本检验真实区分度），并把标定产物回填 Lotus 门禁配置。

## 背景与约束

- 课程方案：L1 零样本判别 → L2 硬负+阈值标定+可解释 → L3 交互 demo。
- Lotus 侧：`paired.py` 的 `pair_cross_modal_alignment` 从 v0 起就是预留的 None 口子，
  本模块把它填成真实分数（方案 B：cross_modal 第三模态键）。
- 本机：RTX 4060 8GB；OpenAI CLIP 官方源不可达，hf-mirror 慢速（~90KB/s），
  CN-CLIP 走 ModelScope 秒级；transformers 5.22 对 legacy CLIP 系有两处行为变更（见开发日志）。

## 关键设计决策

1. **内核放插件层（modalities/image/clip_align.py），评测放 clip_bench/** →
   torch/transformers 只进插件层与独立评测模块，核心层（quality/multimodal.py 等）
   零改动、零新依赖（宪法 + M11 分层边界元测试不破）。否决：把评测塞进
   modalities/image/ ——插件层职责是线上打分，评测是离线实验，混放会污染边界。
2. **双后端自适应（CN-CLIP / OpenAI CLIP）而不是锁死一个** → Lotus 中文语料用
   CN-CLIP；课程评测基准（Flickr8k 英文）用 OpenAI CLIP 才有文献可比性。
   同一内核按 config.json architectures 自动分发。代价：两条预处理路径，
   用 test_real_model 对照测试钉住行为一致。
3. **标定集/测试集分离 + 约登指数定阈值** → 阈值在 500 图标定集上定，指标在
   1500 图测试集上报，防「阈值对着测试集调」的自欺；AUC/F1 手写实现（秩公式），
   不引入 sklearn——指标可审计（每个数字能在 evaluate.py 里指到行）。
4. **硬负样本 = 模型自挖的最易混淆 caption** → 课程方案明示「跨样本随机配对太简单」；
   同时诚实标注自证成分（挖掘用的就是被评模型）——预演实测 hard AUC 0.589 vs
   easy AUC 0.984，证明硬负确实打到了软肋，这个差距本身就是报告的核心分析素材。
5. **clip.enabled 默认 false，配置开关启用** → M11 红线「不破坏现有测试」：
   既有 20 项模态测试断言 alignment=None（v0 行为）保持绿；启用是显式配置动作，
   且启用后缺 torch/模型加载失败 fail-fast 抛错，不静默降级回规则版。
6. **对齐分归一化区间 [cos_lo, cos_hi] 来自标定** → 原始余弦（典型 0.1~0.5）直接
   当质量分会被 min_modality_score=0.30 的木桶线误判；用负样本 p95 → 正样本 p50
   做线性归一，让「0.30 破线」有真实数据语义。标定前配置里是占位默认值并显式标注。
7. **热力图用梯度显著图而不是 attention rollout** → d(cos)/d(pixel) 一步 backward
   即得，无需 hook 多层注意力、无 rollout 的口径争议；诚实命名为「梯度显著图」
   （它回答「哪些像素影响了对齐分」，不声称是「模型注视点」）。

## 架构 / 数据流

```
Flickr8k(真实) → dataset.py: split(seed=2026) → calib(500) / test(1500)
                     │ build_pairs: pos + easy_neg(跨图随机)
                     │ add_hard_negatives: 模型挖掘最混淆 caption
                     ▼
clip_align.ClipAlignScorer.embed_images/embed_texts（批量, GPU）
                     ▼
evaluate.py: score_pairs → 标定集 youden_threshold + 分位数
                     ▼        （calibration_{tag}.json → 回填 multimodal_policy clip.cos_lo/hi）
             测试集 binary_metrics/AUC(整体|仅易负|仅硬负) + retrieval R@k
                     ▼
data/clip/eval_report_{tag}.json → 实验报告-DL大作业.md（数字不手抄）

Lotus 线上路径：paired.PairedQualityScorer.score → clip.enabled?
  → clip_align.score(uri, caption) → per_modality[cross_modal]
  → joint_gate(三模态权重) → pass / remove(low_quality_modality:cross_modal | no_score)
```

## 边界与局限

- 单参考检索口径（每图 1 条 caption 进池），与 Flickr8k 官方 5-caption 口径的 R@k
  数值不可直接对表——报告只用于本实验内部对照，已声明。
- 评测为**零样本**（课程 L1/L2 范围）；微调 CLIP（L3 扩展）未做，不假装做了。
- demo 的显著图对 CPU 用户约 2-5 秒/张（一次 backward），GPU <1 秒。

## 变更记录

- 2026-11-12 初版：dataset/evaluate/run_eval/demo_app + clip_align 内核 +
  paired.py 三模态接线 + 21 项新测试全绿；CN-CLIP 预演跑通（100/200 图，62 秒）。
