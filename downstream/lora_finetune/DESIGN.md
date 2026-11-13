# lora_finetune · DESIGN

## 一句话定位

把「梯度筛选」（课程 P11 内核口径）接到真实治理语料上，用 LoRA 微调双臂对照
（筛选 vs 随机）补上 Lotus「治理 → 下游训练收益」的最后一环证据。

## 背景与约束

- Lotus 已证明「洗得干净」（六项指标全改善 + char-ngram ppl −4.3%），但路线图
  「⬜ 因果链终证：分别微调模型验证清洗策略的训练收益」一直缺真实模型实验。
- 本机 RTX 4060 Laptop 8GB：Qwen3-0.6B fp32 ≈ 2.4GB 权重 + LoRA 梯度/优化器状态，
  单条 256 token 前反向可跑；bf16 更快但打分精度对照需另行验证 → 先 fp32（决策 3）。
- 网络约束：download.pytorch.org 不可达（SSL 层被断），走清华 PyPI（CPU）/
  SJTU pytorch-wheels 镜像（cu126）；HF 模型走 hf-mirror.com（README 已有此纪律）。

## 关键设计决策

1. **双臂对照只比「选择策略」这一个变量** → warm-up 后保存锚点 φ0，两臂各自从 φ0
   **重新加载**再 SFT，训练超参（epochs/lr/seed/截断长度）严格一致；差异只在
   120 篇训练文档的选择方式（梯度 top-5% vs 随机 5%）。否决的替代方案：两臂串行
   共用一个模型对象继续训——φ 状态互相污染，对照失效。代价：每臂多一次模型装载。
2. **随机臂是「同数量对照」而不是「全量对照」** → 课程 PDF 明示「可与相同样本数量的
   随机筛选作对照」；比全量训练回答的问题不同（选择效率 vs 数据数量），不混为一谈。
   全量臂留作可选扩展（时间允许再加，报告分列）。
3. **fp32 而不是 bf16** → 打分是梯度余弦，数值精度直接影响排序边界样本；fp32 下
   0.6B 在 8GB 显存内可跑，用速度换「分数可复算」。代价：全量打分耗时约为 bf16 两倍。
4. **手动训练循环，不用 trl/Trainer** → 每步可审计（loss/steps 进报告）、依赖面小、
   与项目「确定性铁律」（seed 冻结、同序洗牌）一致。代价：没有混合精度/梯度累积等
   现成能力——本实验规模用不上。
5. **评测集复用 ppl 消融的冻结口径（clean 层 seed=2026 前 64 行参考/其余 536 行测试）** →
   与既有 −4.3% 消融同源可比（同一批评测文本，两个引擎各测各的）；候选与评测行做
   SHA-256 防泄漏剔除（实测剔除 6 条——release ⊂ clean 链路真有重叠）。
   否决的替代方案：另抽一批测试集——与既有消融失去可比性，且「评测集一经发布永不改」
   是项目宪法。代价：参考/测试行是 clean 层文档（治理后分布），对 grad_arm 略有利？
   ——不，两臂用同一测试集，偏置对称；但该口径意味着结论是「对 clean 分布的收益」。
6. **打分逐条 backward、即算即释** → 课程附录「候选梯度用于打分后便可释放」；
   LoRA 参数量小（r=8 七目标模块 ≈ 百万级），单条梯度向量 MB 级，无需分块矩阵。
   否决：vmap/functorch 批量逐样本梯度——实现复杂度高，0.6B 规模下循环已够快。

## 架构 / 数据流

```
data.build() ──> candidates(2396) / reference(64) / test(536)   [真实语料+防泄漏]
      │
load_model(φ init, seed=42) → sft(warmup 1%×1ep) → save φ0 → eval baseline(test)
      │
mean_reference_grad(φ0) → score_candidates(φ0, 逐条 backward+释放) → top-⌈0.05N⌉=120
      │
      ├─ grad_arm:   _fresh_from_anchor(φ0) → sft(120篇×4ep) → evaluate_ppl(test)
      └─ random_arm: _fresh_from_anchor(φ0) → sft(120篇×4ep) → evaluate_ppl(test)
      │
lora_report.json：config/env/data/warmup/scoring/selection/baseline/双臂/conclusion
```

## 边界与局限

- 单 seed 单次运行；GPU 上浮点归约顺序不保证跨驱动版本逐位一致——报告记录 env 全量。
- 「收益」的度量 = 独立测试集 token 级 ppl（越低越好）；没有下游任务基准（CEVAL 等）
  ——0.6B + 120 篇文档的规模撑不起基准分显著性，ppl 是该规模下唯一稳定信号。
- warm-up 1% 数据本身来自候选集（课程流程如此），锚点 φ0 已「见过」少量候选——
  两臂对称承受该偏置，不影响对照有效性。

## 变更记录

- 2026-11-12 初版：data/model/run 三件套 + tests/test_lora_data.py（6 passed，真实语料）；
  防泄漏剔除实测 6 条重叠。冒烟与全量运行结果见 开发日志.md。
