# lora_finetune — Lotus 因果链终证（治理 → 选择 → 训练收益 闭环）

## 是什么 / 一句话定位

用 Qwen3-0.6B + LoRA 验证「Lotus 治理 + 梯度筛选出的数据，微调后在独立测试集上
是否优于随机筛选」——补齐 README 路线图最后一块 ⬜「因果链终证」，同时是课程
P11 的扩展内容（模型微调实验）。

## 链路

```
Lotus release 语料（2,396 篇候选，真实治理产物）
  → warm-up：1% × 1 epoch → 锚点 φ0（保存 LoRA 适配器）
  → 梯度打分：φ0 固定，逐候选 backward 取 LoRA 参数梯度，与参考集平均梯度
    ḡref（clean 层冻结评测集前 64 行）算余弦 → top-⌈0.05N⌉ = 120 篇
  → 双臂 SFT（各从 φ0 重新加载，超参严格一致）：
      grad_arm   = 梯度筛选 top-5%        random_arm = 同 seed 随机 5%
  → 独立测试集（评测集其余 536 行，防泄漏剔除后与候选零交集）
  → token 级 ppl 对比 → data/downstream/lora_report.json
```

## 快速上手

```bash
# 前置：pip install -r requirements-grad.txt（torch cu126 装法见该文件头注）
# 模型：Qwen3-0.6B 置于 data/models/Qwen3-0.6B/（ModelScope 下载，gitignored）
.venv/Scripts/python.exe -m downstream.lora_finetune.run --smoke   # 冒烟：40 候选/短截断/1 epoch，分钟级
.venv/Scripts/python.exe -m downstream.lora_finetune.run           # 全量默认配置（GPU 约 34 分钟）
# 多配置对照（复用锚点与打分缓存，各约 15 分钟）：
.venv/Scripts/python.exe -m downstream.lora_finetune.run --lr 5e-5 --epochs 2 --tag lr5e5ep2 --reuse-anchor --reuse-scores
.venv/Scripts/python.exe -m downstream.lora_finetune.run --lr 1e-4 --epochs 3 --tag lr1e4ep3 --reuse-anchor --reuse-scores
# 三臂对照（加全量训练臂，回答 top-5% 是否接近全量；约 21 分钟）：
.venv/Scripts/python.exe -m downstream.lora_finetune.run --lr 5e-5 --epochs 2 --tag lr5e5ep2_3arm --reuse-anchor --reuse-scores --full-arm
```

## 结果（2026-11-13 实跑，原始记录 data/downstream/lora_report*.json）

超参敏感性（grad top-5% vs random 5%，同锚点 φ0=23.51）：

| 配置 | 筛选臂 ppl | 随机臂 ppl | 筛选 vs 随机 | 两臂 vs 基线 |
|---|---|---|---|---|
| lr2e-4 × 4ep | 35.36 | 38.69 | **−8.61%** | 双双大幅变差（遗忘区间） |
| lr1e-4 × 3ep | 24.10 | 23.77 | +1.41%（噪声） | 微差 |
| lr5e-5 × 2ep | 21.67 | 21.35 | +1.51%（噪声） | **双双改善（−7.8%/−9.2%）** |

三臂对照（lr5e-5×2ep 生产配置，加全量臂，`lora_report_lr5e5ep2_3arm.json`）：

| 臂 | 训练文档 | test ppl | vs 基线 φ0=23.51 | vs 全量 |
|---|---|---|---|---|
| grad_arm（top-5%） | 120 | 21.671 | −7.82% | +15.24%（逊于全量） |
| random_arm（随机 5%） | 120 | 21.349 | −9.19% | +13.53%（逊于全量） |
| **full_arm（全量）** | 2,396 | **18.805** | **−20.01%** | — |

**诚实结论**：
1. **治理语料的下游收益成立且强**——全量 SFT 后 test ppl 降 **20%**（因果链终证 ✅）。
2. **梯度筛选的优势只在强微调（遗忘）区间稳健**（lr2e-4×4ep 选对数据减损 8.6%）；
   温和区间 grad≈random（±1.5% 属单 seed 噪声），不声称筛选普遍优于随机。
3. **top-5% 选择未能在本配置逼近全量**：全量（18.81）显著优于任一 5% 子集（~21.5），
   5% 选择是有损的（省 95% 数据、ppl 高 15%）。数据选择真正划算的场景是「全量负担不起」
   或「强微调致遗忘」——后者本实验已证，前者需更大语料/更贵模型，未在 0.6B+2396 篇尺度出现。

## 复现数字

- 三配置对照与结论：上表 ← `lora_report.json` / `lora_report_lr5e5ep2.json` / `lora_report_lr1e4ep3.json`
- 跨运行确定性：打分中位数 0.115816（run1 重算 = run2 重算 = run3 缓存复用，6 位小数一致）；
  baseline ppl 三次运行均 23.5105
- 数据集构建（候选 2,396 / 参考 64 / 测试 536 / 防泄漏剔除 6）：`pytest tests/test_lora_data.py`
- 打分/选择口径与课程基础版一致性：`downstream/grad_select/口径说明.md` §1 + `tests/test_lora_model.py`

## 已知边界（诚实声明）

- **0.6B + LoRA 是「验证数据选择有效」的最小实验**，不是「训出强模型」——结论只能是
  「本配置下筛选是否优于随机」，不外推到其他模型/数据规模。
- 单 seed 单次运行，无显著性检验；若筛选不优于随机，报告如实记录（口径纪律）。
- 训练任务 = 纯文本续写（causal LM loss），非指令微调——语料是百科/新闻正文，
  没有指令-回答对；这与课程 PDF 的 SFT 示例（指令样本）不同，已在口径说明 §2 声明。
- 测试集 ppl 与既有 char-ngram 消融（−4.3%）是**两个引擎**，数字不可互比。

## 文档

DESIGN.md（决策与否决方案）· 口径说明.md（每个数字定义与门禁）· 开发日志.md（过程与坑）
