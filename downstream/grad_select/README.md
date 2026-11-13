# grad_select — 梯度数据筛选内核（课程作业 P11 ⇄ Lotus 因果链终证共用）

## 是什么 / 一句话定位

按「候选样本梯度 vs 参考集平均梯度的余弦相似度」给数据打分并选 top-k 的可复用内核：
课程作业 P11 用线性模型喂它（基础版串行 + PyTorch 并行版），Lotus 用它从治理后语料里
选最利于目标任务的子集做 LoRA 微调验证（扩展版，见 `downstream/lora_finetune/`）。

## 课程口径（来源：并行计算课程项目选题说明 Version3 · P11 及其附录）

- 线性模型：预测 `wᵀx`，损失 `L = ½(wᵀx − y)²`，逐样本梯度 `g = (wᵀx − y)·x`
- 参考平均梯度 `ḡref = (1/M) Σ g′ⱼ`；候选分数 `sᵢ = cos(gᵢ, ḡref)`，**零分母记 0**
- 选 `k = ⌈0.05·N⌉` 条，按分数降序，**同分按编号升序**；输出全部分数及选中编号
- 数据：固定种子，U[-1,1)，FP32 值；参考规模 N=1000/10000/50000，M=64，d=32/128
- 各版本读同一份数据与 w，计算期间不更新模型

## 快速上手

```bash
# 正确性（含课程 PDF 官方调试用例 + 边界用例 + 变异测试）
.venv/Scripts/python.exe -m pytest tests/test_grad_select.py -q

# 性能对照（串行 vs torch 三形态 + 线程扫描 + GPU 臂），产 JSON
.venv/Scripts/python.exe -m downstream.grad_select.benchmark            # 全量五档配置
.venv/Scripts/python.exe -m downstream.grad_select.benchmark --quick    # 冒烟两档
```

## 文件

| 文件 | 职责 |
|---|---|
| `core.py` | 基础版：纯 Python 串行（零依赖、FP64 计算），正确性基准 |
| `datagen.py` | 固定种子数据生成（U[-1,1) → FP32 值，逐位可复现） |
| `torch_kernel.py` | 并行版：`loop`（逐样本对照）/ `batch`（整批向量化）/ `chunk`（分块+及时释放） |
| `benchmark.py` | 计时、加速比、正确性核验，产 `data/downstream/grad_select_bench.json` |

## 复现数字

- 调试用例分数 `1/0/-1/0`、选中编号 `0`：`pytest tests/test_grad_select.py::test_pdf_debug_case`
- 基准耗时/加速比：跑上面的 benchmark 命令，读 JSON（数字不手抄，见 口径说明.md §3）
- 课程实验报告（含结果分析）：`实验报告-P11.md`，原始记录 `data/downstream/grad_select_bench.json`

## 已知边界（诚实声明）

- 串行基准是 FP64、torch 形态是 FP32：分数对照用容差 1e-5，选中编号要求完全一致
- 计时含张量构建（对串行有利 → 加速比是保守口径）；不含数据生成与磁盘 IO
- 扩展版（LoRA 微调验证）不改变本内核——只是换一种「梯度来源」，见 DESIGN.md §4

## 文档

DESIGN.md（为什么这么设计）· 口径说明.md（每个数字的定义与门禁）· 开发日志.md（过程与坑）
