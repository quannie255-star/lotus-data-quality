# grad_select · DESIGN

## 一句话定位

一个「梯度余弦打分 + top-k 筛选」内核，同时服务课程作业 P11（线性模型）与 Lotus
因果链终证（LoRA 模型梯度）——内核只写一次，两处落地。

## 背景与约束

- 课程要求：基础版 + 改进并行版，验证正确性、比较性能；线性模型、OpenMP 或 PyTorch。
- Lotus 路线图缺口：「⬜ 因果链终证——分别微调模型，验证清洗策略的训练收益」（README.md:66）。
- 本机：Windows，RTX 4060 Laptop 8GB，Python 3.11 venv；torch 为新增依赖（仅并行/扩展版需要，
  基础版零依赖可独立跑）。

## 关键设计决策

1. **梯度来源抽象成「打分函数在外、梯度计算可换」** → 线性模型解析梯度（core/torch_kernel）
   与 LoRA 反向传播梯度（lora_finetune）共用同一套「余弦打分 + 零分母记 0 + (−分数,编号)
   排序 + ⌈0.05N⌉」口径。否决的替代方案：课程作业与 Lotus 各写一套——两处口径漂移风险，
   且违背衔接方案「内核只写一次」。代价：lora_finetune 需依赖 core 的口径常量。
2. **基础版用纯 Python 而不是 numpy** → 串行语义无可争议（逐样本、无隐藏向量化），
   课程「基础版 vs 并行版」的对照才干净；numpy 版会引入「库向量化算不算并行」的口径争议。
   代价：大配置下慢（这正是加速比要展示的事实，不是缺陷）。
3. **并行版选 PyTorch 而不是 OpenMP/C 扩展** → 课程两者皆可；PyTorch 与扩展版（LoRA 需要
   autograd）同栈，一套依赖两用；Windows 上 OpenMP 工具链（MSVC + libomp）配置成本高
   （本机有 MSVC link.exe 抢占等已知坑）。代价：CPU 并行度受 torch 线程池限制，
   用线程扫描臂（benchmark `thread_sweep`）如实呈现。
4. **torch 版做三形态（loop/batch/chunk）而不是只做 batch** → 课程评分重点是「并行设计
   是否合理 + 瓶颈分析」：loop 证明「换了库不等于并行」（Python 级迭代仍是瓶颈）、
   batch 是向量化上限、chunk 回答课程附录「4NP 字节存储」的分块+及时释放论据。
   代价：多两条代码路径，但共享 `_scores_from_grads`/`_finish`，漂移面小。
5. **数据生成用标准库 random + ctypes FP32 舍入** → 保证纯 Python 基准与 torch float32
   消费逐位相同的输入（正确性对照的前提）；同种子逐位可复现，满足「附测试数据或生成方法」。
   代价：生成速度慢于 numpy（一次性成本，不进计时）。

## 架构 / 数据流

```
datagen(seed) ──> Dataset(X,y) FP32值 ─┬─> core.select_topk        (串行基准, FP64 计算)
                                       └─> torch_kernel.select_*   (loop/batch/chunk, FP32)
                                              │  device=cpu|cuda
                                              ▼
                     Selection(scores 全量, selected 编号升序, k=⌈0.05N⌉)
                                              ▼
              benchmark.py：计时(中位数) + 加速比 + 正确性核验(容差1e-5/编号全等)
                                              ▼
                     data/downstream/grad_select_bench.json（原始记录）
```

扩展版数据流（Lotus 因果链终证）：release 语料 → lora_finetune 用 Qwen3-0.6B+LoRA 的
autograd 梯度替换解析梯度 → 同口径打分选 top-5% → SFT → 独立测试集 vs 随机筛选对照。

## 边界与局限

- 线性模型基础版只是课程载体，不代表真实数据筛选效果；扩展版结论边界见 lora_finetune 文档。
- FP32 vs FP64：极端近平分样本理论上可能在两实现间翻转选中集合；实测（benchmark
  `correctness.selected_identical`）在全部五档配置上为 true。若未来出现翻转，属数值
  边界而非逻辑缺陷，需在口径说明中记录实例。
- GPU 臂只在有 CUDA 时出现；报告数字以实际运行环境为准，不跨环境外推。

## 变更记录

- 2026-11-12 初版：core/datagen/torch_kernel/benchmark + tests/test_grad_select.py；
  课程调试用例与边界用例全绿（8 passed，torch 5 skipped 待装）。
