"""downstream/grad_select — 梯度数据筛选内核（课程作业 P11 ⇄ Lotus 因果链终证共用）

- core.py        基础版：纯 Python 串行，零依赖，正确性基准
- datagen.py     固定种子数据生成（U[-1,1) → FP32 值）
- torch_kernel.py 并行版：PyTorch 向量化（逐样本循环 / 整批 / 分块三形态）
- benchmark.py   性能对照（串行 vs 并行，加速比与正确性核验，产 JSON）

模块四份文档见本目录 README.md / DESIGN.md / 口径说明.md / 开发日志.md。
"""

from downstream.grad_select.core import Selection, cosine_score, mean_gradient, sample_gradient, select_topk
from downstream.grad_select.datagen import Dataset, gen_dataset, gen_param

__all__ = ["Selection", "cosine_score", "mean_gradient", "sample_gradient", "select_topk",
           "Dataset", "gen_dataset", "gen_param"]
