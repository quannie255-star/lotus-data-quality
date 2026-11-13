"""downstream/grad_select/datagen.py — P11 数据生成（固定种子，U[-1,1)，FP32 值）

课程附录口径：特征、标签和参数均固定随机种子，从 U[-1,1) 生成并保存为 FP32。
参考规模：N=1000/10000/50000，M=64，d=32 或 128。

实现说明：用标准库 random + ctypes 做 FP32 舍入，保证「串行基准（core.py，零依赖）」
与「并行版（torch float32）」消费**逐位相同**的输入值——正确性对照才成立。
numpy 的 float32 转换与此等价（同为 IEEE754 round-to-nearest）。
"""

from __future__ import annotations

import ctypes
import random
from dataclasses import dataclass


def to_fp32(v: float) -> float:
    """把 Python float（FP64）舍入为 FP32 可表示值（仍以 float 返回）。"""
    return ctypes.c_float(v).value


@dataclass(frozen=True)
class Dataset:
    """一组样本：X[i] 是第 i 条的 d 维特征（FP32 值），y[i] 是标签（FP32 值）。"""
    X: list[list[float]]
    y: list[float]

    def __len__(self) -> int:
        return len(self.X)

    def as_pairs(self) -> list[tuple[list[float], float]]:
        return list(zip(self.X, self.y))


def gen_dataset(n: int, d: int, seed: int) -> Dataset:
    """生成 n 条 d 维样本：特征与标签均 U[-1,1) → FP32。同种子逐位可复现。"""
    rng = random.Random(seed)
    X = [[to_fp32(rng.uniform(-1.0, 1.0)) for _ in range(d)] for _ in range(n)]
    y = [to_fp32(rng.uniform(-1.0, 1.0)) for _ in range(n)]
    return Dataset(X=X, y=y)


def gen_param(d: int, seed: int) -> list[float]:
    """生成固定参数向量 w（U[-1,1) → FP32）。各版本读取同一份 w（课程口径）。"""
    rng = random.Random(seed)
    return [to_fp32(rng.uniform(-1.0, 1.0)) for _ in range(d)]
