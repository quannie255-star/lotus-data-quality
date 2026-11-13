"""downstream/grad_select/benchmark.py — P11 性能对照：串行基准 vs PyTorch 并行形态

产出 data/downstream/grad_select_bench.json：每个 (N,d) 配置 × 每种实现形态的
计时（中位数）、相对串行的加速比、以及**正确性核验**（分数最大绝对误差 +
选中编号集合是否与串行基准完全一致）。课程评分「对比实验与结果分析」的原始记录。

口径：
- 数据由 datagen 固定种子生成（U[-1,1) → FP32 值），所有形态读同一份数据与 w
- 计时只含「构建张量 → 打分 → 排序选 top-k」全过程（同一起跑线）；串行基准
  连张量构建都没有，这对其有利，加速比因此是**保守**口径
- time.perf_counter；torch 形态先跑 1 次预热（分配器/内核缓存），再计时取中位数
- 线程数在进程内 torch.set_num_threads 设定；GPU 臂仅在有 CUDA 时运行

用法：python -m downstream.grad_select.benchmark [--quick] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from downstream.grad_select import core, datagen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "downstream" / "grad_select_bench.json"
M_REF = 64
CHUNK_ROWS = 4096
FULL_CONFIGS = [(1000, 32), (10000, 32), (50000, 32), (10000, 128), (50000, 128)]
QUICK_CONFIGS = [(1000, 32), (10000, 32)]


def _make_case(n: int, d: int):
    """同一 (n,d) 的固定数据：候选种子随规模变化，参考集/w 种子固定——可复现。"""
    cand_seed = 20261112 + n * 1000 + d
    cand = datagen.gen_dataset(n, d, seed=cand_seed)
    ref = datagen.gen_dataset(M_REF, d, seed=cand_seed + 1)
    w = datagen.gen_param(d, seed=777 + d)
    return w, cand.as_pairs(), ref.as_pairs()


def _timeit(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _check(result, base) -> dict:
    """正确性核验：分数逐条对照串行基准（FP32 vs FP64 容差），选中编号必须完全一致。"""
    max_dscore = max(abs(a - b) for a, b in zip(result.scores, base.scores))
    return {"max_abs_score_diff": max_dscore,
            "selected_identical": result.selected == base.selected,
            "k_identical": result.k == base.k}


def run_arm(name: str, fn, base, repeats: int, warmup: int) -> dict:
    t = _timeit(fn, repeats, warmup)
    out = {"seconds_median": round(t, 6), "repeats": repeats}
    if base is not None:
        out["correctness"] = _check(fn(), base)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="只跑小规模两档（冒烟）")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU 线程数（默认不动）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    report = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "m_ref": M_REF,
            "chunk_rows": CHUNK_ROWS,
            "repeats": args.repeats,
            "ratio": 0.05,
        },
        "configs": {},
    }

    for n, d in configs:
        w, cand, ref = _make_case(n, d)
        key = f"N{n}_d{d}"
        entry: dict = {}

        # 串行基准（core.py，纯 Python FP64）——大配置只做 1 次预热外计时以控制时长
        serial_repeats = args.repeats if n <= 10000 else 1
        t_serial = _timeit(lambda: core.select_topk(w, cand, ref), serial_repeats, warmup=0)
        base = core.select_topk(w, cand, ref)
        entry["serial_baseline"] = {"seconds_median": round(t_serial, 6),
                                    "repeats": serial_repeats}

        from downstream.grad_select import torch_kernel as tk

        for vname, fn in (("torch_loop", tk.select_loop),
                          ("torch_batch", tk.select_batch),
                          ("torch_chunk", lambda *a, **kw: tk.select_chunk(*a, chunk_rows=CHUNK_ROWS, **kw))):
            entry[vname] = run_arm(
                vname,
                lambda fn=fn: fn(w, cand, ref),
                base, args.repeats, warmup=1)
            entry[vname]["speedup_vs_serial"] = round(t_serial / entry[vname]["seconds_median"], 2)

        if torch.cuda.is_available():
            entry["torch_batch_cuda"] = run_arm(
                "cuda", lambda: tk.select_batch(w, cand, ref, device="cuda"),
                base, args.repeats, warmup=2)
            entry["torch_batch_cuda"]["speedup_vs_serial"] = round(
                t_serial / entry["torch_batch_cuda"]["seconds_median"], 2)

        report["configs"][key] = entry
        print(f"[{key}] serial={t_serial:.4f}s " + " ".join(
            f"{k}={v['seconds_median']:.4f}s(x{v.get('speedup_vs_serial', '-')})"
            for k, v in entry.items() if k != "serial_baseline"))

    # 线程数扫描（最大配置，torch_batch）：并行配置维度的对照
    n, d = max(configs)
    w, cand, ref = _make_case(n, d)
    from downstream.grad_select import torch_kernel as tk
    base = core.select_topk(w, cand, ref)
    sweep = {}
    for th in (1, 2, 4, 8):
        torch.set_num_threads(th)
        arm = run_arm(f"threads{th}", lambda: tk.select_batch(w, cand, ref), base,
                      args.repeats, warmup=1)
        arm["threads"] = th
        sweep[f"threads_{th}"] = arm
        print(f"[threads={th}] batch={arm['seconds_median']:.4f}s")
    report["thread_sweep"] = {"config": f"N{n}_d{d}", "arms": sweep}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
