"""dashboard/autotune.py — 带约束的旋钮自动搜索（离线 CLI，只读预览，不写配置）

问题定义（先写清楚，避免「调参」变成不可解释的数字游戏）：
- 搜索空间：预览开放的 clean / quality 旋钮网格。dedup 不在内——预览跳过 MinHash，
  没有签名就没有真实去重率，把它放进搜索只会制造假信号。
- 硬约束：最终保留率 = quality 放行行数 / 实际读入行数 ≥ min_keep。
  「指标好看」不能靠把数据删光换，先把可接受的数据量钉住。
- 目标：放行行的平均 quality_score 最大；同分则保留率高者胜（多留数据不吃亏）。
- 确定性：固定 seed 的随机抽样网格 + 固定 tie-break，同一份数据同一组参数 → 同一份结果。

口径边界（必须随结果一起报）：
- 1 万行固定前缀抽样（`_sample_raw_rows` 顺序读，不 shuffle）；
- uniqueness 无签名恒为 1.0、non_pollution 无 benchmark 恒为 1.0——目标值只用于横向
  比较候选之间的相对优劣，不等于全量重算后的质量分；
- 搜索结果只是「建议值」，本模块绝不写 configs/*.json；应用需人工确认后用
  `dashboard.knobs.write_knobs`（或控制台）落盘，再走全链重算验证。

用法：
    python -m dashboard.autotune --sample 10000 --evals 24 --min-keep 0.5 \
        --out data/metrics/autotune.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# {类别: {旋钮键: 候选值}}。取值覆盖「默认值附近 ± 一档」，过密只会浪费预算。
DEFAULT_GRID = {
    ("clean", "min_cjk_chars"): [5, 10, 20, 30, 50, 80],
    ("clean", "min_han_ratio"): [0.0, 0.2, 0.3, 0.5],
    ("clean", "max_link_density"): [0.1, 0.2, 0.3, 0.5],
    ("quality", "min_total_score"): [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    ("quality", "min_dim_score"): [0.2, 0.3, 0.4, 0.5],
}

OBJECTIVE = "max mean quality_score of gate-pass rows, subject to keep_rate >= min_keep"


def _combos(grid: dict) -> list[dict]:
    """把网格展开成覆盖 dict 列表；按 key 排序保证顺序确定。"""
    keys = sorted(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        overrides: dict = {}
        for (cat, knob), value in zip(keys, values):
            overrides.setdefault(cat, {})[knob] = value
        out.append(overrides)
    return out


def evaluate_preview(overrides: dict, n_sample: int, seed: int = 2026) -> dict:
    """把预览结果压成搜索需要的三个数：读入行数 / 放行行数 / 放行行分数和。"""
    from dashboard.preview import run_preview

    r = run_preview(overrides or None, n_sample=n_sample, seed=seed)
    return {"n": r["n_read"], "pass": r["quality"]["pass"],
            "score_sum": r["quality"]["pass_score_sum"]}


def search(evaluate, baseline_overrides: dict | None = None,
           grid: dict | None = None, budget: int = 24, min_keep: float = 0.5,
           seed: int = 2026) -> dict:
    """执行约束搜索。evaluate(overrides) -> {n, pass, score_sum}。

    候选 #0 恒为「当前配置」，保证结果可与基线直接对照；其余候选从网格里按 seed
    无放回抽 budget-1 个。budget 是总评估次数（含基线）。
    """
    if budget < 2:
        raise ValueError("budget 至少为 2（基线 + 1 个候选）")
    if not 0.0 <= min_keep <= 1.0:
        raise ValueError(f"min_keep 必须在 [0,1]（给了 {min_keep}）")

    grid = DEFAULT_GRID if grid is None else grid
    combos = _combos(grid)
    rng = random.Random(seed)
    if len(combos) <= budget - 1:
        picked = combos
    else:
        picked = rng.sample(combos, budget - 1)
    candidates = [dict(baseline_overrides or {})] + picked

    history = []
    best = None
    best_key = None
    for i, overrides in enumerate(candidates):
        res = evaluate(overrides)
        n, passed = res["n"], res["pass"]
        keep_rate = passed / n if n else 0.0
        mean_quality = res["score_sum"] / passed if passed else None
        feasible = mean_quality is not None and keep_rate >= min_keep
        rec = {
            "index": i,
            "baseline": i == 0,
            "overrides": overrides,
            "n": n,
            "pass": passed,
            "keep_rate": round(keep_rate, 6),
            "mean_quality": None if mean_quality is None else round(mean_quality, 6),
            "feasible": feasible,
        }
        history.append(rec)
        if feasible:
            key = (rec["mean_quality"], rec["keep_rate"])
            if best_key is None or key > best_key:
                best, best_key = rec, key

    baseline = history[0]
    improvement = {
        "mean_quality": (round(best["mean_quality"] - baseline["mean_quality"], 6)
                         if best is not None and baseline["mean_quality"] is not None
                         and best["mean_quality"] is not None else None),
        "keep_rate": (round(best["keep_rate"] - baseline["keep_rate"], 6)
                      if best is not None else None),
    }
    return {
        "config": {
            "objective": OBJECTIVE,
            "min_keep": min_keep,
            "budget": budget,
            "seed": seed,
            "grid": {f"{cat}.{knob}": list(vals) for (cat, knob), vals in sorted(grid.items())},
            "excluded_knobs": {
                "dedup.verify_threshold": "预览跳过 MinHash，无签名可算去重率；放进来只会造假信号",
            },
        },
        "baseline": baseline,
        "best": best,
        "improvement": improvement,
        "history": history,
    }


def _diff(baseline: dict, candidate: dict) -> dict:
    """候选相对基线的旋钮差异 → {类别: {旋钮: [旧值, 新值]}}（含基线未列出的键）。"""
    from dashboard.preview import read_knobs

    base = baseline or read_knobs()
    out: dict = {}
    for cat, knobs in candidate.items():
        for knob, value in knobs.items():
            old = base.get(cat, {}).get(knob)
            if old != value:
                out.setdefault(cat, {})[knob] = [old, value]
    return out


def _fmt_overrides(overrides: dict) -> str:
    parts = [f"{cat}.{knob}={value}" for cat, knobs in overrides.items()
             for knob, value in knobs.items()]
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="带约束的旋钮自动搜索（只读预览，不写配置）")
    parser.add_argument("--sample", type=int, default=10000, help="抽样行数")
    parser.add_argument("--evals", type=int, default=24, help="总评估次数（含基线）")
    parser.add_argument("--min-keep", type=float, default=0.5, help="最终保留率下限")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None, help="结果 JSON 落盘路径")
    args = parser.parse_args(argv)

    result = search(lambda ov: evaluate_preview(ov, args.sample, args.seed),
                    budget=args.evals, min_keep=args.min_keep, seed=args.seed)
    result["config"]["sample"] = args.sample

    print(f"样本 {args.sample} 行，评估 {len(result['history'])} 个候选，"
          f"min_keep={args.min_keep}")
    b = result["baseline"]
    print(f"基线：保留率 {b['keep_rate']:.3f}，放行行平均分 "
          f"{b['mean_quality'] if b['mean_quality'] is not None else 'n/a'}")
    if result["best"] is None:
        print("无可行候选：所有候选的保留率或放行行数都不满足约束，建议放宽 min_keep")
    else:
        best = result["best"]
        print(f"最优：保留率 {best['keep_rate']:.3f}，放行行平均分 "
              f"{best['mean_quality']}（候选 #{best['index']}）")
        print(f"  旋钮差异：{_fmt_overrides(_diff(result['baseline']['overrides'], best['overrides']))}")
        print(f"  相对基线：平均分 {result['improvement']['mean_quality']:+.4f}，"
              f"保留率 {result['improvement']['keep_rate']:+.4f}")
        print("  注意：建议值不会自动写回 configs；确认后经 knobs.write_knobs 落盘并全链重算验证。")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
