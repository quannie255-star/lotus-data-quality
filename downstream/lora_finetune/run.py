"""downstream/lora_finetune/run.py — 因果链终证编排器（warm-up → 锚点 → 打分 → 双臂 SFT → 评测）

流程（课程 PDF「模型实验定义」+ 衔接方案 §3.2）：
1. 数据集：候选 = Lotus release 语料（治理产物）；参考/测试 = clean 层冻结评测集
   （前 64 行参考、其余测试，防泄漏剔除见 data.py）
2. warm-up：候选的 1% × 1 epoch → 锚点 φ0（保存适配器，双臂共用同一起点）
3. 打分：φ0 固定，逐候选 backward 取 LoRA 梯度，与 ḡref 余弦 → top-⌈0.05N⌉
4. 双臂 SFT（超参严格一致，各从 φ0 重新加载）：
   - grad_arm：梯度筛选 top-5%
   - random_arm：同 seed 随机 5%（同数量对照）
5. 评测：双臂 + φ0 基线在独立测试集上的 token 级 ppl
6. 报告：data/downstream/lora_report.json（全部数字 + 口径 + 诚实结论）

用法：
    python -m downstream.lora_finetune.run --smoke     # 小样本冒烟（验证链路，分钟级）
    python -m downstream.lora_finetune.run             # 全量（GPU 约 20-40 分钟）
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from downstream.lora_finetune import data as data_mod
from downstream.lora_finetune.model import (ModelConfig, evaluate_ppl, load_model,
                                            mean_reference_grad, score_candidates,
                                            select_topk_by_score, sft)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "downstream"
ANCHOR_DIR = OUT_DIR / "lora_anchor_phi0"
SCORES_PATH = OUT_DIR / "lora_scores.json"
REPORT = OUT_DIR / "lora_report.json"
LOCAL_MODEL = ROOT / "data" / "models" / "Qwen3-0.6B"   # ModelScope 下载落点（gitignored）


def _resolve_model(cli_value: str | None) -> str:
    """模型来源优先级：CLI 显式值 > 本地已下载目录 > HF hub id（走 HF_ENDPOINT 镜像）。"""
    if cli_value:
        return cli_value
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL)
    return "Qwen/Qwen3-0.6B"

# 训练超参（grad/random/full 三臂严格一致；已定稿 2026-11-13「不调参、多臂对称」——
# 因变量是数据选择策略，超参必须对齐才成立对照，调参版另立实验，见 口径说明.md §5）
WARMUP_FRAC = 0.01
WARMUP_EPOCHS = 1
SFT_EPOCHS = 4
LR = 2e-4
SELECT_RATIO = 0.05
SEED = 42


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fresh_from_anchor(cfg: ModelConfig, device: str):
    """从锚点 φ0 重新加载一份干净模型（双臂互不污染）。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(cfg.name)
    base = AutoModelForCausalLM.from_pretrained(cfg.name,
                                                dtype=getattr(torch, cfg.dtype))
    model = PeftModel.from_pretrained(base, str(ANCHOR_DIR), is_trainable=True)
    model.to(device)
    model.eval()
    return model, tok


def _scores_fingerprint(cfg: ModelConfig, n_cand: int, m_ref: int) -> dict:
    return {"model": cfg.name, "max_len": cfg.max_len, "lora_r": cfg.lora_r,
            "seed": SEED, "n_candidates": n_cand, "m_ref": m_ref}


def run(smoke: bool = False, model: str | None = None, lr: float = LR,
        sft_epochs: int | None = None, tag: str | None = None,
        reuse_anchor: bool = False, reuse_scores: bool = False,
        full_arm: bool = False) -> dict:
    t_start = time.perf_counter()
    device = _device()
    cfg = ModelConfig(name=_resolve_model(model), max_len=64 if smoke else 256)
    sft_epochs = 1 if smoke else (sft_epochs or SFT_EPOCHS)
    report_path = OUT_DIR / (f"lora_report_{tag}.json" if tag else "lora_report.json")
    report: dict = {
        "config": {**asdict(cfg), "warmup_frac": WARMUP_FRAC, "warmup_epochs": WARMUP_EPOCHS,
                   "sft_epochs": sft_epochs, "lr": lr,
                   "select_ratio": SELECT_RATIO, "seed": SEED, "smoke": smoke, "tag": tag,
                   "full_arm": full_arm},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "platform": platform.platform()},
    }

    # 1. 数据集
    ds = data_mod.build(m_ref=8 if smoke else 64,
                        candidate_limit=40 if smoke else None)
    n_cand = len(ds.candidates)
    k = math.ceil(SELECT_RATIO * n_cand)   # ⌈0.05·N⌉，与 grad_select.core 同口径
    report["data"] = {"n_candidates": n_cand, "k_select": k,
                      "m_ref": len(ds.reference), "n_test": len(ds.test),
                      "excluded_overlap": ds.excluded_overlap, "sources": ds.sources}
    print(f"[data] cand={n_cand} k={k} ref={len(ds.reference)} test={len(ds.test)} "
          f"overlap_excluded={ds.excluded_overlap}", flush=True)

    # 2. warm-up → 锚点 φ0（reuse_anchor：复用已保存的 φ0，跳过 warm-up——
    #    同数据同 seed 的 warm-up 本就是为了产同一个锚点，复用保证多配置对照共享起点）
    anchor_ready = reuse_anchor and (ANCHOR_DIR / "adapter_config.json").exists()
    if anchor_ready:
        report["warmup"] = {"reused": True, "anchor": str(ANCHOR_DIR)}
        print(f"[warmup] 复用锚点 φ0：{ANCHOR_DIR}", flush=True)
        model, tok = _fresh_from_anchor(cfg, device)
    else:
        model, tok = load_model(cfg, device, seed=SEED)
        n_warm = max(1, math.ceil(WARMUP_FRAC * n_cand))
        warm_texts = random.Random(SEED).sample(ds.candidates, n_warm)
        print(f"[warmup] {n_warm} docs x {WARMUP_EPOCHS} epoch", flush=True)
        report["warmup"] = sft(model, tok, warm_texts, cfg, device,
                               epochs=WARMUP_EPOCHS, lr=LR, seed=SEED)
        ANCHOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ANCHOR_DIR))   # 只存 LoRA 适配器（φ0），基础模型不动

    # 3. 基线评测（φ0，未做正式 SFT）
    report["baseline_phi0"] = evaluate_ppl(model, tok, ds.test, cfg, device)
    print(f"[baseline] test ppl={report['baseline_phi0']['ppl']}", flush=True)

    # 4. 打分（φ0 固定）。reuse_scores：锚点与数据指纹一致时复用已保存分数
    #    （打分是确定性计算，同 φ0 同数据 → 同分数；指纹不符则拒绝复用并重算）
    fp = _scores_fingerprint(cfg, n_cand, len(ds.reference))
    scores = None
    if reuse_scores and SCORES_PATH.exists():
        blob = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        if blob.get("fingerprint") == fp and len(blob.get("scores", [])) == n_cand:
            scores = blob["scores"]
            report["scoring"] = {**blob.get("scoring", {}), "reused": True}
            print(f"[scoring] 复用已保存分数（指纹一致，{n_cand} 条）", flush=True)
        else:
            print("[scoring] 指纹不符 → 拒绝复用，重算", flush=True)
    if scores is None:
        t0 = time.perf_counter()
        g_ref = mean_reference_grad(model, tok, ds.reference, cfg, device,
                                    report_every=16 if smoke else 0)
        scores = score_candidates(model, tok, ds.candidates, g_ref, cfg, device,
                                  report_every=0 if smoke else 200)
        report["scoring"] = {
            "wall_seconds": round(time.perf_counter() - t0, 1),
            "score_min": round(min(scores), 6), "score_max": round(max(scores), 6),
            "score_median": round(statistics.median(scores), 6),
            "n_zero_norm": sum(1 for s in scores if s == 0.0),
        }
        if not smoke:
            SCORES_PATH.write_text(json.dumps(
                {"fingerprint": fp, "scores": scores,
                 "scoring": {k: v for k, v in report["scoring"].items()}},
                ensure_ascii=False), encoding="utf-8")
    grad_sel = select_topk_by_score(scores, SELECT_RATIO)
    random_sel = sorted(random.Random(SEED + 1).sample(range(n_cand), k))
    report["selection"] = {"grad_top_indices_head": grad_sel[:20],
                           "random_indices_head": random_sel[:20],
                           "overlap_grad_random": len(set(grad_sel) & set(random_sel))}
    print(f"[scoring] {report['scoring']['wall_seconds']}s, "
          f"median={report['scoring']['score_median']}", flush=True)

    # 5. 双臂（+可选全量臂）SFT（各从 φ0 重新加载）+ 评测
    arms = [("grad_arm", grad_sel), ("random_arm", random_sel)]
    if full_arm:
        # 全量训练臂：全部候选文档（回答「top-5% 选择是否接近全量效果」）
        arms.append(("full_arm", list(range(n_cand))))
    for arm, sel in arms:
        model, tok = _fresh_from_anchor(cfg, device)
        texts = [ds.candidates[i] for i in sel]
        print(f"[{arm}] sft {len(texts)} docs x {sft_epochs} epochs", flush=True)
        train_info = sft(model, tok, texts, cfg, device,
                         epochs=sft_epochs, lr=lr, seed=SEED)
        ev = evaluate_ppl(model, tok, ds.test, cfg, device)
        report[arm] = {**train_info, **ev, "n_train_docs": len(texts)}
        print(f"[{arm}] test ppl={ev['ppl']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 6. 诚实结论（单 seed 单次运行口径）
    g, r = report["grad_arm"]["ppl"], report["random_arm"]["ppl"]
    b = report["baseline_phi0"]["ppl"]
    delta = (g - r) / r * 100
    conclusion = {
        "grad_vs_random_pct": round(delta, 3),
        "grad_vs_baseline_pct": round((g - b) / b * 100, 3),
        "random_vs_baseline_pct": round((r - b) / b * 100, 3),
        "statement": (
            f"单次运行（seed={SEED}）：梯度筛选臂 test ppl {g} vs 随机臂 {r}"
            f"（{delta:+.2f}%）。负值=筛选臂更好。单 seed 无显著性检验，"
            "结论口径=「本配置下筛选是否优于随机」，不外推。"),
        "smoke": smoke,
    }
    if full_arm:
        f_ppl = report["full_arm"]["ppl"]
        conclusion["full_vs_baseline_pct"] = round((f_ppl - b) / b * 100, 3)
        conclusion["grad_vs_full_pct"] = round((g - f_ppl) / f_ppl * 100, 3)
        conclusion["random_vs_full_pct"] = round((r - f_ppl) / f_ppl * 100, 3)
        conclusion["statement_full"] = (
            f"全量臂（{n_cand} 篇全部候选）test ppl {f_ppl} vs 基线 {b}"
            f"（{(f_ppl - b) / b * 100:+.2f}%）；梯度筛选 top-{SELECT_RATIO:.0%} 臂 {g}"
            f"（用 {report['grad_arm']['n_train_docs']}/{n_cand} 篇数据达到的效果相对全量"
            f" {(g - f_ppl) / f_ppl * 100:+.2f}%）。回答「选择是否接近全量」：负值=筛选子集"
            f"不劣于甚至优于全量（数据选择的价值），正值=仍逊于全量（选择有损）。")
    report["conclusion"] = conclusion
    report["wall_seconds_total"] = round(time.perf_counter() - t_start, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                           encoding="utf-8")
    print(f"[done] {report['wall_seconds_total']}s → {report_path}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟（验证链路）")
    ap.add_argument("--model", default=None, help="模型路径或 HF id（默认自动探测本地 data/models）")
    ap.add_argument("--lr", type=float, default=LR, help="双臂 SFT 学习率（默认 2e-4）")
    ap.add_argument("--epochs", type=int, default=None, help="双臂 SFT 轮数（默认 4）")
    ap.add_argument("--tag", default=None, help="报告文件名后缀（多配置对照用）")
    ap.add_argument("--reuse-anchor", action="store_true", help="复用已保存锚点 φ0（跳过 warm-up）")
    ap.add_argument("--reuse-scores", action="store_true", help="指纹一致时复用已保存打分")
    ap.add_argument("--full-arm", action="store_true",
                    help="加跑全量训练臂（全部候选，回答 top-5% 是否接近全量效果）")
    args = ap.parse_args(argv)
    run(smoke=args.smoke, model=args.model, lr=args.lr, sft_epochs=args.epochs,
        tag=args.tag, reuse_anchor=args.reuse_anchor, reuse_scores=args.reuse_scores,
        full_arm=args.full_arm)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
