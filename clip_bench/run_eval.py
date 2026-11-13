"""clip_bench/run_eval.py — 对齐判别评测编排器（课程作业主评测入口）

流程：Flickr8k 固定 seed 切分（标定集/测试集不相交）→ 正/易负/硬负样本构建 →
批量编码打分 → 标定集上做约登阈值标定 + 分布分位数 → 测试集上报
Acc/P/R/F1/AUC（整体 + 仅硬负样本）与 R@1/5/10 检索指标 → JSON 原始记录。

用法：
    python -m clip_bench.run_eval --model data/models/cn-clip-vit-base-patch16 \
        --tag cnclip [--n-calib 500] [--n-test 1500] [--device cpu|cuda]

产出：data/clip/eval_report_{tag}.json + data/clip/calibration_{tag}.json
（后者的 cos_lo/cos_hi 建议值用于回填 configs/multimodal_policy.json 的 clip 归一化区间）
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "clip"

from clip_bench import dataset as dataset_mod
from clip_bench.dataset import add_hard_negatives, build_pairs, load_metadata, split_images  # noqa: E402
from clip_bench.evaluate import (auc_score, binary_metrics, retrieval_metrics,  # noqa: E402
                                 score_pairs, youden_threshold)


def _percentiles(values: list[float], ps=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    s = sorted(values)
    out = {}
    for p in ps:
        idx = min(max(int(round(p / 100 * (len(s) - 1))), 0), len(s) - 1)
        out[f"p{p}"] = round(s[idx], 4)
    return out


def run_eval(model_path: str, tag: str, n_calib: int, n_test: int, device: str | None,
             seed: int = 2026) -> dict:
    from modalities.image.clip_align import ClipAlignScorer

    t0 = time.perf_counter()
    scorer = ClipAlignScorer(model_path, device=device)
    by_img = load_metadata()
    calib_imgs, test_imgs = split_images(by_img, n_calib, n_test, seed=seed)
    report: dict = {
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "device": scorer.device, "backend": scorer.backend},
        "model": {"path": str(Path(model_path).resolve()), "tag": tag},
        "data": {"source": "Flickr8k (ModelScope llm416101531/Flickr8k, 真实照片+人工英文caption)",
                 "n_pool_images": len(by_img), "n_calib": n_calib, "n_test": n_test,
                 "split_seed": seed, "caption_idx": 0,
                 "metadata_excluded_missing_files": dataset_mod.by_img_excluded,
                 "note": "标定集与测试集不相交；阈值只在标定集上定，指标只在测试集上报"},
    }
    print(f"[model] {scorer.backend} @ {scorer.device}", flush=True)

    # ---- 标定集：构对（含硬负挖掘）→ 打分 → 约登阈值 + 分布分位数 ----
    t1 = time.perf_counter()
    calib_pairs = build_pairs(calib_imgs, by_img, seed=seed)
    calib_pairs = add_hard_negatives(calib_pairs, scorer, by_img)
    calib_scores = score_pairs(scorer, calib_pairs)
    pos_s = [s for s, l in zip(calib_scores, calib_pairs.labels) if l == 1]
    neg_s = [s for s, l in zip(calib_scores, calib_pairs.labels) if l == 0]
    cal = {
        "youden": youden_threshold(calib_pairs.labels, calib_scores),
        "pos_cos": _percentiles(pos_s),
        "neg_cos": _percentiles(neg_s),
        "n_pairs": len(calib_scores),
        "wall_seconds": round(time.perf_counter() - t1, 1),
    }
    # 归一化区间建议（约登阈值锚定）：令 align(youden_th) = 0.30（Lotus
    # min_modality_score 门禁线）、align(pos_p95) = 1.0，两点定线性映射：
    #   cos_lo = th − (0.30/0.70)×(pos_p95 − th)，cos_hi = pos_p95
    # 早期用 neg_p95/pos_p50 会退化（实测两模型 neg_p95 > pos_p50，分布重叠区大）——
    # 退化时该公式给出 lo>hi，已弃用；见 开发日志 2026-11-13 条。
    th_, p95 = cal["youden"]["threshold"], cal["pos_cos"]["p95"]
    cal["suggest_cos_lo"] = round(th_ - (0.30 / 0.70) * (p95 - th_), 4)
    cal["suggest_cos_hi"] = p95
    report["calibration"] = cal
    th = cal["youden"]["threshold"]
    print(f"[calib] youden th={th} J={cal['youden']['j']} "
          f"pos_p50={cal['pos_cos']['p50']} neg_p95={cal['neg_cos']['p95']}", flush=True)

    # ---- 测试集：构对（硬负在测试池内独立挖掘）→ 指标 ----
    t2 = time.perf_counter()
    test_pairs = build_pairs(test_imgs, by_img, seed=seed + 1)
    n_easy = len(test_pairs.labels)
    test_pairs = add_hard_negatives(test_pairs, scorer, by_img)
    test_scores = score_pairs(scorer, test_pairs)
    labels = test_pairs.labels
    kinds = test_pairs.kinds

    disc = {
        "at_youden_threshold": binary_metrics(labels, test_scores, th),
        "at_zero_threshold": binary_metrics(labels, test_scores, 0.0),
        "auc_all_negatives": round(auc_score(labels, test_scores), 4),
        "auc_easy_only": round(auc_score(
            [l for l, k in zip(labels, kinds) if k != "hard_neg"],
            [s for s, k in zip(test_scores, kinds) if k != "hard_neg"]), 4),
        "auc_hard_only": round(auc_score(
            [l for l, k in zip(labels, kinds) if k != "easy_neg"],
            [s for s, k in zip(test_scores, kinds) if k != "easy_neg"]), 4),
        "n_pairs": {"pos": kinds.count("pos"), "easy_neg": kinds.count("easy_neg"),
                    "hard_neg": kinds.count("hard_neg")},
    }
    report["discrimination"] = disc
    print(f"[test] AUC all={disc['auc_all_negatives']} easy={disc['auc_easy_only']} "
          f"hard={disc['auc_hard_only']} acc@youden={disc['at_youden_threshold']['accuracy']}",
          flush=True)

    # ---- 检索：测试集图片 × caption[0] 池 ----
    from PIL import Image

    from clip_bench.dataset import FLICKR
    t3 = time.perf_counter()
    chunks = []
    for lo in range(0, len(test_imgs), 128):   # 分批开图，防内存峰值
        batch = []
        for f in test_imgs[lo:lo + 128]:
            with Image.open(FLICKR / f) as im:
                batch.append(im.convert("RGB"))
        chunks.append(scorer.embed_images(batch, batch_size=128))
    import torch
    ie = torch.cat(chunks, dim=0)
    te = scorer.embed_texts([by_img[f][0] for f in test_imgs], batch_size=512)
    report["retrieval"] = retrieval_metrics(ie, te)
    report["retrieval"]["pool_note"] = "候选池=测试集图片与其第1条caption（单参考口径）"
    report["retrieval"]["wall_seconds"] = round(time.perf_counter() - t3, 1)
    print(f"[retrieval] {report['retrieval']}", flush=True)

    report["wall_seconds_total"] = round(time.perf_counter() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"eval_report_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / f"calibration_{tag}.json").write_text(
        json.dumps(cal, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {report['wall_seconds_total']}s → {OUT_DIR / f'eval_report_{tag}.json'}",
          flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    run_eval(args.model, args.tag, args.n_calib, args.n_test, args.device)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
