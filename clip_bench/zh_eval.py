"""clip_bench/zh_eval.py — 真实中文图文对的 CN-CLIP 标定与联合门禁报告

数据源：HF `svjack/Chinese_Children_Image_Captioning_Dataset_Split0`（真实儿童绘本
插图 + 人工中文句子，每图一句；下载脚本见 clip_bench/README.md）。这是**真实中文**
图文对，不是机器翻译语料——跨语言阈值失真的问题（实验报告 §5.3）在这里不成立。
领域边界：儿童绘本短句 ≠ 生产语料的长文档 caption，标定结果按领域口径报，不冒充全量。

流程（与 Flickr8k 英文评测同构，防阈值过拟合）：
固定 seed 切分标定/测试图集（不相交）→ 正/易负/硬负样本 → CN-CLIP 打分 →
标定集约登阈值 + 分位数 → 测试集报 Acc/P/R/F1/AUC 与 R@1/5/10 →
产出 calibration_{tag}.json（可直接被 apply_calibration.py 回填）。

用法：
    python -m clip_bench.zh_eval --corpus data/clip/zh_pairs \
        --model data/models/cn-clip-vit-base-patch16 --tag zh_children \
        --n-calib 120 --n-test 120
    python -m clip_bench.zh_eval --corpus data/clip/zh_pairs --tag zh_children --gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "clip"
METRICS = ROOT / "data" / "metrics"

from clip_bench.dataset import add_hard_negatives, build_pairs, split_images  # noqa: E402
from clip_bench.evaluate import (auc_score, binary_metrics, retrieval_metrics,  # noqa: E402
                                 score_pairs, youden_threshold)
from clip_bench.run_eval import _percentiles  # noqa: E402


def load_metadata(corpus_dir: Path) -> dict[str, list[str]]:
    """metadata.jsonl → {file_name: [caption]}；归一化 = 去掉所有空白（该数据集逐字空格）。

    缺图/空文条目剔除（剔除有账：by_img_excluded）。
    """
    meta_path = Path(corpus_dir) / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"中文图文对未就绪：{meta_path}")
    by_img: dict[str, list[str]] = {}
    excluded: list[str] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = r["file_name"]
        text = "".join(str(r.get("text", "")).split())
        if text and (Path(corpus_dir) / name).exists():
            by_img.setdefault(name, []).append(text)
        else:
            excluded.append(name)
    by_img_excluded.clear()
    by_img_excluded.extend(excluded)
    return by_img


# load_metadata 的剔除账目（报告数据块引用；与 dataset.by_img_excluded 同语义）
by_img_excluded: list[str] = []


def run_eval(corpus_dir: Path, model_path: str, tag: str, n_calib: int, n_test: int,
             device: str | None = None, seed: int = 2026) -> dict:
    from modalities.image.clip_align import ClipAlignScorer

    t0 = time.perf_counter()
    corpus_dir = Path(corpus_dir)
    by_img = load_metadata(corpus_dir)
    scorer = ClipAlignScorer(model_path, device=device)
    calib_imgs, test_imgs = split_images(by_img, n_calib, n_test, seed=seed)
    report: dict = {
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "device": scorer.device, "backend": scorer.backend},
        "model": {"path": str(Path(model_path).resolve()), "tag": tag},
        "data": {"source": "svjack/Chinese_Children_Image_Captioning_Dataset_Split0"
                           "（真实儿童绘本插图+人工中文句子）",
                 "corpus_dir": str(corpus_dir.resolve()),
                 "n_pool_images": len(by_img), "n_calib": n_calib, "n_test": n_test,
                 "split_seed": seed, "caption_norm": "remove_all_whitespace",
                 "metadata_excluded_missing_files": list(by_img_excluded),
                 "note": "标定/测试图集不相交；阈值只在标定集上定；短句领域口径，不外推长文本"},
    }
    print(f"[model] {scorer.backend} @ {scorer.device}", flush=True)

    t1 = time.perf_counter()
    calib_pairs = build_pairs(calib_imgs, by_img, seed=seed, image_root=corpus_dir)
    calib_pairs = add_hard_negatives(calib_pairs, scorer, by_img, image_root=corpus_dir)
    calib_scores = score_pairs(scorer, calib_pairs)
    pos_s = [s for s, lbl in zip(calib_scores, calib_pairs.labels) if lbl == 1]
    neg_s = [s for s, lbl in zip(calib_scores, calib_pairs.labels) if lbl == 0]
    cal = {
        "youden": youden_threshold(calib_pairs.labels, calib_scores),
        "pos_cos": _percentiles(pos_s),
        "neg_cos": _percentiles(neg_s),
        "n_pairs": len(calib_scores),
        "wall_seconds": round(time.perf_counter() - t1, 1),
        "caption_norm": "remove_all_whitespace",
        "corpus": "svjack/Chinese_Children_Image_Captioning_Dataset_Split0",
    }
    th_, p95 = cal["youden"]["threshold"], cal["pos_cos"]["p95"]
    cal["suggest_cos_lo"] = round(th_ - (0.30 / 0.70) * (p95 - th_), 4)
    cal["suggest_cos_hi"] = p95
    report["calibration"] = cal
    print(f"[calib] youden th={th_} J={cal['youden']['j']} "
          f"pos_p50={cal['pos_cos']['p50']} neg_p95={cal['neg_cos']['p95']}", flush=True)

    t2 = time.perf_counter()
    test_pairs = build_pairs(test_imgs, by_img, seed=seed + 1, image_root=corpus_dir)
    test_pairs = add_hard_negatives(test_pairs, scorer, by_img, image_root=corpus_dir)
    test_scores = score_pairs(scorer, test_pairs)
    labels, kinds = test_pairs.labels, test_pairs.kinds
    disc = {
        "at_youden_threshold": binary_metrics(labels, test_scores, th_),
        "at_zero_threshold": binary_metrics(labels, test_scores, 0.0),
        "auc_all_negatives": round(auc_score(labels, test_scores), 4),
        "auc_easy_only": round(auc_score(
            [lab for lab, k in zip(labels, kinds) if k != "hard_neg"],
            [s for s, k in zip(test_scores, kinds) if k != "hard_neg"]), 4),
        "auc_hard_only": round(auc_score(
            [lab for lab, k in zip(labels, kinds) if k != "easy_neg"],
            [s for s, k in zip(test_scores, kinds) if k != "easy_neg"]), 4),
        "n_pairs": {"pos": kinds.count("pos"), "easy_neg": kinds.count("easy_neg"),
                    "hard_neg": kinds.count("hard_neg")},
        "wall_seconds": round(time.perf_counter() - t2, 1),
    }
    report["discrimination"] = disc
    print(f"[test] AUC all={disc['auc_all_negatives']} easy={disc['auc_easy_only']} "
          f"hard={disc['auc_hard_only']} acc@youden={disc['at_youden_threshold']['accuracy']}",
          flush=True)

    from PIL import Image

    import torch

    t3 = time.perf_counter()
    chunks = []
    for lo in range(0, len(test_imgs), 64):
        batch = []
        for f in test_imgs[lo:lo + 64]:
            with Image.open(corpus_dir / f) as im:
                batch.append(im.convert("RGB"))
        chunks.append(scorer.embed_images(batch, batch_size=64))
    ie = torch.cat(chunks, dim=0)
    te = scorer.embed_texts([by_img[f][0] for f in test_imgs], batch_size=256)
    report["retrieval"] = retrieval_metrics(ie, te)
    report["retrieval"]["pool_note"] = "候选池=测试集图片与其唯一 caption（单参考口径）"
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


def gate_report(corpus_dir: Path, tag: str, n: int = 120, seed: int = 2026,
                whatif: bool = False) -> dict:
    """用生产路径（PairedQualityScorer）跑联合门禁抽样报告。

    whatif=True：用本次标定的 [cos_lo, cos_hi] 在**内存里**临时打开 clip（不改配置），
    量化「若开启会误杀多少真实中文对」。生产开关状态照实记在报告里。
    """
    import copy
    import json

    from modalities.image import paired as paired_mod
    from modalities.image.paired import ImageTextSample, PairedQualityScorer
    import quality.multimodal as qm

    policy = copy.deepcopy(qm.policy())
    if whatif:
        cal = json.loads((OUT_DIR / f"calibration_{tag}.json").read_text(encoding="utf-8"))
        policy["image_text"]["clip"]["enabled"] = True
        policy["image_text"]["clip"]["cos_lo"] = cal["suggest_cos_lo"]
        policy["image_text"]["clip"]["cos_hi"] = cal["suggest_cos_hi"]
    elif not policy["image_text"]["clip"]["enabled"]:
        raise RuntimeError("clip.enabled=false —— 先回填 cos_lo/cos_hi 并开启开关")

    from contextlib import ExitStack
    from unittest import mock

    corpus_dir = Path(corpus_dir)
    by_img = load_metadata(corpus_dir)
    names = sorted(by_img)[:n]
    verdicts = {"pass": 0, "remove": 0}
    reasons: dict[str, int] = {}
    aligns, joint_scores, examples = [], [], []
    with ExitStack() as stack:
        if whatif:
            stack.enter_context(mock.patch.object(paired_mod, "mm_policy", lambda: policy))
            stack.enter_context(mock.patch.object(qm, "policy", lambda: policy))
        scorer = PairedQualityScorer()
        for name in names:
            path = corpus_dir / name
            data = path.read_bytes()
            sample = ImageTextSample(
                doc_id=hashlib.sha256(f"zh-corpus:{name}".encode()).hexdigest()[:32],
                source="zh-corpus", source_id=name, caption=by_img[name][0],
                image_uri=str(path), image_bytes_hash=hashlib.sha256(data).hexdigest(),
                image_bytes_len=len(data), image_format=path.suffix.lstrip("."))
            s = scorer.score(sample)
            verdicts[s["joint_verdict"]] = verdicts.get(s["joint_verdict"], 0) + 1
            if s["joint_reason"]:
                reasons[s["joint_reason"]] = reasons.get(s["joint_reason"], 0) + 1
            if s["pair_cross_modal_alignment"] is not None:
                aligns.append(s["pair_cross_modal_alignment"])
            if s["quality_score"] is not None:
                joint_scores.append(s["quality_score"])
            if len(examples) < 5 and s["joint_verdict"] == "remove":
                examples.append({"file": name, "caption": by_img[name][0],
                                 "align": s["pair_cross_modal_alignment"],
                                 "joint_score": s["quality_score"],
                                 "reason": s["joint_reason"]})

    def q(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(max(int(round(p / 100 * (len(s) - 1))), 0), len(s) - 1)], 4)

    out = {
        "tag": tag,
        "mode": "whatif" if whatif else "production",
        "corpus": "svjack/Chinese_Children_Image_Captioning_Dataset_Split0",
        "n_samples": len(names),
        "seed": seed,
        "clip_config": policy["image_text"]["clip"],
        "clip_enabled_in_production": qm.policy()["image_text"]["clip"]["enabled"],
        "joint_gate": policy["joint_gate"]["min_modality_score"],
        "verdicts": verdicts,
        "reasons": reasons,
        "alignment": {"p5": q(aligns, 5), "p50": q(aligns, 50), "p95": q(aligns, 95),
                      "mean": round(statistics.mean(aligns), 4) if aligns else None},
        "joint_score": {"p5": q(joint_scores, 5), "p50": q(joint_scores, 50),
                        "p95": q(joint_scores, 95),
                        "mean": round(statistics.mean(joint_scores), 4) if joint_scores else None},
        "removed_examples": examples,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"multimodal_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_samples", "verdicts", "reasons", "alignment")},
                     ensure_ascii=False), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="中文图文对 CN-CLIP 标定 / 联合门禁报告")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "clip" / "zh_pairs"))
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "cn-clip-vit-base-patch16"))
    ap.add_argument("--tag", default="zh_children")
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gate", action="store_true", help="只跑联合门禁报告（需 clip.enabled=true）")
    ap.add_argument("--whatif-clip", action="store_true",
                    help="联合门禁报告用本次标定区间在内存里临时开 clip，不写配置")
    ap.add_argument("--gate-n", type=int, default=120)
    args = ap.parse_args(argv)
    if args.gate:
        gate_report(Path(args.corpus), args.tag, n=args.gate_n, whatif=args.whatif_clip)
    else:
        run_eval(Path(args.corpus), args.model, args.tag,
                 args.n_calib, args.n_test, args.device)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
