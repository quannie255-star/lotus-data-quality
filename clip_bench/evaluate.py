"""clip_bench/evaluate.py — 对齐判别与图文检索评测（零第三方指标依赖，全部手写可审计）

判别指标：Accuracy / Precision / Recall / F1 / 混淆矩阵（给定阈值）+ AUC（秩公式，
不依赖 sklearn）。检索指标：image→text 与 text→image 的 R@1/R@5/R@10
（Flickr8k 标准口径：每图 5 条 caption 都算正确参考）。
"""

from __future__ import annotations

from pathlib import Path

from clip_bench.dataset import FLICKR, PairSet


def auc_score(labels: list[int], scores: list[float]) -> float:
    """AUC（Mann-Whitney U 秩公式，并列分数取平均秩）。O(n log n)。"""
    paired = sorted(zip(scores, labels))
    # 平均秩处理并列
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2 + 1          # 秩从 1 开始，并列取平均
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("单一类别无法计算 AUC")
    sum_pos_ranks = sum(r for r, (_, lab) in zip(ranks, paired) if lab == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    """给定阈值下的 Acc/P/R/F1/混淆矩阵（判定：score ≥ threshold → 预测匹配）。"""
    tp = fp = fn = tn = 0
    for lab, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and lab == 1: tp += 1
        elif pred == 1 and lab == 0: fp += 1
        elif pred == 0 and lab == 1: fn += 1
        else: tn += 1
    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": threshold, "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def roc_points(labels: list[int], scores: list[float], n_grid: int = 500) -> list[tuple]:
    """ROC 点列（FPR, TPR, threshold），阈值取分数分位网格——供约登指数标定。"""
    lo, hi = min(scores), max(scores)
    grid = [lo + (hi - lo) * i / n_grid for i in range(n_grid + 1)]
    pts = []
    for th in grid:
        m = binary_metrics(labels, scores, th)
        tp, fp, fn, tn = m["confusion"].values()
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        pts.append((fpr, tpr, th))
    return pts


def youden_threshold(labels: list[int], scores: list[float], n_grid: int = 500):
    """约登指数 J = TPR − FPR 最大的阈值（课程方案 §4-L2「不拍脑袋定 0.5」的落点）。"""
    pts = roc_points(labels, scores, n_grid)
    best = max(pts, key=lambda p: p[1] - p[0])
    return {"threshold": round(best[2], 6), "j": round(best[1] - best[0], 4),
            "tpr": round(best[1], 4), "fpr": round(best[0], 4)}


def retrieval_metrics(img_emb, txt_emb, ks=(1, 5, 10)) -> dict:
    """检索口径（单参考）：查询 i 的正确项 = 候选 i（对角线）。

    - image→text：查询=图 embedding，候选池=同序 caption embedding
    - text→image：查询=caption embedding，候选池=图 embedding
    名次 = 严格大于正确项相似度的候选数 + 1（并列按最好名次算，偏乐观口径，
    与主流 clip-benchmark 一致——随机 tie-break 的悲观口径差异在 R@k 三位小数内）。
    """
    n = img_emb.shape[0]
    sim_i2t = img_emb @ txt_emb.T
    sim_t2i = sim_i2t.T
    out: dict = {"n_queries": n}
    for name, sim in (("image2text", sim_i2t), ("text2image", sim_t2i)):
        ranks = (sim > sim.diagonal().unsqueeze(1)).sum(dim=1) + 1
        for k in ks:
            out[f"{name}_R@{k}"] = round(float((ranks <= k).float().mean()), 4)
        out[f"{name}_median_rank"] = int(ranks.median())
    return out


def score_pairs(scorer, pairs: PairSet, batch_size: int = 256) -> list[float]:
    """批量算全部 (image, caption) 对的余弦分数（分批流式开图，防大集合内存峰值）。"""
    from PIL import Image

    sims = []
    n = len(pairs.images)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        with_images = []
        for p in pairs.images[lo:hi]:
            with Image.open(p) as im:
                with_images.append(im.convert("RGB"))
        ie = scorer.embed_images(with_images, batch_size=batch_size)
        te = scorer.embed_texts(pairs.captions[lo:hi], batch_size=512)
        sims.extend((ie * te).sum(dim=1).tolist())
    return sims
