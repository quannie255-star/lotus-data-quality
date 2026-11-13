"""clip_bench/dataset.py — Flickr8k 图文对数据集构建（正/易负/硬负样本）

数据源（真实数据，铁律 1）：Flickr8k（ModelScope llm416101531/Flickr8k，
8,091 张真实照片 × 5 条英文人工 caption，40,460 对），已解压于 data/clip/flickr8k/。

三类样本（课程方案 §2/§4-L2 口径）：
- 正样本 pos：图与其真实 caption
- 易负样本 easy_neg：图 × 另一张图的随机 caption（固定种子）
- 硬负样本 hard_neg：图 × 「模型自己最容易搞混的」非匹配 caption——
  用同一 CLIP 的图文 embedding，对每张图取相似度最高的非配对 caption。
  （挖掘用被评模型本身，存在自证风险——报告须注明该口径，见 口径说明.md §4）

划分纪律：calib 与 test 两个不相交图集合（固定 seed）。阈值只在 calib 上标定，
指标只在 test 上报——标定与评测数据分离，防阈值过拟合。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLICKR = ROOT / "data" / "clip" / "flickr8k"


@dataclass(frozen=True)
class PairSet:
    """一组评测对：images[i] 与 captions[i] 配对；label 1=匹配 0=不匹配；kind 标注负样本类型。"""
    images: list[Path]
    captions: list[str]
    labels: list[int]
    kinds: list[str]          # pos / easy_neg / hard_neg
    image_ids: list[str]      # 对应图片文件名（检索评测复用）


def load_metadata(only_existing: bool = True) -> dict[str, list[str]]:
    """metadata.jsonl → {file_name: [caption×5]}（保持文件内顺序，确定性）。

    已知脏数据：metadata 有 1 条 `….jpg.1` 后缀的重复条目，磁盘上无此文件
    （实际文件是不带 .1 的同名图）——only_existing=True 时剔除并计入
    n_excluded_missing（数据诚实纪律：剔除有账，不静默）。
    """
    if not (FLICKR / "metadata.jsonl").exists():
        raise FileNotFoundError(f"Flickr8k 未就绪：{FLICKR}（下载方式见 clip_bench/README.md）")
    by_img: dict[str, list[str]] = {}
    for line in (FLICKR / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_img.setdefault(r["file_name"], []).append(r["text"].strip())
    if only_existing:
        missing = [k for k in by_img if not (FLICKR / k).exists()]
        for k in missing:
            del by_img[k]
        by_img_excluded.clear()
        by_img_excluded.extend(missing)
    return by_img


# 被剔除的 metadata 条目（load_metadata 填充；进评测报告的数据块账目）
by_img_excluded: list[str] = []


def split_images(by_img: dict, n_calib: int, n_test: int, seed: int = 2026):
    """固定 seed 洗牌 → 前 n_calib 张做标定集，其后 n_test 张做测试集（不相交）。"""
    imgs = sorted(by_img)
    rng = random.Random(seed)
    rng.shuffle(imgs)
    need = n_calib + n_test
    if len(imgs) < need:
        raise ValueError(f"图片不足：需 {need}，实有 {len(imgs)}")
    return imgs[:n_calib], imgs[n_calib:need]


def build_pairs(img_files: list[str], by_img: dict, seed: int = 2026,
                caption_idx: int = 0, image_root: Path = FLICKR) -> PairSet:
    """正样本 + 易负样本（随机跨图配对）。每图取第 caption_idx 条 caption（默认第 1 条）。

    image_root 供其他语料（如中文图文对）复用同一构对口径。
    """
    rng = random.Random(seed)
    images, captions, labels, kinds, ids = [], [], [], [], []
    others = list(img_files)
    for f in img_files:
        cap = by_img[f][caption_idx]
        images.append(image_root / f); captions.append(cap); labels.append(1)
        kinds.append("pos"); ids.append(f)
        g = rng.choice([x for x in others if x != f])
        images.append(image_root / f); captions.append(by_img[g][caption_idx])
        labels.append(0); kinds.append("easy_neg"); ids.append(f)
    return PairSet(images, captions, labels, kinds, ids)


def add_hard_negatives(pairs: PairSet, scorer, by_img: dict,
                       caption_idx: int = 0, image_root: Path = FLICKR) -> PairSet:
    """硬负样本挖掘：对每张图，在全测试池 caption 里找相似度最高的非配对 caption。

    计算口径：图 embedding × 全部候选 caption embedding 的余弦（scorer 批量接口），
    排除该图自己的配对 caption。产出与 pos 等量的 hard_neg 对（追加到 PairSet）。
    """
    # 去重保序（PairSet 里每图出现两次：pos + easy_neg）；用 image_ids 保留
    # metadata 原始 file_name（含 train/ 前缀，与 by_img 键一致）
    seen, uniq = set(), []
    for f, kind in zip(pairs.image_ids, pairs.kinds):
        if kind == "pos" and f not in seen:
            seen.add(f); uniq.append(f)

    pool_caps, pool_owner = [], []
    for f in uniq:
        c = by_img[f][caption_idx]
        pool_caps.append(c); pool_owner.append(f)

    from PIL import Image

    def _embed_paths(paths: list[Path], chunk: int = 128):
        """分批开图编码（防一次性持有上千张 PIL 图的内存峰值）。"""
        import torch
        outs = []
        for lo in range(0, len(paths), chunk):
            batch = []
            for p in paths[lo:lo + chunk]:
                with Image.open(p) as im:
                    batch.append(im.convert("RGB"))
            outs.append(scorer.embed_images(batch, batch_size=chunk))
        return torch.cat(outs, dim=0)

    ie = _embed_paths([image_root / f for f in uniq])
    te = scorer.embed_texts(pool_caps, batch_size=512)
    sim = ie @ te.T   # (n_img, n_cap)，embedding 已 L2 归一 → 点积即余弦

    hard_imgs, hard_caps = [], []
    for i, f in enumerate(uniq):
        row = sim[i].clone()
        row[i] = -2.0   # 排除自身配对（正样本）
        j = int(row.argmax())
        hard_imgs.append(image_root / f)
        hard_caps.append(pool_caps[j])

    return PairSet(
        images=list(pairs.images) + hard_imgs,
        captions=list(pairs.captions) + hard_caps,
        labels=list(pairs.labels) + [0] * len(hard_imgs),
        kinds=list(pairs.kinds) + ["hard_neg"] * len(hard_imgs),
        image_ids=list(pairs.image_ids) + uniq,
    )
