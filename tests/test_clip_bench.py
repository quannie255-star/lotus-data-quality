"""clip_bench 测试 — 指标实现正确性 + 数据集构建口径（不加载 CLIP 模型）

指标测试全部用手算可验证的极小用例（AUC/F1/约登），数据集测试用 Flickr8k 真实
metadata 的前若干条（不碰图片文件——硬负挖掘用替身 embedding）。
"""

from pathlib import Path

import pytest

from clip_bench.dataset import add_hard_negatives, build_pairs, load_metadata, split_images
from clip_bench.evaluate import auc_score, binary_metrics, retrieval_metrics, youden_threshold

FLICKR_META = Path(__file__).resolve().parent.parent / "data" / "clip" / "flickr8k" / "metadata.jsonl"
requires_flickr = pytest.mark.skipif(
    not FLICKR_META.exists(), reason="Flickr8k metadata 缺失（数据集不进仓库，下载见 clip_bench/README.md）")


# ---- 指标实现（手算对照）----

def test_auc_perfect_and_inverse():
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_ties_give_half():
    """全并列 → AUC=0.5（平均秩口径）。"""
    assert auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_single_class_rejected():
    with pytest.raises(ValueError):
        auc_score([1, 1], [0.1, 0.2])


def test_binary_metrics_hand_computed():
    """th=0.5：scores [0.9,0.6,0.3,0.2] labels [1,0,1,0] → tp1 fp1 fn1 tn1。"""
    m = binary_metrics([1, 0, 1, 0], [0.9, 0.6, 0.3, 0.2], 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5
    assert m["recall"] == 0.5 and m["f1"] == 0.5


def test_youden_on_separable_scores():
    """完全可分：正 [0.8,0.9] 负 [0.1,0.2] → J=1，阈值落在两簇之间。"""
    y = youden_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], n_grid=100)
    assert y["j"] == 1.0
    assert 0.2 <= y["threshold"] <= 0.9


def test_retrieval_diagonal_perfect():
    """单位阵相似度（查询 i 与候选 i 最相似）→ R@1=1。"""
    import torch
    e = torch.eye(5)
    m = retrieval_metrics(e, e)
    assert m["image2text_R@1"] == 1.0 and m["text2image_R@1"] == 1.0
    assert m["image2text_median_rank"] == 1


def test_retrieval_second_best():
    """构造正确项排第 2（干扰项相似度 1.0 > 正确项 0.5）→ R@1=0，R@5=1。"""
    import torch
    n = 6
    img = torch.eye(n)
    txt = 0.5 * torch.eye(n) + torch.eye(n).roll(-1, dims=0)
    m = retrieval_metrics(img, txt)
    assert m["image2text_R@1"] == 0.0
    assert m["image2text_R@5"] == 1.0


# ---- 数据集构建（Flickr8k 真实 metadata）----

@pytest.fixture(scope="module")
def by_img():
    meta = load_metadata()
    # 取前 40 张图做构建口径测试（不动图片文件）
    return {k: meta[k] for k in sorted(meta)[:40]}


@requires_flickr
def test_metadata_excludes_missing_files():
    """Flickr8k 已知脏数据：metadata 的 `.jpg.1` 重复条目磁盘无文件——剔除且有账。"""
    from clip_bench import dataset as d

    m = d.load_metadata()
    assert d.by_img_excluded == ["train/2258277193_586949ec62.jpg.1"]
    assert "train/2258277193_586949ec62.jpg.1" not in m
    assert all((d.FLICKR / k).exists() for k in list(m)[:100])


@requires_flickr
def test_split_disjoint_and_sized(by_img):
    calib, test = split_images(by_img, 10, 20, seed=2026)
    assert len(calib) == 10 and len(test) == 20
    assert not (set(calib) & set(test))
    # 同 seed 可复现
    c2, t2 = split_images(by_img, 10, 20, seed=2026)
    assert calib == c2 and test == t2


@requires_flickr
def test_split_rejects_oversize(by_img):
    with pytest.raises(ValueError):
        split_images(by_img, 30, 30)


@requires_flickr
def test_build_pairs_balanced_and_deterministic(by_img):
    imgs = sorted(by_img)[:10]
    p1 = build_pairs(imgs, by_img, seed=7)
    p2 = build_pairs(imgs, by_img, seed=7)
    assert len(p1.labels) == 20
    assert sum(p1.labels) == 10                       # 正负 1:1
    assert p1.kinds.count("easy_neg") == 10
    assert (p1.captions, p1.labels) == (p2.captions, p2.labels)
    # 易负样本的 caption 不属于本图（跨图配对正确性）
    for f, cap, kind in zip(p1.image_ids, p1.captions, p1.kinds):
        if kind == "easy_neg":
            assert cap not in by_img[f]


class _FakeEmbedScorer:
    """替身：图 i 的 embedding = e_i；caption 池第 j 条 = e_{j-1}（错位一格）
    → 图 i 的最硬负样本必须是池内第 i+1 条（其 embedding=e_i 与自己最像）。"""

    def embed_images(self, images, batch_size=32):
        import torch
        return torch.eye(len(images), 16)

    def embed_texts(self, texts, batch_size=64):
        import torch
        return torch.eye(len(texts), 16).roll(1, dims=0)


@requires_flickr
def test_hard_negative_mining_picks_most_confusable(by_img):
    imgs = sorted(by_img)[:8]
    base = build_pairs(imgs, by_img, seed=3)
    out = add_hard_negatives(base, _FakeEmbedScorer(), by_img)
    hard = [(f, c) for f, c, k in zip(out.image_ids, out.captions, out.kinds)
            if k == "hard_neg"]
    assert len(hard) == 8
    assert out.labels.count(1) == 8 and out.labels.count(0) == 16  # pos : easy+hard = 8:16
    # 每图硬负 caption 不得是它自己的 caption（挖掘时已排除对角）
    for f, c in hard:
        assert c not in by_img[f]
