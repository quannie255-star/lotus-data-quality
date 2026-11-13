"""CLIP 跨模态对齐接入测试（DL 大作业 ⇄ Lotus 多模态门禁）

三层：
1. 接线测试（注入替身打分器，不加载模型）：enabled 开关、第三模态键、权重切换、
   None → remove(no_score)、低对齐 → remove(low_quality_modality:cross_modal)
2. 内核纯逻辑：归一化口径、坏输入 → None（空读数纪律）
3. 真实模型对照（本地 CN-CLIP + Flickr8k 真实图片，模型缺失自动跳过）：
   匹配 caption 的对齐分必须显著高于不匹配 caption ——门禁「能红」的实证
"""

import hashlib
from pathlib import Path

import pytest

from modalities.image import paired as paired_mod
from modalities.image.paired import ImageTextSample, PairedQualityScorer

ROOT = Path(__file__).resolve().parent.parent
CN_CLIP_DIR = ROOT / "data" / "models" / "cn-clip-vit-base-patch16"
FLICKR_DIR = ROOT / "data" / "clip" / "flickr8k"

requires_cn_clip = pytest.mark.skipif(
    not (CN_CLIP_DIR / "model.safetensors").exists(),
    reason="本地 CN-CLIP 模型缺失（data/models 不进仓库）")


class FakeScorer:
    """替身打分器：固定返回预设对齐分（接线测试不加载真模型）。"""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def score(self, image_uri, caption):
        self.calls.append((image_uri, caption))
        return self.value


def _policy(enabled: bool) -> dict:
    """构造三模态启用/关闭两态的 policy（权重和=1，与生产配置同构）。"""
    return {
        "joint_gate": {
            "modality_weights": {"text": 0.6, "image": 0.4},
            "modality_weights_cross_modal": {"text": 0.5, "image": 0.2, "cross_modal": 0.3},
            "min_modality_score": 0.30,
            "min_joint_score": 0.55,
        },
        "image_text": {
            "pair_rules": {"orphan_policy": "hold", "min_caption_chars": 5,
                           "min_image_bytes": 1024},
            "clip": {"enabled": enabled},
        },
    }


def _sample(caption="一只金色的狗在草地上奔跑，阳光很好。"):
    return ImageTextSample(doc_id="d1", source="flickr", source_id="1",
                           caption=caption, image_uri="img://1",
                           image_bytes_hash=hashlib.sha256(b"img").hexdigest(),
                           image_bytes_len=20480)


@pytest.fixture()
def policy_enabled(monkeypatch):
    monkeypatch.setattr(paired_mod, "mm_policy", lambda: _policy(enabled=True))


@pytest.fixture()
def policy_disabled(monkeypatch):
    monkeypatch.setattr(paired_mod, "mm_policy", lambda: _policy(enabled=False))


# ---- 1. 接线 ----

def test_disabled_keeps_v0_behavior(policy_disabled):
    """clip 关闭 = v0 行为：alignment 为 None，双模态权重，正常样本 pass。"""
    s = PairedQualityScorer().score(_sample())
    assert s["pair_cross_modal_alignment"] is None
    assert set(s["modality_scores"]) == {"text", "image"}
    assert s["joint_verdict"] == "pass"


def test_enabled_adds_third_modality(policy_enabled):
    """clip 开启：cross_modal 第三键进 per_modality，对齐分不再 None。"""
    fake = FakeScorer(0.82)
    s = PairedQualityScorer(clip_scorer=fake).score(_sample())
    assert s["pair_cross_modal_alignment"] == 0.82
    assert set(s["modality_scores"]) == {"text", "image", "cross_modal"}
    assert fake.calls and s["joint_verdict"] == "pass"


def test_mismatched_caption_removed(policy_enabled):
    """门禁能红：对齐分低于 min_modality_score → remove(low_quality_modality:cross_modal)。"""
    s = PairedQualityScorer(clip_scorer=FakeScorer(0.05)).score(_sample())
    assert s["joint_verdict"] == "remove"
    assert s["joint_reason"] == "low_quality_modality:cross_modal"


def test_none_alignment_removed_no_score(policy_enabled):
    """空读数纪律：打分器 None（坏图/解码失败）→ remove(no_score)，不静默放行。"""
    s = PairedQualityScorer(clip_scorer=FakeScorer(None)).score(_sample())
    assert s["pair_cross_modal_alignment"] is None
    assert s["quality_score"] is None
    assert (s["joint_verdict"], s["joint_reason"]) == ("remove", "no_score")


def test_production_config_weights_sum_to_one():
    """生产配置两套权重和都=1（配置写错直接红，不让 ValueError 潜伏到运行时）。"""
    import json
    cfg = json.loads((ROOT / "configs" / "multimodal_policy.json").read_text(encoding="utf-8"))
    jg = cfg["joint_gate"]
    assert abs(sum(jg["modality_weights"].values()) - 1.0) < 1e-9
    assert abs(sum(jg["modality_weights_cross_modal"].values()) - 1.0) < 1e-9
    assert set(jg["modality_weights_cross_modal"]) == {"text", "image", "cross_modal"}


# ---- 2. 内核纯逻辑 ----

def test_normalize_interval_and_clipping():
    from modalities.image.clip_align import ClipAlignScorer

    norm = lambda cos: ClipAlignScorer.normalize(  # noqa: E731
        type("S", (), {"cos_lo": 0.1, "cos_hi": 0.5})(), cos)
    assert norm(0.1) == 0.0 and norm(0.5) == 1.0
    assert norm(0.3) == pytest.approx(0.5)
    assert norm(-0.4) == 0.0 and norm(0.9) == 1.0   # 截断不越界


@requires_cn_clip
def test_cos_interval_validated():
    from modalities.image.clip_align import ClipAlignScorer
    with pytest.raises(ValueError):
        ClipAlignScorer(CN_CLIP_DIR, cos_lo=0.5, cos_hi=0.5)


def test_load_image_bad_inputs_return_none(tmp_path):
    from modalities.image.clip_align import ClipAlignScorer
    assert ClipAlignScorer._load_image(ROOT / "no_such_file.jpg") is None
    bad = tmp_path / "_corrupt.jpg"
    bad.write_bytes(b"not an image")
    assert ClipAlignScorer._load_image(bad) is None


def test_raw_cosine_empty_caption_none():
    """空 caption → None（不喂垃圾给模型，不假装打分）。"""
    from modalities.image.clip_align import ClipAlignScorer

    class NoModel(ClipAlignScorer):
        def __init__(self):   # 跳过模型加载，只测空输入短路
            pass

    assert NoModel().raw_cosine("whatever.jpg", "   ") is None


# ---- 3. 真实模型对照（本地 CN-CLIP + Flickr8k 真实图；缺失自动跳过）----

requires_clip = pytest.mark.skipif(
    not (CN_CLIP_DIR / "model.safetensors").exists() or not FLICKR_DIR.exists(),
    reason="本地 CN-CLIP 模型或 Flickr8k 数据缺失")


@requires_clip
def test_real_model_matched_beats_mismatched():
    """真实门禁证据：同一张真图，配对 caption 的对齐分必须显著高于错配 caption。

    数据：Flickr8k 真实图片 + 其真实英文 caption（正）/ 另一张图的 caption（负）。
    注：CN-CLIP 以中文为主训，英文 caption 属跨语言弱口径——本测试只断言序关系
    （正 > 负），不断言绝对分值；绝对口径的标定与评测由 clip_bench 用 OpenAI CLIP 完成。
    """
    import json

    from modalities.image.clip_align import ClipAlignScorer

    meta = [json.loads(l) for l in
            (FLICKR_DIR / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    by_img: dict = {}
    for r in meta:
        by_img.setdefault(r["file_name"], []).append(r["text"])
    imgs = sorted(by_img)
    img_a, img_b = imgs[0], imgs[1]
    cap_a, cap_b = by_img[img_a][0], by_img[img_b][0]

    scorer = ClipAlignScorer(CN_CLIP_DIR, device="cpu")
    pos = scorer.raw_cosine(FLICKR_DIR / img_a, cap_a)
    neg = scorer.raw_cosine(FLICKR_DIR / img_a, cap_b)
    assert pos is not None and neg is not None
    assert pos > neg, f"配对 caption 应更对齐：pos={pos:.4f} neg={neg:.4f}"
