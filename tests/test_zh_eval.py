"""clip_bench/zh_eval.py 的测试（中文图文对标定入口）

锁住三件事：
1. metadata.jsonl 的读取纪律：逐字空格归一化、缺图账目（剔除有账，不静默）；
2. 构对复用同一口径但图片根可换（中文语料不必复制 Flickr 的目录结构）；
3. clip.enabled=false 时联合门禁报告必须拒绝运行（不拿未标定的阈值判门禁）。
"""

import json
from pathlib import Path

import pytest

from clip_bench import zh_eval
from clip_bench.dataset import build_pairs


def _write_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.png").write_bytes(b"fake-image-a")
    (tmp_path / "b.png").write_bytes(b"fake-image-b")
    rows = [
        {"file_name": "a.png", "text": "阳 光 下 的 花 朵"},
        {"file_name": "b.png", "text": " 小 猫 在 睡 觉 "},
        {"file_name": "missing.png", "text": "缺图条目"},
        {"file_name": "b.png", "text": ""},
    ]
    (tmp_path / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_load_metadata_normalizes_and_accounts_missing(tmp_path):
    root = _write_corpus(tmp_path)
    by_img = zh_eval.load_metadata(root)
    assert by_img == {"a.png": ["阳光下的花朵"], "b.png": ["小猫在睡觉"]}
    # 缺图与空文都进剔除账目（不静默）
    assert sorted(zh_eval.by_img_excluded) == ["b.png", "missing.png"]


def test_build_pairs_uses_given_image_root(tmp_path):
    root = _write_corpus(tmp_path)
    by_img = zh_eval.load_metadata(root)
    pairs = build_pairs(["a.png", "b.png"], by_img, image_root=root)
    assert len(pairs.images) == 4  # 2 正 + 2 易负
    assert pairs.images[0] == root / "a.png"
    assert pairs.labels == [1, 0, 1, 0]
    assert pairs.kinds == ["pos", "easy_neg", "pos", "easy_neg"]


def test_gate_report_refuses_when_clip_disabled(monkeypatch, tmp_path):
    root = _write_corpus(tmp_path)
    disabled = {"image_text": {"clip": {"enabled": False}}, "joint_gate": {}}

    import quality.multimodal as qm
    monkeypatch.setattr(qm, "policy", lambda: disabled)
    with pytest.raises(RuntimeError, match="clip.enabled=false"):
        zh_eval.gate_report(root, "unit", n=2)
