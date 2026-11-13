"""clip_bench/demo_app.py — 图文对齐判别交互 demo（DL 大作业 L3 产品层）

用法：
    streamlit run clip_bench/demo_app.py

功能：上传图片 + 输入 caption → CLIP 对齐分（原始余弦 + 标定归一 [0,1]）+
匹配判定（标定阈值）+ 梯度显著图热力图（相似度对像素的梯度归因——
「模型看图的哪里来对齐文字」的可视化口径，见 口径说明.md §5）。

模型：默认 CN-CLIP（中文 caption），可切换 OpenAI CLIP（英文，下载就绪后出现）。
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "CN-CLIP（中文 caption）": ROOT / "data" / "models" / "cn-clip-vit-base-patch16",
    "OpenAI CLIP ViT-B/32（英文 caption）": ROOT / "data" / "models" / "clip-vit-base-patch32",
}

st.set_page_config(page_title="图文对齐判别 demo", layout="wide")
st.title("图文跨模态对齐判别（CLIP）")
st.caption("判断「图配的文字对不对」：上传图像 + 输入描述 → 对齐分 / 判定 / 显著图。"
           "内核与 Lotus 多模态质量门禁同源（modalities/image/clip_align.py）。")


@st.cache_resource(show_spinner="加载模型…")
def get_scorer(model_dir: str):
    from modalities.image.clip_align import ClipAlignScorer

    calib = _load_calib(model_dir)
    return ClipAlignScorer(model_dir,
                           cos_lo=calib.get("suggest_cos_lo", 0.0),
                           cos_hi=calib.get("suggest_cos_hi", 0.4))


def _load_calib(model_dir: str) -> dict:
    """标定文件（clip_bench.run_eval 产出）→ 归一化区间与判定阈值；缺失用占位默认。
    tag 优先级：正式 > 全规模 > 预演（用最新可用的标定，不静默用占位）。"""
    name = Path(model_dir).name
    tags = (["clipb32"] if "cn-clip" not in name
            else ["cnclip", "cnclip_full", "cnclip_rehearsal"])
    for tag in tags:
        p = ROOT / "data" / "clip" / f"calibration_{tag}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def saliency_map(scorer, img, caption: str):
    """梯度显著图：d(cos_sim)/d(pixel) 的通道绝对值和，归一到 [0,1]（numpy）。"""
    import numpy as np
    import torch
    from transformers.image_utils import PILImageResampling

    inputs = scorer.image_proc(images=[img], return_tensors="pt",
                               resample=PILImageResampling.BICUBIC)
    pv = inputs["pixel_values"].to(scorer.device).requires_grad_(True)
    feat = scorer._project_image(scorer.model.get_image_features(pixel_values=pv))
    ie = feat / feat.norm(dim=-1, keepdim=True)
    te = scorer.embed_texts([caption])
    sim = (ie[0] * te[0]).sum()
    sim.backward()
    g = pv.grad[0].abs().sum(dim=0)          # (H, W)
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return (g.detach().cpu().numpy() * 255).astype("uint8")


def _ready(d: Path) -> bool:
    """模型就绪 = config + 权重文件都在（权重下载中的模型不出现在选项里）。"""
    return (d / "config.json").exists() and (
        (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists())


available = {k: v for k, v in MODELS.items() if _ready(v)}
if not available:
    st.error("没有可用模型——先按 clip_bench/README.md 下载 CN-CLIP 或 OpenAI CLIP。")
    st.stop()

col_model, col_thresh = st.columns([2, 1])
with col_model:
    pick = st.selectbox("模型", list(available))
with col_thresh:
    calib = _load_calib(available[pick])
    default_th = calib.get("youden", {}).get("threshold", 0.3)
    th = st.number_input("判定阈值（cos，默认=标定集约登阈值）", value=float(default_th),
                         min_value=-1.0, max_value=1.0, step=0.01)

left, right = st.columns(2)
with left:
    up = st.file_uploader("上传图像", type=["jpg", "jpeg", "png", "webp"])
    caption = st.text_input("Caption（与所选模型语言一致）",
                            placeholder="例如：一只黑狗和一只花狗在打架")
with right:
    st.subheader("结果")
    if up is not None and caption.strip():
        from PIL import Image

        img = Image.open(up).convert("RGB")
        st.image(img, caption="输入图像", width="stretch")
        scorer = get_scorer(str(available[pick]))
        cos = scorer.raw_cosine(img, caption)
        if cos is None:
            st.error("无法打分：图像解码失败或 caption 为空（空读数不假装）。")
        else:
            align = scorer.normalize(cos)
            verdict = "匹配 ✅" if cos >= th else "不匹配 ❌"
            st.metric("对齐分（标定归一 [0,1]）", f"{align:.3f}",
                      f"原始余弦 {cos:.3f} · 阈值 {th:.3f}")
            st.success(verdict) if cos >= th else st.error(verdict)
            with st.spinner("计算梯度显著图…"):
                sal = saliency_map(scorer, img, caption)
                st.image(sal, caption="梯度显著图（相似度对像素的归因，亮=对对齐贡献大）",
                         width="stretch")
    else:
        st.info("上传图像并输入 caption 后出结果。")

st.divider()
st.caption("口径：对齐分 = cos(CLIP 图像向量, CLIP 文本向量) 经 [cos_lo, cos_hi] 线性归一；"
           "阈值来自标定集约登指数（data/clip/calibration_*.json），不拍脑袋。"
           "对齐分是「图文语义匹配」弱信号，不判断「文说得对不对」（诚实边界）。")
