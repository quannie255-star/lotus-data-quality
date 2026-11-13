"""modalities/image/clip_align.py — CLIP 图文对齐打分器（跨模态质量维度）

把「图配的文对不对」变成一个 0~1 的读数：图像与 caption 各过 CLIP 编码器取
embedding，余弦相似度经标定区间 [cos_lo, cos_hi] 线性归一到 [0,1]。

纪律（ARCHITECTURE §7 插件层边界）：
- torch/transformers 只在本模块 import（且 lazy）——核心层零感知；
- **空读数不假装**：解码失败/缺图/空 caption → None（联合门禁走 remove(no_score)）；
- 模型加载失败 → 抛异常（fail-fast，不静默降级成规则版）；
- 纯函数幂等：同图同文同分（eval 模式 + no_grad，无随机算子）。

后端自适应：config.json architectures 含 ChineseCLIPModel → 中文 CLIP（CN-CLIP，
Lotus 中文语料默认）；含 CLIPModel → OpenAI CLIP（英文 caption / 课程评测用）。

归一化口径：align = clip((cos − cos_lo)/(cos_hi − cos_lo), 0, 1)。cos_lo/cos_hi
来自真实数据阈值标定（clip_bench/calibrate.py，Flickr8k 实测后回填配置），
**不拍脑袋**——标定前默认值仅作占位并在配置 note 中标注。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CACHE: dict[str, "ClipAlignScorer"] = {}


def _detect_backend(model_path: Path) -> str:
    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    arch = (cfg.get("architectures") or [""])[0]
    if "ChineseCLIP" in arch:
        return "cn_clip"
    if arch == "CLIPModel":
        return "clip"
    raise ValueError(f"不支持的 CLIP 架构: {arch}（{model_path}）")


class ClipAlignScorer:
    """CLIP 对齐打分器（lazy 加载、进程内单例，见 get_scorer）。"""

    def __init__(self, model_path: str | Path, device: str | None = None,
                 cos_lo: float = 0.0, cos_hi: float = 0.4, max_text_len: int = 52):
        import torch
        from transformers import (AutoConfig, BertTokenizer, ChineseCLIPModel,
                                  CLIPModel, CLIPTokenizer)
        from transformers.models.chinese_clip.image_processing_chinese_clip import \
            ChineseCLIPImageProcessor
        from transformers import CLIPImageProcessor

        self.model_path = Path(model_path)
        self.backend = _detect_backend(self.model_path)
        self.cos_lo, self.cos_hi = float(cos_lo), float(cos_hi)
        if self.cos_hi <= self.cos_lo:
            raise ValueError(f"cos_hi({cos_hi}) 必须大于 cos_lo({cos_lo})")
        self.max_text_len = max_text_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.backend == "cn_clip":
            self.model = ChineseCLIPModel.from_pretrained(str(self.model_path))
            self.tokenizer = BertTokenizer.from_pretrained(str(self.model_path))
            self.image_proc = ChineseCLIPImageProcessor.from_pretrained(str(self.model_path))
        else:
            self.model = CLIPModel.from_pretrained(str(self.model_path))
            self.tokenizer = CLIPTokenizer.from_pretrained(str(self.model_path))
            self.image_proc = CLIPImageProcessor.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch
        self._logit_scale = None   # 余弦口径不使用 logit_scale（温度只影响 softmax 检索分布）

    # ---- 编码原语（批量接口供 clip_bench 评测复用，避免逐条加载图像） ----

    def _project_image(self, out):
        """transformers 版本兼容：4.x get_image_features 直接返回投影后张量；
        5.x 返回 BaseModelOutputWithPooling，其 pooler_output 已在模型内部过
        visual_projection（= 最终 embedding），取出即可，不可二次投影。"""
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def _project_text(self, out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def embed_images(self, images: list, batch_size: int = 32):
        """PIL.Image 列表 → L2 归一化 embedding (n, d)。

        注：transformers 5.x 的 legacy CLIP 系 image processor 在 preprocess 里把
        resample=None 直接透传给 resize（bug），须调用时显式传 BICUBIC——
        与 preprocessor_config.json 的 resample=3 同值，口径不变。
        """
        import torch
        from transformers.image_utils import PILImageResampling

        outs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.image_proc(images=batch, return_tensors="pt",
                                         resample=PILImageResampling.BICUBIC)
                feat = self._project_image(
                    self.model.get_image_features(**{k: v.to(self.device)
                                                     for k, v in inputs.items()}))
                outs.append(feat / feat.norm(dim=-1, keepdim=True))
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, 0))

    def embed_texts(self, texts: list[str], batch_size: int = 64):
        """caption 列表 → L2 归一化 embedding (n, d)。"""
        import torch

        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], padding=True,
                                     truncation=True, max_length=self.max_text_len,
                                     return_tensors="pt")
                feat = self._project_text(
                    self.model.get_text_features(**{k: v.to(self.device)
                                                    for k, v in enc.items()}))
                outs.append(feat / feat.norm(dim=-1, keepdim=True))
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, 0))

    # ---- 打分口径 ----

    def raw_cosine(self, image, caption: str) -> float | None:
        """图（PIL.Image 或路径）× caption → 原始余弦。坏输入 → None（不假装）。"""
        if not caption or not caption.strip():
            return None
        img = self._load_image(image)
        if img is None:
            return None
        ie = self.embed_images([img])
        te = self.embed_texts([caption])
        return float((ie[0] * te[0]).sum())

    def score(self, image, caption: str) -> float | None:
        """对齐分 [0,1]：cos 经 [cos_lo, cos_hi] 线性归一 + 截断。坏输入 → None。"""
        cos = self.raw_cosine(image, caption)
        return None if cos is None else self.normalize(cos)

    def normalize(self, cos: float) -> float:
        span = self.cos_hi - self.cos_lo
        return round(min(max((cos - self.cos_lo) / span, 0.0), 1.0), 6)

    @staticmethod
    def _load_image(image):
        """路径/str → RGB PIL.Image；任何失败 → None（空读数纪律）。"""
        try:
            from PIL import Image, UnidentifiedImageError

            if isinstance(image, Image.Image):
                return image.convert("RGB")
            path = Path(str(image))
            if not path.exists() or path.stat().st_size == 0:
                return None
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception:   # noqa: BLE001 —— 解码失败=坏数据 → None，由门禁处置
            return None


def get_scorer(clip_cfg: dict | None = None) -> ClipAlignScorer:
    """按 multimodal_policy.image_text.clip 配置取进程内单例（key=model_path）。

    异常：enabled=false 时抛 ValueError（调用方应先判断 enabled）；模型加载失败原样上抛。
    """
    if clip_cfg is None:
        clip_cfg = json.loads((ROOT / "configs" / "multimodal_policy.json")
                              .read_text(encoding="utf-8"))["image_text"]["clip"]
    if not clip_cfg.get("enabled"):
        raise ValueError("clip.enabled=false —— 调用方应先检查开关再取打分器")
    path = str((ROOT / clip_cfg["model_path"]).resolve())
    if path not in _CACHE:
        _CACHE[path] = ClipAlignScorer(
            path, cos_lo=clip_cfg.get("cos_lo", 0.0), cos_hi=clip_cfg.get("cos_hi", 0.4),
            max_text_len=clip_cfg.get("max_text_len", 52))
    return _CACHE[path]
