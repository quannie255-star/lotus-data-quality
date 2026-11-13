"""downstream/lora_finetune/model.py — Qwen3-0.6B + LoRA 装载与训练/打分原语

课程 PDF「模型实验实现」口径：
- 冻结基础模型 θ_base，只对 LoRA 参数 φ 求导（打分与训练都只动 φ）
- 打分时关闭 dropout 等随机行为；损失归一化方式显式固定（**逐 token 平均**，
  每条样本一个标量损失——见 口径说明.md §2）
- 逐样本梯度打分后立即释放（不整批保存 (N,P) 矩阵——4NP 字节论据的落点）
- 批量计算时保留逐样本梯度，整批平均不能替代逐样本（参考梯度除外：ḡref 本就是平均）

手动训练循环（不用 trl/Trainer）：依赖面小、每步可审计，与项目「确定性铁律」一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0        # 打分要求关闭随机行为；训练同口径（对照公平）
    max_len: int = 256               # 打分/训练统一截断长度（口径说明 §2）
    dtype: str = "float32"           # 8GB 显存下 fp32 0.6B 可跑；bf16 更快但打分精度对照需另证


def load_model(cfg: ModelConfig, device: str, seed: int = 42):
    """装载基础模型 + LoRA 适配器（φ 按 seed 确定性初始化）。返回 (model, tokenizer)。"""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.name)
    base = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=dtype)
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=LORA_TARGETS,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.to(device)
    model.eval()   # 打分口径：dropout 关；训练循环里显式 train() 也只影响 LoRA 的 p=0 dropout
    return model, tok


def lora_param_vector(model) -> torch.Tensor:
    """把所有 requires_grad 参数（= LoRA φ）摊平拼成一个向量（打分用的 gᵢ 就是这个空间里的向量）。"""
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()
                      if p.requires_grad and p.grad is not None])


def encode(tokenizer, text: str, max_len: int, device: str):
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def sample_loss(model, batch) -> torch.Tensor:
    """单条样本损失：逐 token 交叉熵**按有效 token 数平均**（每条样本等权，
    不随长度缩放——长度偏差进 口径说明 §4 已知坑）。labels=input_ids（自回归）。"""
    labels = batch["input_ids"].clone()
    pad = model.config.pad_token_id
    if pad is not None:
        labels[labels == pad] = -100
    out = model(**batch, labels=labels)
    return out.loss


def per_sample_grad(model, tokenizer, text: str, cfg: ModelConfig, device: str) -> torch.Tensor:
    """单条样本对 φ 的梯度向量（backward 后立即从计算图脱离；调用方负责释放）。"""
    model.zero_grad(set_to_none=True)
    loss = sample_loss(model, encode(tokenizer, text, cfg.max_len, device))
    loss.backward()
    return lora_param_vector(model)


def mean_reference_grad(model, tokenizer, reference: list[str], cfg: ModelConfig,
                        device: str, report_every: int = 0) -> torch.Tensor:
    """ḡref = (1/M) Σ ∇φ L(z′ⱼ)（逐条 backward 累加——M=64 开销可控）。"""
    acc = None
    for j, text in enumerate(reference):
        g = per_sample_grad(model, tokenizer, text, cfg, device)
        acc = g.clone() if acc is None else acc + g
        if report_every and (j + 1) % report_every == 0:
            print(f"  [ref grad] {j + 1}/{len(reference)}", flush=True)
    return acc / len(reference)


def cosine_score_torch(g: torch.Tensor, g_ref: torch.Tensor, g_ref_norm: float) -> float:
    """与 grad_select.core.cosine_score 同约定：任一方范数为零 → 0。"""
    ng = float(g.norm())
    if ng == 0.0 or g_ref_norm == 0.0:
        return 0.0
    return float(torch.dot(g, g_ref)) / (ng * g_ref_norm)


def score_candidates(model, tokenizer, candidates: list[str], g_ref: torch.Tensor,
                     cfg: ModelConfig, device: str, report_every: int = 100) -> list[float]:
    """全候选打分：逐条 backward → 余弦 → 释放（课程「打分后便可释放」的实现）。"""
    g_ref_norm = float(g_ref.norm())
    scores = []
    for i, text in enumerate(candidates):
        g = per_sample_grad(model, tokenizer, text, cfg, device)
        scores.append(cosine_score_torch(g, g_ref, g_ref_norm))
        del g
        if report_every and (i + 1) % report_every == 0:
            print(f"  [score] {i + 1}/{len(candidates)}", flush=True)
    return scores


def select_topk_by_score(scores: list[float], ratio: float = 0.05) -> list[int]:
    """与 grad_select.core.select_topk 同口径：k=⌈ratio·N⌉、降序、同分按编号升序。"""
    n = len(scores)
    k = math.ceil(ratio * n)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return sorted(order[:k])


def sft(model, tokenizer, texts: list[str], cfg: ModelConfig, device: str,
        epochs: int, lr: float, seed: int = 42, report_every: int = 10) -> dict:
    """正式 SFT：只更新 LoRA φ，AdamW，样本顺序按 seed 洗牌（每 epoch 同序可复现）。

    返回 {steps, final_loss, wall_seconds}。超参对两臂（梯度筛选/随机）严格一致。
    """
    import time

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    t0 = time.perf_counter()
    steps, last = 0, None
    for ep in range(epochs):
        torch.manual_seed(seed + ep)
        order = torch.randperm(len(texts)).tolist()   # 每 epoch 固定序（可复现）
        for idx in order:
            opt.zero_grad(set_to_none=True)
            loss = sample_loss(model, encode(tokenizer, texts[idx], cfg.max_len, device))
            loss.backward()
            opt.step()
            steps += 1
            last = float(loss.detach())
            if report_every and steps % report_every == 0:
                print(f"  [sft] epoch {ep + 1} step {steps} loss {last:.4f}", flush=True)
    model.eval()
    return {"steps": steps, "final_loss": last,
            "wall_seconds": round(time.perf_counter() - t0, 1)}


@torch.no_grad()
def evaluate_ppl(model, tokenizer, texts: list[str], cfg: ModelConfig, device: str) -> dict:
    """独立测试集评测：逐 token NLL 总和 / 总 token 数 → ppl = exp(NLL)。

    口径：微平均（token 级），与 char-ngram ppl 消融不同引擎不可直接互比，
    只在**本实验两臂之间**可比（口径说明 §3）。
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        batch = encode(tokenizer, text, cfg.max_len, device)
        labels = batch["input_ids"].clone()
        out = model(**batch, labels=labels)
        n_tok = batch["input_ids"].numel()
        # 首 token 无预测目标：HF 的 loss 已按 shift 后有效 token 平均
        n_eff = max(n_tok - 1, 1)
        total_nll += float(out.loss) * n_eff
        total_tokens += n_eff
    nll = total_nll / max(total_tokens, 1)
    return {"token_nll": round(nll, 6), "ppl": round(math.exp(min(nll, 50)), 4),
            "n_texts": len(texts), "n_tokens": total_tokens}
