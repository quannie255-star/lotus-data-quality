"""metrics/multimodal.py — 多模态质量指标（M11）

文本指标（metrics/card.py）不动；本文件新增多模态指标项：配对完整性、图像精确
重复率、跨模态（caption）重复率。口径纪律与指标卡一致：确定性、显式 sampled、
空读数不假装（alignment 类指标未接入就是 None）。
"""

from collections import Counter

from common import ids


def pair_metrics(rows: list) -> dict:
    """配对样本行 → 多模态指标（v0 口径）。

    参数: rows — 行 dict 列表（image_text 样本：含 caption/image_bytes_hash/image_bytes_len）
    返回: {"n_pairs", "orphan_rate", "image_exact_dup_rate", "caption_dup_rate",
           "format_breakdown", "notes"}
    幂等: 纯函数；行序确定 → 同输入同卡
    """
    n = len(rows)
    if n == 0:
        return {"n_pairs": 0, "orphan_rate": None, "image_exact_dup_rate": None,
                "caption_dup_rate": None, "format_breakdown": {},
                "notes": ["零配对样本：空读数，不假装有比例"]}
    orphans = sum(1 for r in rows
                  if r.get("image_bytes_hash") is None or not (r.get("caption") or "").strip())
    seen_img, seen_cap = set(), set()
    img_dup = cap_dup = 0
    for r in rows:  # 行序确定（文件序+行序），dup 定义=与在它之前出现的相同者
        if r.get("image_bytes_hash") is not None:
            if r["image_bytes_hash"] in seen_img:
                img_dup += 1
            else:
                seen_img.add(r["image_bytes_hash"])
        cap_hash = ids.normalized_content_hash(r.get("caption") or "")
        if cap_hash in seen_cap:
            cap_dup += 1
        else:
            seen_cap.add(cap_hash)
    with_img = len(seen_img) + img_dup
    return {
        "n_pairs": n,
        "orphan_rate": round(orphans / n, 5),
        "image_exact_dup_rate": round(img_dup / with_img, 5) if with_img else None,
        "caption_dup_rate": round(cap_dup / n, 5),
        "format_breakdown": dict(sorted(Counter(
            r.get("image_format") or "unknown" for r in rows).items())),
        "notes": [
            "orphan_rate = 缺图或空 caption 的对占比（治理前口径；清洗层会 hold/drop）",
            "image_exact_dup_rate 分母为有图像哈希的对（未检测≠空≠有值）；caption_dup_rate 分母为全部对",
            "感知/语义级重复与跨模态对齐：未接入，不给数",
        ],
    }


def compute_multimodal_card(parts: list, label: str) -> dict:
    """对一组 parquet part 计算多模态指标卡（modality 维度分布 + 配对指标）。

    参数: parts — parquet 文件路径列表（可含纯文本行；只对 image_text 行算配对指标）
    返回: 卡片 dict；无多模态行时 paired 指标为 None（不假装）
    异常: parts 为空 → ValueError
    幂等: 纯读 + 确定性（行序=文件序+批序）
    """
    if not parts:
        raise ValueError(f"[{label}] 没有可计算的数据文件")
    import pyarrow.parquet as pq

    by_modality = Counter()
    paired_rows = []
    for path in parts:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
            for row in batch.to_pylist():
                modality = row.get("modality") or "text"   # 历史文本行无 modality 列 → text
                by_modality[modality] += 1
                if modality == "image_text":
                    paired_rows.append(row)
    paired = pair_metrics(paired_rows) if paired_rows else None
    return {
        "label": label,
        "modality_breakdown": dict(sorted(by_modality.items())),
        "paired": paired,
        "notes": [
            "历史文本行无 modality 列，按 text 计（未检测≠空≠有值的读取侧推论）",
            "paired=None 表示本版本无配对样本，不假装有配对指标",
        ],
    }
