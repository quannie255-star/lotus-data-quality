"""clip_bench/apply_calibration.py — 把标定产物回填 Lotus 门禁配置（可复现、留痕）

读取 data/clip/calibration_{tag}.json，按约登阈值锚定公式换算 [cos_lo, cos_hi]
（align(youden_th)=0.30 ↔ Lotus min_modality_score 门禁线；align(pos_p95)=1.0），
以**字节级精准替换**写回 configs/multimodal_policy.json（复用 dashboard.knobs 的
防格式漂移纪律——整文件重写会污染配置指纹）。

用法：python -m clip_bench.apply_calibration --tag cnclip_full [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data" / "clip"
POLICY = ROOT / "configs" / "multimodal_policy.json"


def compute_interval(calib: dict) -> tuple[float, float]:
    th = calib["youden"]["threshold"]
    p95 = calib["pos_cos"]["p95"]
    lo = round(th - (0.30 / 0.70) * (p95 - th), 4)
    hi = round(p95, 4)
    if not lo < hi:
        raise ValueError(f"标定退化：cos_lo({lo}) >= cos_hi({hi})——分布重叠过大，不可回填")
    return lo, hi


def apply(tag: str, dry_run: bool = False) -> tuple[float, float]:
    path = CALIB / f"calibration_{tag}.json"
    calib = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = compute_interval(calib)
    text = POLICY.read_text(encoding="utf-8")
    new = text
    for key, val in (("cos_lo", lo), ("cos_hi", hi)):
        pat = re.compile(r'("' + key + r'"\s*:\s*)(-?[\d.]+)')
        new, n = pat.subn(lambda m: m.group(1) + repr(val), new, count=1)
        if n != 1:
            raise ValueError(f"定位 {key} 失败——配置结构可能已变")
    if dry_run:
        print(f"[dry-run] {tag}: cos_lo={lo} cos_hi={hi}（未写入）")
        return lo, hi
    tmp = POLICY.with_suffix(".json.tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(POLICY)
    print(f"[applied] {tag}: cos_lo={lo} cos_hi={hi} → {POLICY}")
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    apply(args.tag, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
