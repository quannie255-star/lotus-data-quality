"""clip_bench — 图文跨模态对齐判别评测套件（DL 大作业本体）

dataset.py   Flickr8k 真实数据 → 正/易负/硬负样本对（标定/测试集分离）
evaluate.py  手写可审计指标：AUC(秩公式)/F1/混淆矩阵/约登阈值/R@k
run_eval.py  编排入口 → data/clip/eval_report_{tag}.json + calibration_{tag}.json

对齐分内核复用 modalities/image/clip_align.py（Lotus 插件层，一份内核两处落地）。
文档：README.md / DESIGN.md / 口径说明.md / 开发日志.md / 实验报告（本目录）。
"""
