# dedup

职责：精确去重 + MinHash 近重复检测 + 并查集聚簇，输出代表行与「谁压掉谁」的账。

- 入口：`python -m dedup.run`
- 引擎：`engine.py`（MinHashLSH + union-find）
- 扩展接口：`api.py`（`DedupEngine` 抽象 + 注册表；图像字节级精确哈希已实装，感知哈希/音频指纹为预留）
- 输入 → 输出：`data/clean/source={s}/` → `data/dedup/source={s}/` + `signatures.parquet` + `report.json`
- 账本：每条删除带 `ref_doc_id` / `similarity` / `cluster_id`
- 测试：`pytest tests/test_dedup.py -q`
