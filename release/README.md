# release

职责：把配比候选发布为带版本号、license 执法与溯源卡的数据集。

- 入口：`python -m release.run`
- 版本规则：源集合变 = minor，配置变 = minor，纯重跑 = patch（从 v0.1 起）
- license 执法：`license_status` 不满足发布策略的行剔除并写变更账
- 产物：`data/release/v{semver}/part-*.parquet` + `provenance_card.yaml` + manifest
- 测试：`pytest tests/test_mix_release.py -q`
