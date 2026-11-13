# mix

职责：按体裁/源分层，在固定 token 预算内做确定性抽样，产出配比候选。

- 入口：`python -m mix.run`
- 流程：分层 → 预算分配 → 层内按质量分降序 + 种子洗牌确定性选取 → 枯竭层按剩余层等比再分配并记账
- 输入 → 输出：`data/quality/source={s}/` → `data/mix/candidates/`（附 `token_count` / `sampling_weight` / `mix_batch_id`）
- 配置：`configs/mix_policy.json`
- 测试：`pytest tests/test_mix_release.py -q`
