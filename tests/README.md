# tests

职责：全仓测试。`pytest.ini` 固定 `testpaths = tests`，直接 `pytest` 即可。

- 单测：ID / 账本 / 幂等 / 各阶段规则
- 集成：跨阶段链路（`test_integration.py`）、冒烟（`test_smoke.py`）
- 门禁变异证据：CLIP、配方、梯度筛选等模块测试内含「故意改坏必变红」用例
- 可选依赖：torch 缺失时并行臂与 LoRA 用例自动 skip（`pytest.mark.skipif`）

```bash
python -m pytest -q          # 全量
python -m pytest -q -k clip  # 单模块
```
