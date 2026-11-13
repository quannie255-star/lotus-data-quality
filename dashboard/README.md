# dashboard

职责：Lotus 的「被打开使用」入口——Streamlit + DuckDB 只读看板（旁路观测，不碰 MinIO），
外加控制台旋钮/配方（唯一写配置的入口，写回后由全链重算生效）。

## 一条命令启动

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

浏览器打开后六个 Tab：**控制台 / 治理对比 / 处理漏斗 / 删除归因 / 质量分布 / runs 审计**。

## 能做什么（产品化验收对照）

| 能力 | 位置 | 说明 |
|---|---|---|
| 治理前后六项指标对比 | 「治理对比」Tab | 表格 + 比率类指标 before/after 柱状图；数据源 `data/metrics/comparison.json` |
| 三档配方切换 | 「控制台」Tab | 干净优先 / 多样优先 / 去重优先（`configs/recipes.json`）；先看逐旋钮 diff + 抽样预览代价，再一键应用并后台全链重算。配方数值已定稿（2026-11-13，每档理由见 recipes.json 的 rationale 字段） |
| 导出评估报告 | 「治理对比」Tab | 一键生成自包含 HTML：六项指标卡 + 四臂困惑度消融 + 三档配方代价（抽样口径显式标注）+ 去留账本摘要；缺失块显示占位与产出命令，不编数 |
| 旋钮控制台 | 「控制台」Tab | 单旋钮微调 → 预览（1 万行抽样，秒级）→ 保存写回（`knobs.py` 原子精准替换，防格式漂移污染指纹） |
| 后台全链重算 | 「控制台」Tab | `pipeline_runner.py` subprocess 跑 clean→…→metrics，进度落 `data/meta/pipeline_progress.json` |

## 文件

| 文件 | 职责 |
|---|---|
| `app.py` | 页面装配（六 Tab）；只读查询走 DuckDB over 本地 parquet |
| `knobs.py` | 旋钮读写（KNOB_BOUNDS 范围校验 + 字节级精准替换） |
| `recipes.py` | 三档配方加载/校验/diff/应用（校验失败不写任何文件） |
| `preview.py` | 抽样预览引擎（只读不落盘，绕开 lru_cache 的纯函数通道） |
| `report.py` | 评估报告 HTML 渲染（纯函数，数字全部注入，可离线单测） |
| `pipeline_runner.py` | 后台全链重算（启动/进度/状态） |

## 测试

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard_knobs.py tests/test_dashboard_recipes.py -q
```

变异证据（门禁能红）见 `tests/test_dashboard_recipes.py`：未知旋钮键、越界值、
HTML 注入（XSS 转义）三组用例；`口径说明` 见 `downstream/grad_select/口径说明.md`
（配方相关门禁清单在测试 docstring 内逐条标注）。

## 已知边界

- 配方预览与旋钮预览均为 **1 万行抽样口径**（跳过 dedup——MinHash 是 27–75 分钟的大头），
  只回答「代价方向与幅度」，不替代全量重算后的指标卡。
- 看板查询 fail-fast 让位于可用性（dev-lessons #004）：层产物缺失时显示占位而非报错。
- Streamlit 1.49 起 `use_container_width` 已弃用；本仓看板与 demo 已全部迁移到 `width="stretch"`（2026-11），后续只用新写法。
