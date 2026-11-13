"""dashboard/report.py — 「数据质量评估报告」HTML 导出（业务产出物）

纯函数 build_report_html(...)：所有数字由调用方注入（看板从 data/metrics/comparison.json、
data/downstream/report.json、changelog 聚合、配方预览取），本模块只做渲染——
可离线单测、无 Streamlit 依赖。

口径纪律：
- 报告内每个数字标注来源文件；预览类数字显式标注「1 万行抽样口径，非全量」
- 缺失的数据块渲染成「未产出」占位，不静默省略、不编数
- 自包含 HTML（内联 CSS，无外链），下载即可存档/发送
"""

from __future__ import annotations

import html
from datetime import datetime

_CSS = """
body{font-family:'Microsoft YaHei',system-ui,sans-serif;margin:2rem auto;max-width:960px;
     color:#222;line-height:1.6}
h1{border-bottom:3px solid #1f77b4;padding-bottom:.4rem}
h2{margin-top:2rem;color:#1f77b4}
table{border-collapse:collapse;width:100%;margin:.8rem 0}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}
th{background:#f0f5fa}
.good{color:#0a7a2f;font-weight:bold}.bad{color:#b00020;font-weight:bold}
.src{color:#666;font-size:.85rem}
.missing{background:#fff8e1;border:1px dashed #b8860b;padding:.8rem;border-radius:4px}
footer{margin-top:3rem;color:#888;font-size:.85rem;border-top:1px solid #ddd;padding-top:.8rem}
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _comparison_section(comp: dict | None) -> str:
    if not comp:
        return ('<div class="missing">治理前后指标卡未产出——先跑 '
                '<code>python -m metrics.run</code>（来源：data/metrics/comparison.json）</div>')
    rows = []
    for key, m in sorted(comp["metrics"].items()):
        verdict = m.get("verdict", "?")
        cls = "good" if verdict == "改善" else ("bad" if verdict == "恶化" else "")
        rows.append(
            f"<tr><td>{_esc(key)}</td><td>{_esc(m['before'])}</td><td>{_esc(m['after'])}</td>"
            f"<td>{_esc(m['delta'])}</td><td class='{cls}'>{_esc(verdict)}</td></tr>")
    return (f"<p class='src'>口径：{_esc(comp.get('before'))} → {_esc(comp.get('after'))}，"
            f"来源 data/metrics/comparison.json（metrics 阶段全量产物）</p>"
            "<table><tr><th>指标</th><th>治理前</th><th>治理后</th><th>Δ</th><th>判定</th></tr>"
            + "".join(rows) + "</table>")


def _ppl_section(ppl: dict | None) -> str:
    if not ppl:
        return ('<div class="missing">下游困惑度消融未产出——先跑 '
                '<code>python -m downstream.run</code>（来源：data/downstream/report.json）</div>')
    rows = []
    for arm, a in sorted(ppl.get("arms", {}).items()):
        if "ppl" in a:
            rows.append(f"<tr><td>{_esc(arm)}</td><td>{_esc(a['ppl'])}</td>"
                        f"<td>{_esc(a.get('ci95'))}</td><td>{_esc(a.get('trained_chars'))}</td></tr>")
        else:
            rows.append(f"<tr><td>{_esc(arm)}</td><td colspan='3'>{_esc(a.get('skipped'))}</td></tr>")
    return (f"<p class='src'>引擎 {_esc(ppl.get('engine'))}，评测集 {_esc(ppl.get('eval_n'))} 行"
            f"（冻结），每臂训练语料上限 {_esc(ppl.get('train_chars_cap'))} 字符；"
            f"来源 data/downstream/report.json</p>"
            "<table><tr><th>臂</th><th>ppl</th><th>95%CI</th><th>训练字符</th></tr>"
            + "".join(rows) + "</table>")


def _recipe_section(previews: dict | None) -> str:
    """previews: {配方label: run_preview 结果}。抽样口径必须显式标注。"""
    if not previews:
        return ('<div class="missing">配方对比未生成——在看板控制台点「生成三档配方预览」'
                '后重新导出（预览为 1 万行抽样口径）</div>')
    rows = []
    for label, r in previews.items():
        n = max(r.get("n_sample", 0), 1)
        clean_rm = r.get("clean", {}).get("removed", 0) / n * 100
        q = r.get("quality", {})
        passed = q.get("pass", 0) / n * 100
        removed = q.get("remove", 0) / n * 100
        rows.append(f"<tr><td>{_esc(label)}</td><td>{clean_rm:.2f}%</td>"
                    f"<td>{passed:.2f}%</td><td>{removed:.2f}%</td></tr>")
    return ("<p class='src'><b>抽样口径</b>：raw 层固定种子抽 1 万行跑 clean+quality"
            "（跳过 dedup），是「配方代价」的秒级预览，<b>不是</b>全量重算结果；"
            "来源 dashboard/preview.py</p>"
            "<table><tr><th>配方</th><th>清洗删除率</th><th>门禁放行率</th><th>门禁删除率</th></tr>"
            + "".join(rows) + "</table>")


def _attribution_section(top: list | None, limit: int = 15) -> str:
    if not top:
        return '<div class="missing">changelog 尚无记录（data/meta/changelog.jsonl）</div>'
    rows = "".join(
        f"<tr><td>{_esc(r.get('源'))}</td><td>{_esc(r.get('阶段'))}</td>"
        f"<td>{_esc(r.get('原因'))}</td><td>{_esc(r.get('条数'))}</td></tr>"
        for r in top[:limit])
    return (f"<p class='src'>去留账本 Top{min(limit, len(top))}（源×阶段×原因聚合），"
            f"来源 data/meta/changelog.jsonl</p>"
            "<table><tr><th>源</th><th>阶段</th><th>原因</th><th>条数</th></tr>"
            + rows + "</table>")


def build_report_html(comparison: dict | None, ppl_report: dict | None,
                      recipe_previews: dict | None, attribution_top: list | None,
                      generated_at: datetime | None = None) -> str:
    """渲染完整报告。四个数据块彼此独立缺失——缺哪块哪块显示占位与产出命令，不编数。"""
    ts = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Lotus 数据质量评估报告 · {_esc(ts)}</title><style>{_CSS}</style></head>
<body>
<h1>Lotus 濯莲 · 数据质量评估报告</h1>
<p class="src">生成时间：{_esc(ts)}（dashboard/report.py 渲染，数字全部注入自下列来源文件）</p>
<h2>1. 治理前后六项指标</h2>
{_comparison_section(comparison)}
<h2>2. 下游困惑度消融（四臂）</h2>
{_ppl_section(ppl_report)}
<h2>3. 三档配方代价对比</h2>
{_recipe_section(recipe_previews)}
<h2>4. 去留账本摘要</h2>
{_attribution_section(attribution_top)}
<footer>本报告由 Lotus 看板一键导出。复现：streamlit run dashboard/app.py →「治理对比」→
「导出评估报告」。各数字的分子分母与门禁定义见 downstream/grad_select/口径说明.md 与
docs/（项目文档）。合成/抽样口径均已在对应小节显式标注。</footer>
</body></html>"""
