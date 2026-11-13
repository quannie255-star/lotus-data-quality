"""dashboard/app.py — 数据质量看板（M10，Streamlit + DuckDB，只读）

四视图：处理漏斗（各层行数）/ 删除归因（changelog 按源×规则）/ 质量分布（五维直方图）/
runs 审计（成功/失败/cached）。全部查询走 DuckDB over 本地 parquet 镜像——不碰 MinIO，
不写任何东西（看板是旁路观测，fail-fast 让位于可用性，见 dev-lessons #004 口试 2）。

用法: streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import duckdb
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHANGELOG = DATA / "meta" / "changelog.jsonl"
RUNS = DATA / "meta" / "runs.jsonl"

st.set_page_config(page_title="Lotus 濯莲数据看板", layout="wide")
st.title("濯莲 Lotus 数据质量看板")


def _render_console():
    """控制台 Tab：外行人拧旋钮 → 秒级预览对比 → 保存写回配置 → 后台跑全链。"""
    import pandas as pd

    from dashboard.preview import run_preview
    from dashboard.knobs import read_knobs, knob_specs, write_knobs
    from dashboard.pipeline_runner import read_progress, start_pipeline, is_running

    st.subheader("旋钮控制台 —— 改标准，不动代码")
    st.caption(
        "拧动旋钮 → 「预览」看治理前后对比（真跑样本，秒级）→ 「保存并重算」写回配置 + 后台跑全链。")

    specs = knob_specs()
    current = read_knobs()
    cat_label = {"clean": "清洗规则", "dedup": "去重", "quality": "质量门禁"}

    # 用 form 收集所有旋钮输入，避免每改一个就 rerun
    new_values = {}
    with st.form("knob_form"):
        for cat in ("clean", "dedup", "quality"):
            st.markdown(f"**{cat_label[cat]}**")
            cols = st.columns(len(specs[cat]) if specs[cat] else 1)
            for i, (knob, spec) in enumerate(specs[cat].items()):
                with cols[i]:
                    cur = current[cat][knob]
                    new_values.setdefault(cat, {})[knob] = st.number_input(
                        f"{spec['desc']}", value=float(cur),
                        min_value=float(spec["min"]) if spec["min"] is not None else None,
                        max_value=float(spec["max"]) if spec["max"] is not None else None,
                        key=f"knob_{cat}_{knob}")
        preview_btn = st.form_submit_button("预览效果（治理前后对比）", type="primary")
        save_btn = st.form_submit_button("保存旋钮并后台重算", type="secondary")

    if preview_btn:
        with st.spinner("正在抽 1 万行样本跑规则…"):
            r_base = run_preview(overrides=None, n_sample=10000)
            r_new = run_preview(overrides=new_values, n_sample=10000)
        n = r_new["n_sample"]
        st.success(f"预览完成（样本 {n} 行）")

        # 治理前后对比图（当前配置 vs 新配置）
        rows = [
            {"指标": "清洗删除率", "当前配置": r_base["clean"]["removed"] / n * 100,
             "新配置": r_new["clean"]["removed"] / n * 100},
            {"指标": "门禁放行率", "当前配置": r_base["quality"]["pass"] / n * 100,
             "新配置": r_new["quality"]["pass"] / n * 100},
            {"指标": "门禁删除率", "当前配置": r_base["quality"]["remove"] / n * 100,
             "新配置": r_new["quality"]["remove"] / n * 100},
        ]
        df = pd.DataFrame(rows).set_index("指标")
        st.bar_chart(df, color=["#888888", "#1f77b4"])
        st.dataframe(df.round(2).rename(columns={"当前配置": "当前(%)", "新配置": "新(%)"}),
                     width="stretch")

        # 删除归因对比（新配置下按规则）
        by_rule = r_new["clean"]["by_rule"]
        if by_rule:
            st.markdown("**新配置下删除归因**")
            st.dataframe(pd.DataFrame([{"规则": k, "删除条数": v} for k, v in by_rule.items()]),
                         width="stretch")

    if save_btn:
        try:
            written = write_knobs(new_values)
            if is_running():
                st.warning(f"已保存 {written} 个旋钮，但当前有全链任务在跑，稍后再点重算。")
            else:
                res = start_pipeline()
                if res.get("started"):
                    st.success(f"已保存 {written} 个旋钮，后台开始跑全链（clean→…→metrics）。"
                               f"可在下方看进度。")
                else:
                    st.warning(f"已保存 {written} 个旋钮，但启动失败：{res.get('reason')}")
        except ValueError as e:
            st.error(f"保存失败：{e}")

    # ---- 三档配方：干净 / 多样 / 去重优先（configs/recipes.json）----
    st.divider()
    st.subheader("三档配方 —— 干净 / 多样 / 去重优先")
    st.caption("配方 = 一组旋钮预设。先看代价（抽样预览），再决定是否应用并全链重算。"
               "配方数值已定稿（2026-11-13 owner 采纳建议值，理由见 configs/recipes.json）。")
    from dashboard.recipes import apply_recipe, diff_with_current, load_recipes

    recipes = load_recipes()
    label_to_key = {r["label"]: name for name, r in recipes.items()}
    pick_label = st.selectbox("配方", list(label_to_key), key="recipe_pick")
    pick = label_to_key[pick_label]
    st.markdown(f"_{recipes[pick]['tradeoff']}_")
    st.dataframe(pd.DataFrame(diff_with_current(pick)), width="stretch",
                 hide_index=True)
    rc1, rc2 = st.columns(2)
    if rc1.button("预览该配方代价（抽 1 万行）", key="recipe_preview"):
        with st.spinner("正在抽 1 万行样本对比当前配置与配方…"):
            r_base = run_preview(overrides=None, n_sample=10000)
            r_new = run_preview(overrides=recipes[pick]["knobs"], n_sample=10000)
        n = max(r_new["n_sample"], 1)
        rows = [
            {"指标": "清洗删除率", "当前配置": r_base["clean"]["removed"] / n * 100,
             pick_label: r_new["clean"]["removed"] / n * 100},
            {"指标": "门禁放行率", "当前配置": r_base["quality"]["pass"] / n * 100,
             pick_label: r_new["quality"]["pass"] / n * 100},
            {"指标": "门禁删除率", "当前配置": r_base["quality"]["remove"] / n * 100,
             pick_label: r_new["quality"]["remove"] / n * 100},
        ]
        st.dataframe(pd.DataFrame(rows).set_index("指标").round(2), width="stretch")
    if rc2.button("应用配方并后台重算", type="primary", key="recipe_apply"):
        try:
            written = apply_recipe(pick)
            if is_running():
                st.warning(f"已应用配方「{pick_label}」（{written} 个旋钮），"
                           "但当前有全链任务在跑，稍后再点重算。")
            else:
                res = start_pipeline()
                if res.get("started"):
                    st.success(f"已应用配方「{pick_label}」（{written} 个旋钮），"
                               "后台开始全链重算。")
                else:
                    st.warning(f"已应用配方「{pick_label}」，但启动失败：{res.get('reason')}")
        except ValueError as e:
            st.error(f"应用配方失败（配置未改动）：{e}")

    # 后台全链进度展示
    prog = read_progress()
    if prog.get("status") == "running" or any(
            v.get("status") == "running" for v in prog.get("stages", {}).values()):
        st.info("全链重算进行中…")
        stage_order = ["clean", "dedup", "quality", "mix", "release", "downstream", "metrics"]
        stage_label = {"clean": "清洗", "dedup": "去重", "quality": "门禁", "mix": "配比",
                       "release": "发布", "downstream": "下游验证", "metrics": "指标卡"}
        for s in stage_order:
            v = prog.get("stages", {}).get(s)
            if v is None:
                st.text(f"○ {stage_label[s]}")
            elif v.get("status") == "running":
                st.text(f"⏳ {stage_label[s]} 运行中…")
            elif v.get("status") == "done":
                st.text(f"✅ {stage_label[s]} 完成")
            else:
                st.text(f"❌ {stage_label[s]} 失败")
        st.button("刷新进度")
    elif prog.get("status") == "done":
        st.success("全链重算完成。可到「处理漏斗」「治理对比」看最新结果。")
    elif prog.get("status") == "failed":
        st.error("全链重算失败，见各阶段标记。")





def _con() -> "duckdb.DuckDBPyConnection":
    return duckdb.connect()


@st.cache_data(ttl=60)
def funnel() -> list:
    """处理漏斗：各层行数（层目录存在才计入）。"""
    rows = []
    for zone, label in (("raw", "raw（采集）"), ("clean", "clean（清洗后）"),
                        ("dedup", "dedup（去重后）"), ("quality", "quality（门禁放行）"),
                        ("mix", "mix（配比候选）"), ("release", "release（发布）")):
        zone_dir = DATA / zone
        if not zone_dir.exists():
            continue
        try:
            n = _con().execute(
                f"SELECT count(*) FROM read_parquet('{zone_dir}/**/*.parquet', "
                f"union_by_name=true, filename=true) WHERE regexp_matches(filename, 'part-')"
            ).fetchone()[0]
            rows.append({"层": label, "行数": n})
        except duckdb.Error:
            rows.append({"层": label, "行数": None})
    return rows


@st.cache_data(ttl=60)
def removal_attribution() -> list:
    """删除归因：changelog 按 源×stage×reason 聚合（无文件返回空）。"""
    if not CHANGELOG.exists():
        return []
    agg: dict = {}
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r.get("source", "?"), r.get("stage", "?"), r.get("reason", "?"))
        agg[key] = agg.get(key, 0) + 1
    return [{"源": s, "阶段": st_, "原因": rs, "条数": n} for (s, st_, rs), n in
            sorted(agg.items(), key=lambda kv: -kv[1])]


@st.cache_data(ttl=60)
def quality_distribution() -> "list":
    """质量分布：quality 层的五维与总分（层未产出返回空）。"""
    qdir = DATA / "quality"
    if not qdir.exists():
        return []
    cols = ["quality_completeness", "quality_uniqueness", "quality_consistency",
            "quality_non_pollution", "quality_length", "quality_score"]
    try:
        df = _con().execute(
            f"SELECT {', '.join(cols)}, source FROM read_parquet('{qdir}/**/*.parquet', union_by_name=true)"
        ).df()
        return df
    except duckdb.Error:
        return None


@st.cache_data(ttl=60)
def runs_audit() -> list:
    """runs 审计：start/finish 对齐为每 run 一行（孤儿 start 标 running）。"""
    if not RUNS.exists():
        return []
    runs: dict = {}
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        rid = e.get("run_id")
        rec = runs.setdefault(rid, {"run_id": rid, "stage": e.get("stage", "?"),
                                    "status": "running", "note": ""})
        if e.get("event") == "finish":
            rec.update(status=e.get("status", "?"), note=e.get("note", "") or "")
    return list(runs.values())


@st.cache_data(ttl=60)
def governance_comparison():
    """治理前后指标卡对比（metrics 阶段产物）——控制台的读数面板。"""
    path = DATA / "metrics" / "comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def ppl_report():
    """四臂困惑度消融报告（downstream 阶段产物），供报告导出。"""
    path = DATA / "downstream" / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=300, show_spinner=False)
def run_preview_cached(recipe_key: str, overrides: dict):
    """配方预览缓存：同配方 5 分钟内不重跑抽样（recipe_key 进缓存键）。"""
    from dashboard.preview import run_preview
    return run_preview(overrides=overrides, n_sample=10000)


tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["控制台", "治理对比", "处理漏斗", "删除归因", "质量分布", "runs 审计"])

with tab0:
    _render_console()

with tab1:
    comp = governance_comparison()
    if comp is None:
        st.info("尚无指标卡——先跑 python -m metrics.run")
    else:
        import pandas as pd

        st.subheader(f"{comp['before']} → {comp['after']}（治理前后全指标）")
        st.dataframe([{"指标": k, **{c: v[c] for c in ("before", "after", "delta", "verdict")}}
                      for k, v in comp["metrics"].items()], width="stretch")
        # 比率类指标治理前后对比图（n_docs 量级差 2 个数量级，单独进表格不进图）
        rate_keys = [k for k in comp["metrics"] if k != "n_docs"]
        chart_df = pd.DataFrame(
            [{"指标": k, "治理前": comp["metrics"][k]["before"],
              "治理后": comp["metrics"][k]["after"]} for k in rate_keys]).set_index("指标")
        st.bar_chart(chart_df, color=["#888888", "#1f77b4"])

        st.divider()
        st.subheader("导出「数据质量评估报告」（HTML）")
        st.caption("汇总：六项指标卡 + 四臂困惑度消融 + 三档配方代价预览（抽样口径）+ 去留账本摘要。")
        if st.button("生成报告（含三档配方预览，约 10~30 秒）"):
            from dashboard.recipes import load_recipes
            from dashboard.report import build_report_html

            with st.spinner("抽 1 万行样本跑三档配方预览…"):
                recipes = load_recipes()
                previews = {}
                for name, r in recipes.items():
                    previews[r["label"]] = run_preview_cached(
                        recipe_key=name, overrides=r["knobs"])
            html_doc = build_report_html(
                comparison=comp,
                ppl_report=ppl_report(),
                recipe_previews=previews,
                attribution_top=removal_attribution())
            st.session_state["report_html"] = html_doc
        if st.session_state.get("report_html"):
            st.download_button(
                "下载报告 HTML", data=st.session_state["report_html"],
                file_name=f"lotus_quality_report_{__import__('datetime').date.today():%Y%m%d}.html",
                mime="text/html")

with tab2:
    f = funnel()
    if f:
        st.dataframe(f, width="stretch")
    else:
        st.info("尚无层产物")
with tab3:
    a = removal_attribution()
    if a:
        st.dataframe(a, width="stretch")
    else:
        st.info("changelog 尚无记录")
with tab4:
    q = quality_distribution()
    if q is None or len(q) == 0:
        st.info("quality 层尚未产出")
    else:
        dim = st.selectbox("维度", [c for c in q.columns if c != "source"])
        if dim:
            st.bar_chart(q, x=dim, y="quality_score", color="source")
        st.dataframe(q.describe(), width="stretch")
with tab5:
    r = runs_audit()
    if r:
        st.dataframe(r, width="stretch")
    else:
        st.info("runs 尚无记录")
