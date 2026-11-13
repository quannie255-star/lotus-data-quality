"""dashboard/pipeline_runner.py — 后台跑全链（控制台「保存并重算」用）

把 clean→dedup→quality→mix→release→downstream→metrics 全链包成一个可后台执行的
函数，进度写到 data/meta/pipeline_progress.json，供 Streamlit 前台轮询。

设计：
- 用 subprocess 启动一个独立的 python 进程跑全链（不阻塞 Streamlit 主线程），
  进度文件是唯一的通信信道（进程间解耦，崩了也能从进度文件看到断在哪）。
- 每个阶段跑完写一条进度；全链结束写 status=done/failed。
- 环境处理：继承调用方环境（Streamlit 进程已带好 TMPDIR/代理等），这里不再重复设。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROGRESS_FILE = DATA / "meta" / "pipeline_progress.json"

STAGES = ["clean", "dedup", "quality", "mix", "release", "downstream", "metrics"]


def _python() -> str:
    """当前 venv 的 python（Streamlit 用的同一个，保证依赖一致）。"""
    return sys.executable


def progress_path() -> Path:
    return PROGRESS_FILE


def read_progress() -> dict:
    """读当前进度（无文件 = 未开始）。"""
    if not PROGRESS_FILE.exists():
        return {"status": "idle", "stages": {}}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "idle", "stages": {}}


def is_running() -> bool:
    """是否有一个全链任务正在跑。"""
    p = read_progress()
    return p.get("status") in ("running",)


def _run_script() -> str:
    """生成后台跑的脚本内容（每个阶段调用 run()，写进度）。"""
    return r'''
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(r"{root}")
DATA = ROOT / "data"
PROG = DATA / "meta" / "pipeline_progress.json"

STAGES = ["clean", "dedup", "quality", "mix", "release", "downstream", "metrics"]

def wprog(status, stages):
    PROG.parent.mkdir(parents=True, exist_ok=True)
    PROG.write_text(json.dumps({{"status": status, "stages": stages}},
                               ensure_ascii=False, indent=2), encoding="utf-8")

sys.path.insert(0, str(ROOT))

stages = {{}}
wprog("running", {{}})

for i, stage in enumerate(STAGES):
    stages[stage] = "running"
    wprog("running", stages)
    try:
        import importlib
        mod = importlib.import_module(stage + ".run")
        result = mod.run()
        # 结果里挑关键字段记下来
        info = {{}}
        for k in ("rows", "kept", "version", "report"):
            if k in result:
                info[k] = result[k]
        stages[stage] = {{"status": "done", **info}}
        wprog("running", stages)
    except Exception:
        stages[stage] = {{"status": "failed", "error": traceback.format_exc()[-500:]}}
        wprog("failed", stages)
        sys.exit(1)

wprog("done", stages)
'''.replace("{root}", str(ROOT).replace("\\", "/"))


def start_pipeline() -> dict:
    """启动后台全链。返回启动结果。已在跑则拒绝重复启动。"""
    if is_running():
        return {"started": False, "reason": "已有全链任务在跑"}

    # 清空旧进度
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({"status": "running", "stages": {}}),
                             encoding="utf-8")

    script = _run_script()
    # 写临时脚本到 data/meta/ 下，避免中文路径传给 subprocess 出问题
    script_path = DATA / "meta" / "_run_pipeline.py"
    script_path.write_text(script, encoding="utf-8")

    # 后台启动，完全脱离（start_new_session，Windows 用 CREATE_NEW_PROCESS_GROUP）
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    return {"started": True}
