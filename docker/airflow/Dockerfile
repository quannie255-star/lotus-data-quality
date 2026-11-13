# Lotus 编排镜像：Airflow 官方镜像 + 流水线运行时依赖
# 只装 DAG 任务真正 import 的包（不放 streamlit/pytest，避免调度镜像膨胀）；
# 版本与 requirements.txt 对齐，仓库本身由 compose 挂载到 /opt/airflow/lotus。
FROM apache/airflow:2.10.4

USER airflow
# numpy 必须钉在 1.x：Airflow 2.10 镜像自带的 pandas 是按 numpy 1.x 编译的，
# 直接装 pyarrow 会把 numpy 拉到 2.x，import pandas 报 dtype size changed。
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pyarrow==23.0.1 \
    zhconv==1.4.3 \
    huggingface_hub==0.36.2 \
    pymupdf==1.28.2 \
    requests==2.32.3 \
    datasketch==2.0.0 \
    jsonschema==4.26.0
