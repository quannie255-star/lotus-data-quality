"""dedup/api.py — 去重引擎抽象与注册表（M11 多模态扩展位）

分层：modalities/base.Deduper 是插件层接口；本文件是算法层抽象（DedupEngine）。
文本「精确(normalized hash) + 近重(MinHashLSH)」仍是默认实现（dedup/engine.py，不动），
本文件提供：
- ExactKeyEngine：模态无关的按键精确去重（文本规范化哈希 / 图像字节 sha256 同一实现）；
- 预留接口：PerceptualHashEngine（图像感知哈希）/ AudioFingerprintEngine（音频指纹）——
  调用即 NotImplementedError，显式声明未实装，不静默假装（宪法第 7 条）。

dedup/run.py 的文本阶段编排不变，仍直连 dedup/engine.py；插件层经注册表消费同一实现。
"""

from abc import ABC, abstractmethod

_ENGINES: dict = {}   # name -> DedupEngine 实例


class DedupEngine(ABC):
    """去重引擎抽象：items → 账本口径的报告。

    items 元素是 dict，必含 doc_id 与去重键（键名由引擎定义）；报告结构与
    dedup/engine.dedup_pass 对齐：{"keep": [...], "removed": [...], "clusters": {...}}。
    """

    name: str = "abstract"

    @abstractmethod
    def run(self, items: list) -> dict:
        """幂等要求：全字典序 tiebreak，同输入同输出。"""


class ExactKeyEngine(DedupEngine):
    """按键分组的精确去重（模态无关）。

    items: [{doc_id, source, key, ...payload}]——key 为调用方预算的内容身份
    （文本 normalized_content_hash / 图像字节 sha256 / 图文联合键）。
    簇代表选择与 dedup/engine._pick_canonical 同律：source 信任序 → payload 优先值
    （如内容长度）降序 → doc_id 升序，纯确定性。
    """

    def __init__(self, *, source_trust: list, size_key: str = "size"):
        self.source_trust = source_trust
        self.size_key = size_key
        self.name = "exact_key"

    def run(self, items: list) -> dict:
        trust = {s: i for i, s in enumerate(self.source_trust)}

        def canon_key(item):
            return (trust.get(item.get("source"), len(trust)),
                    -item.get(self.size_key, 0), item["doc_id"])

        by_key = {}
        for it in items:
            by_key.setdefault(it["key"], []).append(it)
        removed, clusters = [], {}
        for key in sorted(by_key):
            group = by_key[key]
            if len(group) < 2:
                continue
            canon = sorted(group, key=canon_key)[0]
            cluster_id = f"exact:{key[:12]}"
            clusters[cluster_id] = sorted(it["doc_id"] for it in group)
            for it in sorted(group, key=lambda i: i["doc_id"]):
                if it["doc_id"] == canon["doc_id"]:
                    continue
                removed.append({"item": it, "reason": "exact_dup",
                                "ref_doc_id": canon["doc_id"], "similarity": 1.0,
                                "cluster_id": cluster_id})
        removed_ids = {r["item"]["doc_id"] for r in removed}
        return {"keep": sorted((it for it in items if it["doc_id"] not in removed_ids),
                               key=lambda i: i["doc_id"]),
                "removed": removed, "clusters": clusters}


class _ReservedEngine(DedupEngine):
    """预留接口基类：显式未实装。调用即抛，错误信息指向路线图——空读数不假装有数。"""

    _what: str = "未指名"

    def run(self, items: list) -> dict:
        raise NotImplementedError(
            f"{self._what}为预留接口（M11 路线图 🚧），未实装；实装前请使用 ExactKeyEngine "
            f"（字节级精确去重）或等待插件引入模态解码依赖")


class PerceptualHashEngine(_ReservedEngine):
    """预留：图像感知哈希（pHash/dHash）近重去重。需图像解码依赖（Pillow 等），
    由 image 插件按需引入，核心层不背。"""

    _what = "图像感知哈希去重 "


class AudioFingerprintEngine(_ReservedEngine):
    """预留：音频指纹（Chromaprint 类）近重去重。需音频解码依赖，由 audio 插件按需引入。"""

    _what = "音频指纹去重 "


def register_engine(engine: DedupEngine, *, replace: bool = False) -> None:
    """登记引擎实例。重名默认拒绝（防注册顺序事故）。"""
    if engine.name in _ENGINES and not replace:
        raise ValueError(f"去重引擎 {engine.name!r} 已注册；确需覆盖请 replace=True")
    _ENGINES[engine.name] = engine


def get_engine(name: str) -> DedupEngine:
    """取引擎；未注册 → KeyError（fail-fast，不静默回退到别的引擎）。"""
    return _ENGINES[name]


def engines() -> tuple:
    """当前已注册引擎名（确定性：注册序）。"""
    return tuple(_ENGINES)
