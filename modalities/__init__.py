"""modalities/ — 模态插件注册表（M11）

核心层不得 import 本包的任何子模块（只允许依赖 base.py 的接口类型）；
插件注册由各插件包 import 时自完成。注册表本身惰性：import modalities 不触发任何
插件加载——load_all() 才绑全量（默认按需 get，未注册模态 fail-fast）。
"""

from modalities.base import (  # noqa: F401  (接口再导出：插件/测试统一从这里拿)
    MODALITY_VALUES, Sample, Collector, Cleaner, Deduper, QualityScorer,
)


class PluginNotRegisteredError(KeyError):
    """模态未注册/未实装——显式报错，不静默降级（宪法第 7 条）。"""


_PLUGINS: dict = {}   # modality(插件键) -> {"collector":..., "cleaner":..., "deduper":..., "scorer":...}
_ROLES = {"collector": Collector, "cleaner": Cleaner,
          "deduper": Deduper, "scorer": QualityScorer}


def register(modality: str, *, collector=None, cleaner=None, deduper=None, scorer=None,
             replace: bool = False) -> None:
    """注册模态插件。重复注册默认拒绝（防 import 顺序事故）；replace=True 显式覆盖。

    参数: modality — 插件键（text/image/audio/video）；四个组件均可选（预留位允许部分实装）
    异常: ValueError — 重复注册且未 replace；组件不是对应基类的实例
    """
    if modality in _PLUGINS and not replace:
        raise ValueError(f"模态 {modality!r} 已注册；确需覆盖请 replace=True")
    components = {"collector": collector, "cleaner": cleaner,
                  "deduper": deduper, "scorer": scorer}
    for role, base_cls in _ROLES.items():
        comp = components[role]
        if comp is not None and not isinstance(comp, base_cls):
            raise ValueError(f"{modality}.{role} 须是 {base_cls.__name__} 实例，"
                             f"得到 {type(comp).__name__}")
    _PLUGINS[modality] = components


def get(modality: str, role: str | None = None):
    """取插件或其单组件。未注册 → PluginNotRegisteredError（预留位 audio/video 走此路径）。

    参数: role — "collector"/"cleaner"/"deduper"/"scorer"；None → 整个插件 dict
    """
    if modality not in _PLUGINS:
        raise PluginNotRegisteredError(
            f"模态 {modality!r} 未注册（预留位未实装也算未注册，不静默）")
    plugin = _PLUGINS[modality]
    if role is None:
        return plugin
    return plugin[role]


def registered() -> tuple:
    """当前已注册模态（确定性：按注册序）。"""
    return tuple(_PLUGINS)


def load_all() -> tuple:
    """import 全部内置插件包触发注册（text/image 已实装；audio/video 为预留位，无包可载）。

    返回: registered() 快照
    """
    from modalities import text, image  # noqa: F401  (import 即注册)

    return registered()
