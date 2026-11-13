"""多模态插件层测试（M11）——接口冒烟 + 图文链路集成 + 分层边界元测试

纪律：预留位（audio/video/感知哈希/CLIP）的测试断言它们**显式不可用**——
未实装不假装，这是被测行为而不是待办。
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 样本基类与注册表

def test_sample_rejects_unknown_modality():
    from modalities.base import Sample

    with pytest.raises(ValueError, match="未知模态"):
        Sample(doc_id="d1", source="wiki", source_id="1", modality="scent")


def test_registry_register_get_and_duplicates():
    import modalities
    from modalities.base import Cleaner

    class ProbeCleaner(Cleaner):
        def clean(self, sample):
            return ("keep",)

    modalities.register("probe", cleaner=ProbeCleaner())
    assert isinstance(modalities.get("probe", "cleaner"), ProbeCleaner)
    with pytest.raises(ValueError, match="已注册"):
        modalities.register("probe", cleaner=ProbeCleaner())
    with pytest.raises(ValueError, match="须是"):
        modalities.register("probe2", cleaner=object(), replace=True)


def test_load_all_registers_text_and_image():
    import modalities

    before = set(modalities.registered())
    modalities.load_all()
    after = set(modalities.registered())
    assert {"text", "image"} <= after
    assert "probe" in after and "probe" in before  # 注册表只增不清


def test_audio_video_reserved_fail_fast():
    import modalities

    with pytest.raises(modalities.PluginNotRegisteredError, match="audio"):
        modalities.get("audio")
    with pytest.raises(modalities.PluginNotRegisteredError, match="video"):
        modalities.get("video")


# ---------------------------------------------------------------- 去重引擎抽象

def test_exact_key_engine_canonical_by_source_trust():
    from dedup.api import ExactKeyEngine

    eng = ExactKeyEngine(source_trust=["wiki", "news", "oscar"])
    items = [
        {"doc_id": "oscar:1", "source": "oscar", "key": "k1", "size": 100},
        {"doc_id": "wiki:1", "source": "wiki", "key": "k1", "size": 10},
        {"doc_id": "wiki:2", "source": "news", "key": "k2", "size": 5},
    ]
    r1, r2 = eng.run(items), eng.run(list(reversed(items)))  # 输入序无关
    assert r1 == r2
    assert [i["doc_id"] for i in r1["keep"]] == ["wiki:1", "wiki:2"]
    removed = r1["removed"][0]
    assert removed["ref_doc_id"] == "wiki:1" and removed["similarity"] == 1.0


def test_reserved_engines_raise_not_implemented():
    from dedup.api import AudioFingerprintEngine, PerceptualHashEngine, get_engine

    with pytest.raises(NotImplementedError, match="预留接口"):
        PerceptualHashEngine().run([])
    with pytest.raises(NotImplementedError, match="预留接口"):
        AudioFingerprintEngine().run([])
    with pytest.raises(KeyError):
        get_engine("no_such_engine")


# ---------------------------------------------------------------- 文本插件绑定

GOOD_TEXT = ("这是一段足够长的中文正文内容，用来通过清洗规则的各项阈值检查。" * 2)


def _text_sample(content, source="wiki", n=1):
    from common import ids
    from modalities.text import TextSample

    return TextSample(doc_id=ids.spine_doc_id(source, f"t{n}"),
                      source=source, source_id=f"t{n}", modality="text",
                      content=content)


def test_text_cleaner_binds_clean_base():
    from modalities import load_all

    load_all()
    from modalities import get

    keep = get("text", "cleaner").clean(_text_sample(GOOD_TEXT))
    assert keep == ("keep",)
    drop = get("text", "cleaner").clean(_text_sample("短文"))
    assert drop[0] == "drop" and drop[1]  # reason 码非空（登记在 reason_registry）


def test_text_scorer_binds_quality_engine():
    from modalities import get, load_all
    from quality.engine import score_row

    load_all()
    s = _text_sample(GOOD_TEXT)
    via_plugin = get("text", "scorer").score(s)
    via_engine = score_row(s.to_row())
    assert via_plugin == via_engine  # 插件不另立标准


def test_text_deduper_binds_dedup_engine():
    from modalities import get, load_all

    load_all()
    a = _text_sample(GOOD_TEXT, n=1)
    b = _text_sample(GOOD_TEXT, source="oscar", n=2)  # 同内容不同源 → exact_dup
    report = get("text", "deduper").dedup([a, b])
    assert len(report["keep"]) == 1 and report["keep"][0].doc_id == a.doc_id
    assert report["removed"][0]["reason"] == "exact_dup"
    assert report["removed"][0]["ref_doc_id"] == a.doc_id  # wiki 信任序在前


# ---------------------------------------------------------------- 联合门禁

def test_joint_score_validates_weights():
    from quality.multimodal import joint_score

    per = {"text": {"quality_score": 0.9}, "image": {"quality_score": 0.5}}
    assert joint_score(per) == pytest.approx(0.9 * 0.6 + 0.5 * 0.4, abs=1e-4)
    with pytest.raises(ValueError, match="权重和"):
        joint_score(per, weights={"text": 0.6, "image": 0.6})
    with pytest.raises(ValueError, match="不一致"):
        joint_score(per, weights={"text": 1.0})
    assert joint_score({"text": {"quality_score": None}, "image": {"quality_score": 0.5}},
                       weights={"text": 0.6, "image": 0.4}) is None


def test_joint_gate_short_board_law():
    from quality.multimodal import joint_gate

    good = {"text": {"quality_score": 0.9}, "image": {"quality_score": 0.8}}
    assert joint_gate(good, gate_policy={
        "modality_weights": {"text": 0.6, "image": 0.4},
        "min_modality_score": 0.3, "min_joint_score": 0.55}) == ("pass", None)
    short = {"text": {"quality_score": 0.9}, "image": {"quality_score": 0.1}}
    verdict, reason = joint_gate(short, gate_policy={
        "modality_weights": {"text": 0.6, "image": 0.4},
        "min_modality_score": 0.3, "min_joint_score": 0.55})
    assert verdict == "remove" and reason == "low_quality_modality:image"


# ---------------------------------------------------------------- 图文配对 v0

def _img_bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pair(source, n, caption, img=b"png-bytes-0001", img_len=4096):
    from modalities.image import ImageTextSample

    return ImageTextSample(
        doc_id=hashlib.sha256(f"{source}:{n}".encode()).hexdigest()[:16],
        source=source, source_id=f"p{n}", modality="image_text",
        modality_pair_id=f"pair-{n}", caption=caption,
        image_uri=None if img is None else f"s3a://raw/img/{n}.png",
        image_bytes_hash=None if img is None else _img_bytes_hash(img),
        image_bytes_len=0 if img is None else img_len,
        image_format=None if img is None else "png")


CAP1 = "这是一张展示古城墙全景的照片，图中可见完整的城垛结构与远处的山脉轮廓。拍摄视角为东南方向，光影层次分明。" 


@pytest.fixture
def clip_off(monkeypatch):
    """v0 行为测试固定在 clip.enabled=false 的浅拷贝策略上。

    生产配置已切 true（2026-11-13 中文在域标定后），但「关闭时对齐分=None、
    双模态权重 pass」这条 v0 契约仍然必须成立——用 fixture 显式声明被测策略，
    不依赖生产开关的当前值。
    """
    import copy
    from modalities.image import paired

    pol = copy.deepcopy(paired.mm_policy())
    pol["image_text"]["clip"]["enabled"] = False
    monkeypatch.setattr(paired, "mm_policy", lambda: pol)
    return pol
CAP2 = "图表展示了近十年城市人口的增减变化，横轴为年份，纵轴为常住人口数，趋势整体向上且在末端加速。"


def test_paired_cleaner_hold_orphan_drop_defects():
    from modalities import get, load_all

    load_all()
    cleaner = get("image", "cleaner")
    assert cleaner.clean(_pair("wiki", 1, CAP1, img=None)) == ("hold", "orphan_pair")
    assert cleaner.clean(_pair("wiki", 2, "")) == ("hold", "orphan_pair")
    assert cleaner.clean(_pair("wiki", 3, CAP1)) == ("keep",)
    assert cleaner.clean(_pair("wiki", 4, "太短"))[0] == "drop"     # caption_too_short
    assert cleaner.clean(_pair("wiki", 5, CAP1, img_len=10)) == ("drop", "image_bytes_too_small")


def test_paired_scorer_alignment_null_and_joint_pass(clip_off):
    from modalities import get, load_all

    load_all()
    scores = get("image", "scorer").score(_pair("wiki", 1, CAP1))
    assert scores["pair_cross_modal_alignment"] is None      # v0 未接入 CLIP：空读数
    assert scores["pair_image_structural"] == 1.0
    assert scores["joint_verdict"] == "pass"
    assert scores["quality_score"] == pytest.approx(
        scores["pair_caption_quality"] * 0.6 + 1.0 * 0.4, abs=1e-3)
    empty = get("image", "scorer").score(_pair("wiki", 2, ""))
    assert empty["pair_caption_quality"] == 0.0
    assert empty["joint_verdict"] == "remove"                # 空文本侧直接 0 分


def test_paired_dedup_pair_and_image_dup():
    from modalities import get, load_all

    load_all()
    s1 = _pair("wiki", 1, CAP1, img=b"img-A")            # 簇代表（wiki 信任序最前）
    s2 = _pair("oscar", 2, CAP1, img=b"img-A")           # 图+文全同 → pair_exact_dup
    s3 = _pair("news", 3, CAP2, img=b"img-A")            # 同图不同文 → image_exact_dup
    s4 = _pair("wiki", 4, CAP2, img=b"img-B")            # 独立对
    report = get("image", "deduper").dedup([s1, s2, s3, s4])
    assert {s.doc_id for s in report["keep"]} == {s1.doc_id, s4.doc_id}
    reasons = {r["sample"].doc_id: r["reason"] for r in report["removed"]}
    assert reasons == {s2.doc_id: "pair_exact_dup", s3.doc_id: "image_exact_dup"}
    assert all(r["ref_doc_id"] == s1.doc_id for r in report["removed"])


def test_paired_dedup_orphans_never_collapse():
    """两个都缺图像哈希的孤儿：未检测 ≠ 空 ≠ 有值——NULL 不与 NULL 相等。"""
    from modalities import get, load_all
    from modalities.image import ImageTextSample

    load_all()
    o1 = ImageTextSample(doc_id="o1", source="wiki", source_id="1",
                         modality="image_text", caption=CAP1)
    o2 = ImageTextSample(doc_id="o2", source="oscar", source_id="2",
                         modality="image_text", caption=CAP1)
    report = get("image", "deduper").dedup([o1, o2])
    assert len(report["keep"]) == 2 and not report["removed"]


# ---------------------------------------------------------------- schema 与指标

def _def_schema(def_name: str) -> dict:
    """取 multimodal_schema 的某个子定义为可独立校验的 schema（$ref 需随 $defs 一起带走）。"""
    schema = json.loads((ROOT / "schema" / "multimodal_schema.json").read_text(encoding="utf-8"))
    return {"$schema": schema["$schema"], "$defs": schema["$defs"],
            "$ref": f"#/$defs/{def_name}"}


def test_multimodal_schema_validates_pair_and_rejects_bad_modality():
    import jsonschema

    good = _pair("wiki", 1, CAP1).to_dict()
    jsonschema.validate(good, _def_schema("image_text_sample"))
    bad = {**good, "modality": "scent"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _def_schema("sample_base"))


def test_schema_enum_matches_code_constant():
    """schema 是模态枚举的事实单一来源；代码常量必须与之对齐（防漂移元测试）。"""
    from modalities.base import MODALITY_VALUES

    schema = json.loads((ROOT / "schema" / "multimodal_schema.json").read_text(encoding="utf-8"))
    enum = schema["$defs"]["sample_base"]["properties"]["modality"]["enum"]
    assert tuple(enum) == MODALITY_VALUES


def test_pair_metrics_and_card(tmp_path):
    from metrics.multimodal import compute_multimodal_card, pair_metrics

    rows = [
        {"caption": CAP1, "image_bytes_hash": "h1", "image_bytes_len": 4096, "image_format": "png"},
        {"caption": CAP1, "image_bytes_hash": "h1", "image_bytes_len": 4096, "image_format": "png"},
        {"caption": CAP2, "image_bytes_hash": None, "image_bytes_len": 0, "image_format": None},
    ]
    m = pair_metrics(rows)
    assert m["n_pairs"] == 3
    assert m["orphan_rate"] == round(1 / 3, 5)    # 指标卡口径统一 round 5
    assert m["image_exact_dup_rate"] == 0.5       # 分母=有哈希的 2 对
    assert m["caption_dup_rate"] == round(1 / 3, 5)

    tbl = pa.table({"modality": ["text", "image_text"],
                    "caption": ["x", CAP1],
                    "image_bytes_hash": [None, "h1"],
                    "image_bytes_len": [0, 4096],
                    "image_format": [None, "png"]})
    part = tmp_path / "part-00000.parquet"
    pq.write_table(tbl, part)
    card = compute_multimodal_card([part], "v0-mm")
    assert card["modality_breakdown"] == {"image_text": 1, "text": 1}
    assert card["paired"]["n_pairs"] == 1

    text_only = tmp_path / "part-00001.parquet"
    pq.write_table(pa.table({"content": ["x"]}), text_only)
    assert compute_multimodal_card([text_only], "legacy")["paired"] is None  # 不假装


# ---------------------------------------------------------------- 图文链路集成

def test_image_text_pipeline_end_to_end_deterministic(clip_off):
    """集成：清洗(孤儿hold) → 去重(精确折叠) → 联合打分+门禁；两次运行字节级一致。"""
    from modalities import get, load_all

    load_all()
    samples = [
        _pair("wiki", 1, CAP1, img=b"img-A"),
        _pair("oscar", 2, CAP1, img=b"img-A"),      # pair_exact_dup（图+文全同）
        _pair("news", 3, CAP2, img=b"img-A"),       # image_exact_dup（同图不同文）
        _pair("wiki", 4, CAP2, img=b"img-B"),       # 独立对
        _pair("oscar", 5, CAP1),                    # 独立对
        _pair("news", 6, CAP1, img=None),           # 孤儿 → hold
    ]

    def run_pipeline():
        cleaner, deduper, scorer = (get("image", r) for r in ("cleaner", "deduper", "scorer"))
        verdicts = [(s, cleaner.clean(s)) for s in samples]
        holds = [(s.doc_id, v) for s, v in verdicts if v[0] == "hold"]
        survivors = [s for s, v in verdicts if v[0] == "keep"]
        dd = deduper.dedup(survivors)
        scored = {s.doc_id: scorer.score(s) for s in dd["keep"]}
        return {
            "holds": holds,
            "removed": [{"doc_id": r["sample"].doc_id, "reason": r["reason"],
                         "ref": r["ref_doc_id"], "sim": r["similarity"]}
                        for r in dd["removed"]],
            "passed": sorted(doc_id for doc_id, sc in scored.items()
                             if sc["joint_verdict"] == "pass"),
        }

    r1, r2 = run_pipeline(), run_pipeline()
    assert json.dumps(r1, sort_keys=True, ensure_ascii=False) == \
           json.dumps(r2, sort_keys=True, ensure_ascii=False)
    assert r1["holds"] == [(samples[5].doc_id, ("hold", "orphan_pair"))]
    assert {x["reason"] for x in r1["removed"]} == {"pair_exact_dup", "image_exact_dup"}
    assert len(r1["passed"]) == 3      # s1、s4、s5 —— 全部过联合门禁


# ---------------------------------------------------------------- 分层边界元测试

def test_core_layer_never_imports_modality_plugins():
    """宪法式边界守卫：核心层（modalities/ 与 tests/ 之外）不得 import 模态插件。

    分层契约的可证伪点——核心层一旦反向依赖插件，「新增模态零改核心」即破产。
    只扫管道源码目录（白名单），venv/缓存/杂物目录不进扫描面。
    """
    core_dirs = {"common", "clean", "dedup", "quality", "metrics", "mix", "release",
                 "ingest", "parse", "downstream", "stream", "dags", "dashboard", "smoke"}
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if not rel.parts or rel.parts[0] not in core_dirs:
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(src.splitlines(), 1):
            code = line.split("#")[0]
            if "import modalities" in code or "from modalities" in code:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"核心层反向依赖模态插件: {offenders}"
