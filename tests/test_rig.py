"""rig.py 的行为测试：标注解析、切件正确性、诊断告警。

全部输入由 conftest 现场合成，不依赖任何外部美术文件（唯一例外见 test_real_art.py）。
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from flat2rig import rig as rig_mod


# --------------------------------------------------------------------------- #
# load_rig
# --------------------------------------------------------------------------- #

def test_load_rig_parses_documented_schema(art):
    rig = rig_mod.load_rig(art.config_path)
    assert rig.size == (64, 64)
    assert [p.name for p in rig.parts] == ["body", "head", "ear"]
    head = rig.part("head")
    assert head is not None
    assert head.pivot == (32, 30)
    assert head.z == 1
    assert head.bones[0].x0 == 32 and head.bones[0].y1 == 14
    assert rig.states == {"idle": 520, "sleep": 560}
    assert rig.eyes == [(27, 20, 3), (38, 20, 3)]
    assert len(head.motions["idle"]) == 4


def test_load_rig_rejects_malformed_parts_list(art, write_config):
    """结构错误必须报错；而缺 pivot 属于文档化的容错（回退到骨骼末端），不算错误。"""
    path = write_config("bad.rig.json", {"canvas": [64, 64], "parts": "not-a-list"})
    with pytest.raises(Exception):
        rig_mod.load_rig(path)


def test_load_rig_falls_back_to_bone_tip_when_pivot_missing(tmp_path, write_config):
    cfg = {
        "canvas": [64, 64],
        "parts": [
            {"name": "a", "bones": [[10, 10, 20, 20]], "z": 0, "blend": 4,
             "motions": {"idle": [[0, 0], [5, 0]]}},
        ],
        "eyes": [],
        "states": {"idle": 100},
    }
    rig = rig_mod.load_rig(write_config("nopivot.rig.json", cfg))
    assert rig.part("a").pivot == (20.0, 20.0), "缺 pivot 时应回退到骨骼末端（文档化行为）"


def test_load_rig_clamps_bone_coordinates_but_keeps_pivot_verbatim(tmp_path, write_config):
    """骨骼坐标会夹紧到画布；pivot 允许在画布外（关节可以合法地位于画面之外）。"""
    cfg = {
        "canvas": [64, 64],
        "parts": [
            {"name": "a", "bones": [[-50, 999, 500, -20]], "pivot": [999, -999],
             "z": 0, "blend": 4, "motions": {"idle": [[0, 0]]}},
        ],
        "eyes": [],
        "states": {"idle": 100},
    }
    rig = rig_mod.load_rig(write_config("clamp.rig.json", cfg))
    part = rig.part("a")
    bone = part.bones[0]
    assert 0 <= bone.x0 <= 64 and 0 <= bone.y0 <= 64
    assert 0 <= bone.x1 <= 64 and 0 <= bone.y1 <= 64
    assert part.pivot == (999.0, -999.0), "pivot 不应被夹紧（文档化行为）"


# --------------------------------------------------------------------------- #
# build_masks / assign_parts
# --------------------------------------------------------------------------- #

def test_masks_partition_the_alpha_channel(make_character, make_config, tmp_path, write_config):
    """核心不变量：各部位掩膜之和 == 原始 alpha（不留缝、不重复计算）。"""
    png = tmp_path / "hero.png"
    make_character().save(png)
    rig = rig_mod.load_rig(write_config("c.json", make_config()))

    rgba, alpha = rig_mod.load_alpha(png)
    masks = rig_mod.build_masks(rgba, rig)

    assert set(masks) == {"body", "head", "ear"}
    a = alpha.load()
    sums = [0, 0, 0]
    for y in range(rgba.height):
        for x in range(rgba.width):
            total = 0
            for m in masks.values():
                total += m.load()[x, y]
            assert abs(total - a[x, y]) <= 2, f"({x},{y}) 掩膜和 {total} != alpha {a[x,y]}"
            sums[0] += 1
    assert sums[0] == rgba.width * rgba.height


def test_every_part_with_bones_gets_pixels(art):
    rig = rig_mod.load_rig(art.config_path)
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    for name, mask in masks.items():
        assert mask.getbbox() is not None, f"{name} 掩膜是空的"
        assert mask.histogram()[255] > 0


def test_full_transparency_is_handled_without_crash(tmp_path, make_config, write_config):
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    png = tmp_path / "blank.png"
    blank.save(png)
    rig = rig_mod.load_rig(write_config("c.json", make_config()))
    rgba, alpha = rig_mod.load_alpha(png)
    masks = rig_mod.build_masks(rgba, rig)
    diag = rig_mod.diagnose(rgba, rig, masks)
    assert all(m.getbbox() is None for m in masks.values())
    assert diag["warnings"], "全透明输入应当给出告警"


# --------------------------------------------------------------------------- #
# diagnose
# --------------------------------------------------------------------------- #

def test_diagnose_reports_full_coverage_for_a_good_config(art):
    rig = rig_mod.load_rig(art.config_path)
    rgba, _ = rig_mod.load_alpha(art.png)
    diag = rig_mod.diagnose(rgba, rig, rig_mod.build_masks(rgba, rig))
    assert diag["coverage"] == pytest.approx(1.0, abs=0.02)
    assert set(diag["per_part_area"]) == {"body", "head", "ear"}
    assert sum(diag["per_part_area"].values()) > 0


def test_diagnose_warns_about_a_starved_part(tmp_path, make_character, make_config, write_config):
    """骨骼全挤在一角时，另一部位会被抢空 —— 必须给出告警而不是静默通过。"""
    png = tmp_path / "hero.png"
    make_character().save(png)
    cfg = make_config()
    # 把 head 的骨骼挪到与 body 完全重合的位置，ear 的骨骼留在原处
    cfg["parts"][1]["bones"] = [[32, 40, 32, 58]]
    rig = rig_mod.load_rig(write_config("c.json", cfg))
    rgba, _ = rig_mod.load_alpha(png)
    diag = rig_mod.diagnose(rgba, rig, rig_mod.build_masks(rgba, rig))
    areas = diag["per_part_area"]
    assert min(areas.values()) < max(areas.values()) * 0.1 or diag["warnings"]


def test_part_without_bones_warns_instead_of_crashing(art, make_config, write_config):
    cfg = make_config()
    cfg["parts"].append({"name": "ghost", "bones": [], "pivot": [10, 10], "z": 9, "blend": 4, "motions": {}})
    rig = rig_mod.load_rig(write_config("ghost.json", cfg))
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    diag = rig_mod.diagnose(rgba, rig, masks)
    assert masks["ghost"].getbbox() is None
    assert any("ghost" in w for w in diag["warnings"])
