"""parts / render / layers / preview / cli 的行为测试。

只测可观察的行为与不变量，不测实现细节；全部输入现场合成。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from flat2rig import parts as parts_mod
from flat2rig import render as render_mod


# --------------------------------------------------------------------------- #
# parts.cut_layers
# --------------------------------------------------------------------------- #

def test_cut_layers_alpha_is_mask_times_source_alpha(art):
    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    rgba, alpha = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    layers = parts_mod.cut_layers(rgba, masks)

    assert set(layers) == set(masks)
    for name, layer in layers.items():
        assert layer.size == rgba.size
        la = layer.getchannel("A").load()
        ma = masks[name].load()
        sa = alpha.load()
        for y in range(0, rgba.height, 3):
            for x in range(0, rgba.width, 3):
                expect = round(ma[x, y] * sa[x, y] / 255)
                assert abs(la[x, y] - expect) <= 2, f"{name} ({x},{y}) {la[x,y]} != {expect}"


def test_cut_layers_has_no_cross_part_contamination(art):
    """部位图层不得互相污染。

    羽化带（``blend``）会在交界处留下 alpha 渐变，这是 :func:`flat2rig.rig.soften_masks`
    的有意设计，因此这里检验的是**真正的不变量**：
      * 离交界较远（> 一个混合带）的地方，另一个部位的图层必须完全透明；
      * 羽化带内的过渡 alpha 必须低于"几乎不透明"的量级（< 200），否则说明发生了泄漏；
      * 任何像素的图层 alpha 都不得超过源图 alpha。
    """
    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    rgba, alpha = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    layers = parts_mod.cut_layers(rgba, masks)

    w, h = rgba.size
    sa = alpha.load()
    worst_feather = 0
    strict = 0
    for name, mask in masks.items():
        other = [n for n in masks if n != name]
        la = layers[name].getchannel("A").load()
        mm = mask.load()
        for y in range(h):
            for x in range(w):
                assert la[x, y] <= sa[x, y], f"{name} 图层 alpha 超过源图 ({x},{y})"
                if mm[x, y] != 0:
                    continue
                near = any(masks[o].load()[min(w - 1, x + dx), min(h - 1, y + dy)] > 0
                           for o in other
                           for dx in (-8, -4, 0, 4, 8) for dy in (-8, -4, 0, 4, 8))
                if near:
                    worst_feather = max(worst_feather, la[x, y])
                else:
                    assert la[x, y] == 0, f"部位外不透明: {name} ({x},{y})={la[x,y]}"
                    strict += 1
    assert strict > 100, "样本太少，测试没有真正覆盖到"
    assert worst_feather < 200, f"羽化带 alpha 高达 {worst_feather}，疑似部位间泄漏"


# --------------------------------------------------------------------------- #
# parts.inpaint_hidden
# --------------------------------------------------------------------------- #

def test_inpaint_fills_the_region_covered_by_a_front_part(tmp_path, make_character,
                                                          make_config, write_config):
    """被前景盖住的下层部位，补全后应当在该区域获得不透明像素。

    这是核心机制之一：前景一旦移动，下层若没补上这块像素就会露出空洞。
    """
    from flat2rig import rig as rig_mod

    png = tmp_path / "hero.png"
    make_character().save(png)
    cfg = make_config()
    # 让前景(ear)在竖直方向压住下层(head)，被遮挡区域才真实存在
    for part in cfg["parts"]:
        if part["name"] == "head":
            part["bones"] = [[32, 34, 32, 28]]
            part["pivot"] = [32, 34]
        if part["name"] == "ear":
            part["bones"] = [[40, 12, 34, 20]]
            part["pivot"] = [40, 14]
    rig = rig_mod.load_rig(write_config("overlap.json", cfg))
    rgba, _ = rig_mod.load_alpha(png)
    masks = rig_mod.build_masks(rgba, rig)
    order = parts_mod.order_parts(rig)
    layers = parts_mod.cut_layers(rgba, masks)
    filled = parts_mod.inpaint_hidden(rgba, masks, order, radius=4)

    back, front = order[0], order[-1]
    assert back != front
    before = layers[back].getchannel("A").load()
    after = filled[back].getchannel("A").load()
    fm = masks[front].load()
    gained = sum(1 for y in range(rgba.height) for x in range(rgba.width)
                 if fm[x, y] > 0 and before[x, y] == 0 and after[x, y] > 0)
    assert gained > 0, f"补全没有为 {back} 增加被 {front} 遮挡的像素"
    assert filled[back].size == rgba.size


def test_inpaint_is_bounded_in_time(art):
    import time

    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    t0 = time.perf_counter()
    parts_mod.inpaint_hidden(rgba, masks, parts_mod.order_parts(rig), radius=6)
    assert time.perf_counter() - t0 < 3.0


def test_order_parts_is_back_to_front(art):
    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    order = parts_mod.order_parts(rig)
    zs = [rig.part(n).z for n in order]
    assert zs == sorted(zs), "order_parts 必须按 z 从后到前排序"
    assert sorted(order) == sorted(p.name for p in rig.parts)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def _solid(size=(64, 64)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 47, 47], fill=(200, 120, 90, 255))
    return img


def test_rotate_about_zero_is_identity():
    img = _solid()
    out = render_mod.rotate_about(img, (32, 32), 0.0)
    assert out.size == img.size
    assert list(out.getdata()) == list(img.getdata())


def test_rotate_about_keeps_canvas_and_content():
    img = _solid()
    out = render_mod.rotate_about(img, (32, 32), 90.0)
    assert out.size == img.size
    assert out.getchannel("A").getbbox() is not None
    # 绕中心转 90° 后，不透明像素数应基本不变
    a0 = img.getchannel("A").histogram()[255]
    a1 = out.getchannel("A").histogram()[255]
    assert abs(a1 - a0) < a0 * 0.35


def test_rotate_about_shifts_by_dy():
    img = _solid()
    out = render_mod.rotate_about(img, (32, 32), 0.0, dy=-8)
    b0 = img.getchannel("A").getbbox()
    b1 = out.getchannel("A").getbbox()
    assert b1[1] < b0[1], "dy 为负应当把内容上移"


def test_compose_frame_paints_front_parts_over_back_ones():
    back = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(back).rectangle([0, 0, 63, 63], fill=(255, 0, 0, 255))
    front = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(front).rectangle([16, 16, 47, 47], fill=(0, 0, 255, 255))
    out = render_mod.compose_frame(
        {"back": back, "front": front}, ["back", "front"],
        {"back": (0, 0, 0), "front": (0, 0, 0)}, {"back": (0, 0), "front": (32, 32)})
    assert out.getpixel((32, 32))[:3] == (0, 0, 255)
    assert out.getpixel((2, 2))[:3] == (255, 0, 0)


def test_overlay_symbol_returns_transparent_canvas_with_content():
    for kind in ("zzz", "sweat", "sparkle"):
        layer = render_mod.overlay_symbol(kind, 1, size=64)
        assert layer.size == (64, 64)
        assert layer.getbbox() is not None, f"{kind} 叠加层是空的"
        assert layer.getchannel("A").getbbox()[2] <= 64
        assert layer.getchannel("A").getbbox()[3] <= 64


def test_overlay_eyes_draws_inside_canvas():
    layer = render_mod.overlay_eyes([(20, 20, 4), (40, 20, 4)])
    assert layer.size == (240, 240)
    assert layer.getbbox() is not None


# --------------------------------------------------------------------------- #
# render.build_frames
# --------------------------------------------------------------------------- #

def test_build_frames_expands_motions(art, make_config, write_config):
    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    layers = parts_mod.cut_layers(rgba, masks)
    frames = render_mod.build_frames(rgba, rig, masks, layers=layers, states=["idle", "sleep"])

    assert set(frames) == {"idle", "sleep"}
    longest = {state: max(len(p.motions[state]) for p in rig.parts if state in p.motions)
               for state in ("idle", "sleep")}
    for state, seq in frames.items():
        assert len(seq) == longest[state], f"{state} 帧数应等于各部位最长动作序列"
        for img, delay in seq:
            assert img.size == rgba.size
            assert delay == rig.states[state]


def test_build_frames_actually_moves(art):
    from flat2rig import rig as rig_mod

    rig = rig_mod.load_rig(art.config_path)
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    layers = parts_mod.cut_layers(rgba, masks)
    seq = render_mod.build_frames(rgba, rig, masks, layers=layers, states=["idle"])["idle"]
    first = list(seq[0][0].getdata())
    assert any(list(img.getdata()) != first for img, _ in seq[1:]), "所有帧完全相同，动画没有生效"


def test_short_motion_list_is_repeated_not_truncated(art, make_config, write_config):
    from flat2rig import rig as rig_mod

    cfg = make_config()
    cfg["parts"][1]["motions"]["idle"] = [[0, 0]]          # 只有 1 帧
    cfg["states"]["idle"] = 300
    rig = rig_mod.load_rig(write_config("short.json", cfg))
    rgba, _ = rig_mod.load_alpha(art.png)
    masks = rig_mod.build_masks(rgba, rig)
    frames = render_mod.build_frames(rgba, rig, masks, layers=None, states=["idle"])
    assert len(frames["idle"]) == 4, "帧数应取各部位中最长的动作序列"


# --------------------------------------------------------------------------- #
# layers（--layers 模式）
# --------------------------------------------------------------------------- #

def test_load_stack_from_png_directory(tmp_path):
    from flat2rig import layers as layers_mod

    d = tmp_path / "stack"
    d.mkdir()
    for name, box in (("body.png", (10, 30, 50, 60)), ("ear_l.png", (5, 5, 25, 25)),
                      ("layer10.png", (0, 0, 4, 4)), ("layer2.png", (60, 60, 64, 64))):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).rectangle(box, fill=(200, 100, 90, 255))
        img.save(d / name)
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(d / "zz_empty.png")

    stack = layers_mod.load_stack(d)
    names = stack.names()
    assert names == ["body", "ear_l", "layer2", "layer10"], f"命名与自然排序不符合约定: {names}"
    assert stack.get("body") is not None
    assert stack.get("zz_empty") is None, "全透明图层不应进入堆栈"


def test_map_parts_matches_exact_alias_and_reports(tmp_path):
    from flat2rig import layers as layers_mod

    d = tmp_path / "stack2"
    d.mkdir()
    for name in ("body.png", "ear_l.png", "ear_r.png", "tail.png"):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).rectangle([10, 10, 40, 40], fill=(180, 160, 90, 255))
        img.save(d / name)

    stack = layers_mod.load_stack(d)
    mapping = layers_mod.map_parts(stack, ["body", "ear", "crown"], {"ear": ["ear_l", "ear_r"]})
    assert mapping["body"].getbbox() is not None
    assert mapping["ear"].getbbox() is not None, "别名 ear_l/ear_r 应当合并到 ear"
    assert mapping["crown"].getbbox() is None, "没有对应图层的部位应当是空图层"
    report = layers_mod.map_report()
    assert "crown" in report.get("unmatched_parts", [])


def test_load_stack_rejects_a_single_flat_png(tmp_path):
    from flat2rig import layers as layers_mod

    p = tmp_path / "flat.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(p)
    with pytest.raises(ValueError):
        layers_mod.load_stack(p)


def test_load_stack_empty_directory_warns_not_crashes(tmp_path):
    from flat2rig import layers as layers_mod

    d = tmp_path / "empty"
    d.mkdir()
    stack = layers_mod.load_stack(d)
    assert stack.names() == []


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #

def test_collect_frames_groups_and_natural_sorts(frames_dir):
    from flat2rig import preview as preview_mod

    groups = preview_mod.collect_frames(frames_dir)
    assert set(groups) == {"idle", "sleep"}
    assert [p.name for p in groups["idle"]] == [f"hero_idle_{i:02d}.png" for i in range(1, 6)]
    assert len(groups["sleep"]) == 3


def test_write_preview_is_self_contained(frames_dir, tmp_path):
    from flat2rig import preview as preview_mod

    out = preview_mod.write_preview(frames_dir, tmp_path / "preview.html")
    html = Path(out).read_text(encoding="utf-8")
    assert "data:image/png;base64," in html, "帧必须以 data URI 内联"
    assert "<img" in html
    assert "src=\"frames/" not in html and "src='frames/" not in html, "不应引用外部文件"
    assert Path(out).stat().st_size > 1000


def test_write_preview_rejects_empty_directory(tmp_path):
    from flat2rig import preview as preview_mod

    d = tmp_path / "nothing"
    d.mkdir()
    with pytest.raises(ValueError):
        preview_mod.collect_frames(d)


# --------------------------------------------------------------------------- #
# cli / autotag
# --------------------------------------------------------------------------- #

def test_cli_build_parser_accepts_documented_invocation():
    from flat2rig import cli

    args = cli.build_parser().parse_args(["build", "hero.png", "-c", "hero.rig.json", "-o", "out"])
    assert args.input == "hero.png"
    assert args.config == "hero.rig.json"
    assert args.output == "out"
    assert args.layers is False
    assert args.func is cli.cmd_build


def test_cli_has_all_documented_subcommands():
    from flat2rig import cli

    parser = cli.build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    assert actions, "找不到子命令"
    cmds = set(actions[0].choices)
    assert {"build", "preview", "inspect", "init", "autotag"} <= cmds


def test_autotag_produces_a_loadable_config(make_character, tmp_path):
    from flat2rig import autotag, rig as rig_mod

    png = tmp_path / "hero.png"
    make_character().save(png)
    cfg = autotag.autotag(png)
    assert cfg["canvas"] == [64, 64]
    assert [p["name"] for p in cfg["parts"]] == ["ear", "head", "body"]
    # pivot 必须落在骨骼段内（早期版本曾出现悬空 pivot）
    for part in cfg["parts"]:
        b = part["bones"][0]
        y0, y1 = sorted((b[1], b[3]))
        assert y0 - 1 <= part["pivot"][1] <= y1 + 1, f"{part['name']} 的 pivot 落在骨骼段之外"
    # 生成的配置必须能被 load_rig 正常解析
    path = tmp_path / "auto.rig.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert rig_mod.load_rig(path).size == (64, 64)


def test_autotag_gives_different_motions_to_different_levels(make_character, tmp_path):
    from flat2rig import autotag

    png = tmp_path / "hero.png"
    make_character().save(png)
    cfg = autotag.autotag(png)
    ear, head, body = cfg["parts"]
    assert ear["motions"]["idle"] != head["motions"]["idle"], "上段与中段应当有不同动作曲线"
    assert head["motions"]["idle"] == body["motions"]["idle"], "中段与下段同为起伏"
