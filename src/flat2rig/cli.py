"""flat2rig 命令行入口。

用法
----
  flat2rig build   <输入> -c <标注.json> -o <输出目录> [--layers]
  flat2rig preview <帧目录> [-o preview.html]
  flat2rig inspect <输入> -c <标注.json> -o debug.png
  flat2rig init    <标注.json>            # 生成一份带注释的标注模板

设计说明：所有重依赖模块都在命令内部延迟导入，因此 `flat2rig --help`
在缺少 numpy/Pillow 的环境里也能正常打印帮助。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STATES_DEFAULT = ["idle", "sleep", "error", "celebrate"]
# 输出里的中文（提示、诊断、告警）需要 UTF-8 控制台。Windows 的默认代码页是 cp936/cp1252，
# 直接 print 会抛 UnicodeEncodeError 把整个命令打挂（CI 的 Windows runner 就是这样挂的）。
_ASCII_FALLBACK = {
    "→": "->", "：": ": ", "，": ", ", "。": ".", "、": ", ",
    "（": " (", "）": ") ", "「": '"', "」": '"', "—": "-",
}


def _force_utf8_streams() -> None:
    """尽量把 stdout/stderr 切到 UTF-8；不支持时退回只打 ASCII。"""
    import io
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if not stream or enc in ("utf8", "utf8mb4", "cp65001"):
            continue
        try:  # 优先：重配置成 UTF-8（不换对象，重定向/管道也能用）
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover - 平台差异
            try:  # 退路：用替换错误处理器包一层
                buffer = getattr(stream, "buffer", None)
                if buffer is None:
                    continue
                setattr(sys, name, io.TextIOWrapper(buffer, encoding=enc or "ascii",
                                                    errors="replace", line_buffering=True))
            except (AttributeError, ValueError, OSError):
                continue


def _safe_print(*args, **kwargs) -> None:
    """print 的兜底：即使流的编码无法表示这些字符，也不能让命令崩掉。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        for bad, good in _ASCII_FALLBACK.items():
            text = text.replace(bad, good)
        text = text.encode("ascii", "replace").decode("ascii")
        print(text, **kwargs)


def _fail(msg: str, code: int = 2) -> int:
    _safe_print(f"flat2rig: {msg}", file=sys.stderr)
    return code


def cmd_build(args: argparse.Namespace) -> int:
    try:
        from .rig import load_rig, load_alpha, build_masks, diagnose
        from .parts import cut_layers, inpaint_hidden, order_parts
        from .render import build_frames
    except ImportError as exc:  # pragma: no cover - 缺依赖时的友好提示
        return _fail(f"缺少依赖或模块未就绪：{exc}（请先 `pip install -e .`）")

    src = Path(args.input)
    if not src.exists():
        return _fail(f"输入不存在：{src}")
    rig = load_rig(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    masks_dir = out / "masks"
    frames_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    if args.layers:
        # 模式 B：输入已经是分层结果（see-through 的 PSD 或 PNG 图层目录）
        try:
            from .layers import load_stack, map_parts
        except ImportError as exc:
            return _fail(f"--layers 需要 layers 模块：{exc}")
        stack = load_stack(src)
        names = [p.name for p in rig.parts]
        mapping = map_parts(stack, names)
        base = None
        for name, layer in mapping.items():
            if layer.getbbox():
                base = layer if base is None else base
        if base is None:
            return _fail("分层输入里没有匹配到任何部位，请检查图层命名（可用 --report 查看匹配情况）")
        rgba = base
        masks = {n: mapping[n].getchannel("A") for n in names}
        layers = mapping
        diag = {"coverage": 1.0, "per_part_area": {n: int((masks[n].point(lambda v: 255 if v > 8 else 0)).histogram()[255]) for n in names}, "warnings": []}
    else:
        # 模式 A：自行切件
        rgba, _alpha = load_alpha(src)
        masks = build_masks(rgba, rig)
        diag = diagnose(rgba, rig, masks)
        order = order_parts(rig)
        layers = cut_layers(rgba, masks)
        layers = inpaint_hidden(rgba, masks, order, radius=args.inpaint_radius)

    # 落盘掩膜（便于在别的工具里复用，也便于人工核对）
    for name, mask in masks.items():
        mask.save(masks_dir / f"{name}.png")

    states = [s for s in (args.states.split(",") if args.states else STATES_DEFAULT)]
    frames = build_frames(rgba, rig, masks, layers=layers, states=states)
    total = 0
    for state, seq in frames.items():
        for i, (img, delay) in enumerate(seq, start=1):
            name = f"{src.stem}_{state}_{i:02d}.png"
            img.save(frames_dir / name, "PNG", optimize=True)
            total += 1
        _safe_print(f"  {state:10s} {len(seq):2d} 帧  每帧 {seq[0][1]} ms")

    # 同时写一份机器可读的 rig.json，方便下游复用
    import json
    (out / "rig.json").write_text(
        json.dumps({
            "source": src.name,
            "size": list(rgba.size),
            "parts": [{"name": p.name, "pivot": list(p.pivot), "z": p.z, "blend": p.blend,
                       "motions": {k: [list(m) for m in v] for k, v in p.motions.items()}}
                      for p in rig.parts],
            "eyes": [list(e) for e in rig.eyes],
            "states": rig.states,
            "diagnostics": diag,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    _safe_print(f"\n共 {total} 帧 → {frames_dir}")
    _safe_print(f"掩膜 → {masks_dir}")
    for w in diag.get("warnings", []):
        _safe_print(f"  警告: {w}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .preview import write_preview
    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        return _fail(f"帧目录不存在：{frames_dir}")
    out = Path(args.output) if args.output else frames_dir.parent / "preview.html"
    write_preview(frames_dir, out)
    _safe_print(f"预览页 → {out}（双击即可打开，素材已内联）")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .rig import load_rig, load_alpha, build_masks, diagnose
    from .debug import draw_diagnostic
    src = Path(args.input)
    if not src.exists():
        return _fail(f"输入不存在：{src}")
    rig = load_rig(args.config)
    rgba, _ = load_alpha(src)
    masks = build_masks(rgba, rig)
    diag = diagnose(rgba, rig, masks)
    out = Path(args.output)
    draw_diagnostic(rgba, rig, masks, out)
    _safe_print(f"切件示意图 → {out}")
    _safe_print(f"覆盖率 {diag['coverage']:.3f}；各部位像素：{diag['per_part_area']}")
    for w in diag.get("warnings", []):
        _safe_print(f"  警告: {w}")
    return 0


TEMPLATE = """{
  "canvas": [240, 240],
  "body": [120, 216],
  "parts": [
    {
      "name": "ear",
      "bones": [[150, 60, 120, 118]],
      "pivot": [120, 118],
      "z": 2,
      "blend": 10,
      "motions": {
        "idle": [[0, 0], [8, -1], [4, -2], [0, 0]],
        "sleep": [[-9, 2], [-7, 4], [-10, 2], [-8, 5]],
        "error": [[-7, 0], [7, 0]],
        "celebrate": [[14, -2], [10, -2], [16, -2]]
      }
    },
    {
      "name": "body",
      "bones": [[120, 150, 120, 216]],
      "pivot": [120, 216],
      "z": 0,
      "blend": 8,
      "motions": {
        "idle": [[0, 0], [0, -1], [0, -2], [0, 0]],
        "sleep": [[0, 2], [0, 4], [0, 2], [0, 5]],
        "celebrate": [[0, -8], [0, -14], [0, -6]]
      }
    }
  ],
  "eyes": [[93, 99, 9], [144, 91, 9]],
  "states": { "idle": 520, "sleep": 560, "error": 170, "celebrate": 140 }
}
"""


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.config)
    if out.exists() and not args.force:
        return _fail(f"{out} 已存在；加 --force 覆盖")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE, encoding="utf-8")
    _safe_print(f"已生成标注模板 → {out}")
    _safe_print("用 inspect 核对切件，再改 motions 调动作：")
    _safe_print(f"  flat2rig inspect <立绘.png> -c {out} -o debug.png")
    return 0


def cmd_autotag(args: argparse.Namespace) -> int:
    from .autotag import autotag
    src = Path(args.input)
    if not src.exists():
        return _fail(f"输入不存在：{src}")
    names = [s.strip() for s in args.parts.split(",") if s.strip()]
    cfg = autotag(src, names)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    _safe_print(f"已生成初始标注 → {out}")
    _safe_print(f"下一步： flat2rig inspect {src} -c {out} -o debug.png")
    return 0


def cmd_petset(args: argparse.Namespace) -> int:
    """导出成桌面宠物插件可直接使用的逐帧素材。"""
    from .petset import export_petset, copy_into_plugin, find_preset, PRESET_DIRS

    if args.preset == "all":
        names = ["daermaodou", "juhuali", "xueyuanguagua"]
    else:
        names = [s.strip() for s in args.preset.split(",") if s.strip()]

    art_dir = Path(args.art)
    entries = []
    for name in names:
        image = art_dir / f"{name}-idle.png"
        if not image.is_file():
            # 也允许直接给 <角色>.png
            alt = art_dir / f"{name}.png"
            image = alt if alt.is_file() else image
        if not image.is_file():
            return _fail(f"找不到 {name} 的立绘：{image}（用 --art 指定目录，文件名 <角色>-idle.png）")
        config = Path(args.config) if args.config else find_preset(name)
        if config is None:
            return _fail(f"找不到 {name} 的标注；用 --config 指定，或先把预设放到 {PRESET_DIRS[0]}")
        entries.append({"name": name, "image": str(image), "config": str(config)})

    states = [s.strip() for s in args.states.split(",") if s.strip()] or None
    export = export_petset(entries, args.output, states=states)
    _safe_print(f"导出 {len(export.characters)} 个角色、共 {export.frame_count} 帧 → {export.out_dir}")
    for who, states_map in export.characters.items():
        detail = "、".join(f"{st}×{len(info['files'])}" for st, info in states_map.items())
        _safe_print(f"  {who:16s} {detail}")
    _safe_print(f"  manifest → {Path(export.out_dir) / 'manifest.json'}")

    if args.plugin:
        dst = copy_into_plugin(export.out_dir, args.plugin)
        _safe_print(f"已复制到插件：{dst}")
        _safe_print("下一步：在该插件仓库里执行  node build-frames.mjs --apply  让插件内联这些帧。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flat2rig",
                                description="把一张平面立绘切成可独立运动的部位，并生成动画帧。")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="切件并生成动画帧")
    b.add_argument("input", help="平面立绘 PNG，或分层目录/PSD（配合 --layers）")
    b.add_argument("-c", "--config", required=True, help="标注 JSON")
    b.add_argument("-o", "--output", default="out", help="输出目录（默认 out）")
    b.add_argument("--layers", action="store_true", help="输入已是分层结果（see-through PSD / PNG 图层目录）")
    b.add_argument("--states", default="", help=f"要生成的状态，逗号分隔（默认 {','.join(STATES_DEFAULT)}）")
    b.add_argument("--inpaint-radius", type=int, default=6, help="遮挡补全的扩散半径")
    b.set_defaults(func=cmd_build)

    at = sub.add_parser("autotag", help="几何启发式生成初始标注（需人工核对）")
    at.add_argument("input", help="平面立绘 PNG")
    at.add_argument("-o", "--output", default="rig.json", help="输出标注 JSON")
    at.add_argument("--parts", default="ear,head,body", help="三段部位名，逗号分隔")
    at.set_defaults(func=cmd_autotag)

    ps = sub.add_parser("petset", help="导出成桌面宠物插件可用的逐帧素材（含 manifest）")
    ps.add_argument("--preset", default="all",
                    help="角色名，逗号分隔；或 all（默认三个示例角色）")
    ps.add_argument("--art", default="examples/art",
                    help="立绘目录，文件名需为 <角色>-idle.png（默认 examples/art）")
    ps.add_argument("--config", default="", help="只处理单个角色时直接指定标注 JSON")
    ps.add_argument("-o", "--output", default="petset-out", help="输出目录")
    ps.add_argument("--states", default="", help="只导出这些状态，逗号分隔")
    ps.add_argument("--plugin", default="", help="给定时，把结果复制到该插件的 petframes/ 下")
    ps.set_defaults(func=cmd_petset)

    pv = sub.add_parser("preview", help="把帧目录生成自包含预览页")
    pv.add_argument("frames", help="帧目录")
    pv.add_argument("-o", "--output", default="", help="输出 HTML 路径")
    pv.set_defaults(func=cmd_preview)

    ins = sub.add_parser("inspect", help="输出切件示意图（人工核对用）")
    ins.add_argument("input", help="平面立绘 PNG")
    ins.add_argument("-c", "--config", required=True, help="标注 JSON")
    ins.add_argument("-o", "--output", default="debug.png", help="输出 PNG")
    ins.set_defaults(func=cmd_inspect)

    ini = sub.add_parser("init", help="生成标注模板")
    ini.add_argument("config", help="要写入的 JSON 路径")
    ini.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    ini.set_defaults(func=cmd_init)

    return p


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
