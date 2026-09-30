"""把 flat2rig 的产物导出成"桌面宠物插件"能直接吃的逐帧素材。

为什么需要这一层：flat2rig 的输出是"每个状态若干帧 PNG"，而桌宠插件的动画系统
期望的是 ``<角色>_<状态>_<序号>.png``（序号 1 起、两位补零）这样一套约定命名，
外加一份"每个状态有几帧"的清单。这个模块负责：

* 按角色/状态生成帧（复用 :mod:`flat2rig.render`），
* 按插件约定改名落盘，
* 写一份 ``manifest.json``（角色 → 状态 → 帧文件名列表 + 每帧时长），

这样插件侧就不必硬编码帧数，改成读清单即可。双方解耦：插件不认识 rig 配置，
flat2rig 也不认识插件内部结构。

用法（命令行）：``flat2rig petset <配置或预设> -o <输出目录>``
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# 插件内部状态 -> 动画状态名。插件的六种事件状态映射到 flat2rig 的运动状态。
PLUGIN_STATE_MAP = {
    "idle": "resting",
    "work": "working",
    "wait": "waiting",
    "celebrate": "celebrate",
    "sleep": "sleeping",
    "error": "error",
    "walk": "walking",
}

#: 预设标注所在目录（仓库内），格式 ``<角色>.rig.json``。
PRESET_DIRS = (
    Path(__file__).resolve().parents[2] / "examples",
    Path.home() / ".flat2rig" / "presets",
)


@dataclass
class PetExport:
    """一次导出的结果：写出的目录、每个角色的帧清单。"""

    out_dir: Path
    characters: dict[str, dict[str, list[str]]]
    frame_count: int

    def to_manifest(self) -> dict:
        return {
            "format": "flat2rig-petset/1",
            "characters": self.characters,
            "frame_count": self.frame_count,
        }


def find_preset(name: str) -> Path | None:
    """按角色名找预设标注；也接受直接给出的路径。"""
    direct = Path(name)
    if direct.is_file():
        return direct
    stem = name if name.endswith(".rig.json") else f"{name}.rig.json"
    for base in PRESET_DIRS:
        candidate = base / stem
        if candidate.is_file():
            return candidate
    return None


def _natural_key(p: Path) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem))


def build_character(config: Path, image: Path, states: list[str] | None = None):
    """生成一个角色的状态帧。返回 ``(frames, rig)``。"""
    from .parts import cut_layers, inpaint_hidden, order_parts
    from .render import build_frames
    from .rig import build_masks, load_alpha, load_rig

    rig = load_rig(config)
    rgba, _alpha = load_alpha(image)
    masks = build_masks(rgba, rig)
    layers = cut_layers(rgba, masks)
    layers = inpaint_hidden(rgba, masks, order_parts(rig))
    frames = build_frames(rgba, rig, masks, layers=layers, states=states)
    return frames, rig


def export_petset(entries: list[dict], out_dir: str | Path,
                  states: list[str] | None = None) -> PetExport:
    """把若干角色导出为插件可用的逐帧素材。

    ``entries`` 每项形如 ``{"name": "daermaodou", "image": "art/daermaodou-idle.png",
    "config": "examples/daermaodou.rig.json"}``（``config`` 可省略，则按预设查找）。
    输出::

        <out_dir>/frames/<角色>_<状态>_<NN>.png
        <out_dir>/manifest.json

    ``manifest.json`` 的 ``characters[角色][状态]`` 是帧文件名列表（按播放顺序），
    ``delays`` 给出对应的每帧毫秒数。
    """
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    total = 0
    for entry in entries:
        name = entry["name"]
        image = Path(entry["image"])
        if not image.is_file():
            raise FileNotFoundError(f"找不到立绘：{image}")
        config = Path(entry["config"]) if entry.get("config") else find_preset(name)
        if config is None or not Path(config).is_file():
            raise FileNotFoundError(f"找不到 {name} 的标注（可用 flat2rig init 生成）")

        frames, _rig = build_character(Path(config), image, states)
        manifest[name] = {}
        for state, seq in frames.items():
            files = []
            for index, (frame, delay) in enumerate(seq, start=1):
                filename = f"{name}_{state}_{index:02d}.png"
                frame.save(frames_dir / filename, "PNG", optimize=True)
                files.append(filename)
                total += 1
            manifest[name][state] = {"files": files, "delays": [int(d) for _f, d in seq]}

    export = PetExport(out_dir=out, characters=manifest, frame_count=total)
    (out / "manifest.json").write_text(
        json.dumps(export.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    return export


def copy_into_plugin(out_dir: str | Path, plugin_dir: str | Path) -> Path:
    """把导出结果复制进插件目录（``<插件>/petframes``）。

    只复制 PNG 与 manifest；不触碰插件的任何代码文件——代码侧由插件的打包脚本
    （``build-frames.mjs``）读取该目录并内联。
    """
    src = Path(out_dir)
    dst = Path(plugin_dir) / "petframes"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "frames"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src / "frames", target)
    shutil.copyfile(src / "manifest.json", dst / "manifest.json")
    return dst


def load_manifest(path: str | Path) -> dict:
    """读取 ``manifest.json``；返回规范化后的 ``{角色: {状态: {"files": [...], "delays": [...]}}}``。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    chars = raw.get("characters", {})
    if not isinstance(chars, dict):
        raise ValueError(f"{path}: manifest 缺少 characters 段")
    out: dict[str, dict] = {}
    for who, states in chars.items():
        if not isinstance(states, dict):
            raise ValueError(f"{path}: characters[{who}] 必须是对象")
        out[who] = {}
        for state, info in states.items():
            files = info.get("files") if isinstance(info, dict) else info
            if not isinstance(files, list) or not files:
                raise ValueError(f"{path}: {who}/{state} 没有帧文件列表")
            delays = info.get("delays") if isinstance(info, dict) else None
            out[who][state] = {"files": [str(f) for f in files],
                               "delays": [int(d) for d in delays] if delays else None}
    return out
