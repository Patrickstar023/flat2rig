"""CI 冒烟测试：不依赖仓库里的任何美术文件（演示立绘是第三方素材，不进仓库），
现场合成一个小角色，跑完整条流水线并断言真的产出了帧。

放在 .github/ 而不是 tests/，因为它产出文件、偏集成而非单元测试；
pytest 套件已覆盖各模块行为，这里只回答一个问题：
"全新环境里 pip install -e . 之后，CLI 能不能真的跑出帧？"
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".ci-out"
STATE_ORDER = ("idle", "sleep", "error", "celebrate")


def make_character(path: Path) -> None:
    """合成一个 240x240 的小角色：身体 + 头 + 头顶附属物，三者留有可分的间距。"""
    img = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([92, 130, 148, 215], radius=18, fill=(126, 170, 236, 255))   # 身体
    d.ellipse([84, 62, 156, 134], fill=(250, 214, 168, 255))                          # 头
    d.ellipse([150, 30, 196, 92], fill=(232, 152, 172, 255))                          # 附属物
    img.save(path)


def write_annotation(path: Path) -> None:
    cfg = {
        "canvas": [240, 240],
        "body": [120, 214],
        "parts": [
            {
                "name": "body",
                "bones": [[120, 150, 120, 212]],
                "pivot": [120, 212],
                "z": 0,
                "blend": 8,
                "motions": {
                    "idle": [[0, 0], [0, -1], [0, -2], [0, 0]],
                    "sleep": [[0, 2], [0, 4], [0, 2], [0, 5]],
                    "error": [[0, 0], [0, 1]],
                    "celebrate": [[0, -5], [0, -9], [0, -4]],
                },
            },
            {
                "name": "head",
                "bones": [[120, 100, 120, 132]],
                "pivot": [120, 134],
                "z": 1,
                "blend": 8,
                "motions": {
                    "idle": [[0, 0], [0, -1], [0, -2], [0, 0]],
                    "sleep": [[0, 2], [0, 4], [0, 2], [0, 5]],
                    "error": [[-3, 0], [3, 0]],
                    "celebrate": [[0, -4], [0, -8], [0, -3]],
                },
            },
            {
                "name": "ear",
                "bones": [[172, 62, 150, 86]],
                "pivot": [150, 92],
                "z": 2,
                "blend": 6,
                "motions": {
                    "idle": [[0, 0], [8, 0], [4, 0], [0, 0]],
                    "sleep": [[-9, 2], [-7, 3], [-10, 2], [-8, 4]],
                    "error": [[-7, 0], [7, 0]],
                    "celebrate": [[12, -2], [8, -2], [14, -2]],
                },
            },
        ],
        "eyes": [],
        "states": {"idle": 520, "sleep": 560, "error": 170, "celebrate": 140},
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def run(*args: str) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    art = OUT / "hero.png"
    rig = OUT / "hero.rig.json"
    make_character(art)
    write_annotation(rig)

    exe = [sys.executable, "-m", "flat2rig.cli"]

    run(*exe, "inspect", str(art), "-c", str(rig), "-o", str(OUT / "diagnostic.png"))
    assert (OUT / "diagnostic.png").stat().st_size > 1000, "inspect 没有产出诊断图"

    run(*exe, "build", str(art), "-c", str(rig), "-o", str(OUT / "out"))

    frames = sorted((OUT / "out" / "frames").glob("hero_*.png"))
    assert frames, "build 没有产出任何帧"
    states = {p.name.split("_")[1] for p in frames}
    assert set(STATE_ORDER) <= states, f"缺少状态：{set(STATE_ORDER) - states}"
    for p in frames:
        with Image.open(p) as im:
            assert im.size == (240, 240), f"{p.name} 尺寸不是 240x240"
            assert im.mode == "RGBA", f"{p.name} 不是 RGBA"
    print(f"frames: {len(frames)} 张，状态 {sorted(states)}")

    run(*exe, "preview", str(OUT / "out" / "frames"), "-o", str(OUT / "preview.html"))
    html = (OUT / "preview.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html, "预览页没有内联帧"
    assert 'src="frames/' not in html, "预览页引用了外部文件，不应自包含"

    run(*exe, "petset", "--preset", "hero", "--art", str(OUT), "--config", str(rig),
        "-o", str(OUT / "petset"))

    manifest_path = OUT / "petset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["characters"]["hero"]
    assert set(files) >= set(STATE_ORDER), "manifest 缺少状态"
    for state, info in files.items():
        assert info["files"], f"{state} 没有帧"
        assert len(info["delays"]) == len(info["files"]), f"{state} 帧数与时长数不一致"
        for index, name in enumerate(info["files"], start=1):
            assert name == f"hero_{state}_{index:02d}.png", name
            assert (OUT / "petset" / "frames" / name).is_file(), f"清单里的 {name} 不存在"
    total = sum(len(v["files"]) for v in files.values())
    print(f"petset: {total} 帧，状态 {sorted(files)}")

    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
