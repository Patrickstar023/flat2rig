"""joint-lab.py —— 关节实验台：快速试"切哪块 / 绕哪个点 / 转多少度"，直接出图。

为什么需要它：单图拆部位做关节动画，效果好不好**只能用眼睛判断**，
所以要有"改参数 → 立刻看图"的闭环，而不是改一次跑一次流水线。

用法：
    python tools/joint-lab.py                # 用默认实验出图
    python tools/joint-lab.py --who juhuali  # 换角色
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECTS = Path(__file__).resolve().parents[2]
FLAT2RIG = PROJECTS / "flat2rig"
ART = FLAT2RIG / "examples" / "art"
OUT = FLAT2RIG / ".tmp"

sys.path.insert(0, str(FLAT2RIG / "src"))
from flat2rig.render import rotate_about  # noqa: E402


def cut(rgba: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """按矩形把一块切出来（返回同尺寸图层，框外透明）。"""
    layer = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    layer.paste(rgba.crop(box), (box[0], box[1]))
    return layer


def cut_along_outline(rgba: Image.Image, box: tuple[int, int, int, int],
                      region: str) -> tuple[Image.Image, Image.Image]:
    """**沿角色轮廓**切出手臂，并把原区域的像素从躯干上抹掉。

    这是关键改进：矩形切块会在原处留下一个方形空洞（一看就是贴纸被撕下来）。
    改成"选中区域内属于角色的像素"作为手臂，并按掩膜把它们从下面那层抠掉，
    胸腔就会露出来——而胸腔是本来就应该在那里的东西，所以看不出缺口。
    """
    w, h = rgba.size
    # 手臂掩膜：区域内的角色像素 + 柔化边缘（避免锯齿）
    region_mask = Image.new("L", (w, h), 0)
    region_mask.paste(255, box)
    alpha = rgba.getchannel("A")
    arm_mask = Image.fromarray(
        np.minimum(np.asarray(region_mask, dtype="uint8"), np.asarray(alpha, dtype="uint8")))
    # 柔化：手臂边缘羽化 2px，避免旋转后出现硬边
    from PIL import ImageFilter
    arm_soft = arm_mask.filter(ImageFilter.GaussianBlur(2))
    arm = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    arm.paste(rgba, (0, 0), arm_soft)
    body = rgba.copy()
    body.paste((0, 0, 0, 0), (0, 0), arm_soft)   # 按掩膜挖掉，不是切方块
    return arm, body


def mask_out(rgba: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """从原图里挖掉一块（返回剩余部分）。"""
    rest = rgba.copy()
    rest.paste((0, 0, 0, 0), box)
    return rest


#: 每只角色的"手臂"大致范围与肩关节。这些坐标是从 240×240 原画上量出来的：
#: 大耳帽兜的双爪举在胸前（约 x95-135 y130-178），呱呱类似，菊花梨没有独立手脚。
EXPERIMENTS = {
    "daermaodou": {
        "arm_box": (95, 132, 138, 180),
        # 关节放在胸口而不是靠近脸：原先放 (120,132) 时，手臂一转就扫过脸，
        # 露出脸上的像素，看着像脸上挂了白纸条。
        "shoulder": (118, 154),
        "angles": [0, -22, 20, -34],
    },
    "xueyuanguagua": {
        # 注意：这只身上是大衣+围巾+衬衫，颜色和层次都复杂。
        # 早先框到 x90-146,y140-194 会把外套下摆一起卷进来，一转就是一块白洞
        # （底下没有"被遮住的衣服"可以露出来）。这里缩到只有手部。
        "arm_box": (108, 150, 140, 186),
        "shoulder": (120, 156),
        "angles": [0, -18, 16, -28],
    },
    "juhuali": {
        # 菊花梨是梨形整体，没有可分离的手臂；这里试"底座"的倾侧，
        # 用来说明"能切开"和"切不开"的差别。
        "arm_box": (78, 150, 200, 224),
        "shoulder": (138, 152),
        "angles": [0, 10, 20, -8],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="关节实验台")
    ap.add_argument("--who", default="daermaodou", choices=list(EXPERIMENTS))
    ap.add_argument("--scale", type=int, default=2, help="输出放大倍数")
    args = ap.parse_args()

    spec = EXPERIMENTS[args.who]
    src = Image.open(ART / f"{args.who}-idle.png").convert("RGBA")

    layer, rest = cut_along_outline(src, spec["arm_box"], args.who)   # 沿轮廓切 + 从躯干抹掉

    tiles: list[tuple[str, Image.Image]] = []
    for angle in spec["angles"]:
        moved = rotate_about(layer, spec["shoulder"], float(angle))
        canvas = rest.copy()
        canvas.alpha_composite(moved)          # 直接叠上去，看接缝有多明显
        bg = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
        bg.alpha_composite(canvas)
        d = ImageDraw.Draw(bg)
        d.rectangle(spec["arm_box"], outline=(255, 0, 0, 90))
        d.ellipse([spec["shoulder"][0] - 3, spec["shoulder"][1] - 3,
                   spec["shoulder"][0] + 3, spec["shoulder"][1] + 3], fill=(0, 128, 255, 220))
        tiles.append((f"{angle}°", bg.convert("RGB")))

    w = 240 * args.scale
    pad, lh = 6, 20
    sheet = Image.new("RGB", (pad + (w + pad) * len(tiles), pad * 2 + lh + w), (245, 246, 251))
    d = ImageDraw.Draw(sheet)
    d.text((pad, pad), f"{args.who}  切开={spec['arm_box']}  关节={spec['shoulder']}"
                       f"（蓝点）红框=切块范围", fill=(30, 30, 45))
    for i, (label, img) in enumerate(tiles):
        x = pad + i * (w + pad)
        sheet.paste(img.resize((w, w), Image.LANCZOS), (x, pad + lh))
        d.text((x + 4, pad + lh + 2), label, fill=(180, 0, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"joint-lab-{args.who}.png"
    sheet.save(path)
    print(f"-> {path}   {sheet.size}")
    print("看什么：① 块动起来有没有'换个姿势'的感觉 ② 原位置留下的缺口/接缝有多明显")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
