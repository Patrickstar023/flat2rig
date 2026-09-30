"""人工核对用的可视化：把切件结果、骨骼与关节画在一张图上。

切件质量完全取决于标注，而标注错了在代码里看不出来——所以必须能"一眼看到"：
哪些像素归了哪个部位、骨骼画在哪里、关节在哪、覆盖率多少。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .rig import Rig

# 部位配色（按 z 序循环取用，尽量可区分）
PALETTE = [
    (255, 122, 122), (122, 180, 255), (140, 220, 140), (240, 200, 90),
    (200, 140, 240), (120, 220, 220), (250, 160, 200), (180, 180, 120),
]
INK = (28, 26, 40)
PAPER = (245, 246, 251)


def draw_diagnostic(rgba: Image.Image, rig: Rig, masks: dict[str, Image.Image],
                    out_path: str | Path) -> Image.Image:
    """产出一张三联图：左=原图，中=部位着色，右=骨骼与关节。

    返回合成后的图片对象，同时写到 out_path。
    """
    w, h = rgba.size
    panels = []

    left = Image.new("RGB", (w, h), PAPER)
    left.paste(rgba, (0, 0), rgba)
    panels.append(left)

    mid = Image.new("RGB", (w, h), PAPER)
    tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for i, (name, mask) in enumerate(sorted(masks.items(), key=lambda kv: kv[0])):
        colour = PALETTE[i % len(PALETTE)]
        solid = Image.new("RGBA", (w, h), colour + (150,))
        # 只在掩膜区域上色，并保留原图的明暗，便于看出部位边界
        shaded = Image.composite(solid, Image.new("RGBA", (w, h), (0, 0, 0, 0)), mask)
        tint.alpha_composite(shaded)
    base = Image.new("RGBA", (w, h), PAPER + (255,))
    base.alpha_composite(rgba)
    base.alpha_composite(tint)
    mid.paste(base.convert("RGB"), (0, 0))
    panels.append(mid)

    right = Image.new("RGB", (w, h), PAPER)
    ghost = rgba.copy()
    ghost.putalpha(ghost.getchannel("A").point(lambda v: v // 3))
    right.paste(ghost, (0, 0), ghost)
    d = ImageDraw.Draw(right)
    for i, part in enumerate(sorted(rig.parts, key=lambda p: p.z)):
        colour = PALETTE[i % len(PALETTE)]
        for bone in part.bones:
            d.line([bone.x0, bone.y0, bone.x1, bone.y1], fill=colour, width=3)
            for (bx, by) in ((bone.x0, bone.y0), (bone.x1, bone.y1)):
                d.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=colour, outline=INK)
        px, py = part.pivot
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 255, 255), outline=colour, width=3)
        d.text((px + 8, py - 6), part.name, fill=INK)
    for (ex, ey, eh) in rig.eyes:
        d.ellipse([ex - eh, ey - eh, ex + eh, ey + eh], outline=(220, 70, 90), width=2)
    panels.append(right)

    gap = 8
    sheet = Image.new("RGB", (w * 3 + gap * 4, h + gap * 2), (225, 228, 236))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (gap + i * (w + gap), gap))

    label = ImageDraw.Draw(sheet)
    for i, text in enumerate(("原始立绘", "部位着色", "骨骼与关节")):
        label.text((gap + i * (w + gap) + 4, 2), text, fill=INK)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return sheet


def draw_frames_strip(frames: list[Image.Image], out_path: str | Path,
                      columns: int = 6, background=(245, 246, 251)) -> Image.Image:
    """把一帧序列排成一张总览图（用于快速肉眼验收动作）。"""
    if not frames:
        raise ValueError("空帧序列")
    w, h = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    gap = 6
    sheet = Image.new("RGB", (columns * w + gap * (columns + 1), rows * h + gap * (rows + 1)), background)
    for i, frame in enumerate(frames):
        r, c = divmod(i, columns)
        cell = Image.new("RGB", (w, h), background)
        cell.paste(frame, (0, 0), frame)
        sheet.paste(cell, (gap + c * (w + gap), gap + r * (h + gap)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return sheet
