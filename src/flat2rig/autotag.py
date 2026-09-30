"""启发式自动标注：给一张立绘，猜出一份可用的初始 rig 配置。

这不是"AI 语义分割"——它只做几何推断，目的是把用户从"面对空白 JSON"变成
"跑一次 autotag，再在 inspect 里微调几个点"。猜错的概率很高，但改比从零标快得多。

推断规则（全部基于 alpha 包围盒与竖直距离变换）：
* 角色包围盒按高度切成三段：上段=头顶附属物(ear/crown)，中段=head，下段=body；
* 每段的"最窄处"（水平宽度局部极小）视为该部位与下部的连接点，作为 pivot；
* bones 取该段的中轴垂直线；z 按 从后到前 = 上段在中段之后；
* 输出还包含 states 默认帧时长；eyes 留空（颜色检测交给调用方或人工）。

CLI:
    python -m flat2rig.autotag art.png -o rig.json [--parts ear,head,body]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_STATES = {"idle": 520, "sleep": 560, "error": 170, "celebrate": 140}

# 不同部位用不同动作曲线，否则"整只一起摆"，失去分部位的意义。
#   swing  —— 绕关节摆动：适合头顶附属物（耳朵/花冠/尾巴）
#   bob    —— 竖直起伏：适合头与身体（呼吸、点头、弹跳）
MOTIONS_SWING = {
    "idle": [[0, 0], [6, 0], [2, 0], [-3, 0]],
    "sleep": [[-9, 2], [-7, 3], [-10, 2], [-8, 4]],
    "error": [[-6, 0], [6, 0]],
    "celebrate": [[12, -2], [8, -2], [14, -2]],
}
MOTIONS_BOB = {
    "idle": [[0, 0], [0, -1], [0, -2], [0, 0]],
    "sleep": [[0, 2], [0, 4], [0, 2], [0, 5]],
    "error": [[0, 0], [0, 1]],
    "celebrate": [[0, -6], [0, -10], [0, -5]],
}


def motions_for(name: str, index: int, total: int) -> dict:
    """按部位名与所处层级挑动作曲线：最上段摆动，其余起伏。"""
    return MOTIONS_SWING if index == 0 else MOTIONS_BOB


def _alpha_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img.getchannel("A"), dtype=np.uint8)


def _row_widths(alpha: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """每一行的非透明像素数（只统计 y0..y1）。"""
    band = alpha[y0:y1] > 8
    return band.sum(axis=1).astype(np.int32)


def _narrowest_row(widths: np.ndarray) -> int:
    """在给定区间里找最窄的那一行，返回相对下标；全为 0 时返回中点。"""
    nonzero = np.nonzero(widths)[0]
    if len(nonzero) == 0:
        return len(widths) // 2
    sub = widths[nonzero[0]:nonzero[-1] + 1]
    return int(nonzero[0] + int(np.argmin(sub)))


def autotag(image_path: str | Path, part_names: list[str] | None = None) -> dict:
    """返回一份可直接写盘的 rig 配置字典。"""
    img = Image.open(image_path).convert("RGBA")
    alpha = _alpha_array(img)
    bbox = img.getchannel("A").getbbox()
    if not bbox:
        raise ValueError(f"{image_path} 的 alpha 全透明，无法标注")
    x0, y0, x1, y1 = bbox
    height = y1 - y0

    names = part_names or ["ear", "head", "body"]
    if len(names) != 3:
        raise ValueError("autotag 目前只支持三个部位（上段/中段/下段），例如 ear,head,body")

    # 三段分界：上 32% 为头顶附属物，中 32%~62% 为头，其余为身体
    split_upper = y0 + int(height * 0.32)
    split_middle = y0 + int(height * 0.62)

    bands = [(y0, split_upper), (split_upper, split_middle), (split_middle, y1)]
    parts = []
    for i, (name, (by0, by1)) in enumerate(zip(names, bands)):
        band_alpha = alpha[by0:by1] > 8
        widths = _row_widths(alpha, by0, by1)
        narrow = by0 + _narrowest_row(widths)
        # pivot 必须落在骨骼段内，否则旋转轴心会悬空（autotag 早期版本就有这个 bug）
        narrow = min(max(narrow, by0 + 2), by1 - 2)
        cx = int(np.mean(np.nonzero(band_alpha)[1])) if np.any(band_alpha) else (x0 + x1) // 2
        parts.append({
            "name": name,
            "bones": [[cx, by0 + 2, cx, by1 - 2]],
            "pivot": [cx, narrow],
            # 越靠上越后画（z 小），中段（头）压在上段之上
            "z": i if name == names[-1] else (0 if name == names[0] else 1),
            "blend": 10,
            "motions": {k: [list(m) for m in v] for k, v in motions_for(name, i, len(names)).items()},
        })

    # body 段用底边作为 pivot（贴地），更自然
    parts[-1]["pivot"] = [parts[-1]["pivot"][0], y1 - 1]

    return {
        "_comment": "由 flat2rig autotag 生成的初始标注：请用 `flat2rig inspect` 核对后再微调 bones/pivot。",
        "canvas": [img.width, img.height],
        "body": [parts[-1]["pivot"][0], parts[-1]["pivot"][1]],
        "parts": parts,
        "eyes": [],
        "states": dict(DEFAULT_STATES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flat2rig-autotag",
                                 description="为一张立绘生成初始 rig 标注（几何启发式，需人工核对）")
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="rig.json")
    ap.add_argument("--parts", default="ear,head,body", help="三段部位名，逗号分隔")
    args = ap.parse_args(argv)

    cfg = autotag(args.image, [s.strip() for s in args.parts.split(",") if s.strip()])
    out = Path(args.output)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成初始标注 → {out}")
    print("下一步： flat2rig inspect " + args.image + f" -c {out} -o debug.png")
    print("看那张三联图，把 bones 与 pivot 调到部位中线和关节上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
