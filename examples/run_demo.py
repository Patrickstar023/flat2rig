"""把 examples/ 里三只角色的立绘跑完整条流水线，产出可直接查看的结果。

用法：
    python examples/run_demo.py

输入：examples/art/<角色>-idle.png（受版权保护的官方立绘，不随仓库分发；
      首次运行会尝试从本机的 whale-pet-redesign/pet-art/ 复制过来，见 examples/README.md）
输出：examples/out/<角色>/
        frames/       动画帧
        masks/        各部位掩膜
        rig.json      解析后的装配信息与诊断
        diagnostic.png  三联核对图（原图 / 部位着色 / 骨骼关节）
        strip.png     该角色 idle 帧的总览图
        preview.html  自包含预览页
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

CHARACTERS = ["daermaodou", "juhuali", "xueyuanguagua"]
# 本机已有的官立绘素材位置（仅用于本地演示；公开仓库不包含这些图）
LOCAL_ART = PROJECT.parent / "whale-pet-redesign" / "pet-art"


def ensure_art() -> None:
    art = ROOT / "art"
    art.mkdir(parents=True, exist_ok=True)
    for name in CHARACTERS:
        dst = art / f"{name}-idle.png"
        if dst.exists():
            continue
        src = LOCAL_ART / f"{name}-idle.png"
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"  复制演示立绘 {src.name} → examples/art/")
    missing = [n for n in CHARACTERS if not (art / f"{n}-idle.png").exists()]
    if missing:
        print(f"缺少演示立绘：{missing}\n请把 PNG 放到 examples/art/<角色>-idle.png（见 examples/README.md）")
        raise SystemExit(1)


def main() -> int:
    from flat2rig.rig import load_rig, load_alpha, build_masks, diagnose
    from flat2rig.parts import cut_layers, inpaint_hidden, order_parts
    from flat2rig.render import build_frames
    from flat2rig.debug import draw_diagnostic, draw_frames_strip
    from flat2rig.preview import write_preview

    print("准备演示素材：")
    ensure_art()
    states = ["idle", "sleep", "error", "celebrate"]

    for name in CHARACTERS:
        img = ROOT / "art" / f"{name}-idle.png"
        cfg = ROOT / f"{name}.rig.json"
        out = ROOT / "out" / name
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)

        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        rig = load_rig(cfg)
        rgba, _alpha = load_alpha(img)
        masks = build_masks(rgba, rig)
        diag = diagnose(rgba, rig, masks)
        t_cut = time.perf_counter()
        layers = cut_layers(rgba, masks)
        layers = inpaint_hidden(rgba, masks, order_parts(rig), radius=6)
        t_fill = time.perf_counter()
        frames = build_frames(rgba, rig, masks, layers=layers, states=states)
        t_frames = time.perf_counter()

        print(f"  覆盖率 {diag['coverage']:.3f}；部位像素 {diag['per_part_area']}")
        for w in diag.get("warnings", []):
            print(f"  警告: {w}")
        print(f"  切件 {t_cut - t0:.2f}s  补全 {t_fill - t_cut:.2f}s  出帧 {t_frames - t_fill:.2f}s")

        for part, mask in masks.items():
            mask.save(out / "masks" / f"{part}.png")
        total = 0
        for state, seq in frames.items():
            for i, (frame, _delay) in enumerate(seq, start=1):
                frame.save(out / "frames" / f"{name}_{state}_{i:02d}.png")
                total += 1
            print(f"  {state:10s} {len(seq):2d} 帧 × {seq[0][1]} ms")
        draw_diagnostic(rgba, rig, masks, out / "diagnostic.png")
        draw_frames_strip([f for f, _ in frames.get("idle", [])], out / "strip.png")
        write_preview(out / "frames", out / "preview.html")
        print(f"  共 {total} 帧 → {out}")

    print("\n完成。打开任一 examples/out/<角色>/preview.html 即可看到动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
