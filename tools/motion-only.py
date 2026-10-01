"""motion-only.py —— 取消旋转，用"纯位移"做出看得见的动作。

用户决定："取消旋转（保证无缝，但动作变整体）"。

原理：关节断口来自**部件之间的相对旋转**——一个部件转开后，边界扫过邻居，
留下错位的轮廓线。**位移不会**：部件平移时边界关系不变，所以结构上不可能产生断口。

第一版把幅度设得太小（实测位移仅 1–3 px，等于没动）。这次放大到能看见的量级：
  * 宠物在插件里显示约 132 px，源图 240 px，所以 1 px 源图 ≈ 0.55 px 屏显
  * 要让"呼吸"看得出来，源图上需要 8–14 px 的位移
  * 各部位相位错开（身体先动、头滞后、耳朵再滞后），避免"整块死板地上下"

产出：改写 examples/*.rig.json 的 motions（rot 一律为 0，脚本会强制校验）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

EX = Path(__file__).resolve().parent.parent / "examples"

#: 位移幅度表（源图像素）。
#:
#: ⚠️ 关键约束：**所有部位必须用同一组位移**。
#: 第一版让各部位位移不同（身体 -12、头 -9、耳朵 -14），本意是"生动"，
#: 但那会在接缝处产生**相对位移**——身体下移时花冠没跟上，边界错开就是台阶。
#: 逐帧补透明像素只能补"完全透明"，接缝处的**半透明**（alpha 30-200）补不到，
#: 于是背景仍会透出一条淡痕。
#: 位移完全一致时，部件之间的相对位置**在数学上不变**，接缝不可能出现。
#: 代价：看起来是"整只在动"，而不是"耳朵在动"——这正是用户选择的方案。
#: 位移幅度（源图像素）。宠物在插件里显示约 132px（源图 240px），
#: 所以 1px 源图 ≈ 0.55px 屏显；呼吸要看得出来，源图上需要 12-16px。
DISPLACEMENT = {
    "idle": [0, -7, -14, -7],
    "walk": [0, -10, 0, -10],
    "work": [0, -6, 0],
    "wait": [0, -9],
    "sleep": [0, 5, 0, 8],
    "error": [0, 4, -4],
    "celebrate": [0, -18, -5],
}

#: 每只角色参与动画的部位（其余部位给零位移，避免残留动作）
PARTS = {
    "daermaodou": ("ear", "head", "body"),
    "xueyuanguagua": ("ear", "head", "body"),
    "juhuali": ("crown", "head", "body"),
}


def motions_for() -> dict[str, list[list[float]]]:
    """整套 motions（rot 恒为 0，所有部位共用同一组位移）。

    格式限制：rig 的 motions 只接受 ``[deg, dy]`` 两元组（见 rig._parse_motions），
    没有横向分量，所以只做竖直位移。
    """
    return {state: [[0, dy] for dy in dys] for state, dys in DISPLACEMENT.items()}


def main() -> int:
    shared = motions_for()
    for who in ("daermaodou", "juhuali", "xueyuanguagua"):
        path = EX / f"{who}.rig.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for part in cfg["parts"]:
            # 所有部位（含未列出的）都用同一组位移：部件之间零相对位移
            part["motions"] = {state: [list(f) for f in seq] for state, seq in shared.items()}
        # 强制校验：不得有任何旋转
        for part in cfg["parts"]:
            for state, seq in (part.get("motions") or {}).items():
                for entry in seq:
                    if abs(float(entry[0])) > 0.001:
                        raise SystemExit(f"{who}/{part['name']}/{state}: 仍有旋转，违反约定")
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {who}: 整只共用同一组位移，旋转全部为 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
