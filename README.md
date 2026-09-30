# flat2rig

**Turn a single flat character illustration into a rigged, part-animated sprite set — no GPU, no model weights, no manual layering.**

**把一张平面角色立绘，自动变成"耳朵会动、头会点、尾巴会摇"的分部位动画素材——不需要显卡、不需要模型权重、不需要手工分层。**

```bash
pip install flat2rig
flat2rig autotag hero.png -c hero.rig.json          # 先自动生成一份初始标注
flat2rig inspect hero.png -c hero.rig.json -o debug.png   # 看三联图核对切件
flat2rig build hero.png -c hero.rig.json -o out/    # 切件 + 生成动画帧
flat2rig preview out/frames                         # 生成自包含预览页
```

不想手写标注？`autotag` 按几何启发式（包围盒分段 + 最窄处作关节）生成一份**可用的初稿**，
再用 `inspect` 微调几个点即可。它不是语义分割——猜错的概率不低，但改比从零标快得多。

---

## Why this exists / 为什么做这个

Single-image layer decomposition is a solved research problem — [see-through](https://github.com/shitagaki-lab/see-through)
(SIGGRAPH 2026) decomposes an anime illustration into ~24 PSD layers, and
[psd2live](https://github.com/tsunehimatoi/psd2live) rigs layered PSDs into Live2D models.

But that stack needs PyTorch 2.8 + transformers 5 + diffusers, several GB of weights, and
(effectively) an NVIDIA GPU. On a laptop with an integrated GPU it is not a 10-second edit loop.

**flat2rig fills the gap underneath**: a classical, deterministic, dependency-light pipeline that
runs in seconds on any machine, and that **can also consume see-through / psd2live output** when
you do have the hardware.

| | see-through + psd2live | **flat2rig** |
| --- | --- | --- |
| Input | single illustration | single illustration **or** see-through layers |
| Decomposition | diffusion model, ~24 semantic layers | skeleton-guided geometric split (you annotate ~6 joints) |
| Weights / GPU | several GB, NVIDIA recommended | **none** |
| Runtime | tens of seconds to minutes per image | **~1–2 s per character** |
| Output | layered PSD | **animation frames (PNG) + preview HTML + rig JSON** |
| Determinism | stochastic | fully deterministic |

They compose: `see-through` for quality decomposition → `flat2rig` for the rig + frames.

---

## How it works / 原理

```
flat PNG  ──┐
            │  1. alpha mask          只处理角色像素
            │  2. skeleton labeling   每个像素归属最近骨骼段 → 部位掩膜（自动，无需逐像素标注）
            │  3. joint wedge cut     在关节处用"关节楔形"做平滑过渡，避免硬边
            │  4. occlusion fill      被前景部位遮住的下层像素用两侧对称信息补全
            │  5. layered compose     每个部位独立绕关节变换后按 z 序合成
            └─► frames/*.png + rig.json + preview.html
```

The only human input is a small JSON with joint positions (~6 points). Everything else is automatic.

### 1. Annotation file / 标注文件

```json
{
  "canvas": [240, 240],
  "body": [120, 216],
  "parts": [
    {
      "name": "ear",
      "bones": [[150, 60, 120, 118]],
      "pivot": [120, 118],
      "z": 2,
      "blend": 10,
      "motions": { "idle": [[0,0],[8,-1],[4,-2],[0,0]], "sleep": [[-9,2],[-7,4],[-10,2],[-8,5]] }
    },
    {
      "name": "body",
      "bones": [[120, 150, 120, 216]],
      "pivot": [120, 216],
      "z": 0,
      "motions": { "idle": [[0,0],[0,-1],[0,-2],[0,0]] }
    }
  ],
  "eyes": [[93, 99, 9], [144, 91, 9]],
  "states": { "idle": 520, "sleep": 560, "error": 170, "celebrate": 140 }
}
```

* `bones` — one or more line segments `[x0,y0,x1,y1]`; every character pixel is assigned to the
  **nearest bone** (Euclidean), which yields the part mask without any manual pixel work.
* `pivot` — the joint this part rotates around.
* `z` — paint order (higher = in front).
* `blend` — half-width (px) of the soft transition band at the part boundary.
* `motions[state]` — per-frame `[rotation_degrees, vertical_offset_px]`.
* `eyes` — optional; used to draw a blink/closed-eye overlay.

### 2. Output

```
out/
├── frames/            <name>_<state>_<i>.png     RGBA 240×240
├── rig.json           resolved rig (masks, pivots, motions)
├── preview.html       self-contained: every frame inlined, plays all states
└── masks/             per-part mask PNGs (debug / reuse in other tools)
```

### 3. Consuming see-through output (optional)

```bash
flat2rig build ./see_through_layers/ --layers -c hero.rig.json -o out/
```

`--layers` treats the input directory as a **back-to-front ordered list of RGBA layers**
(any of `.png` / `.psd`); no decomposition is performed, only rigging + animation.

---

## Install

```bash
pip install flat2rig            # numpy + Pillow only
pip install flat2rig[psd]       # + psd-tools, to read .psd layer stacks
```

Python ≥ 3.10.

---

## Try it

```bash
git clone https://github.com/<you>/flat2rig
cd flat2rig
pip install -e .
python examples/run_demo.py     # writes examples/out/, opens nothing
```

`examples/` ships three annotated characters (the same demo art used in the README GIFs).

实测（本机 Intel Core Ultra 7 256V，纯 CPU，240×240 立绘）：

| 阶段 | 耗时 |
| --- | --- |
| 切件（骨骼归属 + 羽化） | 0.01–0.02 s |
| 遮挡补全 | 0.03–0.26 s |
| 出帧（13 帧 × 4 状态） | 0.10 s |
| 测试套件 | 35 项，0.8 s |

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 35 passed
```

---

## Scope & honesty / 边界

* This is **not** an AI semantic decomposer: it splits along the bones you annotate. It will not
  guess "this is a sleeve" from pixels alone.
* Quality hinges on annotation: ~6 joints per character is usually enough for ears/head/body/tail.
* Occlusion fill is a plausible reconstruction (symmetric + diffusion), not ground truth — for
  heavily occluded art, use see-through layers and `--layers`.
* SVG output, Live2D `.moc3` export and mesh warping are out of scope.

## License

MIT. Demo art in `examples/` keeps its own license (see `examples/README.md`).
