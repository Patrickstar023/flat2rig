# 集成指南

给"要把 flat2rig 接进别的程序"的人看。三种耦合层次，从松到紧任选。

---

## 1. 命令行（最松，推荐先走这条）

```bash
pip install flat2rig

flat2rig autotag hero.png -c hero.rig.json        # 生成标注初稿（几何启发式，需人工核对）
flat2rig inspect hero.png -c hero.rig.json -o debug.png   # 三联图核对切件
flat2rig build   hero.png -c hero.rig.json -o out/        # 出帧
flat2rig preview out/frames -o preview.html               # 自包含预览页
flat2rig petset --art art/ -o petset-out --plugin <宠物壳目录>
```

所有子命令都不联网、不需要显卡，退出码非 0 即失败（预检失败会打印具体原因）。

## 2. Python API（中等耦合）

调用顺序就是 `cli.py` 里 `cmd_build` 的顺序，照抄即可：

```python
from flat2rig.rig import load_rig, load_alpha, build_masks, diagnose
from flat2rig.parts import cut_layers, inpaint_hidden, order_parts
from flat2rig.render import build_frames
from flat2rig.debug import draw_diagnostic

rig = load_rig("hero.rig.json")                 # 标注 -> Rig
rgba, alpha = load_alpha("hero.png")            # 立绘 -> (RGBA 图, alpha 掩膜)
masks = build_masks(rgba, rig)                  # 骨骼归属 + 羽化，掩膜之和 == alpha
report = diagnose(rgba, rig, masks)             # 覆盖率 / 各部位像素数 / 告警

layers = cut_layers(rgba, masks)                            # 掩膜 -> 颜色图层
layers = inpaint_hidden(rgba, masks, order_parts(rig))      # 补接触带，旋转不露缝

frames = build_frames(rgba, rig, masks, layers=layers,
                      states=["idle", "sleep", "error", "celebrate"])
# frames: {状态: [(PIL 图, 该帧毫秒数), ...]}

draw_diagnostic(rgba, rig, masks, "debug.png")  # 出问题时先看这张图
print(report["coverage"], report["warnings"])
```

要点：

* `order_parts(rig)` 是**从后到前**的绘制顺序；`compose_frame` 依赖它。
* `build_frames` 的 `layers` 可传入你自己准备的图层（例如见 `layers.py` 的已分层输入），
  此时跳过切件与补全。
* 全部函数是纯函数式的（不写全局状态、无随机），同一输入必得同一输出。

## 3. 磁盘契约（最紧，给不认识 Python 的工具用）

`flat2rig build -o out/` 产出：

```
out/
├── frames/<立绘文件名>_<状态>_<两位序号>.png    # 序号从 01 开始，1 基
├── masks/<部位名>.png                          # L 模式掩膜，可复用
├── rig.json                                    # 解析后的装配与诊断
└── preview.html                                # 全部帧已内联
```

`out/rig.json`：

| 字段 | 含义 |
| --- | --- |
| `source` | 源立绘文件名 |
| `size` | `[宽, 高]`，所有帧同一画布 |
| `parts[]` | 每个部位：`name`、`pivot`、`z`（越大越靠前）、`blend`、`motions`（状态 → 逐帧 `[角度, 竖直位移]`） |
| `eyes` | `[cx, cy, 半径]`，用于眨眼叠加；可为空 |
| `states` | 状态 → 每帧毫秒数 |
| `diagnostics` | 覆盖率、各部位像素数、告警 |

`flat2rig petset -o petset-out` 产出给桌面宠物壳用的版本：

```
petset-out/
├── frames/<角色>_<状态>_<两位序号>.png
└── manifest.json      # {"format":"flat2rig-petset/1","characters":{角色:{状态:{"files":[...],"delays":[...]}}}}
```

**插件/引擎侧只需读 `manifest.json`，不要硬编码帧数**——加状态、改帧数都不必动它的代码。
（本项目配套的 `whale-pet-redesign/build-frames.mjs` 就是扫描该目录 + 读清单来内联的。）

## 4. 在引擎 / 网页 / 桌宠壳里播放

* 帧就是普通 PNG（RGBA、同一画布尺寸），不需要图集，直接一张张换 `src` 即可。
* 播放节奏用 `manifest.json` 的 `delays`（每帧可不同），或 `rig.json` 的 `states`（每状态一个值）。
* 抖动来自切帧时重新解码位图。**建议启动时预解码该角色的全部帧**（`new Image(); img.src = uri`），
  本项目配套插件就是这么做（`preloadFrames()`），否则低端机上会看到闪。
* 帧之间只差一个部位的角度，画布不变，因此不会出现尺寸跳动。

## 5. 无头 / CI 环境

* 无 GPU、无网络、无外部权重；依赖只有 numpy / Pillow / scipy。
* 输出完全确定：没有随机数、没有时间戳。同样的标注 + 同样的立绘 = 同样的字节。
* 想复现就固定标注文件（它是唯一的"输入参数"）。
* 240×240 实测：切件 0.01–0.02 s、补全 0.03–0.26 s、13 帧 × 4 状态 0.10 s。

## 6. 排错表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 某个部位整个消失 | 骨骼与别的部位挤在一起，被抢空 | 跑 `inspect` 看"部位着色"；把骨骼挪到该部位中线上 |
| 帧数比动作列表少 | 帧数取**各部位中最长**的动作序列，短列表重复最后一帧 | 这是设计行为；要对齐就补齐最长的那个 |
| 旋转时露出空隙 | 接触带没补上：`blend` 太小或补全距离太小 | 调大该部位的 `blend`，或增大 `--inpaint-radius` |
| 旋转时露出"假补丁" | 补全把被遮挡区域整块划给了下层 | 已修（只补接触带）；若仍明显，缩小 `blend` |
| 部位带上了别人的像素 | 骨骼侵入相邻部位 | `inspect` 对照，收紧骨骼 |
| 播放时闪烁 | 切帧重新解码 | 启动时预解码全部帧 |
| 立绘四周被裁 | 部件旋转越出画布 | 出图时给画布留边距（`make-pet-art` 用的是四周各 14px） |
| `--layers` 说没匹配到部位 | 图层命名与部位名不一致 | 用 `map_report()` 看匹配情况，或传 `aliases` |
