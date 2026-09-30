# examples

本目录只包含**标注文件**（`*.rig.json`）与运行脚本；**不包含**任何游戏立绘。

## 为什么不含立绘

`examples/art/` 下的演示图来自《洛克王国世界》官方素材，**版权归腾讯/相关权利人**，
不适合放进公开仓库。因此 `.gitignore` 排除了 `examples/art/`。

## 怎么跑

```bash
pip install -e .
python examples/run_demo.py
```

脚本会：

1. 若 `examples/art/` 为空，尝试从本机 `../whale-pet-redesign/pet-art/` 复制三张立绘过去；
2. 逐角色执行"切件 → 补全遮挡 → 出帧 → 出预览"；
3. 把结果写到 `examples/out/<角色>/`。

想换成你自己的角色：

```bash
flat2rig init my.rig.json                                  # 生成标注模板
flat2rig inspect my-art.png -c my.rig.json -o debug.png    # 看切件对不对
flat2rig build   my-art.png -c my.rig.json -o out/         # 出帧
flat2rig preview out/frames                                # 出预览页
```

## 标注怎么标

* `bones`：一个或多个线段，把该部位的"主干"画出来即可（角色像素会自动归属最近的骨骼）。
* `pivot`：该部位绕哪个点转——耳朵标在耳根、花冠标在花茎、头标在颈点。
* `z`：越大越靠前。皮肤/身体 0，耳/冠 1，头 2。
* `blend`：交界处的柔和过渡宽度（像素），8–12 比较自然。
* `motions[state]`：逐帧 `[旋转角度, 竖直位移]`；帧数就是动作帧数。
* `eyes`：`[cx, cy, half]`，用于眨眼/闭眼叠加（可选）。

**判断标注对不对最快的办法**：跑 `flat2rig inspect`，看那张三联图里"部位着色"是否干净、
骨骼是否落在部位中线上、关节是否在转动轴心。
