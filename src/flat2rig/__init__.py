"""flat2rig —— 单张平面立绘 → 分部位动画素材。

公开 API 见 README。各子模块按职责拆分：

* :mod:`flat2rig.rig`      标注解析 + 骨骼归属切件（骨骼→部位掩膜）
* :mod:`flat2rig.parts`    掩膜→颜色图层，并补全被前景遮挡的像素
* :mod:`flat2rig.render`   分层合成 + 绕关节变换 + 状态帧展开
* :mod:`flat2rig.layers`   读取"已分层"输入（see-through 的 PSD / PNG 图层目录）
* :mod:`flat2rig.cli`      命令行入口
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
