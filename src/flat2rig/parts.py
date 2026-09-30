"""Turn part masks into colour layers, and reconstruct what foreground parts hide.

:mod:`flat2rig.rig` answers *which* pixels belong to which part.  This module
answers the two follow-up questions the animation needs:

1. :func:`cut_layers` -- give every part a full-canvas RGBA layer whose alpha is
   ``mask * source_alpha / 255``, so feathered mask edges keep a feathered alpha
   and compositing the layers back together reproduces the source image.
2. :func:`inpaint_hidden` -- the source art is a *flat composite*: a part drawn
   behind another part is missing exactly the pixels the front part covers.
   Rotating the back part exposes that hole, so the missing pixels are
   reconstructed from the part's own visible pixels by mirror symmetry first and
   by iterative diffusion second, then blended in with a soft falloff.

Everything is deterministic (no randomness) and works on numpy arrays with
Pillow only; the expensive loops are restricted to the bounding box of the
region that actually needs filling, which keeps a 240x240 / 4-part character
well under two seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps this module importable on its own
    from .rig import Rig

__all__ = ["cut_layers", "inpaint_hidden", "order_parts"]

#: Mask value above which a pixel counts as "belonging to" the part.
SUPPORT_THRESHOLD = 0.35
#: Mask/alpha value above which a pixel counts as visible material.
VISIBLE_EPS = 0.02
#: Jacobi iterations for the nearest-neighbour propagation (200 is plenty at 240 px).
DIFFUSION_ITERS = 200
#: Max per-channel difference for a mirror sample to be trusted (0-255).
MIRROR_TOLERANCE = 26.0
#: 图层像素 alpha 不超过此值才算"空的"、允许被补全写入；
#: 高于它的（含羽化带）一律不动 —— 否则补全会覆盖部位自己的真实像素（实测过的事故）。
FILL_ALPHA_GATE = 2
#: How many pixels of background colour get bled outwards (anti-halo margin).
BLEED_ITERS = 4


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def order_parts(rig: "Rig") -> list[str]:
    """Return the part names in paint order, back to front.

    Sorted by ``(z, name)``: ascending ``z`` is "painted first / furthest back",
    and the name breaks ties so the order is stable and deterministic even when
    two parts share a ``z``.  Later entries are painted over earlier ones.
    """
    return [part.name for part in sorted(rig.parts, key=lambda p: (float(p.z), str(p.name)))]


def cut_layers(rgba: Image.Image, masks: dict[str, Image.Image]) -> dict[str, Image.Image]:
    """Cut ``rgba`` into one full-canvas RGBA layer per part.

    For every entry of ``masks`` the returned layer contains only that part's
    pixels, with ``alpha = mask * source_alpha / 255`` so a feathered mask edge
    yields a correspondingly feathered alpha (the layers sum back to the source).

    The colour channels keep the *source* colour also where the layer is
    transparent, and a few pixels of background colour are bled outwards from
    the silhouette.  That matters because rotation resamples R, G, B and A
    independently: with the usual transparent-black (or transparent-(2,2,2))
    padding, bicubic rotation would smear dark fringes into every edge.
    """
    size = rgba.size
    src = _rgba_array(rgba)
    src_alpha = src[..., 3] * (1.0 / 255.0)
    rgb = _bleed_colour(src[..., :3], src_alpha)

    layers: dict[str, Image.Image] = {}
    for name, mask in masks.items():
        m = _mask_array(mask, size)
        arr = np.empty((size[1], size[0], 4), dtype=np.float32)
        arr[..., :3] = rgb
        arr[..., 3] = np.rint(np.clip(m * src_alpha, 0.0, 1.0) * 255.0)
        layers[name] = _to_image(arr)
    return layers


def rotate_mask(mask: np.ndarray, pivot: tuple[float, float], deg: float) -> np.ndarray:
    """把布尔掩膜绕 pivot 旋转 deg 度（最近邻），用于推算动作会露出哪块区域。"""
    if abs(deg) < 1e-6:
        return mask.copy()
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    rotated = img.rotate(deg, resample=Image.NEAREST,
                         center=(float(pivot[0]), float(pivot[1])), fillcolor=0)
    return np.asarray(rotated, dtype=np.uint8) > 127


def inpaint_hidden(rgba: Image.Image, masks: dict[str, Image.Image], order: list[str],
                   radius: int = 6, angles: dict[str, float] | None = None,
                   pivots: dict[str, tuple[float, float]] | None = None) -> dict[str, Image.Image]:
    """Return :func:`cut_layers` outputs extended by each part's hidden pixels.

    ``order`` is the paint order, back to front (see :func:`order_parts`).

    **需要补哪块**：不是"整块被遮挡区域"，也不是拍一个固定距离，而是
    :func:`exposure_band` 算出的"会被别的部位移动掀开"的那块——只有这里会露成空洞。
    传入 ``angles``（见 :func:`motion_extent`）与 ``pivots`` 启用该判据；不传时退回
    "沿自身轮廓限距生长"的粗略做法（用在动画幅度未知的场合）。

    **怎么补**：先在与主体色的色差可接受时采信镜像样本，其余按"最近已知像素"传播色值。
    这里刻意不用迭代扩散——扩散把周围颜色不断平均，补出来的块会明显发白。

    ``radius`` 是补全区向真实像素过渡的宽度（也用于孔洞外扩）；越大越柔和、稍慢。

    ``masks`` 里每个名字都会出现在返回值中；``order`` 里没有的名字会被追加到最前
    （它们遮挡不了调用者描述的那些部位）。
    """
    size = rgba.size
    height, width = size[1], size[0]
    src = _rgba_array(rgba)
    src_alpha = src[..., 3] * (1.0 / 255.0)

    names = list(masks)
    ordered = [n for n in order if n in masks]
    ordered += [n for n in names if n not in ordered]

    soft = {n: _mask_array(masks[n], size) for n in ordered}
    boxes = {n: _bbox(soft[n]) for n in ordered}
    base = cut_layers(rgba, {n: masks[n] for n in ordered})

    # Mirror axis: the vertical line through the character's alpha centroid.
    centroid_x = _centroid_x(src_alpha, width)
    mirror_src = np.clip(np.rint(2.0 * centroid_x - np.arange(width)).astype(np.intp), 0, width - 1)

    out: dict[str, Image.Image] = {}
    for index, name in enumerate(ordered):
        own = soft[name]
        own_bin = own > SUPPORT_THRESHOLD
        if not own_bin.any():
            out[name] = base[name]  # empty part: nothing to reconstruct
            continue

        front = _front_union(soft, boxes, ordered, index, height, width)
        if not front.any():
            out[name] = base[name]  # nothing is painted over this part
            continue

        support = _plausible_support(own_bin, mirror_src)
        # 补全范围 = 会被其它部位的移动掀开、又没有别人接手的那块（见 exposure_band）。
        # 没有动作信息时退回"沿自身轮廓限距生长"。
        if angles and pivots:
            peaks = {n: float(angles.get(n, 0.0)) for n in ordered}
            band = exposure_band({n: soft[n] > SUPPORT_THRESHOLD for n in ordered},
                                 ordered, name, peaks, pivots)
            if not band.any():
                out[name] = base[name]   # 这个部位不会被掀开，无需补
                continue
            # 用二值的"该像素是否属于本部位"而不是羽化权重：交界处的羽化值可能只有 0.1，
            # 若按 (1-own) 折算，补全会被压到阈值以下而完全失效（真实踩过的坑）。
            occ = np.clip((band & ~own_bin).astype(np.float32) + support * (1.0 - own), 0.0, 1.0)
        else:
            grown = _grow_from_own(own_bin, front, max_dist=max(8, int(radius) * 3))
            occ = np.clip(front * np.maximum(support, grown) * (1.0 - own), 0.0, 1.0)
        occ_bin = occ > VISIBLE_EPS
        if not occ_bin.any():
            out[name] = base[name]
            continue

        arr = _refill_layer(_rgba_array(base[name]), src, src_alpha, own, own_bin,
                            occ, occ_bin, mirror_src, int(radius))
        out[name] = _to_image(arr)
    return out


# --------------------------------------------------------------------------- #
# occlusion fill internals
# --------------------------------------------------------------------------- #
def motion_extent(rig) -> dict[str, float]:
    """每个部位在所有状态里的最大"动作幅度"：旋转角度与平移像素取较大者。

    只用旋转角度会漏掉"只平移不旋转"的部位（例如头随呼吸上下移动），
    那样它的补全区会被算成空集 —— 真实缺陷：头被耳朵腾出来的那块就没人补。
    """
    out: dict[str, float] = {}
    for part in getattr(rig, "parts", []):
        peak = 0.0
        for sequence in (part.motions or {}).values():
            for move in sequence:
                try:
                    angle = abs(float(move[0]))
                    shift = max((abs(float(v)) for v in move[1:]), default=0.0)
                except (TypeError, ValueError, IndexError):
                    continue
                peak = max(peak, angle, shift)
        out[part.name] = peak
    return out


def swept_region(mask: np.ndarray, pivot: tuple[float, float], peak: float,
                 steps: int = 6) -> np.ndarray:
    """该部位在 ±peak（角度或像素）内摆动时扫过的区域。

    "需要补全的范围"不是拍脑袋的距离，而是这个扫掠区减去原始掩膜：
    只有这块会因为动作而露出来。
    """
    swept = mask.copy()
    for index in range(1, steps + 1):
        offset = peak * index / steps
        swept |= _shift_mask(rotate_mask(mask, pivot, offset), offset, 0)
        swept |= _shift_mask(rotate_mask(mask, pivot, -offset), -offset, 0)
    return swept


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """平移布尔掩膜（最近邻取整）。"""
    ix, iy = int(round(dx)), int(round(dy))
    if ix == 0 and iy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    xs0, xs1 = (ix, w) if ix >= 0 else (0, w + ix)
    xd0 = xs0 - ix
    ys0, ys1 = (iy, h) if iy >= 0 else (0, h + iy)
    yd0 = ys0 - iy
    out[yd0:yd0 + (ys1 - ys0), xd0:xd0 + (xs1 - xs0)] = mask[ys0:ys1, xs0:xs1]
    return out


def exposure_band(masks: dict[str, np.ndarray], order: list[str],
                  name: str, peaks: dict[str, float],
                  pivots: dict[str, tuple[float, float]]) -> np.ndarray:
    """``name`` 里"会被任意其它部位的移动掀开、因而必须补上"的区域。

    推导：某像素现在被部位 Q 占着，但 Q 摆动后不再盖住它、又没有别人接手，
    那么合成时这里就是透明的空洞。所以需要补的区域是

        (Q 的扫掠区 - Q 现在的覆盖) ∩ name 的覆盖

    对所有 Q 取并集。注意这里用的是**所有其它部位**的扫掠区，不是"paint 在前面的
    那些" —— 耳朵往旁边转，被腾出来的是画在前面的**头**的地盘，只按 z 序取并集会漏掉。
    """
    mine = masks[name]
    band = np.zeros_like(mine, dtype=bool)
    for other, other_mask in masks.items():
        if other == name:
            continue
        peak = float(peaks.get(other, 0.0))
        if peak <= 1e-6:
            continue
        pivot = pivots.get(other)
        if pivot is None:
            continue
        swept = swept_region(other_mask, pivot, peak)
        band |= swept & ~other_mask & mine
    return band


def _grow_from_own(own_bin: np.ndarray, front: np.ndarray, max_dist: int = 24) -> np.ndarray:
    """没有动作信息时的退路：从自身可见像素出发，沿前景区域限距生长。

    只补"离本体不远、且被前景盖住"的连通区域，避免把整块被遮挡区域都占为己有
    （那等于把本部位的空白填充版整片盖到前景之上，是真实出现过的缺陷）。
    """
    allowed = front > 0 if front.dtype != np.bool_ else front
    grown = np.zeros_like(own_bin, dtype=bool)
    frontier = own_bin & allowed
    grown |= frontier
    for _ in range(max(0, max_dist)):
        nxt = np.zeros_like(frontier)
        nxt[1:, :] |= frontier[:-1, :]
        nxt[:-1, :] |= frontier[1:, :]
        nxt[:, 1:] |= frontier[:, :-1]
        nxt[:, :-1] |= frontier[:, 1:]
        nxt &= allowed
        nxt &= ~grown
        if not nxt.any():
            break
        grown |= nxt
        frontier = nxt
    return grown


def _refill_layer(base: np.ndarray, src: np.ndarray, src_alpha: np.ndarray,
                  own: np.ndarray, own_bin: np.ndarray, occ: np.ndarray,
                  occ_bin: np.ndarray, mirror_src: np.ndarray, radius: int) -> np.ndarray:
    """Paint the occluded region into ``base`` (a cut layer) and return it.

    ``base`` is modified in place and returned; all work happens inside the
    bounding box of the hole plus the cross-fade ring.
    """
    height, width = occ_bin.shape
    rgb = src[..., :3]

    # 1. reconstruct the hole plus a ring of the part's real pixels around it,
    #    so the guess can be cross-faded into the original instead of butting
    #    against it.
    ring = _dilate(occ_bin, max(1, radius)) & own_bin & ~occ_bin
    domain = occ_bin | ring

    ys, xs = np.nonzero(domain)
    y0, y1 = max(0, int(ys.min()) - 1), min(height, int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(width, int(xs.max()) + 2)
    sl = (slice(y0, y1), slice(x0, x1))
    dom = domain[sl]

    # 2. seed the field with the part's own pixels that stay untouched.
    seed = own_bin[sl] & ~dom
    if not seed.any():  # tiny part fully covered by the ring: fall back to any real pixel
        seed = own_bin[sl] & ~occ_bin[sl]

    val = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    wgt = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    val[seed] = rgb[sl][seed]
    wgt[seed] = np.maximum(src_alpha[sl][seed], 0.05)

    # 3. 镜像对称只用于"小范围对齐校正"，不参与大面积取色。
    #    真实缺陷：把镜像样本按全权重铺满被遮挡区，会把对侧（甚至眼睛）的颜色搬过来，
    #    旋转后露出一块与周围不搭的色斑。因此这里只接受"距离很近且色差不大"的镜像样本，
    #    其余一律交给第 4 步的最近邻传播（取本部位真实边界的颜色）。
    mirrored_own = own[:, mirror_src][sl]
    mir = dom & (mirrored_own > VISIBLE_EPS) & (wgt <= 0.0)
    if mir.any():
        mval = rgb[:, mirror_src][sl]
        # 与已知种子的色差：超过阈值就不采信（避免把对侧的眼睛/深色轮廓搬过来）
        if wgt.max() > 0.0:
            known = wgt > 0.0
            ref = val[known].mean(axis=0)
            close = np.abs(mval - ref).max(axis=2) <= MIRROR_TOLERANCE
            mir = mir & close
        if mir.any():
            val[mir] = mval[mir]
            wgt[mir] = 1.0

    # 4. 剩余区域按"最近已知像素"传播色值。
    #    这里用最近邻传播而不是迭代扩散：扩散把周围颜色不断平均，补出来的块会明显发白
    #    （真实缺陷：旋转后露出的补丁比周围浅一大截）；最近邻传播取的是边界真实色值，
    #    与邻域一致，看不出补丁。
    unknown = dom & (wgt <= 0.0)
    if unknown.any():
        idx = np.full(unknown.shape, -1, dtype=np.intp)   # 该像素取色用的种子下标
        known_flat = np.flatnonzero(wgt.ravel() > 0.0)
        if known_flat.size:
            idx.ravel()[known_flat] = known_flat
            frontier = idx >= 0
            for _ in range(DIFFUSION_ITERS):
                if not (unknown & ~frontier).any():
                    break
                nxt = np.full(unknown.shape, -1, dtype=np.intp)
                for src_slice, dst_slice in (
                    ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                    ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                    ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                    ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                ):
                    from_src = frontier[src_slice]
                    take = from_src & (nxt[dst_slice] < 0)
                    if take.any():
                        nxt[dst_slice] = np.where(take, idx[src_slice], nxt[dst_slice])
                fresh = (nxt >= 0) & ~frontier
                if not fresh.any():
                    break
                idx[fresh] = nxt[fresh]
                frontier = frontier | fresh
            filled = unknown & (idx >= 0)
            if filled.any():
                flat_idx = np.clip(idx[filled], 0, val.reshape(-1, 3).shape[0] - 1)
                val[filled] = val.reshape(-1, 3)[flat_idx]
                wgt[filled] = 1.0
            unknown = unknown & (idx < 0)

    # 5. falloff: full weight inside the hole, ramping to ~0 at the outer edge of
    #    the ring (geodesic distance measured from the hole itself).
    fall = np.ones_like(wgt)
    if ring[sl].any():
        reached = occ_bin[sl].copy()
        step = 0
        limit = max(1, radius)
        while step < limit:
            step += 1
            grown = _dilate(reached, 1) & dom
            fresh = grown & ~reached
            if not fresh.any():
                break
            fall[fresh] = 1.0 - float(step) / float(limit + 1)
            reached = reached | fresh
        fall[occ_bin[sl]] = 1.0

    # 6. 把补出来的内容写回图层。
    #
    # 只写"洞"里的像素，而且**只写图层上原本没有内容的地方**——这是本条流水线上最容易
    # 踩的坑：早先的写入范围是 `dom & (fall > 0)`，而 dom 还包含洞外那一圈部位自己的
    # 真实像素（ring），补全于是把这些真实像素的颜色也覆盖掉了：实测 body 684px +
    # ear 1087px 的"原本完全不透明"像素被改写，表现就是形象发脏、脸被涂抹。
    # 这里再加一道闸：图层上已有 alpha 的像素（含羽化带）绝不触碰，
    # 洞与真实像素之间的过渡交给部位自身 alpha 的羽化。
    out_rgb = base[..., :3][sl]
    out_a = base[..., 3][sl]
    hole = occ_bin[sl] & (out_a <= FILL_ALPHA_GATE)
    if hole.any():
        out_rgb[hole] = val[hole]
        out_a[hole] = np.maximum(out_a[hole], src_alpha[sl][hole] * occ[sl][hole])
    return base


def _front_union(soft: dict[str, np.ndarray], boxes: dict[str, tuple[int, int, int, int] | None],
                 ordered: list[str], index: int, height: int, width: int) -> np.ndarray:
    """Masks of the parts painted in front of ``ordered[index]`` that overlap its box."""
    own_box = boxes[ordered[index]]
    acc = np.zeros((height, width), dtype=np.float32)
    for j in range(index + 1, len(ordered)):
        other = ordered[j]
        if own_box is None or boxes[other] is None:
            continue
        if not _boxes_overlap(own_box, boxes[other]):
            continue  # cannot cover this part: skip the (larger) array work
        acc += soft[other]
    return np.clip(acc, 0.0, 1.0)


def _plausible_support(own_bin: np.ndarray, mirror_src: np.ndarray) -> np.ndarray:
    """Where the part could plausibly have existed before it was occluded.

    Its own pixels, their mirror image across the character axis (characters are
    roughly symmetric, and the occluder usually hides the counterpart of a part
    that is visible on the other side), plus every concavity enclosed by that
    union.  A 1 px dilation bridges the soft mask's feathered edge.
    """
    support = own_bin | own_bin[:, mirror_src]
    support = support | _enclosed_holes(support)
    return _dilate(support, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# small array helpers
# --------------------------------------------------------------------------- #
def _rgba_array(image: Image.Image) -> np.ndarray:
    """``HxWx4`` float32 array (0..255) of an image, converting the mode if needed."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return np.asarray(image, dtype=np.float32)


def _mask_array(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """``HxW`` float32 mask in ``[0, 1]``; alpha channels are accepted as masks."""
    if mask.mode in ("RGBA", "LA"):
        mask = mask.getchannel("A")
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.BILINEAR)
    return np.clip(np.asarray(mask, dtype=np.float32) * (1.0 / 255.0), 0.0, 1.0)


def _to_image(arr: np.ndarray) -> Image.Image:
    """Float RGBA array (0..255) -> 8-bit RGBA image."""
    return Image.fromarray(np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8), "RGBA")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive-exclusive ``(x0, y0, x1, y1)`` of the mask's visible pixels."""
    ys, xs = np.nonzero(mask > SUPPORT_THRESHOLD)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _centroid_x(alpha: np.ndarray, width: int) -> float:
    """Alpha-weighted horizontal centroid of the character."""
    column = alpha.sum(axis=0)
    total = float(column.sum())
    if total <= 0.0:
        return (width - 1) * 0.5
    return float((column * np.arange(width, dtype=np.float64)).sum() / total)


def _dilate(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """4-connected binary dilation (``iterations`` >= 1 keeps the input untouched)."""
    out = binary
    for _ in range(max(1, int(iterations))):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _enclosed_holes(binary: np.ndarray) -> np.ndarray:
    """Background pixels fully enclosed by ``binary`` (4-connected flood fill).

    Runs on the mask's own bounding box, so the O(iterations) propagation stays
    cheap for a 240x240 canvas.
    """
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return np.zeros_like(binary)
    y0 = max(0, int(ys.min()) - 1)
    x0 = max(0, int(xs.min()) - 1)
    y1 = min(binary.shape[0], int(ys.max()) + 2)
    x1 = min(binary.shape[1], int(xs.max()) + 2)
    roi = binary[y0:y1, x0:x1]
    free = ~roi

    reach = np.zeros_like(free)
    reach[0, :] = free[0, :]
    reach[-1, :] = free[-1, :]
    reach[:, 0] = free[:, 0]
    reach[:, -1] = free[:, -1]
    for _ in range(roi.shape[0] + roi.shape[1]):
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if np.array_equal(grown, reach):
            break
        reach = grown

    holes = np.zeros_like(binary)
    holes[y0:y1, x0:x1] = free & ~reach
    return holes


def _bleed_colour(rgb: np.ndarray, alpha: np.ndarray, iterations: int = BLEED_ITERS) -> np.ndarray:
    """Copy the nearest silhouette colour a few pixels into the transparent area.

    Rotation resamples RGBA channels independently, so transparent padding (this
    demo art stores ``(2, 2, 2)`` there) bleeds dark fringes into every moving
    edge.  Only RGB is touched -- alpha stays exactly as authored.
    """
    out = rgb.copy()
    known = alpha > VISIBLE_EPS
    if known.all():
        return out
    for _ in range(max(0, int(iterations))):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(alpha)
        acc[1:, :] += out[:-1, :] * known[:-1, :, None]
        cnt[1:, :] += known[:-1, :]
        acc[:-1, :] += out[1:, :] * known[1:, :, None]
        cnt[:-1, :] += known[1:, :]
        acc[:, 1:] += out[:, :-1] * known[:, :-1, None]
        cnt[:, 1:] += known[:, :-1]
        acc[:, :-1] += out[:, 1:] * known[:, 1:, None]
        cnt[:, :-1] += known[:, 1:]
        upd = (~known) & (cnt > 0.0)
        if not upd.any():
            break
        out[upd] = acc[upd] / cnt[upd, None]
        known = known | upd
    return out
