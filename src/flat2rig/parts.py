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
#: Jacobi iterations for the diffusion fill (200 is plenty at 240 px).
DIFFUSION_ITERS = 200
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


def inpaint_hidden(rgba: Image.Image, masks: dict[str, Image.Image], order: list[str],
                   radius: int = 6) -> dict[str, Image.Image]:
    """Return :func:`cut_layers` outputs extended by each part's hidden pixels.

    ``order`` is the paint order, back to front (see :func:`order_parts`).
    For part ``P`` at index ``i`` the region it must additionally cover is the
    union of the masks of every part painted *in front of* it (index ``> i``)
    whose bounding box overlaps ``P``'s own bounding box.  That region is
    intersected with a plausible silhouette for ``P`` -- its own pixels, its
    mirror image across the character's vertical axis, and any concavity fully
    enclosed by either -- so a part never grows a blob where it never existed
    simply because a foreground part happened to overhang the outline.

    The missing pixels are then reconstructed:

    1. mirror symmetry -- reflect across the vertical axis through the
       character's alpha centroid and sample the part's own visible pixels;
    2. iterative diffusion -- remaining pixels are filled with the
       alpha-weighted average of their already-known 8/4-neighbours
       (:data:`DIFFUSION_ITERS` Jacobi sweeps, bounded to the fill bounding box);
    3. soft falloff -- the reconstruction is cross-faded over ``radius`` pixels
       of the part's real pixels around the hole, and its alpha follows the mask
       feather, so no seam appears when the front part moves away.

    ``radius`` is the width in pixels of that cross-fade band (and of the
    dilation used around the hole); larger is smoother and slightly slower.

    Every name in ``masks`` gets an entry.  Names absent from ``order`` are
    appended at the front (they cannot occlude anything the caller described).
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
        # 被遮挡区域按定义位于本部位"可见像素之外"，因此不能要求它落在自身轮廓内
        # （否则补全会被全部否决）。合理判据：从可见像素出发，沿着被前景覆盖的区域
        # 做连通生长——只补"前景确实压在上面、且与本体相连"的地方。
        grown = _grow_into_front(own_bin, front)
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
def _grow_into_front(own_bin: np.ndarray, front: np.ndarray,
                     max_iter: int = 4096) -> np.ndarray:
    """从本部位的可见像素出发，在"被前景覆盖"的区域里做四邻连通生长。

    被遮挡的像素按定义不在 ``own_bin`` 内，所以"补全范围必须落在自身轮廓内"这类约束
    会把补全全部否决。正确判据是沿 ``front`` 连通生长：既不越出前景覆盖范围，也不会
    凭空长出与本体不相连的大块。
    """
    allowed = front > 0 if front.dtype != np.bool_ else front
    grown = np.zeros_like(own_bin, dtype=bool)
    frontier = own_bin & allowed
    grown |= frontier
    for _ in range(max_iter):
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

    # 3. mirror-symmetric guess, where the reflected pixel is part of this part.
    mirrored_own = own[:, mirror_src][sl]
    mir = dom & (mirrored_own > VISIBLE_EPS)
    if mir.any():
        val[mir] = rgb[:, mirror_src][sl][mir]
        wgt[mir] = 1.0

    # 4. diffusion for everything the mirror could not supply.
    unknown = dom & (wgt <= 0.0)
    for _ in range(DIFFUSION_ITERS):
        if not unknown.any():
            break
        acc = np.zeros_like(val)
        accw = np.zeros_like(wgt)
        acc[1:, :] += val[:-1, :] * wgt[:-1, :, None]
        accw[1:, :] += wgt[:-1, :]
        acc[:-1, :] += val[1:, :] * wgt[1:, :, None]
        accw[:-1, :] += wgt[1:, :]
        acc[:, 1:] += val[:, :-1] * wgt[:, :-1, None]
        accw[:, 1:] += wgt[:, :-1]
        acc[:, :-1] += val[:, 1:] * wgt[:, 1:, None]
        accw[:, :-1] += wgt[:, 1:]
        upd = unknown & (accw > 0.0)
        if not upd.any():
            break  # hole is not connected to any known pixel
        val[upd] = acc[upd] / accw[upd, None]
        wgt[upd] = 1.0
        unknown = unknown & ~upd

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

    # 6. composite the reconstruction into the layer.
    out_rgb = base[..., :3][sl]
    out_a = base[..., 3][sl]
    touch = dom & (fall > 0.0)
    if touch.any():
        w = fall[touch][:, None]
        out_rgb[touch] = val[touch] * w + out_rgb[touch] * (1.0 - w)
        out_a[touch] = np.maximum(out_a[touch], src_alpha[sl][touch] * occ[sl][touch] * fall[touch])
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
