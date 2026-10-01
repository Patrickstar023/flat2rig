"""Animate the part layers: joint rotation, back-to-front compositing, state frames.

The rig says *what* moves (degrees + offsets per frame); this module turns that
into pictures.  Three ideas carry the whole animation:

* :func:`rotate_about` -- a full-canvas layer rotated around an arbitrary joint
  with bicubic resampling, canvas size untouched;
* :func:`compose_frame` -- paint the layers back to front, optionally sealing
  each rotating joint with a small half-angle patch so the wedge a rotation
  opens at the pivot does not show;
* :func:`build_frames` -- expand ``Part.motions[state]`` into ``(image, delay)``
  frames, padding short motion lists and compositing the state decoration
  (blink / zzz / sweat / sparkle) on top of every frame.

Diagonstics: a part whose rotated bounding box no longer fits the canvas is
reported through :mod:`warnings` (``RuntimeWarning``) instead of being silently
clipped, and nothing is printed outside ``__main__`` blocks.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .parts import cut_layers, inpaint_hidden, motion_extent, order_parts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .rig import Rig

__all__ = ["rotate_about", "compose_frame", "overlay_eyes", "overlay_symbol", "build_frames"]

#: Canvas the overlays are designed for; other sizes are scaled proportionally.
CANVAS = 240
#: 关节补丁半径上限。旧值是 ``CANVAS // 8``（=30）；抽成常量是为了让"为什么是这个数"
#: 有地方可查，也方便 rig 用 ``seam`` 字段显式指定更大的补丁。
MAX_SEAM = 40
#: Colour of the source art's outline, reused by the eye / decoration overlays.
INK = (30, 27, 46, 235)
#: Sweat-drop fill.
DROP = (127, 166, 232, 235)
#: Sparkle colours.
GOLD = (242, 179, 43, 240)
CREAM = (255, 233, 166, 240)
#: Sleep "Z" colour.
ZED = (124, 140, 228, 240)
#: Default closed-eye frame duration (capped by the state's own delay).
BLINK_MS = 150
#: Default frame delay for a state missing from ``rig.states``.
FALLBACK_MS = 100
#: Angles below this count as "not moving" (avoids a needless resample).
EPS = 1e-3
#: Canonical state order; states found in the rig but not listed here follow it.
CANONICAL_STATES = ("idle", "sleep", "error", "celebrate")


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def rotate_about(layer: Image.Image, pivot: tuple[float, float], deg: float,
                 dy: float = 0.0, dx: float = 0.0) -> Image.Image:
    """Rotate a full-canvas RGBA layer around ``pivot``, then shift by ``(dx, dy)``.

    PIL rotates around the image centre, so this is the usual
    translate-pivot-rotate-translate-back -- except that ``Image.rotate`` accepts
    ``center=``, which does the same maths in a *single* bicubic resampling pass
    (no double interpolation, no rounding of the translation).  The canvas size
    is preserved and everything rotated outside it is clipped; use
    :func:`build_frames`, which warns when that would happen.

    ``deg`` follows PIL's convention: positive is counter-clockwise on screen.
    ``dy`` is positive downwards, ``dx`` positive to the right.
    """
    pivot = (float(pivot[0]), float(pivot[1]))
    deg = float(deg)
    dy = float(dy)
    dx = float(dx)
    if abs(deg) < EPS and abs(dy) < EPS and abs(dx) < EPS:
        return layer.copy()

    out = layer
    if abs(deg) >= EPS:
        out = out.rotate(deg, resample=Image.BICUBIC, center=pivot)
    if abs(dx) >= EPS or abs(dy) >= EPS:
        whole = abs(dx - round(dx)) < EPS and abs(dy - round(dy)) < EPS
        out = out.transform(
            out.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, -dx, 0.0, 1.0, -dy),  # output -> input, i.e. content moves by (+dx, +dy)
            resample=Image.NEAREST if whole else Image.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
    return out


def compose_frame(layers: dict[str, Image.Image], order: list[str],
                  moves: dict[str, tuple[float, float, float]],
                  joints: dict[str, tuple[float, float]],
                  seam: int = 0,
                  base: Image.Image | None = None) -> Image.Image:
    """Paint ``order`` back to front, each layer transformed by ``moves``.

    ``moves[name]`` is ``(deg, dy, dx)``; missing names are drawn untransformed.
    ``joints[name]`` is the pivot that part rotates about (missing names pivot
    around the canvas centre).

    If ``seam > 0`` a rotating part additionally gets a small circular *joint
    patch*: the part's own pixels within ``seam`` px of the pivot, feathered at
    the rim and rotated by **half** the part's angle.  It is composited just
    *under* the part, so it covers the wedge the full rotation opens at the
    joint while the moved part stays crisp on top.  Patches are only used for
    real rotations -- a pure translation cannot open a wedge, and patching it
    would only smear the drawing.

    Missing layers are skipped with a warning (a partially matched layer stack
    should still produce frames), which leaves the canvas transparent there.

    ``base`` (optional) is the original artwork, composited **first — underneath every
    part**.  It closes the joint gap a rotating part can leave behind: the base has real
    content exactly there (it is what the part was covering), and because it sits
    underneath, the moved part still covers it wherever the part now lies.  Painting the
    same pixels *on top* instead would put the original outline next to the rotated one
    and produce a visible double edge — which is what it looked like when the repair ran
    after composition.
    """
    if not order:
        raise ValueError("order must contain at least one part name")
    size = _canvas_size(layers, order)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))

    # 底图只补"部件没盖到"的位置（见 base 的说明）。
    # 不能无条件垫底：部件自己也带 alpha，叠上去会让第 0 帧比原画更实
    # （原画是柔边画风，重复叠加会改变观感）。
    base_fill: Image.Image | None = None
    if base is not None:
        base_rgba = _fit(base if base.mode == "RGBA" else base.convert("RGBA"), size)
        probe = Image.new("RGBA", size, (0, 0, 0, 0))
        for name in order:
            layer = layers.get(name)
            if layer is None:
                continue
            deg, dy, dx = _triple(moves.get(name, (0.0, 0.0, 0.0)))
            p = joints.get(name, (size[0] * 0.5, size[1] * 0.5))
            probe.alpha_composite(rotate_about(_fit(layer, size), (float(p[0]), float(p[1])), deg, dy, dx))
        pa = np.asarray(probe.getchannel("A"), dtype=np.uint8)
        gap = np.where(pa <= 2, 255, 0).astype(np.uint8)
        if gap.any():
            base_fill = base_rgba.copy()
            base_fill.putalpha(Image.fromarray(
                (np.asarray(base_rgba.getchannel("A"), dtype=np.uint16)
                 * np.asarray(Image.fromarray(gap), dtype=np.uint16) // 255).astype(np.uint8)))
        # 垫在所有部件之下，只出现在"部件没盖到"的位置
        if base_fill is not None:
            canvas.alpha_composite(base_fill)

    for name in order:
        layer = layers.get(name)
        if layer is None:
            warnings.warn(f"compose_frame: no layer for part {name!r}; skipped",
                          RuntimeWarning, stacklevel=2)
            continue
        layer = _fit(layer, size)
        deg, dy, dx = _triple(moves.get(name, (0.0, 0.0, 0.0)))
        pivot_raw = joints.get(name, (size[0] * 0.5, size[1] * 0.5))
        pivot = (float(pivot_raw[0]), float(pivot_raw[1]))

        if seam and abs(deg) > 0.5:
            patch = _joint_patch(layer, pivot, int(seam), deg * 0.5, dy * 0.5, dx * 0.5)
            if patch is not None:
                canvas.alpha_composite(patch)
        canvas.alpha_composite(rotate_about(layer, pivot, deg, dy, dx))
    return canvas


# --------------------------------------------------------------------------- #
# overlays
# --------------------------------------------------------------------------- #
def overlay_eyes(eyes: list[tuple[float, float, float]], droop: bool = False,
                 colour: tuple[int, int, int, int] = INK) -> Image.Image:
    """Closed-eye arcs for blink and sleep, as a transparent overlay layer.

    ``eyes`` is ``[(cx, cy, half), ...]`` in the character's canvas coordinates.
    The overlay is square and at least :data:`CANVAS` px, growing if an eye sits
    further out; callers compositing onto a differently sized frame should paste
    it at ``(0, 0)`` (as :func:`build_frames` does).  ``droop=True`` draws the
    flatter, lower arcs used while asleep.
    """
    size = _overlay_canvas(eyes)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = size / float(CANVAS)
    width = max(2, int(round(5 * scale)))
    start, end = (190, 350) if droop else (200, 340)
    for eye in eyes:
        cx, cy, half = float(eye[0]), float(eye[1]), float(eye[2])
        radius = max(4.0, half * (1.25 if droop else 1.10))
        offset = (0.30 if droop else 0.10) * max(half, 3.0)
        draw.arc([cx - radius, cy + offset - radius, cx + radius, cy + offset + radius],
                 start=start, end=end, fill=tuple(colour), width=width)
    return layer


def overlay_symbol(kind: str, step: int, size: int = CANVAS) -> Image.Image:
    """A small procedural decoration for one animation state.

    ``kind`` is ``"zzz"`` (sleep), ``"sweat"`` (error) or ``"sparkle"``
    (celebrate); ``step`` is the frame index, which cycles the animation phase.
    Everything is drawn inside a ``size`` x ``size`` transparent canvas and every
    glyph is clamped to stay there, so positions never leave the frame.
    """
    if kind not in ("zzz", "sweat", "sparkle"):
        raise ValueError(f"unknown overlay symbol {kind!r}; expected 'zzz', 'sweat' or 'sparkle'")
    size = max(8, int(size))
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = size / float(CANVAS)
    step = int(step)

    if kind == "zzz":
        for index, (x, y, glyph) in enumerate(((150, 96, 17), (176, 66, 23), (198, 32, 29))):
            phase = (step - index) % 3
            if phase == 0:
                continue  # each "Z" is dark for one frame of the cycle
            alpha = 255 if phase == 2 else 150
            gx, gy = _clamp_box(x * scale, y * scale, glyph * scale, size)
            _draw_z(draw, gx, gy, glyph * scale, (ZED[0], ZED[1], ZED[2], alpha), scale)
    elif kind == "sweat":
        for index, (x, y, radius) in enumerate(((196, 74, 7), (212, 96, 5))):
            bob = 0.0 if (step + index) % 2 == 0 else 4.0 * scale
            sx, sy = _clamp_box(x * scale, y * scale + bob, radius * scale, size)
            _draw_drop(draw, sx, sy, radius * scale, scale)
    else:  # sparkle
        pulse = (0.80, 1.00, 0.85)[step % 3]
        for x, y, radius, colour in ((38, 60, 16, GOLD), (202, 92, 13, CREAM)):
            sx, sy = _clamp_box(x * scale, y * scale, radius * scale * pulse, size)
            _draw_star(draw, sx, sy, radius * scale * pulse, colour, scale)
    return layer


# --------------------------------------------------------------------------- #
# state -> frames
# --------------------------------------------------------------------------- #
def build_frames(rgba: Image.Image, rig: "Rig", masks: dict[str, Image.Image],
                 layers: dict[str, Image.Image] | None = None,
                 states: list[str] | None = None,
                 seam: int | None = None,
                 repair_holes: bool = True) -> dict[str, list[tuple[Image.Image, int]]]:
    """Expand every state into ``[(frame, delay_ms), ...]``.

    ``layers`` defaults to :func:`flat2rig.parts.inpaint_hidden`, i.e. the
    occlusion-completed layers; pass an explicit dict (for example a see-through
    layer stack) to override.

    ``states`` defaults to the states present in ``rig.states`` (canonical
    ``idle / sleep / error / celebrate`` first).  A state no part has motions for
    is skipped.  Frame count is the longest motion list of that state; a part
    with a shorter list repeats its last entry.

    ``seam`` is the joint-patch radius forwarded to :func:`compose_frame`
    (``None`` derives it from the parts' ``blend``).  ``repair_holes`` fills pixels that
    compose as fully transparent although the base artwork has content there — the joint
    gap a rotating part can leave behind.  Decorations are composited
    last, on top of everything: a blink frame is appended to ``idle`` when the
    rig has eyes, ``sleep`` gets closed eyes plus "zzz", ``error`` gets sweat and
    ``celebrate`` gets sparkles.  A part whose rotated bounds would leave the
    canvas raises a ``RuntimeWarning`` (the frames themselves are still 240x240
    and clipped, exactly like the source art is at the border).
    """
    parts = list(rig.parts)
    order = order_parts(rig)
    pivots = {p.name: (float(p.pivot[0]), float(p.pivot[1])) for p in parts}
    if layers is None:
        # 补全范围要按每个部位实际的动作幅度来算（见 parts.motion_extent）
        layers = inpaint_hidden(rgba, masks, order,
                                angles=motion_extent(rig), pivots=pivots)
    size = rgba.size
    if seam is None:
        seam = _default_seam(parts, getattr(rig, "seam", None))

    frames: dict[str, list[tuple[Image.Image, int]]] = {}
    for state in _select_states(rig, states):
        sequences = [(part, list(part.motions.get(state) or [])) for part in parts]
        count = max((len(seq) for _, seq in sequences), default=0)
        if count == 0:
            continue  # no part says anything about this state
        delay = int(rig.states.get(state, FALLBACK_MS))

        frame_moves: list[dict[str, tuple[float, float, float]]] = []
        for index in range(count):
            moves: dict[str, tuple[float, float, float]] = {}
            for part, sequence in sequences:
                if not sequence:
                    continue
                moves[part.name] = _triple(sequence[min(index, len(sequence) - 1)])
            frame_moves.append(moves)

        clip_reported: set[str] = set()
        for index, moves in enumerate(frame_moves):
            for name, move in moves.items():
                if name in clip_reported or name not in layers:
                    continue
                message = _clipping_warning(name, state, index, layers[name], pivots.get(name), move, size)
                if message:
                    clip_reported.add(name)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)

        sequence_frames: list[tuple[Image.Image, int]] = []
        for index, moves in enumerate(frame_moves):
            frame = compose_frame(layers, order, moves, pivots, seam=seam)
            # 补缝：部件转开后，接缝处可能只剩下另一半的权重，背景就透了进来
            # （表现为"头裂开"）。用底图把**完全透明**的像素补上——底图在那里本来
            # 就有内容，补上是"把被遮挡的部分露出来"，而不是凭空造像素。
            # 只补 alpha≈0 的像素，半透明的柔边不动，所以不会把画面糊掉。
            if repair_holes:
                frame = _fill_transparent(frame, rgba)
            overlay = _state_overlay(rig, state, index, size)
            if overlay is not None:
                frame.alpha_composite(overlay)
            sequence_frames.append((frame, delay))

        # 说明：这里**不再**往 idle 首帧叠加闭眼弧。
        # 曾经的做法是第一帧叠加 overlay_eyes，但原画本身已经画好了眼睛，
        # 再叠一层深色弧线会表现为"眼睛变了、整体发脏"（用户实际反馈）。
        # 需要眨眼时请在标注里显式提供 eyes，并自行确认叠加效果；默认不动原画。

        frames[state] = sequence_frames
    return frames


# --------------------------------------------------------------------------- #
# frame helpers
# --------------------------------------------------------------------------- #
def _fill_transparent(frame: Image.Image, base: Image.Image) -> Image.Image:
    """Fill fully transparent pixels of ``frame`` from ``base``.

    When a rigid part rotates away, the feathered boundary can leave the neighbour with
    only part of the joint's pixels, so the background shows through along the joint ("the
    head splits apart").  The base artwork has real content exactly there — it is what the
    moving part was covering — so copying it in reveals hidden pixels rather than inventing
    them.

    Only ``alpha <= 2`` pixels are touched: the soft edges stay as they are, so the frame
    keeps its antialiasing, and the repair cannot smear the drawing.  Pixels where ``base``
    is also transparent are left alone (nothing to reveal).
    """
    out = frame if frame.mode == "RGBA" else frame.convert("RGBA")
    src = base if base.mode == "RGBA" else base.convert("RGBA")
    if out.size != src.size:
        src = src.resize(out.size, Image.LANCZOS)
    fa = np.asarray(out.getchannel("A"), dtype=np.uint8)
    sa = np.asarray(src.getchannel("A"), dtype=np.uint8)
    hole = (fa <= 2) & (sa > 0)
    if not hole.any():
        return out
    fixed = out.copy()
    fixed.paste(src, (0, 0), Image.fromarray(np.where(hole, 255, 0).astype(np.uint8)))
    return fixed


def _state_overlay(rig: "Rig", state: str, step: int, size: tuple[int, int]) -> Image.Image | None:
    """Decoration composited on top of every frame of ``state`` (or ``None``)."""
    symbol_size = max(size)
    if state == "sleep":
        # 只放 Z，不画闭眼：原画本身只有"睁眼"一种眼型，程序画的闭眼弧叠上去
        # 就是一团深色（用户反馈"眼睛变了、发脏"）。宁可让眼睛保持原样。
        return _fit(overlay_symbol("zzz", step, symbol_size), size)
    if state == "error":
        return _fit(overlay_symbol("sweat", step, symbol_size), size)
    if state == "celebrate":
        return _fit(overlay_symbol("sparkle", step, symbol_size), size)
    return None


def _joint_patch(layer: Image.Image, pivot: tuple[float, float], radius: int,
                 deg: float, dy: float, dx: float) -> Image.Image | None:
    """Feathered disc of ``layer`` around ``pivot``, rotated by the half-angle."""
    if radius <= 0:
        return None
    width, height = layer.size
    x0 = int(math.floor(pivot[0] - radius))
    y0 = int(math.floor(pivot[1] - radius))
    side = 2 * radius
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    piece = np.asarray(layer.crop((x0, y0, x0 + side, y0 + side)), dtype=np.float32)

    yy, xx = np.mgrid[0:side, 0:side]
    dist = np.hypot(x0 + xx + 0.5 - pivot[0], y0 + yy + 0.5 - pivot[1])
    piece[..., 3] *= np.clip((radius - dist) / max(1.0, radius * 0.45), 0.0, 1.0)

    patch = np.zeros((height, width, 4), dtype=np.float32)
    patch[sy0:sy1, sx0:sx1] = piece[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    if not patch[..., 3].any():
        return None
    return rotate_about(Image.fromarray(np.clip(np.rint(patch), 0.0, 255.0).astype(np.uint8), "RGBA"),
                        pivot, deg, dy, dx)


def _clipping_warning(name: str, state: str, index: int, layer: Image.Image,
                      pivot: tuple[float, float] | None, move: tuple[float, float, float],
                      size: tuple[int, int]) -> str | None:
    """Message when ``layer``'s rotated bounds leave the canvas, else ``None``."""
    if pivot is None:
        return None
    box = layer.getchannel("A").getbbox()
    if box is None:
        return None
    deg, dy, dx = move
    if abs(deg) < EPS and abs(dy) < EPS and abs(dx) < EPS:
        return None
    theta = math.radians(deg)
    cos, sin = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for px, py in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        rx, ry = px - pivot[0], py - pivot[1]
        # PIL's rotate(deg) turns content counter-clockwise on screen (y grows down).
        xs.append(pivot[0] + rx * cos + ry * sin + dx)
        ys.append(pivot[1] - rx * sin + ry * cos + dy)
    over = (
        min(xs) < -0.5 or min(ys) < -0.5
        or max(xs) > size[0] + 0.5 or max(ys) > size[1] + 0.5
    )
    if not over:
        return None
    return (f"{name}: rotated bounds [{min(xs):.1f},{min(ys):.1f},{max(xs):.1f},{max(ys):.1f}] "
            f"leave the {size[0]}x{size[1]} canvas in state {state!r} frame {index + 1}; "
            f"the part is clipped there -- add canvas margin or reduce the motion")


def _default_seam(parts: Iterable[object], explicit: int | None = None) -> int:
    """Joint-patch radius from the rig's ``seam`` field, else from ``blend``.

    A rig can set ``"seam": N`` to name the radius directly.  That matters when a part
    turns far: the wedge a rotation opens grows with the angle and with the distance
    from the pivot to the joint boundary, so a 20 deg ear rotation needs a wider patch
    than the 8-10 px that ``blend`` implies — otherwise a sliver of background opens
    between the part and whatever sits behind it ("the head splits apart").
    """
    if explicit and explicit > 0:
        return max(0, min(int(explicit), MAX_SEAM))
    blends = [float(getattr(part, "blend", 0.0) or 0.0) for part in parts]
    blends = [b for b in blends if b > 0.0]
    if not blends:
        return 0
    blends.sort()
    radius = int(round(blends[len(blends) // 2]))
    return max(0, min(radius, MAX_SEAM))


def _select_states(rig: "Rig", states: list[str] | None) -> list[str]:
    """States to expand: explicit request, else the rig's (canonical order first)."""
    if states is not None:
        return [str(s) for s in states]
    present = list(getattr(rig, "states", {}) or {})
    ordered = [s for s in CANONICAL_STATES if s in present]
    ordered += [s for s in present if s not in CANONICAL_STATES]
    return ordered


def _motion_triple(entry: Sequence[float]) -> tuple[float, float, float]:
    """``[deg, dy]`` (or ``[deg, dy, dx]``) -> ``(deg, dy, dx)``."""
    values = list(entry)
    deg = float(values[0]) if len(values) > 0 else 0.0
    dy = float(values[1]) if len(values) > 1 else 0.0
    dx = float(values[2]) if len(values) > 2 else 0.0
    return deg, dy, dx


_triple = _motion_triple  # internal shorthand


def _canvas_size(layers: Mapping[str, Image.Image], order: Sequence[str]) -> tuple[int, int]:
    for name in order:
        layer = layers.get(name)
        if layer is not None:
            return layer.size
    for layer in layers.values():
        return layer.size
    raise ValueError("compose_frame: no layers given")


def _fit(layer: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop/pad a layer to ``size`` at the top-left origin (overlays, odd stacks)."""
    if layer.size == size:
        return layer
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(layer if layer.mode == "RGBA" else layer.convert("RGBA"), (0, 0))
    return out


# --------------------------------------------------------------------------- #
# glyph helpers
# --------------------------------------------------------------------------- #
def _draw_z(draw: ImageDraw.ImageDraw, x: float, y: float, glyph: float,
            colour: tuple[int, int, int, int], scale: float) -> None:
    """One "Z": top bar, diagonal, bottom bar."""
    width = max(2, int(round(glyph / 5.0)))
    draw.line([(x, y), (x + glyph, y)], fill=colour, width=width)
    draw.line([(x + glyph, y), (x, y + glyph)], fill=colour, width=width)
    draw.line([(x, y + glyph), (x + glyph, y + glyph)], fill=colour, width=width)


def _draw_drop(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float,
               scale: float) -> None:
    """One sweat drop: a triangular tip over a round belly."""
    outline = tuple(INK)
    draw.polygon([(x, y - radius * 1.6), (x + radius, y + radius * 0.6),
                  (x - radius, y + radius * 0.6)], fill=tuple(DROP), outline=outline)
    draw.ellipse([x - radius, y - radius * 0.6, x + radius, y + radius * 1.4],
                 fill=tuple(DROP), outline=outline)


def _draw_star(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float,
               colour: tuple[int, int, int, int], scale: float) -> None:
    """One four-point sparkle (8 alternating vertices)."""
    points = []
    for index in range(8):
        angle = math.pi / 2.0 * (index / 2.0)
        length = radius if index % 2 == 0 else radius * 0.30
        points.append((x + length * math.cos(angle), y - length * math.sin(angle)))
    draw.polygon(points, fill=tuple(colour), outline=tuple(INK))


def _clamp_box(cx: float, cy: float, half: float, size: int) -> tuple[float, float]:
    """Keep a square glyph of half-extent ``half`` inside a ``size`` canvas."""
    low = 2.0 + half
    high = max(low, size - 2.0 - half)
    return min(max(cx, low), high), min(max(cy, low), high)


def _overlay_canvas(eyes: Sequence[Sequence[float]]) -> int:
    """Overlay size: :data:`CANVAS`, grown until every eye (plus its arc) fits."""
    size = CANVAS
    for eye in eyes:
        cx, cy, half = float(eye[0]), float(eye[1]), float(eye[2])
        size = max(size, int(math.ceil(max(cx, cy) + 2.5 * max(half, 1.0) + 8.0)))
    return size
