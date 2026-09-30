"""Skeleton-guided part masking — flat2rig pipeline stages 1-2.

A small annotation (a few bone segments per named part) plus the character's alpha
channel are turned into one soft mask per part:

1. :func:`assign_parts` labels every opaque pixel with the part whose bone segment is
   nearest.  The search is a multi-source wavefront (chamfer 3-4 distance transform)
   over the pixel grid, seeded from the rasterised bones; ties go to the part with the
   lower ``z`` so a background part never loses pixels to a foreground one.
2. :func:`soften_masks` feathers every mask boundary with a Gaussian transition band and
   renormalises across parts, so the masks still sum back to the character alpha.

Everything is deterministic, needs only numpy + Pillow (scipy is used when installed)
and runs in milliseconds for typical sprite sizes.

No output is produced outside :func:`_main`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # scipy is optional: flat2rig's base install is numpy + Pillow only.
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
except ImportError:  # pragma: no cover - only hit in a scipy-free install
    _scipy_gaussian = None

__all__ = [
    "Bone",
    "Part",
    "Rig",
    "load_rig",
    "load_alpha",
    "assign_parts",
    "soften_masks",
    "build_masks",
    "diagnose",
    "DEFAULT_BLEND",
]

#: Transition half-width (px) used when a part does not specify ``blend``.
DEFAULT_BLEND = 8.0

#: Chamfer 3-4 weights: orthogonal neighbour costs 3, diagonal neighbour costs 4.
_CHAMFER_ORTH = 3
_CHAMFER_DIAG = 4

#: Sentinel for "not reached yet"; real keys stay far below :data:`_FINITE_LIMIT`.
_INF = 1 << 50
_FINITE_LIMIT = 1 << 40


# --------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------


@dataclass
class Bone:
    """A straight bone segment in canvas pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def length(self) -> float:
        """Euclidean length of the segment in pixels."""
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def clamped(self, width: int, height: int) -> "Bone":
        """Return a copy whose endpoints lie inside a ``width`` x ``height`` canvas."""
        max_x = float(max(width - 1, 0))
        max_y = float(max(height - 1, 0))
        return Bone(
            min(max(self.x0, 0.0), max_x),
            min(max(self.y0, 0.0), max_y),
            min(max(self.x1, 0.0), max_x),
            min(max(self.y1, 0.0), max_y),
        )


@dataclass
class Part:
    """One animated part: its bones, its pivot joint and its motion table.

    Attributes:
        name: unique part name (also the mask key); duplicates are merged.
        bones: segments used to label pixels; an empty list yields an empty mask.
        pivot: joint ``(x, y)`` the part rotates around.
        z: paint order, higher = in front; also the tie-break priority for labeling.
        blend: half-width (px) of the soft transition band around this part's boundary.
        motions: ``state -> [[rotation_deg, dy_px], ...]``, one entry per frame.
    """

    name: str
    bones: list[Bone]
    pivot: tuple[float, float]
    z: int = 0
    blend: float = DEFAULT_BLEND
    motions: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


@dataclass
class Rig:
    """A parsed annotation: canvas size, parts, eye anchors and state timings."""

    size: tuple[int, int]
    parts: list[Part]
    eyes: list[tuple[float, float, float]] = field(default_factory=list)
    states: dict[str, int] = field(default_factory=dict)
    body: tuple[float, float] | None = None

    def part(self, name: str) -> Part | None:
        """Return the first part called ``name``, or ``None``."""
        for part in self.parts:
            if part.name == name:
                return part
        return None

    def paint_order(self) -> list[Part]:
        """Parts sorted back-to-front (ascending ``z``, stable on declaration order)."""
        return sorted(self.parts, key=lambda part: part.z)


# --------------------------------------------------------------------------------------
# annotation loading
# --------------------------------------------------------------------------------------


def load_rig(path: str | Path) -> Rig:
    """Parse an annotation JSON file into a :class:`Rig`.

    The canonical schema is the one documented in the README::

        {"canvas": [240, 240], "body": [120, 216],
         "parts": [{"name": "ear", "bones": [[150, 60, 120, 118]], "pivot": [120, 118],
                    "z": 2, "blend": 10, "motions": {"idle": [[0, 0], [8, -1]]}}],
         "eyes": [[93, 99, 9]], "states": {"idle": 520}}

    A few tolerant extensions are accepted (none of them is required):

    * a character-keyed wrapper ``{"<name>": {"parts": [...]}}`` — the exporter format of
      ``whale-pet-redesign/tools/rig-config.json``; when several characters are present
      the one whose key appears in the file name wins, otherwise the first one;
    * ``region`` instead of ``bones`` (``ellipse`` / ``rect`` / ``rest``): an ellipse or
      rectangle becomes a bone running from its centre towards the pivot, and a ``rest``
      region a bone from the pivot towards the canvas centre.  ``bones`` always wins;
    * a missing ``pivot`` falls back to the first bone's far end (the README convention);
    * a missing ``canvas`` is inferred from the annotation's bounding box.

    Bone coordinates are clamped to the canvas; pivots are kept verbatim, because an
    animation may legitimately pivot around a point outside the canvas.

    Raises:
        ValueError: if the file is not a flat2rig annotation or a field is malformed.
    """
    annotation_path = Path(path)
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{annotation_path}: annotation must be a JSON object")
    doc = _select_character(raw, annotation_path)

    parts_raw = doc.get("parts", [])
    if not isinstance(parts_raw, list):
        raise ValueError(f"{annotation_path}: 'parts' must be a list")

    pending: list[tuple[str, list[Bone], Any, tuple[float, float], int, float, dict[str, list[tuple[float, float]]]]] = []
    for index, entry in enumerate(parts_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{annotation_path}: parts[{index}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{annotation_path}: parts[{index}] needs a non-empty 'name'")
        where = f"{annotation_path}: part '{name}'"
        bones = _parse_bones(entry.get("bones"), where)
        pivot_raw = entry.get("pivot")
        if pivot_raw is None:
            pivot = (bones[-1].x1, bones[-1].y1) if bones else (0.0, 0.0)
        else:
            pivot = _as_pair(pivot_raw, f"{where} pivot")
        z = int(round(_as_float(entry.get("z", 0), f"{where} z")))
        blend = max(_as_float(entry.get("blend", DEFAULT_BLEND), f"{where} blend"), 0.0)
        motions = _parse_motions(entry.get("motions"), where)
        pending.append((name, bones, entry.get("region"), pivot, z, blend, motions))

    eyes = _parse_eyes(doc.get("eyes"), annotation_path)
    states = _parse_states(doc.get("states"), annotation_path)
    body_raw = doc.get("body")
    body = None if body_raw is None else _as_pair(body_raw, f"{annotation_path}: body")

    size = _parse_canvas(doc.get("canvas"), annotation_path)
    if size is None:
        size = _infer_canvas(pending, eyes, body)
    width, height = size

    parts: list[Part] = []
    for name, bones, region, pivot, z, blend, motions in pending:
        if not bones:
            bones = _bones_from_region(region, pivot, size)
        parts.append(
            Part(
                name=name,
                bones=[bone.clamped(width, height) for bone in bones],
                pivot=pivot,
                z=z,
                blend=blend,
                motions=motions,
            )
        )
    return Rig(size=size, parts=parts, eyes=eyes, states=states, body=body)


def load_alpha(image_path: str | Path) -> tuple[Image.Image, Image.Image]:
    """Open an image and return ``(rgba_image, alpha_mask)``.

    The image is converted to RGBA, so an input without an alpha channel (flat JPEG or
    PNG) is treated as fully opaque and its alpha mask is 255 everywhere.  The mask keeps
    the original alpha values, so anti-aliased edges stay anti-aliased; the "is character"
    test used by the mask builders is ``alpha > 0``.

    Raises:
        OSError: if the file cannot be opened or decoded by Pillow.
    """
    with Image.open(image_path) as opened:
        rgba = opened.convert("RGBA")
        rgba.load()
    return rgba, rgba.getchannel("A")


def _as_float(value: Any, what: str) -> float:
    """Coerce ``value`` to a finite float or raise :class:`ValueError`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def _as_pair(value: Any, what: str) -> tuple[float, float]:
    """Coerce ``value`` to a pair of finite floats."""
    if isinstance(value, Mapping):
        return (_as_float(value.get("x"), f"{what} x"), _as_float(value.get("y"), f"{what} y"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{what} must be [x, y], got {value!r}")
    return (_as_float(value[0], f"{what} x"), _as_float(value[1], f"{what} y"))


def _as_triple(value: Any, what: str) -> tuple[float, float, float]:
    """Coerce ``value`` to a triple of finite floats."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{what} must be [a, b, c], got {value!r}")
    return (
        _as_float(value[0], f"{what} a"),
        _as_float(value[1], f"{what} b"),
        _as_float(value[2], f"{what} c"),
    )


def _as_bone(value: Any, what: str) -> Bone:
    """Coerce one of the accepted bone spellings to a :class:`Bone`."""
    if isinstance(value, Mapping):
        return Bone(
            _as_float(value.get("x0"), f"{what} x0"),
            _as_float(value.get("y0"), f"{what} y0"),
            _as_float(value.get("x1"), f"{what} x1"),
            _as_float(value.get("y1"), f"{what} y1"),
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{what} must be a segment, got {value!r}")
    if len(value) == 2 and all(
        isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) == 2
        for point in value
    ):
        start = _as_pair(value[0], f"{what} start")
        end = _as_pair(value[1], f"{what} end")
        return Bone(start[0], start[1], end[0], end[1])
    if len(value) == 4:
        return Bone(
            _as_float(value[0], f"{what} x0"),
            _as_float(value[1], f"{what} y0"),
            _as_float(value[2], f"{what} x1"),
            _as_float(value[3], f"{what} y1"),
        )
    raise ValueError(f"{what} must be [x0, y0, x1, y1] or [[x0, y0], [x1, y1]], got {value!r}")


def _parse_bones(value: Any, where: str) -> list[Bone]:
    """Parse a ``bones`` field (a list of segments); missing means "no bones"."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where} 'bones' must be a list of segments")
    return [_as_bone(item, f"{where} bone[{index}]") for index, item in enumerate(value)]


def _bones_from_region(
    region: Any, pivot: tuple[float, float], size: tuple[int, int] | None
) -> list[Bone]:
    """Derive a bone from an ``ellipse`` / ``rect`` / ``rest`` region (exporter format)."""
    if not isinstance(region, Mapping):
        return []
    kind = str(region.get("type", "")).lower()
    if kind == "ellipse":
        centre = (_as_float(region.get("cx"), "region cx"), _as_float(region.get("cy"), "region cy"))
    elif kind == "rect":
        centre = (
            (_as_float(region.get("x0"), "region x0") + _as_float(region.get("x1"), "region x1")) / 2.0,
            (_as_float(region.get("y0"), "region y0") + _as_float(region.get("y1"), "region y1")) / 2.0,
        )
    else:
        if size is None:
            return []
        canvas_centre = (size[0] / 2.0, size[1] / 2.0)
        return [
            Bone(
                pivot[0],
                pivot[1],
                pivot[0] + (canvas_centre[0] - pivot[0]) / 2.0,
                pivot[1] + (canvas_centre[1] - pivot[1]) / 2.0,
            )
        ]
    return [Bone(centre[0], centre[1], pivot[0], pivot[1])]


def _parse_motions(value: Any, where: str) -> dict[str, list[tuple[float, float]]]:
    """Parse ``motions: {state: [[deg, dy], ...]}``."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} 'motions' must be an object of state -> frames")
    motions: dict[str, list[tuple[float, float]]] = {}
    for state, frames in value.items():
        if not isinstance(frames, list):
            raise ValueError(f"{where} motions[{state!r}] must be a list of [deg, dy] frames")
        parsed: list[tuple[float, float]] = []
        for index, frame in enumerate(frames):
            parsed.append(_as_pair(frame, f"{where} motions[{state!r}][{index}]"))
        motions[str(state)] = parsed
    return motions


def _parse_eyes(value: Any, path: Path) -> list[tuple[float, float, float]]:
    """Parse the optional ``eyes: [[cx, cy, half], ...]`` field."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: 'eyes' must be a list")
    return [_as_triple(item, f"{path}: eyes[{index}]") for index, item in enumerate(value)]


def _parse_states(value: Any, path: Path) -> dict[str, int]:
    """Parse the optional ``states: {state: frame_delay_ms}`` field."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: 'states' must be an object of state -> delay in ms")
    return {str(state): int(round(_as_float(delay, f"{path}: states[{state!r}]"))) for state, delay in value.items()}


def _parse_canvas(value: Any, path: Path) -> tuple[int, int] | None:
    """Parse ``canvas: [w, h]``; ``None`` when the field is absent."""
    if value is None:
        return None
    width, height = _as_pair(value, f"{path}: canvas")
    return (max(int(round(width)), 1), max(int(round(height)), 1))


def _infer_canvas(
    pending: Sequence[tuple[str, list[Bone], Any, tuple[float, float], int, float, Any]],
    eyes: Sequence[tuple[float, float, float]],
    body: tuple[float, float] | None,
) -> tuple[int, int]:
    """Infer a canvas from the annotation when ``canvas`` is missing (ceil of the bbox)."""
    points: list[tuple[float, float]] = []
    for _name, bones, _region, pivot, _z, _blend, _motions in pending:
        for bone in bones:
            points.append((bone.x0, bone.y0))
            points.append((bone.x1, bone.y1))
        if not bones:
            points.append(pivot)
    for cx, cy, half in eyes:
        points.append((cx + abs(half), cy + abs(half)))
    if body is not None:
        points.append(body)
    if not points:
        return (1, 1)
    width = max(int(math.ceil(max(x for x, _ in points))), 1)
    height = max(int(math.ceil(max(y for _, y in points))), 1)
    return (width, height)


def _select_character(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return the document holding ``parts``, unwrapping a character-keyed file."""
    if "parts" in raw:
        return raw
    candidates = [(key, value) for key, value in raw.items() if isinstance(value, dict) and "parts" in value]
    if not candidates:
        raise ValueError(f"{path}: no 'parts' list found — not a flat2rig annotation")
    if len(candidates) > 1:
        stem = path.name.lower()
        for key, value in candidates:
            if key.lower() in stem:
                return value
    return candidates[0][1]


# --------------------------------------------------------------------------------------
# mask computation
# --------------------------------------------------------------------------------------


@dataclass
class _Group:
    """One unique part name with all of its bones (duplicate names are merged)."""

    name: str
    bones: list[Bone]
    z: int


def _part_groups(rig: Rig) -> list[_Group]:
    """Collapse ``rig.parts`` into unique-name groups, keeping first-appearance order."""
    groups: list[_Group] = []
    seen: dict[str, _Group] = {}
    for part in rig.parts:
        group = seen.get(part.name)
        if group is None:
            group = _Group(name=part.name, bones=list(part.bones), z=int(part.z))
            seen[part.name] = group
            groups.append(group)
        else:
            group.bones.extend(part.bones)
            group.z = min(group.z, int(part.z))
    return groups


def _paint_ranks(groups: Sequence[_Group]) -> list[int]:
    """Label ids ordered by ``(z, declaration order)`` so ties go to the lower ``z``."""
    order = sorted(range(len(groups)), key=lambda index: (groups[index].z, index))
    ranks = [0] * len(groups)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _rasterize_segment(bone: Bone) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise a bone into the grid pixels it passes through (8-connected DDA)."""
    steps = int(max(abs(bone.x1 - bone.x0), abs(bone.y1 - bone.y0)))
    if steps <= 0:
        return (
            np.array([int(round(bone.x0))], dtype=np.intp),
            np.array([int(round(bone.y0))], dtype=np.intp),
        )
    t = np.linspace(0.0, 1.0, steps + 1)
    xs = np.rint(bone.x0 + (bone.x1 - bone.x0) * t).astype(np.intp)
    ys = np.rint(bone.y0 + (bone.y1 - bone.y0) * t).astype(np.intp)
    return xs, ys


def _chamfer_keys(keys: np.ndarray, labels: int) -> np.ndarray:
    """Multi-source wavefront distance transform over the pixel grid.

    ``keys`` is an int64 grid holding ``distance * labels + part_label`` — zero for seed
    pixels and :data:`_INF` elsewhere.  A forward raster pass (top-left to bottom-right)
    and a backward pass then relax every pixel with the chamfer 3-4 neighbourhood, using
    ``np.minimum.accumulate`` for the within-row propagation, so the whole transform is
    vectorised numpy.  Minimising the encoded key minimises the distance first and the
    label id second, which is exactly the required "ties go to the smaller ``z``" rule
    (ids are pre-sorted by ``z``).  Seeds are modified in place.
    """
    height, width = keys.shape
    orth = np.int64(_CHAMFER_ORTH) * labels
    diag = np.int64(_CHAMFER_DIAG) * labels
    offsets = np.arange(width, dtype=np.int64) * orth

    for y in range(height):  # forward raster pass
        row = keys[y]
        if y:
            up = keys[y - 1]
            candidate = up + orth
            if width > 1:
                np.minimum(candidate[1:], up[:-1] + diag, out=candidate[1:])
                np.minimum(candidate[:-1], up[1:] + diag, out=candidate[:-1])
            np.minimum(row, candidate, out=row)
        tmp = row - offsets
        np.minimum.accumulate(tmp, out=tmp)
        row[...] = tmp + offsets

    for y in range(height - 1, -1, -1):  # backward raster pass
        row = keys[y]
        if y + 1 < height:
            down = keys[y + 1]
            candidate = down + orth
            if width > 1:
                np.minimum(candidate[1:], down[:-1] + diag, out=candidate[1:])
                np.minimum(candidate[:-1], down[1:] + diag, out=candidate[:-1])
            np.minimum(row, candidate, out=row)
        flipped = row[::-1] - offsets
        np.minimum.accumulate(flipped, out=flipped)
        row[::-1] = flipped + offsets
    return keys


def assign_parts(rgba: Image.Image, rig: Rig) -> dict[str, Image.Image]:
    """Label every opaque pixel with its nearest bone and return one ``L`` mask per part.

    Distances are computed over the whole canvas (straight-line, as in the README) and
    the result is clipped to ``alpha > 0``, so the masks never bleed outside the
    silhouette and their union equals the character.  Parts sharing a name are merged;
    a part without bones yields an empty mask (see :func:`diagnose`, which reports it).

    Returns:
        ``{part_name: "L" image}`` with 255 where the pixel belongs to that part.
    """
    rgba = rgba if rgba.mode == "RGBA" else rgba.convert("RGBA")
    width, height = rgba.size
    groups = _part_groups(rig)
    masks = {group.name: Image.new("L", (width, height), 0) for group in groups}
    if not groups or width <= 0 or height <= 0:
        return masks

    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8) > 0
    ranks = _paint_ranks(groups)
    keys = np.full((height, width), _INF, dtype=np.int64)
    seeded = False
    for index, group in enumerate(groups):
        rank = ranks[index]
        for bone in group.bones:
            xs, ys = _rasterize_segment(bone.clamped(width, height))
            if xs.size:
                np.minimum.at(keys, (ys, xs), rank)
                seeded = True
    if not seeded:
        return masks

    labels = _chamfer_keys(keys, len(groups) + 1)
    finite = labels < _FINITE_LIMIT
    decoded = np.where(finite, labels % (len(groups) + 1), -1)
    for index, group in enumerate(groups):
        selected = (decoded == ranks[index]) & alpha
        masks[group.name] = Image.fromarray(np.where(selected, np.uint8(255), np.uint8(0)))
    return masks


def soften_masks(
    masks: dict[str, Image.Image], blend: float | Mapping[str, float]
) -> dict[str, Image.Image]:
    """Feather every mask boundary and renormalise the parts so their sum is preserved.

    Each mask is blurred with a Gaussian of ``sigma = blend / 2`` — a "blend half-width"
    of ``b`` pixels yields a transition band of roughly ``b`` pixels on either side of the
    boundary — then every pixel is divided by the total blurred weight and multiplied by
    the incoming total, i.e. the union of the input masks.  Nothing is lost or counted
    twice: where the input masks tile the character alpha, the outputs sum to that alpha
    exactly (up to ``uint8`` rounding) and stay 0 outside it.

    Args:
        masks: ``{name: mask}``; single channel, 255 = belongs to the part.
        blend: transition half-width in pixels, or a ``{name: half-width}`` mapping.

    Returns:
        New ``"L"`` masks with the same keys and order.
    """
    names = list(masks)
    if not names:
        return {}
    arrays = [_mask_array(masks[name]) for name in names]
    shape = arrays[0].shape
    for name, array in zip(names, arrays):
        if array.shape != shape:
            raise ValueError(f"mask '{name}' has shape {array.shape}, expected {shape}")

    target = np.zeros(shape, dtype=np.float32)
    for array in arrays:
        target += array
    if not np.any(target > 0):
        return {name: Image.fromarray(np.zeros(shape, dtype=np.uint8)) for name in names}

    blurred = [_blur(array, max(_blend_for(name, blend), 0.0) / 2.0) for name, array in zip(names, arrays)]
    total = np.zeros(shape, dtype=np.float32)
    for item in blurred:
        total += item

    outputs: list[np.ndarray] = []
    for item in blurred:
        weight = np.divide(item, total, out=np.zeros(shape, dtype=np.float32), where=total > 1e-6)
        outputs.append(np.clip(np.rint(weight * target), 0.0, 255.0))

    # uint8 rounding can drop a pixel that the input covered; hand it to the strongest part.
    stacked = np.stack(blurred, axis=0)
    covered = np.zeros(shape, dtype=np.float32)
    for out in outputs:
        covered += out
    gap = (target > 0.5) & (covered <= 0.5)
    if np.any(gap):
        best = np.argmax(stacked, axis=0)
        for index, out in enumerate(outputs):
            selected = gap & (best == index)
            out[selected] = np.rint(target[selected])

    return {name: Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8)) for name, out in zip(names, outputs)}


def build_masks(rgba: Image.Image, rig: Rig) -> dict[str, Image.Image]:
    """Full stage 1-2: nearest-bone labeling plus per-part boundary softening.

    Uses each :class:`Part`'s own ``blend`` width.  The returned masks are clipped to the
    character silhouette and, for a well-formed annotation, sum to the alpha there.
    """
    hard = assign_parts(rgba, rig)
    blends: dict[str, float] = {}
    for part in rig.parts:
        blends[part.name] = float(part.blend)
    soft = soften_masks(hard, blends)

    rgba = rgba if rgba.mode == "RGBA" else rgba.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8) > 0
    clipped: dict[str, Image.Image] = {}
    for name, image in soft.items():
        array = np.asarray(image, dtype=np.uint8)
        clipped[name] = Image.fromarray(np.where(alpha, array, np.uint8(0)))
    return clipped


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------


def diagnose(rgba: Image.Image, rig: Rig, masks: dict[str, Image.Image]) -> dict[str, Any]:
    """Summarise mask quality and return coverage, per-part areas and warnings.

    ``coverage`` is the fraction of ``alpha > 0`` pixels touched by at least one mask;
    ``per_part_area`` counts a part's pixels as the integral ``sum(mask) / 255`` (soft
    edges count fractionally, so the areas of a well-formed rig add up to the character
    area), while ``per_part_pixels`` counts strictly non-zero mask pixels.

    Warnings are emitted for a part covering less than 1% or more than 90% of the
    character, for parts without bones or without pixels, for coverage deviating from 1.0
    by more than 0.02, for masks covering pixels outside the silhouette and for a rig
    canvas that does not match the image.

    Returns:
        A dict with ``coverage``, ``per_part_area``, ``per_part_pixels``,
        ``per_part_share``, ``character_area``, ``covered_area``, ``total_area``,
        ``image_size``, ``canvas`` and ``warnings``.
    """
    rgba = rgba if rgba.mode == "RGBA" else rgba.convert("RGBA")
    width, height = rgba.size
    alpha_values = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    alpha = alpha_values > 0
    character_area = int(np.count_nonzero(alpha))

    names = list(masks)
    if names:
        arrays = np.stack([_mask_array(masks[name]) for name in names], axis=0)
        if arrays.shape[1:] != (height, width):
            raise ValueError(f"masks have shape {arrays.shape[1:]}, expected {(height, width)}")
    else:
        arrays = np.zeros((0, height, width), dtype=np.float32)

    union = arrays.max(axis=0) > 0 if names else np.zeros((height, width), dtype=bool)
    covered_area = int(np.count_nonzero(union & alpha))
    coverage = covered_area / character_area if character_area else 1.0

    per_part_area: dict[str, int] = {}
    per_part_pixels: dict[str, int] = {}
    per_part_share: dict[str, float] = {}
    for index, name in enumerate(names):
        soft_area = int(round(float(arrays[index].sum()) / 255.0))
        per_part_area[name] = soft_area
        per_part_pixels[name] = int(np.count_nonzero(arrays[index] > 0))
        per_part_share[name] = soft_area / character_area if character_area else 0.0

    sums = arrays.sum(axis=0) if names else np.zeros((height, width), dtype=np.float32)
    total_area = int(round(float(sums.sum()) / 255.0))

    warnings: list[str] = []
    if not rig.parts:
        warnings.append("rig has no parts — nothing to mask")
    if character_area == 0:
        warnings.append("character alpha is empty (0 px) — nothing to mask")
    if tuple(rig.size) != (width, height):
        warnings.append(
            f"rig canvas {rig.size[0]}x{rig.size[1]} does not match the image {width}x{height} "
            "— bones are clamped to the image"
        )

    seen: set[str] = set()
    for part in rig.parts:
        if part.name in seen:
            continue
        seen.add(part.name)
        if not part.bones:
            warnings.append(f"part '{part.name}' has no bones")
        if part.name not in masks:
            warnings.append(f"part '{part.name}' has no mask — masks do not match this rig")
            continue
        share = per_part_share[part.name]
        if part.bones and per_part_pixels[part.name] == 0:
            warnings.append(f"part '{part.name}' has bones but covers no pixels — check that they are inside the canvas")
        if share < 0.01:
            warnings.append(f"part '{part.name}' covers {share * 100:.1f}% of the character — check bones")
        elif share > 0.90:
            warnings.append(f"part '{part.name}' covers {share * 100:.1f}% of the character — check bones / z order")

    if abs(coverage - 1.0) > 0.02:
        warnings.append(
            f"coverage {coverage * 100:.1f}% of the character — "
            f"{character_area - covered_area} of {character_area} px are not covered by any mask"
        )
    outside = int(np.count_nonzero(union & ~alpha))
    if outside:
        warnings.append(f"masks cover {outside} px outside the character silhouette")
    deviation = int(np.abs(sums - np.where(alpha, np.float32(255.0), np.float32(0.0))).max()) if names else 0
    if deviation > 2:
        warnings.append(f"masks sum to alpha ± {deviation} levels — parts may double-count pixels")

    return {
        "coverage": float(coverage),
        "per_part_area": per_part_area,
        "per_part_pixels": per_part_pixels,
        "per_part_share": per_part_share,
        "character_area": character_area,
        "covered_area": covered_area,
        "total_area": total_area,
        "image_size": (width, height),
        "canvas": tuple(rig.size),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _mask_array(image: Image.Image) -> np.ndarray:
    """Return a mask as a float32 array of 0..255 values."""
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.float32)


def _blend_for(name: str, blend: float | Mapping[str, float]) -> float:
    """Resolve the blend width for ``name`` (mapping form) or use the scalar."""
    if isinstance(blend, Mapping):
        return float(blend.get(name, DEFAULT_BLEND))
    return float(blend)


def _blur(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2D array; uses scipy when available, boxes otherwise."""
    if sigma <= 0:
        return values.astype(np.float32, copy=True)
    if _scipy_gaussian is not None:
        return np.asarray(_scipy_gaussian(values, sigma=float(sigma), mode="nearest"), dtype=np.float32)
    return _box_blur(values, sigma)


def _box_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    """Three box blurs per axis — a scipy-free Gaussian approximation."""
    radius = max(1, int(round(sigma - 0.5)))
    out = values.astype(np.float32, copy=True)
    for _ in range(3):
        out = _box_blur_axis(out, radius, 0)
        out = _box_blur_axis(out, radius, 1)
    return out


def _box_blur_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Moving average of window ``2 * radius + 1`` along one axis (edge-clamped)."""
    if radius <= 0:
        return values
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zero_shape, dtype=np.float64), cumulative], axis=axis)
    size = 2 * radius + 1
    length = values.shape[axis]
    high = np.take(cumulative, np.arange(size, size + length), axis=axis)
    low = np.take(cumulative, np.arange(0, length), axis=axis)
    return np.asarray((high - low) / size, dtype=np.float32)


# --------------------------------------------------------------------------------------
# smoke entry point
# --------------------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m flat2rig.rig <image> -c <annotation.json>`` — print a mask report."""
    parser = argparse.ArgumentParser(
        prog="python -m flat2rig.rig",
        description="Compute part masks for a flat character and print a diagnosis.",
    )
    parser.add_argument("image", help="flat RGBA character image")
    parser.add_argument("-c", "--config", required=True, help="annotation JSON (see README)")
    args = parser.parse_args(argv)

    rgba, _alpha = load_alpha(args.image)
    rig = load_rig(args.config)
    masks = build_masks(rgba, rig)
    report = diagnose(rgba, rig, masks)

    print(f"{args.image} {rgba.size[0]}x{rgba.size[1]} — coverage {report['coverage'] * 100:.1f}%")
    for name, area in report["per_part_area"].items():
        share = report["per_part_share"][name] * 100
        print(f"  {name:<16} {area:>7} px  {share:5.1f}%")
    for warning in report["warnings"]:
        print(f"  ! {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
