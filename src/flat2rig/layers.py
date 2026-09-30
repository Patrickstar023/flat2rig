"""Layer-stack loading for the "bring your own layers" mode of flat2rig.

flat2rig can obtain its part images in two ways:

* **mode A** -- decompose a single flat illustration itself (skeleton-guided
  geometric split, implemented elsewhere); or
* **mode B** -- consume an *already layered* character, i.e. the ~24-layer PSD
  produced by `see-through <https://github.com/shitagaki-lab/see-through>`_ or
  any directory of back-to-front RGBA layer PNGs, and only do the rigging and
  animation.

This module is the front door of mode B:

* :func:`load_stack` discovers, orders (back-to-front) and loads a stack from a
  ``.psd``/``.psb`` file or from a directory of layer images;
* :func:`map_parts` decides which source layer(s) feed each part name the rig
  asks for, and :func:`map_report` reports what happened;
* :func:`suggest_aliases` offers a first guess at the alias table by bucketing
  layer names with a small keyword table.

Conventions
-----------
* A :class:`LayerStack` is always *back-to-front*: ``layers[0]`` is painted
  first (the most distant layer), ``layers[-1]`` is the closest to the viewer.
  This is the order of both see-through's PSD layer panel and filenames sorted
  naturally.
* Every layer image is ``RGBA`` and fills the whole canvas (``stack.size``), so
  a caller never has to deal with per-layer offsets.
* Layer *names* are file stems for a directory and PSD layer names for a PSD.
* Only the standard library, Pillow and numpy are required. ``psd-tools`` is an
  optional extra imported lazily inside :func:`load_stack`, so a base install
  keeps working without it (``pip install flat2rig[psd]`` adds PSD reading).

The only module-level mutable state is the report of the last :func:`map_parts`
call, returned by :func:`map_report`; see its docstring.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

__all__ = [
    "LayerStack",
    "SEMANTIC_KEYWORDS",
    "load_stack",
    "map_parts",
    "map_report",
    "suggest_aliases",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Suffixes treated as "one RGBA layer per file" inside a directory.
_LAYER_SUFFIXES: frozenset[str] = frozenset({".png", ".webp"})

#: Suffixes of a single flattened image: a *part mask source*, never a stack.
_FLAT_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".webp", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
)

#: Suffixes handled by psd-tools (PSB is the "big document" variant of PSD).
_PSD_SUFFIXES: frozenset[str] = frozenset({".psd", ".psb"})

_PSD_EXTRA_HINT = "pip install flat2rig[psd]"

# Natural sort: split a name into digit / non-digit chunks so "layer2" < "layer10".
_NATURAL_CHUNK = re.compile(r"(\d+)")

# Name normalisation used by map_parts.
_SEPARATORS = re.compile(r"[\s_\-.·・]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CJK_RANGES = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
_NON_WORD = re.compile(rf"[^0-9A-Za-z{_CJK_RANGES}]+")
_CJK_CHAR = re.compile(rf"[{_CJK_RANGES}]")

#: Keyword table used by :func:`suggest_aliases`, in evaluation order: the first
#: bucket whose keyword matches wins. Keys are the canonical semantic buckets;
#: values are keywords matched against the *tokens* of a layer name (see
#: :func:`_bucket_for`). Extend it by adding keywords to an existing bucket, or
#: add a new bucket -- put more specific buckets earlier, because "ear" must win
#: over "head" for a layer called ``head_ear``. ASCII keywords match a whole
#: token or a token prefix/suffix (so ``left_ear`` -> ear, ``forearm`` -> arm);
#: CJK keywords match single characters (so ``左耳`` -> ear).
SEMANTIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ear": ("ear", "ears", "耳", "みみ", "耳朵"),
    "tail": ("tail", "尻尾", "しっぽ", "尾", "尾巴"),
    "hair": ("hair", "bangs", "ahoge", "髪", "前髪", "头发", "劉海", "刘海"),
    "head": ("head", "neck", "頭", "头", "首", "脖子"),
    "face": (
        "face",
        "eye",
        "eyes",
        "brow",
        "brows",
        "lash",
        "lashes",
        "mouth",
        "nose",
        "cheek",
        "blush",
        "顔",
        "脸",
        "眼",
        "目",
        "口",
        "鼻",
        "眉",
    ),
    "arm": ("arm", "hand", "shoulder", "sleeve", "elbow", "wrist", "腕", "手", "肩", "袖", "胳膊", "手臂"),
    "leg": ("leg", "thigh", "knee", "foot", "feet", "shoe", "boot", "sock", "脚", "足", "腿", "靴", "鞋"),
    "body": ("body", "torso", "chest", "waist", "hip", "hips", "体", "胴", "身体", "躯干"),
    "accessory": (
        "cloth",
        "clothes",
        "clothing",
        "dress",
        "shirt",
        "skirt",
        "coat",
        "cape",
        "hat",
        "cap",
        "glasses",
        "ribbon",
        "bow",
        "accessory",
        "acc",
        "weapon",
        "sword",
        "bag",
        "wing",
        "wings",
        "服",
        "衣",
        "帽",
        "眼镜",
        "眼鏡",
        "饰",
        "配件",
        "翅膀",
    ),
}

#: Bucket for layer names that no keyword matched.
_OTHER_BUCKET = "other"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _natural_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Return a sort key where digit runs compare numerically.

    ``"layer2.png"`` therefore sorts before ``"layer10.png"``, while text keeps
    its case-insensitive alphabetical order. Used to order a PNG directory.
    """
    key: list[tuple[int, Any]] = []
    for chunk in _NATURAL_CHUNK.split(name.casefold()):
        if not chunk:
            continue
        key.append((1, int(chunk)) if chunk.isdecimal() else (0, chunk))
    return tuple(key)


def _key(name: str) -> str:
    """Case-insensitive comparison key: ``" Ear_L "`` -> ``"ear_l"``."""
    return name.strip().casefold()


def _squash(name: str) -> str:
    """Separator-insensitive key: ``"Ear L"`` -> ``"earl"`` (used by fuzzy match)."""
    return _SEPARATORS.sub("", name.casefold())


def _alpha_has_content(image: Image.Image) -> bool:
    """True when the image's alpha channel has at least one non-zero pixel."""
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    return bool(alpha.any())


def _transparent(size: tuple[int, int]) -> Image.Image:
    """Return a fully transparent ``RGBA`` image of ``size`` (width, height)."""
    return Image.new("RGBA", (max(int(size[0]), 0), max(int(size[1]), 0)), (0, 0, 0, 0))


def _common_size(images: Sequence[Image.Image]) -> tuple[int, int]:
    """Largest (width, height) among ``images``, or ``(0, 0)`` when empty."""
    if not images:
        return (0, 0)
    return (max(image.width for image in images), max(image.height for image in images))


def _place_on_canvas(
    image: Image.Image,
    size: tuple[int, int],
    offset: tuple[int, int] | None = None,
) -> Image.Image:
    """Return ``image`` as an ``RGBA`` full-canvas layer of ``size``.

    ``offset`` is the layer's ``(left, top)`` in canvas coordinates (used for
    PSD layers, whose stored pixels may sit at an offset, possibly negative).
    When ``offset`` is ``None`` the image is centred, which is the policy for a
    PNG directory whose layers do not all share one size.
    """
    if image.size == size:
        return image
    canvas = _transparent(size)
    if offset is None:
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
    else:
        left, top = int(offset[0]), int(offset[1])
    # ``paste`` clips sources that stick out of the canvas, so negative offsets
    # (a layer larger than the document) are fine.
    canvas.paste(image, (left, top), image)
    return canvas


def _merge_max_alpha(images: Sequence[Image.Image], size: tuple[int, int]) -> Image.Image:
    """Union several full-canvas layers into one by per-pixel maximum alpha.

    Where two layers overlap, the colour of the more opaque one wins; alpha is
    the maximum of the inputs. The union is not an alpha *blend* -- it is meant
    to gather e.g. ``ear_l`` + ``ear_r`` into a single "ear" part image without
    darkening their overlap.
    """
    if not images:
        return _transparent(size)
    if len(images) == 1:
        # Cheap path: the caller gets the stack's own image. Treat the result of
        # map_parts as read-only.
        return images[0]
    width, height = size
    best_alpha = np.zeros((height, width), dtype=np.uint8)
    best_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for image in images:
        array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if array.shape[:2] != (height, width):  # pragma: no cover - defensive
            continue
        alpha = array[..., 3]
        take = alpha > best_alpha
        best_rgb[take] = array[..., :3][take]
        best_alpha = np.where(take, alpha, best_alpha)
    merged = np.empty((height, width, 4), dtype=np.uint8)
    merged[..., :3] = best_rgb
    merged[..., 3] = best_alpha
    return Image.fromarray(merged)


# --------------------------------------------------------------------------- #
# LayerStack
# --------------------------------------------------------------------------- #


@dataclass
class LayerStack:
    """An ordered, full-canvas set of RGBA layers.

    Attributes:
        size: ``(width, height)`` of the canvas every layer fills.
        layers: ``(name, image)`` pairs in **back-to-front** paint order; each
            image is mode ``RGBA`` and exactly ``size`` pixels.
        source: where the stack came from: ``"psd"`` or ``"png-dir"``.
    """

    size: tuple[int, int]
    layers: list[tuple[str, Image.Image]]
    source: str

    def __len__(self) -> int:
        """Number of layers (convenience, so ``if stack:`` means "non-empty")."""
        return len(self.layers)

    def names(self) -> list[str]:
        """Layer names in back-to-front order (duplicates are preserved)."""
        return [name for name, _ in self.layers]

    def get(self, name: str) -> Image.Image | None:
        """Return the layer called ``name``, or ``None``.

        The look-up is exact first, then case-insensitive. When a stack holds
        several layers with the same name (common in PSDs) the first one in
        back-to-front order wins; use :attr:`layers` to reach the others.
        """
        for layer_name, image in self.layers:
            if layer_name == name:
                return image
        wanted = _key(name)
        for layer_name, image in self.layers:
            if _key(layer_name) == wanted:
                return image
        return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_stack(path: str | Path) -> LayerStack:
    """Load a back-to-front layer stack from ``path``.

    Two inputs are accepted (mode B -- "the character is already layered"):

    * a ``.psd`` / ``.psb`` file: every visible leaf layer is composited to the
      full document canvas, back-to-front. Needs the optional ``psd-tools``
      dependency; without it an :class:`ImportError` explains how to install it.
    * a directory: every ``*.png`` / ``*.webp`` file whose alpha is not entirely
      zero, natural-sorted by filename (``layer2`` < ``layer10``) and therefore
      back-to-front. Layer names are the file stems.

    A single flat image (``foo.png``) is *not* a stack: that is mode A, and a
    :class:`ValueError` says so. An empty (or fully transparent) directory is
    not an error either -- the result is an empty stack plus a
    :class:`RuntimeWarning`, so a CLI can report it instead of crashing.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ImportError: a PSD was given but ``psd-tools`` is not installed.
        ValueError: ``path`` is a single flat image or an unsupported file type.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"layer stack not found: {target}")
    if target.is_dir():
        return _load_png_dir(target)
    suffix = target.suffix.casefold()
    if suffix in _PSD_SUFFIXES:
        return _load_psd(target)
    if suffix in _FLAT_SUFFIXES:
        raise ValueError(
            f"{target} is a single flat image, not a layer stack. flat2rig can "
            "decompose a flat image itself (mode A): run the build pipeline "
            "without the layer-stack flag. Mode B expects a see-through .psd "
            "or a directory of back-to-front RGBA layer PNGs."
        )
    raise ValueError(
        f"unsupported layer-stack input {str(target)!r}: expected a .psd/.psb "
        "file or a directory of *.png / *.webp layer images."
    )


def _load_png_dir(directory: Path) -> LayerStack:
    """Load ``*.png`` / ``*.webp`` layers from ``directory``, back-to-front."""
    files = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.casefold() in _LAYER_SUFFIXES
    ]
    # Natural order by stem, with the original name as a deterministic tie-break
    # (e.g. "layer02" vs "layer2").
    files.sort(key=lambda entry: (_natural_key(entry.stem), entry.stem.casefold()))

    layers: list[tuple[str, Image.Image]] = []
    for file in files:
        try:
            with Image.open(file) as handle:
                handle.load()
                rgba = handle.convert("RGBA")
        except Exception as exc:  # noqa: BLE001 - skip a broken file, keep the rest
            warnings.warn(
                f"skipping unreadable layer file {file.name!r}: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not _alpha_has_content(rgba):
            warnings.warn(
                f"skipping fully transparent layer file {file.name!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        layers.append((file.stem, rgba))

    if not layers:
        psd_hint = ""
        if any(entry.is_file() and entry.suffix.casefold() in _PSD_SUFFIXES for entry in directory.iterdir()):
            psd_hint = " (found a .psd/.psb in there -- pass that file itself, not its folder)"
        warnings.warn(
            f"no usable layers in {directory}: expected *.png or *.webp files with "
            f"non-empty alpha, found {len(files)} candidate file(s){psd_hint}. "
            "Returning an empty layer stack.",
            RuntimeWarning,
            stacklevel=2,
        )

    size = _common_size([image for _, image in layers])
    if layers and any(image.size != size for _, image in layers):
        warnings.warn(
            f"layer images in {directory} do not share one size; smaller ones are "
            f"centred on the {size[0]}x{size[1]} canvas",
            RuntimeWarning,
            stacklevel=2,
        )
    layers = [(name, _place_on_canvas(image, size)) for name, image in layers]

    keys = [_key(name) for name, _ in layers]
    if len(set(keys)) != len(keys):
        duplicates = sorted({name for name in keys if keys.count(name) > 1})
        warnings.warn(
            f"duplicate layer names in {directory} (case-insensitively): "
            f"{duplicates}; only the first of each is reachable via LayerStack.get()",
            RuntimeWarning,
            stacklevel=2,
        )

    return LayerStack(size=size, layers=layers, source="png-dir")


def _require_psd_tools() -> Any:
    """Import psd-tools lazily and return ``PSDImage``.

    ``psd-tools`` is an optional extra: keeping the import inside the function
    means the base install (numpy + Pillow) never needs it.
    """
    try:
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading a .psd/.psb layer stack needs the optional 'psd-tools' "
            f"package: {_PSD_EXTRA_HINT} (or: pip install psd-tools). "
            "Alternatively export the layers as RGBA PNGs and pass that "
            f"directory to load_stack(). Original error: {exc}"
        ) from exc
    return PSDImage


def _psd_layer_visible(layer: Any) -> bool:
    """Visibility of a psd-tools layer, accounting for hidden parent groups."""
    is_visible = getattr(layer, "is_visible", None)
    if callable(is_visible):
        try:
            return bool(is_visible())
        except Exception:  # noqa: BLE001 - fall back to the raw flag  # pragma: no cover
            pass
    return bool(getattr(layer, "visible", True))


def _iter_psd_leaves(node: Any) -> Iterable[Any]:
    """Yield the visible leaf layers of a PSD/group, back-to-front.

    psd-tools (>= 1.7) iterates "from background to foreground", so a
    depth-first walk in iteration order already yields the whole document
    back-to-front; a group is expanded in place, which keeps its children in
    their z-slot. Hidden groups are skipped entirely, hidden leaves too.
    """
    for layer in node:
        if not _psd_layer_visible(layer):
            continue
        if layer.is_group():
            yield from _iter_psd_leaves(layer)
        else:
            yield layer


def _render_psd_layer(layer: Any, name: str, size: tuple[int, int]) -> Image.Image | None:
    """Composite one psd-tools layer to a full-canvas RGBA image, or ``None``."""
    viewport = (0, 0, size[0], size[1])
    try:
        try:
            image = layer.composite(viewport=viewport)
        except TypeError:  # pragma: no cover - very old psd-tools without viewport
            image = layer.composite()
    except Exception as exc:  # noqa: BLE001 - one bad layer must not kill the stack
        warnings.warn(
            f"skipping PSD layer {name!r}: {exc!r} "
            "(vector strokes / gradient fills need 'pip install psd-tools[composite]')",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if image is None:
        return None
    rgba = image.convert("RGBA")
    if rgba.size != size:
        offset = getattr(layer, "offset", None)
        rgba = _place_on_canvas(rgba, size, None if offset is None else tuple(offset))
    return rgba


def _load_psd(path: Path) -> LayerStack:
    """Load every visible leaf layer of a PSD/PSB, back-to-front."""
    PSDImage = _require_psd_tools()
    document = PSDImage.open(path)
    size = (int(document.width), int(document.height))

    layers: list[tuple[str, Image.Image]] = []
    for index, layer in enumerate(_iter_psd_leaves(document)):
        name = str(getattr(layer, "name", "") or "").strip() or f"layer_{index}"
        image = _render_psd_layer(layer, name, size)
        if image is None:
            continue
        if not _alpha_has_content(image):
            warnings.warn(
                f"skipping fully transparent PSD layer {name!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        layers.append((name, image))

    if not layers:
        warnings.warn(
            f"{path} has no visible, non-transparent layers; returning an empty "
            "layer stack (a flattened PSD is a single flat image -- mode A only)",
            RuntimeWarning,
            stacklevel=2,
        )
    return LayerStack(size=size, layers=layers, source="psd")


# --------------------------------------------------------------------------- #
# Mapping stack layers onto rig part names
# --------------------------------------------------------------------------- #

#: Report of the most recent :func:`map_parts` call in this process. Module-level
#: state is intentional here: the public API is ``map_report()`` with no
#: arguments, so the last call has to be remembered somewhere. It is overwritten
#: by every call and is therefore not thread-safe.
_LAST_MAP_REPORT: dict[str, Any] = {
    "matched": {},
    "unmatched_parts": [],
    "unused_layers": [],
}


def _build_alias_table(aliases: Mapping[str, Sequence[str]] | None) -> dict[str, set[str]]:
    """Normalise the caller's alias mapping into ``{part_key: {layer_key, ...}}``.

    Both the case-insensitive key and the separator-insensitive key are stored
    for each alias, so ``"left_ear"``, ``"Left Ear"`` and ``"leftear"`` all
    reach the same layer.
    """
    table: dict[str, set[str]] = {}
    if not aliases:
        return table
    for part, alias_names in aliases.items():
        if isinstance(alias_names, str):  # tolerate a bare string (documented)
            alias_names = [alias_names]
        keys: set[str] = set()
        for alias in alias_names:
            if not isinstance(alias, str) or not alias.strip():
                continue
            keys.add(_key(alias))
            keys.add(_squash(alias))
        if keys:
            table.setdefault(_key(str(part)), set()).update(keys)
    return table


def map_parts(
    stack: LayerStack,
    part_names: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Image.Image]:
    """Pick the source layer(s) for each requested part name.

    Each part name is resolved in this order; the first stage that finds
    anything wins:

    1. **exact** -- the part name equals a layer name, case-insensitively;
    2. **aliases** -- the part name is a key of ``aliases`` (e.g.
       ``{"ear": ["ear_l", "ear_r"], "body": ["body", "torso", "base"]}``) and
       the listed alias names match layer names (case- and separator-insensitive);
    3. **fuzzy containment** -- the part name appears inside the layer name or
       vice versa, ignoring case and separators (``"ear"`` matches
       ``"ear_l"``); single-character part names are ignored as too ambiguous.

    All layers matched in the winning stage are merged into one image with a
    per-pixel maximum-alpha union (see :func:`_merge_max_alpha`), so
    ``ear_l`` + ``ear_r`` become a single "ear" part. A part that matches
    nothing is returned as a fully transparent ``RGBA`` image of
    ``stack.size``. A single-hit part is returned as the stack's own image, not
    a copy -- treat the results as read-only.

    Args:
        stack: the loaded layer stack.
        part_names: part names the rig expects, in caller order.
        aliases: optional ``{part_name: [alternative layer names]}``.

    Returns:
        ``{part_name: RGBA image}`` in the order of ``part_names``. Call
        :func:`map_report` afterwards for the matched/unmatched/unused summary.
    """
    global _LAST_MAP_REPORT

    entries = [
        (index, name, _key(name), _squash(name))
        for index, (name, _) in enumerate(stack.layers)
    ]
    alias_table = _build_alias_table(aliases)

    result: dict[str, Image.Image] = {}
    matched: dict[str, list[str]] = {}
    unmatched: list[str] = []
    used: set[int] = set()

    for part in part_names:
        part_key = _key(part)
        part_squash = _squash(part)

        # 1. exact (case-insensitive) name match.
        hits = [index for index, _, key, _ in entries if key == part_key]

        # 2. alias mapping.
        if not hits and alias_table:
            wanted = alias_table.get(part_key, set())
            if wanted:
                hits = [
                    index
                    for index, _, key, squashed in entries
                    if key in wanted or squashed in wanted
                ]

        # 3. fuzzy containment, both directions.
        if not hits and len(part_squash) >= 2:
            hits = [
                index
                for index, _, _, squashed in entries
                if squashed and (part_squash in squashed or squashed in part_squash)
            ]

        if hits:
            images = [stack.layers[index][1] for index in hits]
            result[part] = _merge_max_alpha(images, stack.size)
            matched[part] = [stack.layers[index][0] for index in hits]
            used.update(hits)
        else:
            result[part] = _transparent(stack.size)
            unmatched.append(part)

    _LAST_MAP_REPORT = {
        "matched": matched,
        "unmatched_parts": unmatched,
        "unused_layers": [
            name for index, (name, _) in enumerate(stack.layers) if index not in used
        ],
    }
    return result


def map_report() -> dict[str, Any]:
    """Human-readable report of the most recent :func:`map_parts` call.

    Module-level state: the report describes the last call in *this process*,
    is overwritten by every new call and is **not** thread-safe. Before any
    call it is ``{"matched": {}, "unmatched_parts": [], "unused_layers": []}``.

    Returns:
        ``{"matched": {part: [layer names]}, "unmatched_parts": [...],
        "unused_layers": [...]}`` -- a fresh copy, safe to mutate.
    """
    return {
        "matched": {part: list(names) for part, names in _LAST_MAP_REPORT["matched"].items()},
        "unmatched_parts": list(_LAST_MAP_REPORT["unmatched_parts"]),
        "unused_layers": list(_LAST_MAP_REPORT["unused_layers"]),
    }


# --------------------------------------------------------------------------- #
# Alias suggestions
# --------------------------------------------------------------------------- #


def _tokenize(name: str) -> list[str]:
    """Split a layer name into lowercase tokens.

    Separators (``_``, ``-``, spaces, dots) and camelCase boundaries split
    tokens, and CJK characters become one token each, so ``hairBack`` ->
    ``["hair", "back"]`` and ``左耳`` -> ``["左", "耳"]``.
    """
    tokens: list[str] = []
    for chunk in _NON_WORD.split(_CAMEL_BOUNDARY.sub(" ", name)):
        if not chunk:
            continue
        current = ""
        for char in chunk.casefold():
            if _CJK_CHAR.match(char):
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
    return tokens


def _bucket_for(name: str) -> str:
    """Best-guess semantic bucket of ``name``; ``"other"`` when nothing matches.

    A keyword matches when a token equals it, starts with it or ends with it
    (``left_ear`` -> ear, ``forearm`` -> arm). Buckets are tried in
    :data:`SEMANTIC_KEYWORDS` order, so put specific buckets first.
    """
    tokens = _tokenize(name)
    for bucket, keywords in SEMANTIC_KEYWORDS.items():
        for keyword in keywords:
            needle = keyword.casefold()
            for token in tokens:
                if token == needle or token.startswith(needle) or token.endswith(needle):
                    return bucket
    return _OTHER_BUCKET


def suggest_aliases(stack: LayerStack) -> dict[str, list[str]]:
    """Group the stack's layer names into semantic buckets.

    Intended as a starting point for :func:`map_parts`' ``aliases`` argument:
    ``suggest_aliases(stack)["ear"]`` lists every layer that looks like an ear,
    so a rig can use it directly or after a manual edit.

    Buckets are ``ear``, ``tail``, ``hair``, ``head``, ``face``, ``arm``,
    ``leg``, ``body``, ``accessory`` and ``other``, matched by the keyword
    table :data:`SEMANTIC_KEYWORDS` (first matching bucket wins; see
    :func:`_bucket_for` for the matching rule). Empty buckets are omitted, the
    remaining ones keep the canonical order, and names without separators
    (e.g. ``"leftear"``) may land in ``"other"`` -- pass an explicit ``aliases``
    mapping to :func:`map_parts` when that matters.

    Returns:
        ``{bucket: [layer names]}`` in back-to-front order within each bucket.
    """
    buckets: dict[str, list[str]] = {bucket: [] for bucket in SEMANTIC_KEYWORDS}
    buckets[_OTHER_BUCKET] = []
    for name, _ in stack.layers:
        buckets[_bucket_for(name)].append(name)
    return {bucket: names for bucket, names in buckets.items() if names}


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m flat2rig.layers <stack.psd | layer-dir>
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m flat2rig.layers <stack.psd | layer-dir>")
        raise SystemExit(2)

    _stack = load_stack(sys.argv[1])
    print(f"{len(_stack.layers)} layer(s) from {_stack.source}, canvas {_stack.size}")
    for _index, (_name, _image) in enumerate(_stack.layers):
        print(f"  [{_index:>3}] {_name}  {_image.size[0]}x{_image.size[1]} {_image.mode}")
    for _bucket, _names in suggest_aliases(_stack).items():
        print(f"  {_bucket:<10} {', '.join(_names)}")
