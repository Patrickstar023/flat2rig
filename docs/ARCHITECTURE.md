# Architecture

For developers who will modify the code. `README.md` explains *what* flat2rig does and how to use
the CLI; this document explains *how it is built* and *why each choice was made*, so that a change
in one module does not silently break the contract with the next.

Read time: ~15 minutes. Everything here is normative unless marked **intent** (design target for
code that is still being written) or **estimate** (a number with a stated dependency).

---

## 1. Module map

```
src/flat2rig/
├── __init__.py    version + package docstring
├── rig.py         annotation JSON → Rig object → part masks (+ diagnose)
├── parts.py       masks → colour layers (cut + occlusion fill + z order)
├── render.py      layered transforms → frames, per state
├── layers.py      consume pre-layered input (see-through PSD / PNG stack)
├── preview.py     frames directory → self-contained preview.html
├── debug.py       human-verification images (diagnostic sheet, frame strip)
└── cli.py         argparse surface; the only module that does I/O orchestration
```

Data flows one way:

```
                   hero.rig.json
                         │
   hero.png ──► rig.load_alpha ──► rgba ──► rig.build_masks ──► masks ──┐
                         │                    │                        │
                         │                    └──► rig.diagnose ──► diag │
                         │                                             │
                         │             parts.order_parts ◄─────────────┤
                         │                                             │
                         └──► parts.cut_layers(rgba, masks) ──► layers │
                                        │                            │
                              parts.inpaint_hidden(...) ──► layers     │
                                                                      │
                              render.build_frames(rgba, rig, masks, layers, states)
                                        │
                                        ├──► frames/*.png        (cli.py or caller saves)
                                        ├──► debug.draw_frames_strip  (strip.png)
                                        └──► debug.draw_diagnostic    (diagnostic.png)
```

`cli.cmd_build` is the reference orchestration — copy its call order, not `examples/run_demo.py`'s
slightly different one (see §3.5).

### 1.1 `rig.py` — annotation → masks

| Symbol | Signature | Contract |
| --- | --- | --- |
| `Bone` | `dataclass(x0: float, y0: float, x1: float, y1: float)` | One line segment in source-image pixel coordinates. No normalisation, no y-flip. |
| `Part` | `dataclass(name: str, bones: list[Bone], pivot: tuple[float, float], z: int, blend: float, motions: dict[str, list[tuple[float, float]]])` | `blend` is a **half-width in px** (see §2.3). `motions[state][i] == (degrees, dy_px)`. |
| `Rig` | `dataclass(parts: list[Part], eyes: list[tuple[float, float, float]], states: dict[str, int], canvas: tuple[int, int] | None, body: tuple[float, float] | None)` | `parts` keeps **file order** (tie-break for equal `z`). `states[state]` is the per-frame delay in ms. |
| `load_rig(path)` | `str \| Path → Rig` | Parses + validates the annotation JSON. Raises `ValueError` on: unparsable JSON, missing `parts`, a part without `name` or `bones`, a bone that is not 4 numbers, `pivot` missing, a motion entry that is not `[angle, dy]`. **Must not import numpy/Pillow** — `flat2rig --help` has to work in a bare environment. |
| `load_alpha(path)` | `str \| Path → tuple[Image.Image, Image.Image]` | Returns `(rgba, alpha)`; the image is converted to `RGBA`, and `alpha` is the same image's `A` channel. Coordinates in the annotation are interpreted in this image's pixel space, so **never resize here** — it would silently invalidate every pivot the user annotated. |
| `build_masks(rgba, rig)` | `→ dict[str, Image.Image]` | One `L`-mode mask per part, keys in `rig.parts` order. Values are 0–255 weights, **not** boolean: `sum(masks.values()) == 255` on every opaque pixel, and `== 0` where the source alpha is 0 (see §2.2). |
| `diagnose(rgba, rig, masks)` | `→ dict` | Returns `{"coverage": float, "per_part_area": dict[str, int], "warnings": list[str]}`. `coverage` is the fraction of opaque source pixels that ended up in some mask; `per_part_area` counts mask pixels above a small threshold. Keys and shape are a public contract — `cli.py` prints them and writes them into `rig.json`. |

### 1.2 `parts.py` — masks → colour layers

| Symbol | Signature | Contract |
| --- | --- | --- |
| `order_parts(rig)` | `Rig → list[int]` | Part indices sorted back-to-front: ascending `z`, ties broken by file order. A stable sort — do not use `sorted(..., key=z)` on a dict. |
| `cut_layers(rgba, masks)` | `→ dict[str, Image.Image]` | For each part, RGB = source RGB, A = `mask`. Same keys/order as `masks`. Opaque pixels only; colour under transparent pixels is undefined and callers must not read it. |
| `inpaint_hidden(rgba, masks, order, radius=6)` | `→ dict[str, Image.Image]` | Returns a *new* layer dict where pixels hidden by a higher part are filled: mirror-symmetric copy first, boundary diffusion within `radius` as fallback (§2.4). Pixels that stay unknown keep alpha 0 — a hole is better than a smear. Does not mutate its inputs. |

`inpaint_hidden` is the only O(parts × pixels) stage in the pipeline; if a profile points there,
that is expected, not a bug (see §4).

### 1.3 `render.py` — layered transforms → frames

| Symbol | Signature | Contract |
| --- | --- | --- |
| `build_frames(rgba, rig, masks, layers=None, states=None)` | `→ dict[str, list[tuple[Image.Image, int]]]` | Outer key = state name, inner list = frames in play order. Each frame is `(RGBA image, delay_ms)`. Frame count per state = `len(motions[state])` of the part with the most entries for that state; delay = `rig.states.get(state, <default>)`. Results are fully deterministic; the same inputs always produce byte-identical PNGs. |

Behaviours a caller may rely on:

* `layers=None` → render slices `rgba` with `masks` on the fly. Passing `layers` (from
  `cut_layers` + `inpaint_hidden`) is what makes rotation reveal real content instead of the source
  pixels of the occluding part.
* A part that has no entry for a requested state **does not move in that state** (rotation 0,
  dy 0) rather than being dropped — `examples/*.rig.json` rely on this: `body` has no `error` key
  and `build --states idle,sleep,error,celebrate` still emits `error` frames.
* Rotation is clockwise-positive, in degrees, around `Part.pivot`, with the canvas size unchanged.
  Content rotated outside the canvas is clipped, not re-fitted.
* `states=None` means "all states in `rig.states`", in declaration order.
* The function never writes to disk; the caller owns all file I/O.

### 1.4 `layers.py` — consume pre-layered input

Used only by `flat2rig build --layers` (and by anyone with real see-through output).

| Symbol | Signature | Contract |
| --- | --- | --- |
| `load_stack(src)` | `str \| Path → list[Image.Image]` | Ordered **back-to-front**. `src` may be a directory of `.png` frames sorted by filename, a single `.psd` (requires `flat2rig[psd]`, i.e. `psd-tools`), or, **intent**, a single multi-layer file. Every returned image is RGBA at a common size. Raise a clear error when `psd-tools` is missing instead of a bare `ImportError`. |
| `map_parts(stack, names)` | `→ dict[str, Image.Image]` | Maps each requested part name to one layer from `stack`. Matching is by layer name, normalised (case-folded, non-alphanumerics stripped), with substring matching as a fallback. Unmatched names must be reported, and an empty `getbbox()` result means "this layer carries no pixels". This function is where a bad see-through layer naming convention surfaces as a wrong rig — keep it loud. |

### 1.5 `preview.py`

| Symbol | Signature | Contract |
| --- | --- | --- |
| `write_preview(frames_dir, out_path, fps=...)` | `→ Path` | Scans `frames_dir` for `<stem>_<state>_<NN>.png`, groups by `stem` and `state`, and writes one self-contained HTML file: every frame inlined as a base64 `data:` URI, no external requests, no JS dependencies. Opens correctly from `file://`. Returns the written path. |

### 1.6 `debug.py`

Already implemented; treat as frozen apart from additive changes:

| Symbol | Signature | Contract |
| --- | --- | --- |
| `draw_diagnostic(rgba, rig, masks, out_path)` | `→ Image.Image` | Three-panel sheet: original / per-part tint / bones + joints + pivots + eye circles. Colours come from `PALETTE` in `mask`-name sort order (panel 2) and `z` order (panel 3) — deliberately *not* the same indexing, because panel 3 must show paint order. Writes the PNG and also returns it. |
| `draw_frames_strip(frames, out_path, columns=6, background=(245,246,251))` | `→ Image.Image` | Contact sheet of one state's frames. Raises `ValueError` on an empty list. |

### 1.7 `cli.py`

`main(argv) -> int`; exit codes: `0` success, `2` (`_fail`) for input/annotation errors, `1` for an
uncaught exception (e.g. a missing module). Subcommands and the exact flags:

| Command | Positional | Options | Writes |
| --- | --- | --- | --- |
| `build` | `input` (PNG, or dir/PSD with `--layers`) | `-c/--config` (required), `-o/--output` (default `out`), `--layers`, `--states` (comma-separated, default `idle,sleep,error,celebrate`), `--inpaint-radius` (default `6`) | `frames/`, `masks/`, `rig.json` |
| `preview` | `frames` (directory) | `-o/--output` (default `<frames>/../preview.html`) | `preview.html` |
| `inspect` | `input` (PNG) | `-c/--config` (required), `-o/--output` (default `debug.png`) | the diagnostic PNG; prints coverage + per-part areas + warnings |
| `init` | `config` (path) | `--force` | the annotation template |

Two structural notes for anyone editing the CLI:

* Heavy imports (`numpy` via the other modules, `PIL`) are **deferred into the command bodies** so
  that `flat2rig --help` and `flat2rig init` work without dependencies. Keep it that way: only
  `argparse`, `sys` and `pathlib` may be imported at module scope.
* Only `cmd_build` wraps its imports in `try/except ImportError` and turns them into a friendly
  message. `cmd_preview` and `cmd_inspect` raise a raw `ModuleNotFoundError` when the package is
  not installed — a known rough edge, not intended behaviour. `cmd_build` also creates
  `<output>/{frames,masks}` *before* validating the input path, so a failed run can leave empty
  directories behind.

---

## 2. The algorithm

### 2.1 Nearest-bone labelling

**What happens.** Each bone is a segment. Every pixel that the source alpha marks as character is
assigned to the segment with the smallest Euclidean distance to that pixel (point-to-segment, so
the perpendicular projection is clamped to the segment's endpoints). Pixels with alpha 0 are never
labelled. Within one part, all bones vote for the same part, so the label is a per-part mask; the
pivot of the winning part is the joint that later rotates the pixel.

**Why the distance field, not per-pixel Python.** The naive form is a loop over `H × W`, computing
`B` distances per pixel. In pure Python that is ~10⁵–10⁶ interpreted iterations per character —
seconds to minutes, which destroys the "seconds per character" property. Two vectorised shapes are
acceptable:

* **Per-bone vectorised distance** over the character-pixel index list (the three example
  characters, `B = 3–3` bones): one temporary of `P` floats per bone, `B` passes, `argmin` at the
  end. Cheap for `B ≲ 16`.
* **Multi-source BFS** on the alpha support with 4-connected steps, all bones seeded at distance 0
  and expanding together. Each pixel is settled once, so the cost is `O(P)` with a single `int32`
  label map and a `deque` — no `(B, H, W)` volume at all. This is the form to reach for when the
  annotation has many bones, and it is the reason the contract is phrased as "nearest bone" rather
  than "nearest pixel": BFS gives the geodesic distance *inside the silhouette*, so a bone that
  belongs to an arm crossing behind the head cannot grab head pixels across an air gap.

The forbidden shape is materialising a full `(B, H, W)` float volume and taking the `argmin`: for
`B = 8` at 1024² that is a 67 MB temporary per distance evaluation, and it is recomputed for every
part. The mask contract only cares about the **partition**, so only the winning index has to be
kept.

Consequence to remember when changing this: the two forms disagree on concave silhouettes and can
disagree on ties (a pixel equidistant from two bones). Break ties deterministically — by part order
in the annotation — otherwise rigs stop being reproducible.

### 2.2 Why masks are renormalised to sum to the alpha channel

The labeller produces an unnormalised weight per mask; after the soft blend band (§2.3) a boundary
pixel can belong to two parts at once. Let `m_i(p)` be the raw weight of part `i` at pixel `p` and
`a(p)` the source alpha (0–255). The masks are stored as

```
m_i'(p) = m_i(p) / Σ_j m_j(p) · a(p)        (where Σ_j m_j(p) > 0)
```

Two failure modes are prevented by the `a(p)` factor:

* **Pixel loss** — if masks are normalised to sum to 1 but never multiplied by alpha,
  semi-transparent edge pixels become opaque, which shows up as a hard, over-dark outline.
* **Pixel double counting** — if they are not normalised at all, a boundary pixel is written into
  two layers at full weight, and source-over compositing adds it twice: the seam appears as a
  darker/brighter ridge whenever the two layers are not perfectly aligned (i.e. as soon as anything
  rotates).

With the formula above, `Σ_i alpha_i'(p) == a(p)` exactly in real arithmetic, so the identity that
"the union of the parts is the whole character" survives the split. In practice the layers are
quantised to `uint8`, so a boundary pixel can be off by ±1 level; that is 1/255 of the alpha and is
not visible. If you ever see the character get *lighter* at a seam, you have introduced rounding
into the renormalisation (e.g. by converting to `uint8` before dividing) — do the arithmetic in
`float32`/`float64` and convert once, at the end.

### 2.3 Soft blend bands, and how the band width interacts with rotation

A hard label boundary is a step edge. Rotating one side of a step edge opens a visible slit at the
seam. The `blend` field (default 8–12 px in the examples) is the **half-width** of a transition
band placed along the part boundary, so a pixel at the boundary gets, e.g., 0.7 of part A and 0.3 of
part B. The band is then renormalised as in §2.2, and both sides rotate by the same transform
where the weight is 1 and by progressively less across the band: the colour cross-fades instead of
tearing.

The quantitative rule of thumb: a point at radius `r` from the pivot moves by `d ≈ r · θ` for a
rotation of `θ` radians. The band hides a displacement of at most about its own width, so

```
blend_half_width  ≳  2 · r_max · θ_max · π/180        (blend in px, θ in degrees)
```

Worked example: an ear whose tip is 40 px from the ear base, swinging ±14° (see
`examples/daermaodou.rig.json`) moves ~10 px at the tip, ~2 px at 8 px from the pivot. The band
only has to cover the *joint-adjacent* displacement, not the tip, which is why `blend: 10` is
comfortable there. Two consequences worth internalising:

* **`blend` is local, rotation is global.** A large amplitude on a far-away part does not need a
  wide band; a large amplitude *near the pivot* does.
* **Wider is not better.** A band wider than the small parts it touches makes those parts
  semi-transparent as a whole (a 10 px band on a 12 px ear tip dissolves it). If a seam still
  shows, first reduce the rotation amplitude near the joint, then widen the band, then consider a
  joint stub patch (§2.5).

### 2.4 Occlusion fill

When the arm is cut out and rotated, the pixels it used to cover on the body are gone. The source
illustration does not contain them, so the only options are to synthesise plausible content or to
leave a hole; flat2rig synthesises, per part, back-to-front, only where the source is opaque and
the target mask is not:

1. **Mirror-symmetric copy.** For a pixel hidden by a higher part, reflect it across the axis of
   the covering part's bone (equivalently, across the part's local symmetry line) and, if the
   reflected source pixel is opaque *and* already assigned to this lower part, copy its colour.
   Animals and cartoon characters are near-symmetric at the scale of a part boundary, so this is
   the highest-quality cheap guess and it preserves the exact palette and line weight.
2. **Boundary diffusion (fallback).** Pixels that mirroring cannot resolve grow inward from known
   neighbours: iterate up to `--inpaint-radius` (default 6) times, each pass averaging known
   neighbours into one more ring of unknown pixels. This is a small inpainting kernel, not a
   texture synthesiser: it fills flat regions convincingly and smears fine detail.

**Why this is a plausible reconstruction and not ground truth.** Both branches invent pixels that
were never observed. Mirroring assumes local symmetry that the artist may not have drawn (a
deliberately asymmetric marking mirrored across the seam is simply wrong), and diffusion is a
smoothness prior that cannot recover a pattern it has never seen. The output is *consistent* — it
keeps palette, lighting and line style — and it is *stable* — same input, same output — but it is
an extrapolation. Anything that needs the true hidden content (a striped tail behind the body, a
logo on a covered wing) must come from real layers: run see-through, then `--layers`. Keeping this
honest in the UI is a feature: `diagnose()` should be able to report the filled pixel count, and
users should be told that a large filled fraction means a low-confidence rig.

### 2.5 Paint order, z semantics, and the joint "stub" patch

`z` is the paint order: **higher `z` is in front**. Rendering sorts back-to-front by ascending `z`
and composites with alpha, i.e. `z = 0` is drawn first and can be covered by everything. Equal `z`
is resolved by file order (first declared wins the back position). Because every frame's canvas is
the same size as the source and nothing is re-fitted, a part with a high `z` can cover another
part entirely — that is a legal but usually unintended rig; `diagnose()` should warn when a part's
visible area is a small fraction of its mask area.

Two properties follow from compositing *masks* rather than *groups*:

* Mask overlap is resolved by `z`, not by the artist's intention. If two parts both claim a pixel
  at the same `z`, whichever is declared first is hidden; there is no blending between them beyond
  the `blend` band.
* A part that rotates pulls its mask away from the joint it used to cover, exposing a wedge of the
  part behind it (and the background). Widening `blend` reduces the visible seam but cannot fill a
  wedge whose apex is at the pivot.

The **joint stub patch** addresses this: for each part, a thin patch of pixels is copied around the
`pivot` from the shared boundary region and is *not* rotated (or is rotated by a fraction of the
part's rotation), so the exposed wedge is covered by content that plausibly belongs there — the
same colour as the pixels either side of the joint. It hides rotation gaps in the common case where
the pivot sits on a smooth, locally uniform region (a neck, an ear base, a stem). It is a patch,
not a fix: for a pivot on a high-contrast boundary the stub either mismatches colour or is wide
enough to look like a smear. Keeping the stub small and taking its colour from the band-normalised
masks (rather than from the source pixels under the pivot, which may belong to the covering part)
is what keeps it invisible.

---

## 3. Data formats

### 3.1 Annotation JSON (`hero.rig.json`)

Input file, written by the user (or by `flat2rig init`) and parsed by `rig.load_rig`. All
coordinates are in **source-image pixels**, origin top-left, y down.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `canvas` | `[w, h]` | no | Declared design size. Used as a validation hint — a `bones`/`pivot` outside it is a likely mistake. Not used to resize anything. |
| `body` | `[x, y]` | no | Ground contact point (the "feet"): handy for placing the character in a window and for a sanity check that the body bone ends near it. |
| `parts` | array | **yes** | One object per animatable part. At least one; order is significant (z tie-break). |
| `parts[].name` | string | **yes** | Key used for mask filenames (`masks/<name>.png`), the `--layers` name match, and the preview grouping. Keep it filesystem-safe: `[a-z0-9_-]`. |
| `parts[].bones` | array of `[x0,y0,x1,y1]` | **yes** | One or more segments that "skeletonise" the part. Two segments for a V-shaped ear, three for a crown with leaves. Pixel ownership is nearest-bone. |
| `parts[].pivot` | `[x, y]` | **yes** | The joint this part rotates around: ear base, stem root, neck point. Must lie at or inside the part's mask, otherwise rotation exposes a gap (§2.5). |
| `parts[].z` | int | no (default 0) | Paint order, higher = in front. Body/skin 0, ears/crown 1, head 2 in the examples. |
| `parts[].blend` | number | no (default 8) | Half-width of the soft boundary band, px (§2.3). |
| `parts[].motions` | `{state: [[deg, dy], ...]}` | **yes** | Per state, the frame list. Length = frame count for that state. `deg` is clockwise-positive rotation about `pivot`; `dy` is a vertical offset in px (negative = up). A state missing here means "this part is static in that state". |
| `eyes` | `[[cx, cy, half], ...]` | no (default `[]`) | Blink/closed-eye overlays. `half` is the circle radius in px. Empty is legal and used by two of the three shipped examples. |
| `states` | `{state: delay_ms}` | no | Playback delay per frame. Also fixes the *set* of states the CLI builds when `--states` is omitted (in declaration order). |
| `_comment` | string | no | Free-form note. Ignored by the loader, kept for humans; the shipped examples use it to record which character and calibration the numbers came from. |

### 3.2 `rig.json` (output, resolved rig)

Written by `cli.cmd_build` next to `frames/`. It is **not** re-readable as an annotation: it carries
the resolved information and no bone geometry (bones belong to the annotation, masks belong to the
PNG files). Field by field, matching the writer in `cli.py`:

| Field | Notes |
| --- | --- |
| `source` | `src.name` — the input filename only, not a path. Good for provenance, not for re-running. |
| `size` | `[w, h]` of the loaded RGBA, i.e. the actual source size (may differ from `canvas`). |
| `parts[]` | `{name, pivot, z, blend, motions}` — the resolved part list in rig order; `motions` values are arrays of `[deg, dy]`. |
| `eyes` | The eye circles, as arrays. |
| `states` | The delay table from the annotation. |
| `diagnostics` | `{coverage, per_part_area, warnings}` — the exact `diagnose()` result. When `--layers` is used, `coverage` is `1.0` by construction and `per_part_area` is recomputed from the supplied layer alphas, because no decomposition happened. |

### 3.3 Frames layout

```
out/
├── frames/
│   └── <stem>_<state>_<NN>.png     RGBA, same size as the source, NN 1-based, zero-padded to 2
├── masks/
│   └── <name>.png                  L-mode (grayscale) weight map, 0–255, one per part
├── rig.json                        resolved rig + diagnostics
└── preview.html                     written by `flat2rig preview`, not by `build`
```

* `<stem>` is the input filename without extension (`hero.png` → `hero`).
* `<state>` is the state name verbatim, so a state name must not contain `_`; the preview parses
  the filename by splitting on it.
* Frame delay is **not** in the filename. It is in `rig.json` (`states`) — write a manifest if you
  need it per frame, otherwise the PNG mtimes lie.
* `build` writes `frames/`, `masks/` and `rig.json`; `diagnostic.png` and `strip.png` come from
  `debug.py`, which the CLI does not call (only `examples/run_demo.py` does). If you want them for
  a normal build, call `draw_diagnostic`/`draw_frames_strip` explicitly.

### 3.4 `--layers` input

A directory of RGBA `.png` (or `.psd`, with `flat2rig[psd]`) sorted by filename, interpreted
back-to-front. No decomposition is performed; layer names must match part names closely enough for
`layers.map_parts`. When this path is taken, `coverage` in `rig.json` is trivially `1.0`, so read
`per_part_area` instead to see whether the name matching actually found anything.

### 3.5 Note for API users

`examples/run_demo.py` and `cli.cmd_build` call the same modules but differ in two details:
`run_demo.py` imports from `flat2rig.*` after `sys.path.insert(0, "src")` (works without an
install), and it writes `diagnostic.png` / `strip.png` itself. If you are adding a stage, add it to
`cmd_build` first — that is the path users' scripts depend on.

---

## 4. Performance

**What dominates.** Three costs, in this order:

1. **Occlusion fill** (`inpaint_hidden`) — `parts × hidden_pixels × radius` neighbour reads. This is
   the one stage that grows with the *part count* as well as the pixel count, and it is the stage
   most likely to be optimised later (it is embarrassingly parallel per part).
2. **Per-frame rotation + resampling** — PIL's `rotate`/`affine` per part per frame. 13 frames ×
   3 parts = 39 resamples of a full-canvas RGBA image, most of them 90 % transparent. To speed it
   up, crop each layer to its bounding box, transform the crop, and paste back at the transformed
   offset.
3. **Mask building** — vectorised distance or BFS over the character pixels (~22 k pixels for a
   240² demo character, ~400 k for 1024²).

**Expected timings.** These are **estimates from a throwaway benchmark that mirrors the described
stages** (multi-source labelling, per-part Gaussian-widened blend, boundary diffusion, one rotate +
composite), run on this machine (Windows, CPython 3.12, numpy 2.5, Pillow 12.3, no GPU). They are
one `perf_counter` sample per stage, so treat them as order-of-magnitude only.

| Stage (synthetic 3-bone character) | 240 px | 1024 px |
| --- | --- | --- |
| Nearest-bone labelling (vectorised, 3 bones) | ~1.5 ms | ~60 ms |
| Soft blend + renormalise (per-part full-canvas blur) | ~3 ms | ~70 ms |
| Boundary diffusion, radius 6 | ~0.3 ms | ~4 ms |
| One rotate + composite of one layer | ~4 ms | ~73 ms |
| **Full `build`, 4 states, 13 frames, 3 parts** | **~0.15–0.4 s** | **~1.5–4 s** |

The full-build figures are extrapolated from the per-stage numbers, not measured end-to-end (the
end-to-end path needs the real modules). What moves them: number of parts and frames (linear),
whether `--layers` is used (skips labour-heavy stages only if the input is already layered), the
`--inpaint-radius` (linear in fill cost), and whether your Pillow is SIMD-optimised. PNG *encoding*
is a separate cost and is often comparable to the rendering at 1024² — `cli.cmd_build` saves with
`optimize=True`, which buys size at the cost of CPU.

The README's "~1–2 s per character" is consistent with these numbers at 240 px with a real
character and 4 states.

**Memory notes.** Roughly, with `P` = character pixels, `H×W` = canvas, `K` = parts:

| Buffer | 240² | 1024² |
| --- | --- | --- |
| Source RGBA (`uint8`) | 0.23 MB | 4 MB |
| `K` masks (`L`, `uint8`) | 0.06 MB × K | 1 MB × K |
| `K` colour layers (RGBA, `uint8`) | 0.23 MB × K | 4 MB × K |
| **`K` blend weights as `float64` over the full canvas** | **0.46 MB × K** | **8.4 MB × K** |
| One `float64` `H×W` temporary (e.g. a distance field) | 0.46 MB | 8.4 MB |

The dominant term at high resolution is *any* float buffer allocated over the full canvas rather
than over the character mask: at 1024² each is 8.4 MB, and it is easy to hold `K` of them plus a
few temporaries. Rules to keep memory flat: compute on the character-pixel index list, use
`float32` where the extra precision buys nothing, and never allocate a `(B, H, W)` distance volume.
One frame is ~4 MB as RGBA at 1024², so holding every state in memory (as `build_frames` does) is
`frames × 4 MB` — a 100-frame character at 1024² is ~400 MB. Stream to disk instead if that becomes
real.

---

## 5. Extension points

### 5.1 A new motion curve type

Today `motions[state]` is an explicit per-frame list, which is why the annotation stays tiny and
reproducible. To add procedural motion (sine sway, breathing, damped bounce, a springy follow):

1. Add a curve family in `render.py`, e.g. `def sample_curve(kind, amp, phase, period, t) -> float`.
   It must be **pure and deterministic** (no wall clock, no unseeded RNG) and must return
   `(deg, dy)` in the same units as the explicit lists.
2. Add an annotation form for it that `rig.load_rig` accepts *in addition to* the list form, and
   normalise it to a list of frames at load time. That keeps `render.build_frames` unchanged, keeps
   `rig.json` output stable, and means a curve is expanded once rather than re-sampled by every
   consumer.
3. Keep the numeric contract: frames are clockwise degrees and vertical px offsets. Curves that
   need horizontal motion or scale are a different feature — add them to the tuple *and* bump the
   documented schema, do not overload `dy`.
4. Test determinism: the same annotation must give byte-identical PNGs across runs.

### 5.2 A new backend (input source)

`layers.load_stack` is the seam. To add a format (OpenRaster, Krita, WebP layers, a sprite sheet
plus a rect manifest, a clip-studio file):

1. Write a loader that returns `list[Image.Image]` in **back-to-front** order, all RGBA, all the
   same size, and register it in `load_stack`'s dispatch by suffix/type.
2. Never crop or rescale — the annotation coordinates must keep meaning the same thing as in the
   single-image path.
3. Handle the optional-dependency case explicitly: raise a message naming the extra
   (`pip install flat2rig[psd]`) instead of letting `ImportError` escape into `cli.py`.
4. Consider whether `map_parts` can match your names; if not, extend the normalisation there rather
   than special-casing per format.

For a *new decomposition* backend in the other direction (calling a model to produce layers, then
rigging them), keep the model out of this package: produce a layer directory and use `--layers`.
That preserves the "no weights, no GPU" property of the default install.

### 5.3 Exporting to another format

The frame list from `build_frames` is the interchange point. To add GIF/APNG/sprite-sheet/CSS or a
game-engine atlas:

1. Write an exporter next to `preview.py` (e.g. `export.py`) taking `frames` (or a frames
   directory) plus an output path.
2. Get the per-frame delays from `rig.states` / the `(image, delay)` tuples, not from PNG metadata.
3. Keep it dependency-light — Pillow can write GIF and APNG; a sprite sheet is a paste plus a JSON
   rect list; do not pull in a package for this.
4. Add a CLI subcommand only if it is useful standalone; otherwise document the API call.
5. Do not put frame pacing in the exporter that the renderer should own, and vice versa.

A Live2D `.moc3` exporter or an SVG exporter are explicitly out of scope (see `README.md`): they
need runtime formats and vectorisation decisions that the layer model here does not carry.

---

## 6. Failure modes and diagnostics

### 6.1 What `diagnose()` warns about

`diagnose(rgba, rig, masks)` returns `{"coverage", "per_part_area", "warnings"}`. `coverage` is the
fraction of opaque source pixels covered by at least one mask; `per_part_area` is the per-part pixel
count; `warnings` is a list of human-readable strings, printed by `inspect`/`build` and copied into
`rig.json`. The intended checks:

| Warning | Trigger | Usually means |
| --- | --- | --- |
| Low coverage | `coverage ≲ 0.98` | Some opaque pixels are in no mask — a bone missing from the annotation, or a part whose bones never win a region. |
| Part is nearly empty | `per_part_area[name]` is a tiny fraction of the character | The bone is buried inside another part's territory (often a nested part: an ear inside a head) or the pivot/bone is off the character. |
| Part overlaps a neighbour too much | A large share of two parts' masks coincide | Two bones run too close together; the boundary is arbitrary. Move the bones apart or remove the redundant one. |
| Mask outside the silhouette | A mask has weight where source alpha is 0 | The part will paint over transparent areas when it rotates — visible as a coloured halo. |

Treat a non-empty `warnings` list as "the rig will build, but check the diagnostic sheet before
trusting it": every warning here is cheaper to fix in the annotation than in the output.

### 6.2 How to read `inspect` output

`flat2rig inspect hero.png -c hero.rig.json -o debug.png` prints coverage and per-part areas and
writes the three-panel sheet from `debug.draw_diagnostic`:

| Panel | What a healthy picture looks like | What a broken one looks like |
| --- | --- | --- |
| Left, labelled 原始立绘 ("original illustration") | Your illustration, unchanged. | If it is blank, the alpha channel is missing or fully transparent — a palette/indexed PNG saved without alpha. |
| Middle, labelled 部位着色 ("part tint") | Flat, contiguous colour regions with a *soft* but narrow border; the tint follows the drawn shapes. | Ragged stripes or a jagged frontier = bones that do not lie along the part's medial axis. A region coloured like a neighbouring part = a bone of the other part is closer to those pixels. |
| Right, labelled 骨骼与关节 ("bones and joints") | Each bone runs down the middle of its part; every pivot circle sits on the joint (ear base, neck, stem root); eye circles land on the eyes. | A bone floating off the body; a pivot outside the part or displaced away from the joint (rotation will expose a gap); a pivot at the part's *centre of mass* instead of its *joint* — the part then rotates about the wrong point and slides. |

`inspect` and `build` compute masks the same way (`load_alpha` → `build_masks` → `diagnose`), so if
`inspect` looks wrong, `build` will be wrong. Use it as a fast loop: edit JSON → `inspect` → look →
repeat. Only then run `build`.

### 6.3 The three most common annotation mistakes

1. **Pivot at the centre of mass instead of the joint.** The part rotates about its middle, so it
   translates away from the neighbour it should stay attached to, and the seam opens even at small
   angles. Fix: put the pivot at the ear base / neck / stem root, then confirm on the right panel
   that the circle sits on the joint.
2. **A bone that does not follow the part's medial axis.** "Nearest bone wins" is unforgiving: a
   bone drawn along the outline of an ear steals half the head, and a single bone for a V-shaped
   ear splits the V between the two prongs. Fix: draw one segment per prong, and make sure each
   segment is inside the region you want to own.
3. **Fighting over the joint region (missing or excessive `blend`).** With no band the seam is a
   hard step and any rotation tears it; with a band much wider than the small parts it touches, the
   ear tip becomes semi-transparent. Fix: `blend` 8–12 px for a 240 px character, scaled with
   `r_max · θ_max` (§2.3), and re-check the middle panel — the colour cross-fade should be visible
   but narrow.

Honourable mentions: a `motions` entry whose list length differs between parts (the shorter part
simply stops early — usually fine, but confusing when it is not intended); a state listed in
`--states` that no part defines (all parts static → a still frame); and a state name containing `_`,
which breaks both the frames filename scheme and the preview parser.
