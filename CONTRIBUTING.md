# Contributing to flat2rig

Practical guide. If you only read one section, read §1 (environment) — this machine has a quirk
that breaks `pip` and `pytest` in confusing ways if you skip it.

By contributing you agree your changes are licensed under the project's MIT license (`LICENSE`).

---

## 1. Development setup

### 1.1 The TEMP quirk (read this first)

On this Windows machine, builds and test runs that write to the system temp directory are not
reliable: the sandbox/permissions state of `C:\temp` depends on which account and which shell
started the process, and when a write is denied, `pip` and `pytest` fail in ways that look
unrelated to temp: `Access is denied`, `[Errno 13]`, a download that hangs during "Installing build
dependencies", or `pytest` dying while collecting tests. That is why this repo keeps a
project-local `.tmp/` (gitignored via the `.gitignore` entry) and sets the temp variables at it
per shell — it also keeps wheel caches out of your user profile, which matters on a machine where
`C:\temp` may be readable but shared.

Run this **once per shell, before creating the venv or installing anything**:

```powershell
# From the repo root: C:\Users\<you>\Desktop\毕业论文\flat2rig
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TMP  = (Resolve-Path .tmp).Path
$env:TEMP = (Resolve-Path .tmp).Path
```

`New-Item ... -Force` is not optional: `pip` will not create the directory chain for you, it will
just fail to write. Verify it took effect (`Resolve-Path` returns an absolute path):

```powershell
$env:TEMP
# C:\Users\<you>\Desktop\毕业论文\flat2rig\.tmp
```

If you prefer it permanent for this checkout, set it in the same shell you use for every
install/test command, or export it from your PowerShell profile. Do not commit anything from
`.tmp/` — it holds wheel caches (`pipcache/`) and throwaway scripts.

### 1.2 Create the venv and install

```powershell
# still in the repo root, with TMP/TEMP pointed at .tmp\
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,psd]"
```

Notes:

* Prefer invoking `.\.venv\Scripts\python.exe -m pip ...` over activating the venv. It is one less
  thing that can silently point at the wrong interpreter, and it is what the examples/CI commands
  below use.
* `[dev]` adds `pytest`; `[psd]` adds `psd-tools` and is only needed when you touch
  `flat2rig.layers` or test `build --layers` with a real PSD. Use plain
  `pip install -e .` if you want the minimal dependency surface (numpy + Pillow + scipy).
* Python **3.10+** is required (`pyproject.toml`, `requires-python = ">=3.10"`). The local venv here
  is CPython 3.12.
* `pyproject.toml` uses a `src/` layout: the package lives in `src/flat2rig/`, and the editable
  install is what makes `import flat2rig` work. Without it, use
  `$env:PYTHONPATH = "src"` or the `sys.path.insert` trick in `examples/run_demo.py`.
* If you add a lint/format step locally, install `ruff` into the venv too — it is not a project
  dependency, only the configured tool (`[tool.ruff]` in `pyproject.toml`).

### 1.3 Smoke-test the install

```powershell
.\.venv\Scripts\python.exe -m flat2rig.cli --help     # must work with no deps installed at all
.\.venv\Scripts\python.exe -m flat2rig.cli init .tmp\smoke.rig.json --force
```

`--help` and `init` import nothing heavy on purpose (see `docs/ARCHITECTURE.md` §1.7): they are the
fastest way to tell "package not importable" apart from "dependency missing".

---

## 2. Running tests

```powershell
.\.venv\Scripts\python.exe -m pytest                      # everything
.\.venv\Scripts\python.exe -m pytest -m "not slow"        # skip tests marked slow
.\.venv\Scripts\python.exe -m pytest tests/test_rig.py -k pivot -x
```

`pyproject.toml` already configures pytest: `testpaths = ["tests"]` and a `slow` marker ("takes more
than a couple of seconds"). Use `@pytest.mark.slow` for anything full-pipeline or 1024².

**Current state: `tests/` does not exist yet**, so `pytest` exits with "no tests ran". Treat the test
layout below as the target and add the file for whatever you touch:

```
tests/
├── test_rig.py       annotation parsing/validation, mask partition properties
├── test_parts.py     cut + occlusion fill invariants
├── test_render.py    frame counts, delays, determinism (same input ⇒ same bytes)
└── test_cli.py       subcommand surface, exit codes, output layout
```

Rules that keep the suite useful and fast:

* **Synthesise your input image in the test** (`PIL.Image.new` + `ImageDraw`, a handful of shapes).
  Do not add art to the repo: `examples/art/` is gitignored because the demo illustrations are
  third-party copyrighted (`examples/README.md`).
* Assert on **properties**, not golden pixels, wherever you can: masks sum to the alpha channel,
  every opaque pixel is labelled, `build_frames` returns `len(motions[state])` frames, equal inputs
  produce equal output. Golden-image tests are brittle across Pillow versions.
* Keep the whole non-slow suite under a couple of seconds; move full `build` runs to `slow`.

### 2.1 The manual loop you will actually use

While the rig modules are in flux, the fastest verification is the CLI itself:

```powershell
.\.venv\Scripts\python.exe -m flat2rig.cli inspect .tmp\hero.png -c .tmp\hero.rig.json -o .tmp\debug.png
.\.venv\Scripts\python.exe -m flat2rig.cli build   .tmp\hero.png -c .tmp\hero.rig.json -o .tmp\out --states idle,sleep
.\.venv\Scripts\python.exe -m flat2rig.cli preview .tmp\out\frames
```

`inspect` prints coverage, per-part pixel counts and `diagnose()` warnings, and writes the
three-panel diagnostic sheet. Never run `build` first — always look at the sheet
(`docs/ARCHITECTURE.md` §6.2 explains how to read it).

`examples/run_demo.py` runs the full pipeline over the three shipped characters and writes
`diagnostic.png`, `strip.png`, `frames/`, `masks/`, `rig.json` and `preview.html` per character. It
needs `examples/art/<name>-idle.png` (see `examples/README.md`; the script tries to copy them from
the sibling `whale-pet-redesign/pet-art/` checkout, and asserts if they are missing).

---

## 3. Project layout

```
flat2rig/
├── pyproject.toml            build config, deps, pytest + ruff settings
├── README.md                 user-facing: pitch, CLI, annotation schema, scope
├── docs/
│   ├── ARCHITECTURE.md       module contracts, algorithm, formats, performance
│   └── COMPARISON.md         positioning vs see-through / psd2live / VPet / BongoCat
├── src/flat2rig/
│   ├── __init__.py           version
│   ├── rig.py                annotation → part masks (+ diagnose)
│   ├── parts.py              masks → colour layers + occlusion fill
│   ├── render.py             layered transforms → frames
│   ├── layers.py             pre-layered input (see-through PSD / PNG stack)
│   ├── preview.py            frames → self-contained preview.html
│   ├── debug.py              diagnostic sheet + frame strip
│   └── cli.py                the only I/O orchestration; argparse surface
├── examples/
│   ├── README.md             how to run, how to annotate
│   ├── run_demo.py           full pipeline for three characters
│   └── *.rig.json            annotation files (art itself is not committed)
└── .tmp/                     local scratch: TMP/TEMP target, wheel cache (gitignored)
```

`docs/ARCHITECTURE.md` §1 lists the exact function-level contract of each module. If your change
alters a signature, a return shape or a documented invariant, update that section **in the same
pull request** — the doc is the contract, not an afterthought.

---

## 4. Code style

| Rule | Detail |
| --- | --- |
| Line length | **110** columns (`[tool.ruff] line-length = 110`). |
| Target | Python 3.10 (`target-version = "py310"`): use `X \| Y` unions, `list[str]`, `dict[str, int]`. |
| Type hints | On every public function, including `-> None`. Start each module with `from __future__ import annotations`. |
| Docstrings | **English**, module-level and on public functions/classes. Say what the contract is and why the non-obvious choice was made; skip restating the signature. |
| Imports | Standard library, then third-party, then local — and keep heavy imports (`numpy`, `PIL`) out of module scope in `cli.py` and `rig.py` so the CLI keeps working without dependencies. |
| Text | Keep user-facing CLI strings and comments consistent with what is already there; all *new* prose in code and docs is English. |
| Dependencies | Do not add a runtime dependency for something the standard library or `numpy`/`Pillow` already does. Optional features go behind an extra in `pyproject.toml` (as `psd-tools` does) and must fail with a message that names the extra. |

Run the linter before pushing:

```powershell
.\.venv\Scripts\python.exe -m ruff check src examples
.\.venv\Scripts\python.exe -m ruff format --check src examples   # if you use the formatter
```

Known inconsistency, so you are not surprised: the existing modules (`cli.py`, `debug.py`,
`__init__.py`) carry **Chinese docstrings and comments**, while this guide mandates English for new
work. Both can be true only during the transition — when you touch a file, translate the lines you
are already editing. Do not open a PR that is only a mass re-translation; that review is not worth
the merge conflict.

Performance changes: `docs/ARCHITECTURE.md` §4 says what dominates and which allocations are
forbidden (no `(bones, H, W)` float volume, no full-canvas float64 buffers you can avoid). If you
optimise, keep the output byte-identical — a "faster" rig that renders differently is a bug.

---

## 5. Adding an example character

You do not need to add art to the repo to try this; use any transparent PNG and a scratch
annotation under `.tmp/`. Only `*.rig.json` files are committed under `examples/`.

1. **Prepare the art.** One flat illustration on a transparent background, character roughly
   filling the canvas, saved as RGBA PNG (not indexed/palette: a palette PNG without alpha will
   look blank in the diagnostic sheet). Note the pixel size — every coordinate you type is in that
   pixel space.
2. **Write a starting annotation.**

   ```powershell
   .\.venv\Scripts\python.exe -m flat2rig.cli init .tmp\mychar.rig.json
   ```

   `init` writes the template from `cli.TEMPLATE`: a 240×240 canvas, an `ear` (z 2, blend 10) and a
   `body` (z 0), four states with delays, and two eye circles. Edit it — see §5.1.
3. **Verify with `inspect`, fix, repeat.**

   ```powershell
   .\.venv\Scripts\python.exe -m flat2rig.cli inspect .tmp\mychar.png -c .tmp\mychar.rig.json -o .tmp\debug.png
   ```

   Read the three panels left to right (original / part tint / bones+joints) and the printed
   coverage, per-part areas and warnings. Iterate on the JSON only — do not touch code to fix a
   bad split. `docs/ARCHITECTURE.md` §6.2 has the "what broken looks like" column, §6.3 the three
   most common mistakes (pivot at centre of mass, bone off the medial axis, wrong `blend`).
4. **Tune the motion.** Only once the split is clean:

   ```powershell
   .\.venv\Scripts\python.exe -m flat2rig.cli build .tmp\mychar.png -c .tmp\mychar.rig.json -o .tmp\out
   .\.venv\Scripts\python.exe -m flat2rig.cli preview .tmp\out\frames
   ```

   `motions[state]` is a list of `[degrees, dy]`; the list length is the frame count. Start small
   (±4–8° on ears, ±1–2 px on the body) — large amplitudes near a pivot are what tear the seam
   (§2.3 of the architecture doc). A part with no entry for a state simply holds still in it.
5. **Add it to the demo** (optional, only if you have redistributable art). Put the annotation at
   `examples/mychar.rig.json` and add the character name to the `CHARACTERS` list in
   `examples/run_demo.py`; the script expects `examples/art/mychar-idle.png`, which stays
   gitignored. Mention in `examples/README.md` where the art comes from.
6. **Commit** the JSON (and any doc updates), never the art.

### 5.1 Annotation walkthrough (fields that matter)

| Field | Practical guidance |
| --- | --- |
| `canvas` | The design size. Validation hint only; keep it equal to the art size so "coordinate 260 in a 240 canvas" stands out. |
| `body` | Ground-contact point (feet/root). Useful anchor; keep it near the bottom of the canvas. |
| `parts[].name` | `[a-z0-9_-]` only — it becomes a filename (`masks/<name>.png`) and a preview key. |
| `parts[].bones` | One segment per "prong", drawn down the middle of the region that should own pixels. Two bones for a V ear, three for a crown. |
| `parts[].pivot` | The **joint**, not the centre: ear base, neck point, stem root. Confirm the circle in panel 3 sits on the joint. |
| `parts[].z` | Higher is in front. Typical: body 0, ears/crown/accessories 1, head 2. |
| `parts[].blend` | Half-width in px of the soft seam. 8–12 at 240 px; scale it with `r_max · θ_max` (architecture §2.3). |
| `parts[].motions` | Per state, a list of `[degrees clockwise, dy px]`. Missing state ⇒ static part. |
| `eyes` | `[cx, cy, half]` circles for the blink overlay. Optional; `[]` is normal for characters without a visible eye overlay. |
| `states` | `{name: delay_ms}`. Also defines the state set when `--states` is omitted. State names must not contain `_`. |
| `_comment` | Free-form provenance note (what the numbers were calibrated against). Loaders ignore it; reviewers appreciate it. |

### 5.2 The verification loop, condensed

```
init  →  edit bones/pivot  →  inspect  →  (look at panel 2 and 3)  →  edit  →  inspect  →  build  →  preview
         └───────────────── only JSON changes ─────────────────┘        └── only motions from here ──┘
```

If `inspect`'s middle panel is clean and its right panel has every pivot on a joint, the rig will
render correctly. If not, no `motions` tuning will save it.

---

## 6. Commit messages

Conventional Commits, English, imperative mood, subject ≤ 72 characters, no trailing period:

```
<type>(<scope>): <subject>

<body: what changed and why, wrapped at ~80 columns>

<footer: Closes #12 / BREAKING CHANGE: ...>
```

| Type | Use for |
| --- | --- |
| `feat` | New behaviour (a flag, a curve type, a loader). |
| `fix` | A bug in existing behaviour. |
| `docs` | README / `docs/` / docstrings only. |
| `test` | Test-only changes. |
| `perf` | Behaviour-preserving speed/memory work — say which stage and quote the before/after. |
| `refactor` | Internal restructuring with no output change. |
| `build` / `chore` | `pyproject.toml`, dependencies, tooling, `.gitignore`. |

`<scope>` is the module or area: `rig`, `parts`, `render`, `layers`, `preview`, `debug`, `cli`,
`examples`, `docs`. Keep each commit to one scope when you can; the CLI surface change and the
module change behind it belong together, but the unrelated formatting sweep does not.

Examples:

```
feat(render): add a sine motion curve for idle sway
fix(rig): break bone-distance ties by part order, not dict order
docs(architecture): document the blend-width / rotation rule of thumb
perf(parts): crop layers to their bbox before rotating
```

Note: this convention is being introduced with these docs. Older history (if any) does not follow
it; do not rewrite it. Match the convention going forward.

---

## 7. What a good pull request includes

1. **The change, scoped.** One topic. Module contract changes get a matching
   `docs/ARCHITECTURE.md` update in the same PR.
2. **Tests** (`tests/`, per §2) or an explicit statement of why the change is not testable without
   art you cannot commit.
3. **For any rigging change — rendering, masks, blending, occlusion fill, motion curves, z order —
   attach both images:**
   * `diagnostic.png` from `flat2rig inspect` (the three-panel sheet: proves the split is still
     correct),
   * `strip.png` from `debug.draw_frames_strip` (the contact sheet of one state's frames: proves the
     motion still looks right).
   Put the *before* and *after* pair in the PR description. A reviewer cannot judge a seam or a
   rotation gap from a diff, and "coverage went from 0.971 to 0.998" does not tell them whether the
   ear still tears.
4. **Numbers if you claim performance.** State the machine, the resolution, the part/frame counts,
   and whether PNG encoding is included (see the estimate table in `docs/ARCHITECTURE.md` §4).
   Unqualified "2× faster" claims get sent back.
5. **A note on compatibility** if the annotation schema, `rig.json`, the frames layout or the CLI
   flags changed: say exactly what breaks and whether the loader accepts the old form.
6. **Green checks:** `pytest` (or "no tests yet, ran the manual `inspect`/`build`/`preview` loop
   instead") and `ruff check`. Paste the output.

Small, reviewable, evidence-backed PRs get merged. Large refactors bundled with a feature do not.
