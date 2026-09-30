# Comparison: where flat2rig sits

Honest positioning against the tools people usually mention in the same sentence. The short
version: **these are complementary, not competitors.** Each one solves a different problem, and for
most real projects the answer is "use two of them", not "pick one".

Everything below is a design-position document, not a benchmark. Numbers for other projects are
taken from their own documentation and from the survey in `whale-pet-redesign/research/README.md`
(see the caveat in §7); numbers for flat2rig are described in `docs/ARCHITECTURE.md` §4 and are
estimates with their dependencies stated.

---

## 1. The niche, stated plainly

flat2rig does one thing: **take one flat illustration plus ~6 annotated joints and emit a few dozen
animation frames in seconds, on a CPU, with numpy and Pillow.**

Its properties, in the order that matters to people who choose it:

* **No GPU, no model weights, no downloads.** The default install is `numpy` + `Pillow` (plus
  `scipy` as declared); the wheels are tens of megabytes, not gigabytes.
* **Deterministic.** Same PNG + same JSON ⇒ byte-identical frames. No sampling, no seeds, no
  "rerun and it looks slightly different".
* **Seconds per character** (estimate, README-scale art at ~240 px, 3–4 parts, ~13 frames: low
  single-digit seconds including PNG encoding — see architecture §4 for what that depends on).
* **Procedural part motion, not semantic decomposition.** It splits along bones *you* draw. It has
  no idea that a region is a sleeve.
* **Output is frames** — `frames/*.png` plus a self-contained `preview.html`, plus `rig.json` and
  per-part masks. Not a model file, not a PSD.

That is a real but narrow niche. It is the right tool for game jams, prototypes, plugin mascots,
CI-generated art, and any environment where "install PyTorch" is not on the table. It is the wrong
tool when the art must be decomposed semantically, warped smoothly, or driven by a runtime engine.

---

## 2. The alternatives, one by one

### see-through (shitagaki-lab) — single-image semantic layer decomposition

[see-through](https://github.com/shitagaki-lab/see-through) (SIGGRAPH 2026) decomposes a single
anime illustration into roughly two dozen semantic PSD layers (hair, face, each sleeve, each
accessory, …) with a diffusion model. This is a genuine capability gap, not a matter of tuning: the
model infers *what* a region is from pixels and global context, which is exactly what a geometric
splitter cannot do. The cost is the stack around it — PyTorch 2.8, transformers 5, diffusers, and
several GB of weights, with an NVIDIA GPU effectively required for a usable loop; tens of seconds to
minutes per image, and stochastic output. If you need the semantic layers, there is no CPU
substitute here — annotate more bones, or accept that flat2rig will guess wrong on nested clothing.
The key point for this project is the interface: see-through's layered PSD output is exactly what
`flat2rig build --layers` consumes, so the two compose into "quality decomposition + deterministic
rig and frames".

### psd2live (tsunehimatoi) — layered PSD → Live2D model

[psd2live](https://github.com/tsunehimatoi/psd2live) takes an *already layered* PSD and rigs it into
a Live2D model, generating meshes and deformers automatically. It assumes you have layers; flat2rig
can produce a similar starting point (part masks + layers) for art that has none, so the natural
pipeline is flat2rig → psd2live when you want a mesh-based model rather than frames. Where they
differ is output physics: a Live2D model deforms continuously and interpolates between key values,
while flat2rig bakes discrete frames at fixed delays. Mesh warping means a Live2D model can bend a
sleeve smoothly around an arm; flat2rig rotates a rigid part and hides the seam with a blend band
and a joint stub (architecture §2.5). Flat2rig does not export `.moc3` and will not: the layer model
here has no mesh, no deformer hierarchy, and no runtime.

### VPet (LorisYounger) — a full desktop-pet application

[VPet](https://github.com/LorisYounger/VPet) is a complete desktop-pet *runtime*: a window that
lives on your desktop, with mod support, a pet editor, interaction, and stock pets drawn as
thousands of hand-drawn PNG frames. It is a different category of software — an application, not a
pipeline — and comparing them is like comparing a game engine to a sprite exporter. What VPet shows
is what flat2rig deliberately does **not** do: heavy per-frame authoring and a behaviour layer
(roaming, interaction, state machines). Use flat2rig to *generate* a small frame set that a runtime
like VPet — or a plugin, or a web page — then plays. VPet's frame naming and folder conventions are
a useful target for an exporter if anyone wants one (`docs/ARCHITECTURE.md` §5.3).

### BongoCat / DyberPet — other desktop-pet apps, different focuses

The earlier survey (`whale-pet-redesign/research/README.md`) compared four desktop-pet routes:
frame sequences (`1ilit/Desktop-Cat`), sprite sheets (`OpenPetsHQ/openpets`), Shimeji behaviour XML
(`pixelomer/Shijima-Qt`), and Live2D (`liwenka1/bongo-cat-next`). BongoCat-style apps are
input-driven mascots — the animation is a response to your keyboard/mouse, so the interesting
engineering is the event plumbing, not the art. DyberPet-style apps are pet *frameworks* with a
plugin/behaviour focus and hand-drawn art. Both need frames; neither generates them. That is the
gap flat2rig fills: it turns one drawing into the frames these runtimes expect. The conventions the
survey settled on — `<character>_<state>_<index>.png` naming, per-state frame delays, frame index
resetting on state change, overlay layers drawn separately — are what flat2rig's output format and
`rig.json` reproduce on purpose.

---

## 3. Comparison table (expanded from the README)

| | see-through + psd2live | VPet | BongoCat / DyberPet | **flat2rig** |
| --- | --- | --- | --- | --- |
| Category | decomposition + rigging pipeline | desktop-pet application | desktop-pet application / framework | rig + frame generator (library + CLI) |
| Input | single illustration | hand-drawn frame sets | hand-drawn frame sets | single illustration **or** see-through layers |
| Decomposition | diffusion model, ~24 semantic layers | none (artist draws each part) | none (artist draws each part) | skeleton-guided geometric split, ~6 annotated joints |
| Rigging | auto mesh/deformer (psd2live) | none | none | rigid per-part rotation about a pivot |
| Output | layered PSD → Live2D `.moc3` | app runs it directly | app runs it directly | animation frames (PNG) + `rig.json` + `preview.html` |
| Weights / GPU | several GB, NVIDIA recommended | none | none | **none** |
| Runtime per character | tens of seconds to minutes | authoring time in hours+ | authoring time in hours+ | **~1–2 s** (estimate; scales with part/frame count and resolution) |
| Determinism | stochastic | deterministic (static art) | deterministic (static art) | fully deterministic |
| Dependencies | PyTorch 2.8 + transformers 5 + diffusers + weights | full app + runtime | full app + runtime | `numpy` + `Pillow` (+ `scipy`); `psd-tools` optional |
| Animation model | continuous deformation (Live2D params) | discrete frames + behaviour logic | discrete frames + input events | discrete frames from per-state motion lists |
| Semantic understanding | yes (learned) | n/a (human) | n/a (human) | **no** (bones you annotate) |
| Effort before first animation | install stack, wait for inference | draw/collect frames, configure app | draw/collect frames, configure app | annotate ~6 joints (~10 min), run CLI |
| Best fit | high-quality assets, GPU available | shipping a desktop pet today | shipping an input-reacting mascot | prototypes, jams, plugins, low-power/CI environments |

Composition, concretely:

```bash
# No GPU: geometry split → frames.
flat2rig build hero.png -c hero.rig.json -o out/

# GPU available: semantic layers, then deterministic rig + frames.
flat2rig build ./see_through_layers/ --layers -c hero.rig.json -o out/
```

---

## 4. What flat2rig cannot do that they can

Stated without hedging, because these are the reasons someone should pick something else:

| Capability | Who has it | Why flat2rig does not |
| --- | --- | --- |
| **Semantic layer guessing** — "this is a sleeve", nested clothing, per-strand hair | see-through | It has no model. A pixel belongs to the nearest bone you drew; there is no label vocabulary beyond your part names. |
| **Mesh warping / smooth deformation** — bending, squash-and-stretch, cloth follow-through | psd2live / Live2D | flat2rig parts are rigid; each part gets a rotation and a vertical offset. A seam is hidden by a blend band and a stub patch, not by deforming geometry. |
| **`.moc3` / Live2D export** | psd2live | Out of scope by design: the layer model carries no mesh, deformer tree, parameter set, or physics. |
| **Continuous motion at runtime** — arbitrary angles, parameter blending, physics | Live2D runtimes | flat2rig bakes a fixed frame list. Interpolation happens only in whatever player consumes the PNGs. |
| **Window roaming behaviours** — walking along the taskbar, climbing window edges, reacting to the mouse | VPet, Shimeji/Qt | flat2rig produces assets, not an application. No event loop, no behaviour tree, no window handle. |
| **Input-driven reactions** (BongoCat-style keyboard/mouse animation) | BongoCat-likes | Same reason: the state machine lives in the app. flat2rig only defines the states and their delays. |
| **Artist-grade control per frame** — hand-tuned keyframes, effects, overlays | VPet-style hand-drawn sets, and any manual pipeline | Procedural lists of `[degrees, dy]` are deliberately tiny. If a frame needs a bespoke pose, draw it. |
| **Learning from examples / your style** | see-through and friends | Nothing in the default path is trained. |
| **High-quality inpainting of large occlusions** | see-through (real layers exist) | Mirror + diffusion is a small, stable guess (architecture §2.4) and is labelled as such. |

---

## 5. What flat2rig does better, honestly scoped

Not "better in general" — better *on these axes*, for people who care about them:

* **Cold-start cost.** `pip install flat2rig` in an environment with no GPU, no CUDA, no multi-GB
  download; first animation within minutes of cloning.
* **Reproducibility.** Rigs are JSON, output is byte-stable, so frames can be regenerated in CI and
  diffed. That makes a rig reviewable in a pull request (hence `diagnostic.png` / `strip.png` in
  the PR checklist).
* **Tiny annotation surface.** ~6 joints per character, versus hours of layer preparation. The
  quality ceiling is lower, but the floor is much higher than "no animation at all".
* **Composability.** It is a library first (`rig` / `parts` / `render` are importable) and a CLI
  second, and `--layers` makes it a good downstream consumer of the heavy tools.

---

## 6. Choosing (decision table)

| Situation | Use |
| --- | --- |
| Laptop, no GPU, need *some* animation this afternoon | flat2rig |
| GPU + need correct semantic layers for hand-off to an artist | see-through → (optionally) flat2rig `--layers` |
| Need a Live2D model that deforms smoothly at runtime | psd2live (flat2rig can supply layers) |
| Need a complete desktop pet with roaming and interaction, today | VPet / DyberPet, with frames from anywhere |
| Need a keyboard-reacting mascot overlay | BongoCat-style runtime; flat2rig supplies frames |
| Need pixel-perfect bespoke poses and effects | Commission/draw the frames; flat2rig only for the repetitive idle/sleep loops |

---

## 7. Sources and caveats

* Upstream capabilities and version requirements (PyTorch 2.8, transformers 5, diffusers, ~24
  layers, auto mesh/deformer, frame volume) come from each project's own README and from the
  survey at `whale-pet-redesign/research/README.md`; they were **not** re-measured in this
  repository. Re-check the linked repositories before quoting them in a paper or a release note —
  ML stacks move fast and "requires Python X / CUDA Y" statements age in weeks.
* Every flat2rig timing here is an **estimate** with its dependency stated (resolution, part count,
  frame count, CPU, whether PNG encoding is included). See `docs/ARCHITECTURE.md` §4.
* The comparison is about capability, not quality. see-through's layers are *better* than flat2rig's
  geometric split on exactly the axis it was trained for; flat2rig's advantage is that it runs at
  all on the machines most people have.
