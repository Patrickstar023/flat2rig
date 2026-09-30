"""Self-contained preview output for rendered animation frames.

The frame directory written by ``flat2rig build`` is a flat pile of PNGs::

    frames/<character>_<state>_<index>.png      e.g. daermaodou_idle_01.png

Two renderings are produced from it:

* :func:`write_preview` — one HTML file with every frame inlined as a
  ``data:`` URI.  It opens straight from ``file://`` with no server, no sibling
  assets and no network access, and it *plays* the animation: one ``<img>`` per
  state is cycled by a few dozen lines of vanilla JS.
* :func:`write_contact_sheet` — one PNG contact sheet of a single state, for
  eyeballing a cut or dropping into a README.

This module intentionally depends on nothing but the standard library and
Pillow (not on :mod:`flat2rig.rig` / :mod:`flat2rig.parts` /
:mod:`flat2rig.render`), so ``flat2rig preview`` keeps working even when the
rig pipeline is unavailable or the frames came from another tool.

Per-frame delays
----------------
PNG has no portable delay field, so a frame's delay is resolved in this order:

1. ``duration`` from the PNG metadata (APNG / ``dURATION`` chunk) when present;
2. ``states[<state>] / frame_count`` from the ``rig.json`` that
   ``flat2rig build`` writes next to the frames directory;
3. :data:`DEFAULT_FRAME_DELAY_MS`.

Everything here is deterministic: no timestamps, no randomness, no printing.
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
from pathlib import Path

from PIL import Image

__all__ = [
    "DEFAULT_FRAME_DELAY_MS",
    "FRAME_RE",
    "collect_frames",
    "state_label",
    "write_contact_sheet",
    "write_preview",
]

#: Fallback delay (ms) for a frame with no delay information of its own.
DEFAULT_FRAME_DELAY_MS = 120

#: Browsers clamp very small timeouts; keep delays above that floor.
MIN_FRAME_DELAY_MS = 16

#: ``<character>_<state>_<index>.png`` — the character name may contain ``_``,
#: the state may not (the state is the segment right before the numeric index).
FRAME_RE = re.compile(r"^(?P<name>.+)_(?P<state>[^_]+)_(?P<index>\d+)\.png$", re.IGNORECASE)

#: Chinese labels for the state names used by the CLI defaults and common pets.
STATE_LABELS = {
    "idle": "待机",
    "sleep": "睡眠",
    "error": "出错",
    "celebrate": "庆祝",
    "walk": "行走",
    "run": "奔跑",
    "jump": "跳跃",
    "fall": "落下",
    "happy": "开心",
    "sad": "难过",
    "angry": "生气",
    "surprised": "惊讶",
    "attack": "攻击",
    "hurt": "受伤",
    "blink": "眨眼",
    "talk": "说话",
    "wave": "挥手",
    "drag": "拖拽",
    "wake_up": "醒来",
    "wakeup": "醒来",
    "think": "思考",
    "sit": "坐下",
    "eat": "进食",
    "play": "玩耍",
}


def state_label(state: str) -> str:
    """Return the Chinese UI label for *state* (falls back to the raw name)."""
    return STATE_LABELS.get(state.lower(), state)


def collect_frames(frames_dir: str | Path) -> dict[str, list[Path]]:
    """Group ``<name>_<state>_<NN>.png`` files by state.

    Files that do not match :data:`FRAME_RE` are ignored.  Each state's frames
    are sorted by their numeric suffix (natural order, so ``..._2.png`` comes
    before ``..._10.png``); the returned mapping is sorted by state name so the
    result is deterministic.

    Args:
        frames_dir: Directory holding the rendered frames.

    Returns:
        ``{state: [frame, ...]}`` with :class:`pathlib.Path` values.

    Raises:
        FileNotFoundError: *frames_dir* does not exist or is not a directory.
        ValueError: No file in *frames_dir* matches the frame pattern.
    """
    root = Path(frames_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"帧目录不存在或不是目录：{root}")

    grouped: dict[str, list[tuple[int, str, Path]]] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if match is None:
            continue
        grouped.setdefault(match.group("state"), []).append(
            (int(match.group("index")), path.name, path)
        )

    if not grouped:
        raise ValueError(f"目录中没有匹配 <name>_<state>_<NN>.png 的帧：{root}")

    return {
        state: [entry[2] for entry in sorted(entries, key=lambda item: (item[0], item[1]))]
        for state, entries in sorted(grouped.items())
    }


def _png_duration_ms(path: Path) -> int | None:
    """Return the delay stored inside *path* (APNG / ``dURATION``), else ``None``."""
    try:
        with Image.open(path) as image:
            image.load()
            duration = image.info.get("duration")
    except Exception:  # unreadable file: fall back to rig.json / default
        return None
    if isinstance(duration, (int, float)) and duration > 0:
        return int(duration)
    return None


def _state_totals_ms(frames_dir: Path) -> dict[str, float]:
    """Read ``states`` (state -> total duration in ms) from a nearby ``rig.json``.

    ``flat2rig build`` writes ``out/rig.json`` next to ``out/frames/``; both
    locations are accepted.  A missing or malformed file simply yields ``{}``.
    """
    for candidate in (frames_dir / "rig.json", frames_dir.parent / "rig.json"):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(states, dict):
            continue
        totals: dict[str, float] = {}
        for key, value in states.items():
            try:
                totals[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if totals:
            return totals
    return {}


def _frame_delays(paths: list[Path], state: str, totals: dict[str, float]) -> list[int]:
    """Resolve one delay per frame, in milliseconds."""
    fallback: int | None = None
    total = totals.get(state)
    if total is None:  # tolerate a differently-cased state key
        lowered = {key.lower(): value for key, value in totals.items()}
        total = lowered.get(state.lower())
    if total and total > 0 and paths:
        fallback = max(MIN_FRAME_DELAY_MS, int(round(total / len(paths))))
    if fallback is None:
        fallback = DEFAULT_FRAME_DELAY_MS

    delays: list[int] = []
    for path in paths:
        delays.append(_png_duration_ms(path) or fallback)
    return delays


_CARD_TEMPLATE = """      <figure class="card" data-state="{state}">
        <div class="stage"><img class="frame" alt="{label}动画" src="{first_uri}"></div>
        <figcaption>
          <span class="label">{label}</span>
          <span class="badge">1/{count}</span>
          <span class="ms">{delay} ms/帧</span>
        </figcaption>
      </figure>
"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flat2rig 动画预览</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f4f5fa;
    --fg: #1c1a28;
    --muted: #6b7280;
    --card: #ffffff;
    --line: #dfe3ee;
    --accent: #4a6cf7;
    --stage-a: #eceffa;
    --stage-b: #dfe4f4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14151b;
      --fg: #ecedf3;
      --muted: #9aa3b2;
      --card: #1e2029;
      --line: #2d303c;
      --accent: #7c9cff;
      --stage-a: #23252f;
      --stage-b: #2b2e3a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px clamp(12px, 4vw, 48px) 40px;
    background: var(--bg);
    color: var(--fg);
    font: 15px/1.5 "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif;
  }}
  header {{ max-width: 1100px; margin: 0 auto 18px; }}
  h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 650; }}
  .meta {{ margin: 0; color: var(--muted); font-size: 13px; }}
  .toolbar {{
    position: sticky; top: 0; z-index: 2;
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    max-width: 1100px; margin: 0 auto 18px; padding: 10px 14px;
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  }}
  button, select {{
    font: inherit; color: var(--fg); background: var(--bg);
    border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px;
  }}
  button {{ cursor: pointer; }}
  button:hover {{ border-color: var(--accent); color: var(--accent); }}
  .toolbar label {{ display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 13px; }}
  .hint {{ margin-left: auto; color: var(--muted); font-size: 12px; }}
  .grid {{
    display: grid; gap: 16px; max-width: 1100px; margin: 0 auto;
    grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  }}
  .card {{
    margin: 0; padding: 10px; background: var(--card);
    border: 1px solid var(--line); border-radius: 12px;
  }}
  .stage {{
    display: grid; place-items: center; border-radius: 8px; overflow: hidden;
    background-color: var(--stage-a);
    background-image:
      linear-gradient(45deg, var(--stage-b) 25%, transparent 25%, transparent 75%, var(--stage-b) 75%),
      linear-gradient(45deg, var(--stage-b) 25%, transparent 25%, transparent 75%, var(--stage-b) 75%);
    background-size: 16px 16px; background-position: 0 0, 8px 8px;
  }}
  .frame {{ display: block; width: 100%; height: auto; image-rendering: pixelated; }}
  figcaption {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin-top: 8px; font-size: 12px; }}
  .label {{ font-size: 14px; font-weight: 600; }}
  .badge {{ color: var(--accent); font-variant-numeric: tabular-nums; }}
  .ms {{ color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }}
  footer {{ max-width: 1100px; margin: 22px auto 0; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>flat2rig 动画预览</h1>
  <p class="meta">{summary}</p>
</header>
<div class="toolbar">
  <button id="toggle" type="button" aria-pressed="false">⏸ 暂停</button>
  <label>速度
    <select id="speed">
      <option value="0.25">0.25×</option>
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
  </label>
  <span class="hint">空格键：暂停 / 播放</span>
</div>
<main class="grid">
{cards}</main>
<footer>flat2rig · 单文件预览页，素材已内联，可离线打开</footer>
<script>
var DATA = {data};
(function () {{
  "use strict";
  var cards = Array.prototype.slice.call(document.querySelectorAll(".card"));
  var players = DATA.states.map(function (state, index) {{
    var card = cards[index];
    return {{
      frames: state.frames,
      img: card ? card.querySelector("img.frame") : null,
      badge: card ? card.querySelector(".badge") : null,
      index: 0,
      timer: 0
    }};
  }});
  var playing = true;
  var speed = 1;
  var toggle = document.getElementById("toggle");
  var speedSelect = document.getElementById("speed");

  function show(player) {{
    var frame = player.frames[player.index];
    if (player.img && player.img.getAttribute("src") !== frame[0]) {{
      player.img.setAttribute("src", frame[0]);
    }}
    if (player.badge) {{
      player.badge.textContent = (player.index + 1) + "/" + player.frames.length;
    }}
  }}

  function stop(player) {{
    if (player.timer) {{ clearTimeout(player.timer); player.timer = 0; }}
  }}

  function schedule(player) {{
    stop(player);
    if (!playing || player.frames.length < 2) {{ return; }}
    var delay = Math.max(16, player.frames[player.index][1] / speed);
    player.timer = setTimeout(function () {{
      player.timer = 0;
      player.index = (player.index + 1) % player.frames.length;
      show(player);
      schedule(player);
    }}, delay);
  }}

  function setPlaying(value) {{
    playing = value;
    toggle.textContent = value ? "⏸ 暂停" : "▶ 播放";
    toggle.setAttribute("aria-pressed", value ? "false" : "true");
    players.forEach(value ? schedule : stop);
  }}

  toggle.addEventListener("click", function () {{ setPlaying(!playing); }});
  speedSelect.addEventListener("change", function () {{
    speed = parseFloat(speedSelect.value) || 1;
    if (playing) {{ players.forEach(schedule); }}
  }});
  document.addEventListener("keydown", function (event) {{
    if (event.key === " " || event.code === "Space") {{
      event.preventDefault();
      setPlaying(!playing);
    }}
  }});
  document.addEventListener("visibilitychange", function () {{
    if (document.hidden) {{ players.forEach(stop); }}
    else if (playing) {{ players.forEach(schedule); }}
  }});

  players.forEach(function (player) {{ show(player); schedule(player); }});
}})();
</script>
</body>
</html>
"""


def write_preview(frames_dir: str | Path, out_path: str | Path) -> Path:
    """Write one self-contained HTML player for every frame in *frames_dir*.

    Every frame is inlined as a base64 ``data:`` URI, so the page has no
    external references and works from ``file://``.  There is exactly one
    ``<img>`` per state; the player swaps that image's ``src``, which keeps the
    DOM tiny even for hundreds of frames.

    Args:
        frames_dir: Directory holding ``<name>_<state>_<NN>.png`` frames.
        out_path: HTML file to write (parent directories are created).

    Returns:
        The resolved output path.

    Raises:
        FileNotFoundError: *frames_dir* does not exist.
        ValueError: *frames_dir* holds no matching frames.
    """
    frames = collect_frames(frames_dir)
    root = Path(frames_dir)
    totals = _state_totals_ms(root)

    cards: list[str] = []
    states: list[dict[str, object]] = []
    total_frames = 0

    for state, paths in frames.items():
        label = state_label(state)
        delays = _frame_delays(paths, state, totals)
        entries: list[list[object]] = []
        for path, delay in zip(paths, delays):
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            entries.append([f"data:image/png;base64,{encoded}", int(delay)])
        total_frames += len(entries)
        states.append({"name": state, "label": label, "frames": entries})
        cards.append(
            _CARD_TEMPLATE.format(
                state=html.escape(state, quote=True),
                label=html.escape(label, quote=True),
                first_uri=entries[0][0],
                count=len(entries),
                delay=delays[0],
            )
        )

    payload = json.dumps({"states": states}, ensure_ascii=False, separators=(",", ":"))
    # A JSON string cannot be broken out of with "</script>": `<` is escaped.
    payload = payload.replace("<", "\\u003c")

    summary = (
        f"共 {len(states)} 个状态 · {total_frames} 帧 · 素材已内联，单文件离线可播放"
    )
    document = _PAGE_TEMPLATE.format(
        data=payload, cards="".join(cards), summary=html.escape(summary)
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # write bytes: no platform newline translation, so the HTML is byte-identical
    # for identical inputs (LF everywhere).
    out.write_bytes(document.encode("utf-8"))
    return out


def write_contact_sheet(
    frames_dir: str | Path,
    out_path: str | Path,
    state: str = "idle",
    columns: int = 6,
    gap: int = 6,
    background: tuple[int, int, int] = (245, 246, 251),
) -> Path:
    """Lay one state's frames out in a grid and save it as a single PNG.

    Layout (deterministic, easy to eyeball): every frame gets a cell sized to
    the largest frame; cells are separated by *gap* pixels, including a *gap*
    border around the whole sheet.  Cell size is ``max(w) x max(h)`` and frames
    are centred in their cell, so mixed frame sizes still line up.

    Args:
        frames_dir: Directory holding ``<name>_<state>_<NN>.png`` frames.
        out_path: PNG file to write (parent directories are created).
        state: State to lay out.
        columns: Cells per row (``>= 1``).
        gap: Padding between cells, in pixels.
        background: RGB background colour for transparency.

    Returns:
        The resolved output path.

    Raises:
        FileNotFoundError: *frames_dir* does not exist.
        ValueError: No matching frames, unknown *state*, or ``columns < 1``.
    """
    if columns < 1:
        raise ValueError(f"columns 必须 >= 1，收到 {columns}")

    frames = collect_frames(frames_dir)
    if state not in frames:
        available = ", ".join(sorted(frames)) or "（无）"
        raise ValueError(f"没有状态 {state!r} 的帧；可用状态：{available}")
    paths = frames[state]

    loaded: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            loaded.append(image.convert("RGBA").copy())

    cell_w = max(image.width for image in loaded)
    cell_h = max(image.height for image in loaded)
    rows = math.ceil(len(loaded) / columns)
    sheet_w = columns * cell_w + gap * (columns + 1)
    sheet_h = rows * cell_h + gap * (rows + 1)

    sheet = Image.new("RGB", (sheet_w, sheet_h), background)
    for index, image in enumerate(loaded):
        row, column = divmod(index, columns)
        left = gap + column * (cell_w + gap) + (cell_w - image.width) // 2
        top = gap + row * (cell_h + gap) + (cell_h - image.height) // 2
        sheet.paste(image, (left, top), image)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG", optimize=True)
    return out
