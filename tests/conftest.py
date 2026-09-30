"""Shared fixtures for the flat2rig test suite.

Every input is generated on the fly with Pillow — the suite never depends on
art files shipped elsewhere, with one deliberate exception: ``real_art`` points
at the demo character used on this machine and the single test using it skips
when the file is absent.

Fixtures are factory-style so each test gets fresh, independent data.  Nothing
here imports :mod:`flat2rig.*` at module level on purpose: a missing or broken
source module must show up as a test failure inside the test that needs it, not
as a collection error for the whole suite.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # runnable without `pip install -e .`
    sys.path.insert(0, str(SRC))

CANVAS = (64, 64)

#: Demo art used by the optional end-to-end test (env var wins when set).
REAL_ART_CANDIDATES = [
    os.environ.get("FLAT2RIG_REAL_ART", ""),
    str(Path(r"C:\Users\马贺想\Desktop\毕业论文\whale-pet-redesign\pet-art\daermaodou-idle.png")),
    str(ROOT.parent / "whale-pet-redesign" / "pet-art" / "daermaodou-idle.png"),
]


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker so ``-m slow`` / ``-m 'not slow'`` work."""
    config.addinivalue_line("markers", "slow: test takes longer than about two seconds")


# --------------------------------------------------------------------------- #
# synthetic character
# --------------------------------------------------------------------------- #

def _draw_character(size: tuple[int, int] = CANVAS) -> Image.Image:
    """A tiny flat character: body bar + head disc + ear blob on transparency.

    The three shapes are separated enough for nearest-bone assignment to give
    each part a decent area, and the ear overlaps the head so the mask boundary
    has something to do.
    """
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 30, 39, 57], fill=(120, 170, 240, 255))   # body
    draw.ellipse([21, 9, 43, 31], fill=(250, 210, 160, 255))      # head
    draw.ellipse([38, 6, 50, 18], fill=(230, 150, 170, 255))      # ear
    return image


def _base_config() -> dict:
    """Annotation matching :func:`_draw_character` (both states have 4 frames)."""
    return {
        "canvas": [CANVAS[0], CANVAS[1]],
        "body": [32, 60],
        "parts": [
            {
                "name": "body",
                "bones": [[32, 40, 32, 58]],
                "pivot": [32, 58],
                "z": 0,
                "blend": 6,
                "motions": {
                    "idle": [[0, 0], [0, -1], [0, -2], [0, 0]],
                    "sleep": [[0, 2], [0, 4], [0, 1], [0, 3]],
                },
            },
            {
                "name": "head",
                "bones": [[32, 24, 32, 14]],
                "pivot": [32, 30],
                "z": 1,
                "blend": 6,
                "motions": {
                    "idle": [[0, 0], [8, 0], [-6, 0], [0, 0]],
                    "sleep": [[-5, 1], [-3, 2], [-6, 1], [-4, 3]],
                },
            },
            {
                "name": "ear",
                "bones": [[44, 12, 46, 8]],
                "pivot": [42, 14],
                "z": 2,
                "blend": 4,
                "motions": {
                    "idle": [[0, 0], [14, -1], [8, -2], [0, 0]],
                    "sleep": [[-9, 2], [-7, 4], [-6, 1], [-8, 5]],
                },
            },
        ],
        "eyes": [[27, 20, 3], [38, 20, 3]],
        "states": {"idle": 520, "sleep": 560},
    }


def _deep_update(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def make_character():
    """Factory: ``make_character(size=(w, h)) -> PIL.Image`` (RGBA)."""
    return _draw_character


@pytest.fixture()
def make_config():
    """Factory: ``make_config(**overrides) -> dict`` (deep-merged annotation)."""
    def factory(**overrides) -> dict:
        return _deep_update(deepcopy(_base_config()), deepcopy(overrides))

    return factory


@pytest.fixture()
def write_config(tmp_path):
    """Factory: ``write_config(name, config) -> Path`` writing annotation JSON."""
    def factory(name: str, config: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    return factory


@pytest.fixture()
def art(tmp_path, make_character, make_config, write_config):
    """Synthetic character: ``.png``, ``.config`` dict and ``.config_path``."""
    png = tmp_path / "hero.png"
    make_character().save(png)
    config = make_config()
    return SimpleNamespace(png=png, config=config, config_path=write_config("hero.rig.json", config))


@pytest.fixture()
def frames_dir(tmp_path):
    """A directory of synthetic frames plus the ``rig.json`` written next to it.

    Layout mirrors ``flat2rig build``::

        tmp/out/frames/hero_idle_01.png ... hero_idle_05.png
                       hero_sleep_01.png ... hero_sleep_03.png
        tmp/out/rig.json           {"states": {"idle": 500, "sleep": 600}}

    so idle frames carry a 100 ms delay and sleep frames 200 ms.
    """
    out = tmp_path / "out"
    frames = out / "frames"
    frames.mkdir(parents=True)
    for index in range(1, 6):
        _frame(24, index, (240, 120, 90)).save(frames / f"hero_idle_{index:02d}.png")
    for index in range(1, 4):
        _frame(24, index, (90, 140, 240)).save(frames / f"hero_sleep_{index:02d}.png")
    (out / "rig.json").write_text(
        json.dumps({"states": {"idle": 500, "sleep": 600}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return frames


def _frame(size: int, index: int, colour: tuple[int, int, int]) -> Image.Image:
    """One small RGBA frame whose content depends on *index*."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    offset = (index - 1) % 6
    draw.ellipse([2 + offset, 2, size - 3, size - 3], fill=colour + (255,))
    return image


@pytest.fixture()
def real_art():
    """Path to the real demo art, or skip when it is not on this machine."""
    for candidate in REAL_ART_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    pytest.skip("demo art not available: " + (REAL_ART_CANDIDATES[1] or "<unset>"))


@pytest.fixture()
def load_rigged(art):
    """Factory returning ``(rig, rgba, alpha, masks)`` for the synthetic art."""
    def factory():
        from flat2rig import rig as rig_mod

        rig = rig_mod.load_rig(art.config_path)
        rgba, alpha = rig_mod.load_alpha(art.png)
        masks = rig_mod.build_masks(rgba, rig)
        return rig, rgba, alpha, masks

    return factory
