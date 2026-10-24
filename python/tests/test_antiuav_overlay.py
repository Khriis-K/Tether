"""Spotting Anti-UAV frames by their turret overlay (#45), so they can be dropped from Seraphim test.
The reticle's twelve line segments sit at fixed places in every 1920x1080 frame, drawn black or
white to stand out from what is behind them. The check on real frames is `tether-transfer-sets
overlay-check` (docs/benchmarks/drone-transfer-overlay-check.json), since the datasets aren't in git."""

import cv2
import numpy as np
import pytest

from tether.antiuav_overlay import RETICLE, has_antiuav_overlay, reticle_score, seraphim_like

def sky(seed=0, w=1920, h=1080, texture=8.0):
    """A gray sky with a gradient and noise."""
    rng = np.random.default_rng(seed)
    base = np.linspace(90, 150, w)[None, :] + np.linspace(-20, 20, h)[:, None]
    return np.clip(base + rng.normal(0, texture, (h, w)), 0, 255).astype(np.uint8)


def draw_reticle(gray, level=255, shift=(0, 0)):
    out = gray.copy()
    dx, dy = shift
    for kind, at, start, end in RETICLE:  # 2-px lines centered on `at`
        if kind == "h":
            out[at - 1 + dy:at + 1 + dy, start + dx:end + dx] = level
        else:
            out[start + dy:end + dy, at - 1 + dx:at + 1 + dx] = level
    return out


def jpeg(gray, quality=75):
    return cv2.imdecode(cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_GRAYSCALE)


# --- Synthetic frames ---

def test_a_frame_with_the_reticle_is_flagged_at_full_size():
    assert has_antiuav_overlay(draw_reticle(sky()))


def test_the_reticle_is_flagged_whether_drawn_white_black_or_both():
    frame = draw_reticle(sky(), level=0)
    frame[:, 960:] = draw_reticle(sky(), level=255)[:, 960:]

    assert has_antiuav_overlay(draw_reticle(sky(), level=0))
    assert has_antiuav_overlay(frame)


@pytest.mark.parametrize("width", [640, 416])
def test_the_reticle_is_flagged_after_seraphims_resize_and_pad_and_jpeg(width):
    assert has_antiuav_overlay(jpeg(seraphim_like(draw_reticle(sky()), width)))


def test_the_reticle_is_flagged_on_a_frame_stretched_to_a_square():
    assert has_antiuav_overlay(cv2.resize(draw_reticle(sky()), (640, 640), interpolation=cv2.INTER_AREA))


def test_the_reticle_is_flagged_on_a_color_image():
    assert has_antiuav_overlay(cv2.cvtColor(draw_reticle(sky()), cv2.COLOR_GRAY2BGR))


def test_a_frame_without_the_reticle_is_not_flagged():
    assert not has_antiuav_overlay(sky())
    assert not has_antiuav_overlay(seraphim_like(sky(texture=40.0), 640))


def test_the_same_lines_moved_off_the_reticles_places_are_not_flagged():
    # A frame-wide grid of thin lines or a reticle-like shape elsewhere is not the overlay.
    assert not has_antiuav_overlay(draw_reticle(sky(), shift=(60, 45)))


def test_a_grid_of_lines_everywhere_is_not_flagged():
    grid = sky()
    grid[::36, :] = 255
    grid[:, ::36] = 255

    assert not has_antiuav_overlay(grid)


def test_edges_along_the_reticles_places_are_not_lines():
    # A bright panel (a window, a screen) whose border runs exactly where the corner brackets are drawn:
    # each bracket segment sits on a step from dark to bright, not on a thin line, so it isn't the overlay.
    panel = sky(texture=2.0)
    panel[325:757, 577:1345] = 230

    assert not has_antiuav_overlay(panel)


def test_the_score_rises_with_the_reticles_contrast():
    faint = reticle_score(draw_reticle(sky(), level=150))
    strong = reticle_score(draw_reticle(sky(), level=255))

    assert reticle_score(sky()) < faint < strong
