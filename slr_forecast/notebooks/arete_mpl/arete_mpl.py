"""
arete_mpl  –  Matplotlib styles and colormaps for Arête Glacier Initiative
===========================================================================

Quick start
-----------
    import arete_mpl          # registers everything on import
    import matplotlib.pyplot as plt

    arete_mpl.use('notebook')                     # style context
    plt.imshow(data, cmap='arete_blues')          # sequential blue cmap
    plt.scatter(x, y, c=z, cmap='arete_blue2earth')  # perceptually-uniform

Styles
------
    arete-paper      journal figures   (6.4 × 4.4 in)
    arete-notebook   Jupyter / screen  (8.0 × 5.5 in)
    arete-talk       slide projection  (10.4 × 7.15 in)
    arete-poster     conference poster (12.8 × 8.8 in)

Colormaps (all registered + reversed _r variants)
-------------------------------------------------
  Sequential (single-hue families):
    arete_blues, arete_reds, arete_greens, arete_purples, arete_oranges

  Perceptually-uniform blends (CIELAB-interpolated):
    arete_blue2earth, arete_blue2green, arete_blue2purple,
    arete_blue2red, arete_blue2white

  Diverging (symmetric L*, white midpoint):
    arete_div_blue_red, arete_div_blue_earth, arete_div_blue_green,
    arete_div_blue_purple, arete_div_blue_orange

  Cyclic (start == end, for periodic data):
    arete_cyclic_phase, arete_cyclic_twilight, arete_cyclic_ice
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ---------------------------------------------------------------------------
#  Arête brand hex values
# ---------------------------------------------------------------------------
GLACIER_BLUE = {
    100: "#D8E7F1",
    200: "#B5D5E9",
    300: "#72A3C3",
    400: "#036C9A",
    500: "#07456C",
    600: "#031D3A",
}
EARTH   = "#AB6638"
WHITE   = "#FFFFFF"
BLACK   = "#000000"

# Default color cycle used in the .mplstyle files
CYCLE = ["#036C9A", "#AB6638", "#2A9D8F", "#C44E52", "#72A3C3", "#D4A24E"]

# ---------------------------------------------------------------------------
#  Colour-space helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(np.clip(rgb * 255, 0, 255).astype(int)))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0-1] to CIELAB via XYZ (D65).  Input shape (..., 3)."""
    # Linearise sRGB
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB -> XYZ  (D65 reference white)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ M.T
    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    # XYZ -> Lab
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to sRGB [0-1].  Input shape (..., 3)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([x, y, z], axis=-1) * np.array([0.95047, 1.00000, 1.08883])
    # XYZ -> linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    lin = xyz @ M_inv.T
    lin = np.clip(lin, 0, None)
    # Gamma-compress
    rgb = np.where(lin > 0.0031308, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(rgb, 0.0, 1.0)


def _make_perceptual_cmap(hex_start: str, hex_end: str, name: str, N: int = 256) -> LinearSegmentedColormap:
    """Interpolate linearly in CIELAB, return a matplotlib LinearSegmentedColormap."""
    lab0 = _rgb_to_lab(_hex_to_rgb(hex_start))
    lab1 = _rgb_to_lab(_hex_to_rgb(hex_end))
    t = np.linspace(0, 1, N).reshape(-1, 1)
    lab_path = lab0 * (1 - t) + lab1 * t
    rgb_path = _lab_to_rgb(lab_path)
    return LinearSegmentedColormap.from_list(name, rgb_path, N=N)


def _make_perceptual_multi(hex_anchors: list[str], name: str, N: int = 256) -> LinearSegmentedColormap:
    """Multi-stop perceptual cmap: interpolate piecewise in CIELAB."""
    n_seg = len(hex_anchors) - 1
    per_seg = max(N // n_seg, 2)
    rgb_all = []
    for i in range(n_seg):
        lab0 = _rgb_to_lab(_hex_to_rgb(hex_anchors[i]))
        lab1 = _rgb_to_lab(_hex_to_rgb(hex_anchors[i + 1]))
        t = np.linspace(0, 1, per_seg, endpoint=(i == n_seg - 1)).reshape(-1, 1)
        lab_seg = lab0 * (1 - t) + lab1 * t
        rgb_all.append(_lab_to_rgb(lab_seg))
    rgb_path = np.concatenate(rgb_all, axis=0)
    return LinearSegmentedColormap.from_list(name, rgb_path, N=len(rgb_path))


# ---------------------------------------------------------------------------
#  Build colormaps
# ---------------------------------------------------------------------------

def _build_colormaps() -> dict[str, LinearSegmentedColormap]:
    cmaps = {}

    # ---- Sequential single-hue families ----

    # Arête Blues:  light (GB100) -> dark (GB600)
    cmaps["arete_blues"] = _make_perceptual_multi(
        [GLACIER_BLUE[100], GLACIER_BLUE[200], GLACIER_BLUE[300],
         GLACIER_BLUE[400], GLACIER_BLUE[500], GLACIER_BLUE[600]],
        "arete_blues",
    )

    # Complementary reds  (light blush -> deep crimson)
    cmaps["arete_reds"] = _make_perceptual_multi(
        ["#F5DADA", "#E8A6A6", "#C44E52", "#8B1A1E", "#3D0B0D"],
        "arete_reds",
    )

    # Complementary greens  (pale sage -> deep teal-green)
    cmaps["arete_greens"] = _make_perceptual_multi(
        ["#DFF0EB", "#A8D8C8", "#2A9D8F", "#1A6B61", "#0B3530"],
        "arete_greens",
    )

    # Complementary purples  (lavender -> deep aubergine)
    cmaps["arete_purples"] = _make_perceptual_multi(
        ["#E8E0F0", "#B8A4D0", "#7B5EA7", "#4E3175", "#251638"],
        "arete_purples",
    )

    # Complementary oranges  (pale sand -> deep earth)
    cmaps["arete_oranges"] = _make_perceptual_multi(
        ["#F5E6D8", "#D9B896", EARTH, "#7A4826", "#3D2413"],
        "arete_oranges",
    )

    # ---- Perceptually-uniform bivariate blends (Arête blue -> X) ----
    # Strategy: interpolate in CIELAB with a *monotonic* L* ramp.
    # L* rises linearly from the dark-blue start to a lighter end colour.
    # a* and b* are blended with a sigmoid-weighted transition so hue shifts
    # smoothly over the upper third of the map, keeping blues dominant.

    def _blend_cmap(name, hex_start, hex_end, N=256, transition_centre=0.65, transition_width=0.15):
        """Build a cmap from hex_start to hex_end with monotonic L* and
        sigmoid-weighted hue transition."""
        lab0 = _rgb_to_lab(_hex_to_rgb(hex_start))
        lab1 = _rgb_to_lab(_hex_to_rgb(hex_end))
        t = np.linspace(0, 1, N)
        # L* ramps linearly
        L = lab0[0] + (lab1[0] - lab0[0]) * t
        # a*, b* transition via a sigmoid centred at transition_centre
        sig = 1 / (1 + np.exp(-(t - transition_centre) / transition_width))
        a = lab0[1] * (1 - sig) + lab1[1] * sig
        b = lab0[2] * (1 - sig) + lab1[2] * sig
        lab_path = np.column_stack([L, a, b])
        rgb_path = _lab_to_rgb(lab_path)
        return LinearSegmentedColormap.from_list(name, rgb_path, N=N)

    # 1. Blue -> Earth  (dark navy -> warm tan;  L* 10.6 -> 86)
    #    useful for bed topography, bathymetry-to-land
    cmaps["arete_blue2earth"] = _blend_cmap(
        "arete_blue2earth", GLACIER_BLUE[600], "#E8D5C0",
        transition_centre=0.6, transition_width=0.15,
    )

    # 2. Blue -> Green  (dark navy -> light sage;  L* 10.6 -> 80)
    #    useful for ice-to-vegetation transitions
    cmaps["arete_blue2green"] = _blend_cmap(
        "arete_blue2green", GLACIER_BLUE[600], "#A8D8C8",
        transition_centre=0.6, transition_width=0.15,
    )

    # 3. Blue -> Purple  (dark navy -> light lavender;  L* 10.6 -> 80)
    #    useful for geothermal / anomaly data
    cmaps["arete_blue2purple"] = _blend_cmap(
        "arete_blue2purple", GLACIER_BLUE[600], "#C8B4E0",
        transition_centre=0.6, transition_width=0.15,
    )

    # 4. Blue -> Red  (dark navy -> blush;  L* 10.6 -> 74)
    #    useful for cold-to-warm without a white centre
    cmaps["arete_blue2red"] = _blend_cmap(
        "arete_blue2red", GLACIER_BLUE[600], "#E8A6A6",
        transition_centre=0.55, transition_width=0.15,
    )

    # 5. Blue -> White  (dark navy -> white;  L* 10.6 -> 100)
    #    pure luminance ramp in Arête blue hue, fading to white
    cmaps["arete_blue2white"] = _blend_cmap(
        "arete_blue2white", GLACIER_BLUE[600], WHITE,
        transition_centre=0.75, transition_width=0.20,
    )

    # ---- Diverging colormaps ----
    # Design: symmetric L* profile peaking at a near-white midpoint (~L*97).
    # Each half ramps L* linearly from a dark endpoint up to the midpoint.
    # Hue/chroma transitions via sigmoid so the outer ~60% stays chromatic
    # and the inner ~40% fades toward neutral.  Both halves are forced to
    # the same L*_dark so the profile is exactly symmetric.

    def _diverging_cmap(name, hex_left, hex_right, hex_mid="#F5F5F5",
                        N=256, transition_width=0.12):
        """Build a perceptually-uniform diverging cmap.

        Parameters
        ----------
        hex_left, hex_right : str
            Dark saturated endpoints (should have similar L*).
        hex_mid : str
            Near-white midpoint.
        """
        lab_L = _rgb_to_lab(_hex_to_rgb(hex_left))
        lab_R = _rgb_to_lab(_hex_to_rgb(hex_right))
        lab_M = _rgb_to_lab(_hex_to_rgb(hex_mid))

        # Force symmetric L*: use the lower of the two endpoints
        L_dark = min(lab_L[0], lab_R[0])
        L_mid = lab_M[0]

        half = N // 2

        def _half_ramp(lab_end, n):
            """Build one half: dark endpoint -> midpoint."""
            t = np.linspace(0, 1, n)
            L = L_dark + (L_mid - L_dark) * t
            # Sigmoid: 0 at t=0 (endpoint hue) -> 1 at t=1 (neutral mid)
            sig = 1 / (1 + np.exp(-(t - 0.7) / transition_width))
            a = lab_end[1] * (1 - sig) + lab_M[1] * sig
            b = lab_end[2] * (1 - sig) + lab_M[2] * sig
            return np.column_stack([L, a, b])

        left_lab = _half_ramp(lab_L, half)          # dark_left -> mid
        right_lab = _half_ramp(lab_R, N - half)      # dark_right -> mid

        # Left half reads dark_left -> mid; right half reversed to mid -> dark_right
        full_lab = np.concatenate([left_lab, right_lab[::-1]], axis=0)
        full_rgb = _lab_to_rgb(full_lab)
        return LinearSegmentedColormap.from_list(name, full_rgb, N=len(full_rgb))

    # Midpoint colours: slightly tinted toward each pair for warmth
    NEUTRAL = "#F5F5F5"       # L*≈96.5, nearly achromatic

    # All endpoints at L*≈28 so hue is clearly visible.
    # Blue side is always GB500 (#07456C, L*≈28).

    # 1. Blue ↔ Red  (cold/warm — the workhorse diverging map)
    cmaps["arete_div_blue_red"] = _diverging_cmap(
        "arete_div_blue_red",
        GLACIER_BLUE[500], "#792228", NEUTRAL,
    )

    # 2. Blue ↔ Earth  (ocean/land, bathymetry/topography)
    cmaps["arete_div_blue_earth"] = _diverging_cmap(
        "arete_div_blue_earth",
        GLACIER_BLUE[500], "#653513", NEUTRAL,
    )

    # 3. Blue ↔ Green  (two cool hues, useful for signed velocity anomalies)
    cmaps["arete_div_blue_green"] = _diverging_cmap(
        "arete_div_blue_green",
        GLACIER_BLUE[500], "#034a43", NEUTRAL,
    )

    # 4. Blue ↔ Purple  (signed anomalies, geothermal-related fields)
    cmaps["arete_div_blue_purple"] = _diverging_cmap(
        "arete_div_blue_purple",
        GLACIER_BLUE[500], "#4c376d", NEUTRAL,
    )

    # 5. Blue ↔ Orange  (high contrast, better for colour-vision deficiency)
    cmaps["arete_div_blue_orange"] = _diverging_cmap(
        "arete_div_blue_orange",
        GLACIER_BLUE[500], "#673400", NEUTRAL,
    )

    # ---- Cyclic colormaps ----
    # For periodic data (phase, azimuth, direction).  Start colour == end colour.

    def _cyclic_constant_L(name, L, C, h_start_deg=255.0, N=256):
        """Full 360° hue rotation at constant L* and chroma C.
        h_start_deg defaults to the Arête blue hue (~255°)."""
        h = np.linspace(h_start_deg, h_start_deg + 360, N, endpoint=False)
        h_rad = np.radians(h)
        a = C * np.cos(h_rad)
        b = C * np.sin(h_rad)
        lab = np.column_stack([np.full(N, L), a, b])
        rgb = _lab_to_rgb(lab)
        # Append the first colour to close the loop
        rgb = np.vstack([rgb, rgb[0:1]])
        return LinearSegmentedColormap.from_list(name, rgb, N=N)

    def _cyclic_twilight_style(name, h_dark_deg, h_warm_deg,
                               L_dark, L_peak, C_dark, C_peak, N=256):
        """Dark-light-dark cyclic map.  Both endpoints are the SAME colour
        (at h_dark_deg, L_dark, C_dark), guaranteeing true cyclicity.

        First half:  dark (h_dark) -> light peak, sweeping through cool hues.
        Second half: light peak -> dark (h_dark), sweeping through warm hues.

        Hue goes h_dark -> h_peak_cool -> h_peak_warm -> h_dark, covering
        the full cool-to-warm arc while returning to the start.
        """
        half = N // 2

        # The light peak sits at a hue midway between the outgoing and
        # returning arcs.  For cool outgoing (blue) and warm return, the
        # peak hue is roughly the average of h_dark and h_warm.
        # But the two halves approach from different sides.
        h_peak_out  = (h_dark_deg + h_warm_deg) / 2        # cool side of peak
        h_peak_back = h_peak_out                             # warm side of peak

        t_up   = np.linspace(0, 1, half)          # dark -> peak
        t_down = np.linspace(1, 0, N - half)      # peak -> dark

        def _build_half(t, h_start, h_end):
            L = L_dark + (L_peak - L_dark) * t
            C = C_dark + (C_peak - C_dark) * t
            h = np.radians(np.linspace(h_start, h_end, len(t)))
            a = C * np.cos(h)
            b = C * np.sin(h)
            return np.column_stack([L, a, b])

        # Outgoing half: dark_blue -> peak, hue sweeps h_dark -> h_peak
        lab1 = _build_half(t_up, h_dark_deg, h_peak_out)

        # Return half: peak -> dark_blue, hue sweeps h_peak -> h_warm -> h_dark
        # We go the "long way" through warm hues back to h_dark.
        # Since h_warm is on the other side, the path is:
        #   h_peak -> h_warm -> h_dark
        # Concatenate two sub-segments for the return:
        n_return = N - half
        n_ret1 = n_return // 2
        n_ret2 = n_return - n_ret1
        t_ret1 = np.linspace(1, 0.5, n_ret1)   # peak -> mid-return
        t_ret2 = np.linspace(0.5, 0, n_ret2)   # mid-return -> dark
        lab2a = _build_half(t_ret1, h_peak_back, h_warm_deg)
        lab2b = _build_half(t_ret2, h_warm_deg, h_dark_deg)

        full_lab = np.concatenate([lab1, lab2a, lab2b], axis=0)
        full_rgb = _lab_to_rgb(full_lab)
        # Force exact closure: last sample = first sample
        full_rgb[-1] = full_rgb[0]
        return LinearSegmentedColormap.from_list(name, full_rgb, N=len(full_rgb))

    # 1. arete_cyclic_phase  — constant L*, full hue rotation
    #    L*=60, C=40.  Vivid and in-gamut at all hue angles.
    #    at all hue angles.  Starts at Arête blue hue (255°).
    #    Use for: InSAR phase, tidal phase, flow azimuth, wind direction.
    cmaps["arete_cyclic_phase"] = _cyclic_constant_L(
        "arete_cyclic_phase", L=60, C=40, h_start_deg=255.0,
    )

    # 2. arete_cyclic_twilight  — dark-light-dark, cool/warm split
    #    Wrap point at L*=45, peak at L*=92.
    #    Outgoing half sweeps through cool blues; return half through warm earth tones.
    #    Both endpoints are the same colour, ensuring true cyclicity.
    cmaps["arete_cyclic_twilight"] = _cyclic_twilight_style(
        "arete_cyclic_twilight",
        h_dark_deg=255, h_warm_deg=55,
        L_dark=45, L_peak=92, C_dark=30, C_peak=10,
    )

    # 3. arete_cyclic_ice  — more chromatic, smaller L* swing
    #    Wrap at L*=50, peak at L*=85.  Return through rose-red (h=25°).
    cmaps["arete_cyclic_ice"] = _cyclic_twilight_style(
        "arete_cyclic_ice",
        h_dark_deg=255, h_warm_deg=25,
        L_dark=50, L_peak=85, C_dark=35, C_peak=15,
    )

    return cmaps


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

_STYLES_DIR = Path(__file__).resolve().parent / "styles"

_registered = False


def register():
    """Register all Arête styles and colormaps with matplotlib."""
    global _registered
    if _registered:
        return
    _registered = True

    # ---- register style sheets ----
    style_dir = _STYLES_DIR
    if style_dir.is_dir():
        # Add individual style files to matplotlib's style library
        for sty in style_dir.glob("*.mplstyle"):
            name = sty.stem  # e.g. "arete-paper"
            mpl.style.core.read_style_directory(str(style_dir))
            break  # read_style_directory loads the whole dir at once
        # Merge into the global library
        user_styles = mpl.style.core.read_style_directory(str(style_dir))
        mpl.style.library.update(user_styles)
        mpl.style.core.available[:] = sorted(mpl.style.library.keys())

    # ---- register colormaps ----
    for name, cmap in _build_colormaps().items():
        try:
            mpl.colormaps.register(cmap, name=name)
            mpl.colormaps.register(cmap.reversed(), name=name + "_r")
        except ValueError:
            pass  # already registered


def use(context: str = "notebook"):
    """Apply Arête base style + a sizing context.

    Parameters
    ----------
    context : str
        One of 'paper', 'notebook', 'talk', 'poster'.
    """
    register()
    valid = ("paper", "notebook", "talk", "poster")
    if context not in valid:
        raise ValueError(f"context must be one of {valid}, got {context!r}")
    # Use full paths to avoid relying on matplotlib's style library cache,
    # which can be cleared between cells in "Run All" mode.
    base = _STYLES_DIR / "arete-base.mplstyle"
    ctx = _STYLES_DIR / f"arete-{context}.mplstyle"
    plt.style.use([str(base), str(ctx)])


def get_cmap(name: str):
    """Return an Arête colormap by name (registers if needed)."""
    register()
    return mpl.colormaps[name]


def list_cmaps() -> list[str]:
    """Return names of all Arête colormaps (excluding reversed)."""
    return [
        "arete_blues", "arete_reds", "arete_greens",
        "arete_purples", "arete_oranges",
        "arete_blue2earth", "arete_blue2green", "arete_blue2purple",
        "arete_blue2red", "arete_blue2white",
        "arete_div_blue_red", "arete_div_blue_earth",
        "arete_div_blue_green", "arete_div_blue_purple",
        "arete_div_blue_orange",
        "arete_cyclic_phase", "arete_cyclic_twilight", "arete_cyclic_ice",
    ]


# Auto-register on import
register()
