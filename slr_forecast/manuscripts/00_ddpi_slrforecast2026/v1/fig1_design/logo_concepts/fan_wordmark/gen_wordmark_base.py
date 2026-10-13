"""Generate SPECTRUM wordmark concepts as outlined SVGs (Rubik Black -> paths)."""
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
import os

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "/Users/minchew/Documents/Research/MyPapers/Conferences/arete_working_files/brand_and_style/static/Rubik-Black.ttf"
f = TTFont(FONT)
gs = f.getGlyphSet()
cmap = f.getBestCmap()

GB = {100: "#D8E7F1", 200: "#B5D5E9", 300: "#72A3C3", 400: "#036C9A",
      500: "#07456C", 600: "#031D3A"}
EARTH = "#AB6638"
DARK = "#031D3A"

CAP = 700  # cap height, font units
# hand kerning (font units) applied after advance of first glyph
KERN = {("S", "P"): -10, ("P", "E"): 0, ("E", "C"): -6, ("C", "T"): -22,
        ("T", "R"): -22, ("R", "U"): -4, ("U", "M"): 0}
TRACK = 12


def glyph_path(ch, dx):
    """Return SVG path for glyph ch translated by dx, y flipped so baseline=CAP."""
    g = gs[cmap[ord(ch)]]
    pen = SVGPathPen(gs)
    tp = TransformPen(pen, (1, 0, 0, -1, dx, CAP))
    g.draw(tp)
    bp = BoundsPen(gs)
    g.draw(TransformPen(bp, (1, 0, 0, -1, dx, CAP)))
    return pen.getCommands(), bp.bounds, g.width


def layout(word):
    x = 0
    items = []
    for i, ch in enumerate(word):
        d, b, w = glyph_path(ch, x)
        items.append((ch, d, b))
        x += w + TRACK
        if i + 1 < len(word):
            x += KERN.get((ch, word[i + 1]), 0)
    xmin = min(b[0] for _, _, b in items)
    xmax = max(b[2] for _, _, b in items)
    return items, xmin, xmax


def svg(vb, body, defs="", bg=None, title="SPECTRUM"):
    x0, y0, w, h = vb
    bgr = f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {w} {h}">'
            f'<title>{title}</title><defs>{defs}</defs>{bgr}{body}</svg>\n')


def write(name, s):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(s)


def hard_stops(colors, offsets=None, y1=0, y2=CAP, gid="g"):
    """Vertical gradient with hard bands. colors listed top->bottom."""
    n = len(colors)
    if offsets is None:
        offsets = [i / n for i in range(n + 1)]
    stops = []
    for i, c in enumerate(colors):
        stops.append(f'<stop offset="{offsets[i]:.4f}" stop-color="{c}"/>')
        stops.append(f'<stop offset="{offsets[i+1]:.4f}" stop-color="{c}"/>')
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="{y1}" '
            f'x2="0" y2="{y2}">{"".join(stops)}</linearGradient>')


PAD = 40
items, XMIN, XMAX = layout("SPECTRUM")
WORD_W = XMAX - XMIN
ALL_D = " ".join(d for _, d, _ in items)


def wm_vb(extra_top=0, extra_bot=0):
    return (XMIN - PAD, -PAD - extra_top, WORD_W + 2 * PAD, CAP + 2 * PAD + extra_top + extra_bot)


# ---------------------------------------------------------------------------
# Concept 1: RISING WATERLINE
# Whole word, light-above / deep-below a waterline that rises left->right.
# A thin knockout gap is the waterline. Below-water part gets the
# light-to-dark "melt-water" ramp. An Earth sliver marks the upper tail.
# ---------------------------------------------------------------------------
def concept1(dark):
    p = "c1d" if dark else "c1"
    top = GB[200] if dark else GB[300]
    deep = [GB[300], GB[400]] if dark else [GB[400], GB[600]]
    # waterline: rises from y=470 at left to y=330 at right (svg y is down)
    xl, xr = XMIN - PAD, XMAX + PAD
    yl, yr = 470, 330
    gap = 28

    def yline(x):
        return yl + (yr - yl) * (x - xl) / (xr - xl)
    # polygons above and below the line (with gap)
    above = f"M{xl},{-PAD} L{xr},{-PAD} L{xr},{yr - gap/2} L{xl},{yl - gap/2} Z"
    below = f"M{xl},{yl + gap/2} L{xr},{yr + gap/2} L{xr},{CAP + PAD} L{xl},{CAP + PAD} Z"
    defs = (f'<clipPath id="{p}-a"><path d="{above}"/></clipPath>'
            f'<clipPath id="{p}-b"><path d="{below}"/></clipPath>'
            f'<linearGradient id="{p}-g" gradientUnits="userSpaceOnUse" x1="0" y1="{yr}" x2="0" y2="{CAP}">'
            f'<stop offset="0" stop-color="{deep[0]}"/><stop offset="1" stop-color="{deep[1]}"/></linearGradient>')
    body = (f'<path d="{ALL_D}" fill="{top}" clip-path="url(#{p}-a)"/>'
            f'<path d="{ALL_D}" fill="url(#{p}-g)" clip-path="url(#{p}-b)"/>')
    vb = (XMIN - PAD, -PAD, WORD_W + 2 * PAD, CAP + 2 * PAD)
    return svg(vb, body, defs, DARK if dark else None)


def mono1(dark):
    p = "c1md" if dark else "c1m"
    d, b, w = glyph_path("S", 0)
    top = GB[200] if dark else GB[300]
    deep = [GB[300], GB[400]] if dark else [GB[400], GB[600]]
    x0, x1 = b[0] - 60, b[2] + 60
    yl, yr, gap = 470, 300, 50
    above = f"M{x0},-100 L{x1},-100 L{x1},{yr-gap/2} L{x0},{yl-gap/2} Z"
    below = f"M{x0},{yl+gap/2} L{x1},{yr+gap/2} L{x1},{CAP+100} L{x0},{CAP+100} Z"
    defs = (f'<clipPath id="{p}-a"><path d="{above}"/></clipPath>'
            f'<clipPath id="{p}-b"><path d="{below}"/></clipPath>'
            f'<linearGradient id="{p}-g" gradientUnits="userSpaceOnUse" x1="0" y1="{yr}" x2="0" y2="{CAP}">'
            f'<stop offset="0" stop-color="{deep[0]}"/><stop offset="1" stop-color="{deep[1]}"/></linearGradient>')
    body = (f'<path d="{d}" fill="{top}" clip-path="url(#{p}-a)"/>'
            f'<path d="{d}" fill="url(#{p}-g)" clip-path="url(#{p}-b)"/>')
    s = max(b[2] - b[0], b[3] - b[1]) + 160
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    vb = (cx - s / 2, cy - s / 2, s, s)
    return svg(vb, body, defs, DARK if dark else None, "SPECTRUM monogram")


# ---------------------------------------------------------------------------
# Concept 2: EIGHT-BAND STACK
# Every letter is filled with eight horizontal bands, light at top to dark
# at bottom (one band per contributor). The top band is Earth: the WAIS
# fast-storyline tail sits on top of the stack.
# Band thickness loosely varies, like a stacked-area chart.
# ---------------------------------------------------------------------------
BANDS_LIGHT = [EARTH, GB[100], GB[200], GB[300], "#3A88B0", GB[400], GB[500], GB[600]]
BANDS_DARK = [EARTH, "#FFFFFF", GB[100], GB[200], GB[300], "#3A88B0", GB[400], GB[500]]
# thicknesses top->bottom (sum normalised)
BAND_W = [0.10, 0.10, 0.11, 0.12, 0.12, 0.14, 0.15, 0.16]


def band_offsets():
    tot = sum(BAND_W)
    o = [0.0]
    for w in BAND_W:
        o.append(o[-1] + w / tot)
    return o


def concept2(dark):
    p = "c2d" if dark else "c2"
    cols = BANDS_DARK if dark else BANDS_LIGHT
    defs = hard_stops(cols, band_offsets(), 0, CAP, f"{p}-g")
    body = f'<path d="{ALL_D}" fill="url(#{p}-g)"/>'
    return svg(wm_vb(), body, defs, DARK if dark else None)


def mono2(dark):
    p = "c2md" if dark else "c2m"
    cols = BANDS_DARK if dark else BANDS_LIGHT
    d, b, w = glyph_path("S", 0)
    defs = hard_stops(cols, band_offsets(), b[1], b[3], f"{p}-g")
    body = f'<path d="{d}" fill="url(#{p}-g)"/>'
    s = max(b[2] - b[0], b[3] - b[1]) + 160
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return svg((cx - s / 2, cy - s / 2, s, s), body, defs, DARK if dark else None,
               "SPECTRUM monogram")


# ---------------------------------------------------------------------------
# Concept 3: THE U VESSEL
# Solid navy wordmark; the U is a vessel filled with eight stacked bands up
# to a fill line below its rim, dark at the bottom, Earth band on top (the
# uncertain WAIS tail). The empty part of the U is drawn as an outline.
# ---------------------------------------------------------------------------
def concept3(dark):
    p = "c3d" if dark else "c3"
    solid = GB[100] if dark else GB[600]
    cols = BANDS_DARK if dark else BANDS_LIGHT
    body = ""
    defs = ""
    for ch, d, b in items:
        if ch != "U":
            body += f'<path d="{d}" fill="{solid}"/>'
        else:
            fill_top = 250  # fill level (svg y); U rim at 0
            offs = band_offsets()
            defs += hard_stops(cols, offs, fill_top, CAP, f"{p}-g")
            defs += (f'<clipPath id="{p}-c"><rect x="{b[0]-10}" y="{fill_top}" '
                     f'width="{b[2]-b[0]+20}" height="{CAP}"/></clipPath>'
                     f'<clipPath id="{p}-e"><rect x="{b[0]-10}" y="-50" '
                     f'width="{b[2]-b[0]+20}" height="{fill_top+50}"/></clipPath>')
            # empty upper part: the letter in the solid colour (vessel walls)
            body += f'<path d="{d}" fill="{solid}" clip-path="url(#{p}-e)"/>'
            body += f'<path d="{d}" fill="url(#{p}-g)" clip-path="url(#{p}-c)"/>'
    return svg(wm_vb(), body, defs, DARK if dark else None)


def mono3(dark):
    """Monogram: the S with the same fill line and eight bands as the U."""
    p = "c3md" if dark else "c3m"
    solid = GB[100] if dark else GB[600]
    cols = BANDS_DARK if dark else BANDS_LIGHT
    d, b, w = glyph_path("S", 0)
    fill_top = 250
    defs = hard_stops(cols, band_offsets(), fill_top, b[3], f"{p}-g")
    defs += (f'<clipPath id="{p}-c"><rect x="{b[0]-10}" y="{fill_top}" width="{b[2]-b[0]+20}" height="{CAP+50}"/></clipPath>'
             f'<clipPath id="{p}-e"><rect x="{b[0]-10}" y="-50" width="{b[2]-b[0]+20}" height="{fill_top+50}"/></clipPath>')
    body = (f'<path d="{d}" fill="{solid}" clip-path="url(#{p}-e)"/>'
            f'<path d="{d}" fill="url(#{p}-g)" clip-path="url(#{p}-c)"/>')
    s = max(b[2] - b[0], b[3] - b[1]) + 160
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return svg((cx - s / 2, cy - s / 2, s, s), body, defs, DARK if dark else None,
               "SPECTRUM monogram")


for n, (wf, mf) in {1: (concept1, mono1), 2: (concept2, mono2), 3: (concept3, mono3)}.items():
    write(f"c{n}_wordmark.svg", wf(False))
    write(f"c{n}_wordmark_dark.svg", wf(True))
    write(f"c{n}_mono.svg", mf(False))
    write(f"c{n}_mono_dark.svg", mf(True))

# kerning check: plain path wordmark
write("kerncheck_paths.svg", svg(wm_vb(), f'<path d="{ALL_D}" fill="#000"/>'))
print("ok", WORD_W)
