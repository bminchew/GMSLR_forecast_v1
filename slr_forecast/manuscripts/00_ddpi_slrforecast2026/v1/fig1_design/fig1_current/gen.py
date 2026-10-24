#!/usr/bin/env python3
"""Generate figure.html: SPECTRUM Figure 1, flowing lanes that stack into a schematic
stacked-area hindcast and then fan out to 2100 (final merged variant: pictograms, new order and palette)."""
import os

W, H = 1344, 880
NAVY = "#031D3A"; B500 = "#07456C"; B400 = "#036C9A"; B300 = "#72A3C3"
B200 = "#B5D5E9"; B100 = "#D8E7F1"; EARTH = "#AB6638"; OVER = "#D9D9D9"
GREY = "#6b7a88"
GREEN = "#2F6B4F"; TWSG = "#9A9A9A"
SMB_BG = "#DDE9F1"; SMB_ST = "#9DC0D7"   # light tints of #72A3C3 for the hatch

out = []
def add(s): out.append(s)

def ribbon(x0, t0, b0, x1, t1, b1, fill, extra=""):
    """Filled band whose top/bottom edges are horizontal-tangent cubic Beziers."""
    xm = (x0 + x1) / 2
    d = (f"M{x0:.1f},{t0:.1f} C{xm:.1f},{t0:.1f} {xm:.1f},{t1:.1f} {x1:.1f},{t1:.1f} "
         f"L{x1:.1f},{b1:.1f} C{xm:.1f},{b1:.1f} {xm:.1f},{b0:.1f} {x0:.1f},{b0:.1f} Z")
    add(f'<path d="{d}" fill="{fill}" {extra}/>')

def curve(x0, y0, x1, y1):
    xm = (x0 + x1) / 2
    return f"C{xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"

# ---------------- pictograms (40x40 design box, single stroke colour) ----------------
import json
_S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes.json")))
ANT = _S["ANT"]  # Natural Earth 110m, polar stereographic, Greenwich up (mkshapes.py)
ANT_REG = {k: _S[k] for k in ("wais", "eais", "ap")}
GRL = _S["GRL"]

def picto(k, cx, cy, col, ring=False):
    """White disk with a monochrome line pictogram, centred at (cx, cy)."""
    r = 22
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff"'
        + (f' stroke="{col}" stroke-width="2"' if ring else '') + '/>')
    sc = 0.86
    g = (f'<g transform="translate({cx-20*sc:.1f},{cy-20*sc:.1f}) scale({sc})" fill="none" stroke="{col}" '
         f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round">')
    if k == "th":       # ocean waves
        body = "".join(f'<path d="M4,{y} q4,-5 8,0 t8,0 t8,0 t8,0"/>' for y in (12, 21, 30))
    elif k == "gl":     # mountains with snow line
        body = ('<path d="M2,34 L15,10 L22,22 L27,15 L38,34 Z"/>'
                '<path d="M10.5,18.5 L14,21 L17,17.5 L19.5,20"/>')
    elif k == "tws":    # reservoir basin with water line
        body = ('<path d="M4,9 L9,33 L31,33 L36,9"/>'
                '<path d="M7.5,19 q3,-3.5 6.25,0 t6.25,0 t6.25,0 t6.25,0"/>')
    elif k in ANT_REG:  # Antarctica outline, sub-region filled
        body = (f'<path d="{ANT_REG[k]}" fill="{col}" stroke="{col if k == "ap" else "none"}" stroke-width="6"/>'
                f'<path d="{ANT}"/>')
    elif k == "smb":    # Greenland + melt drop
        body = (f'<g transform="translate(-4,1) scale(0.92)"><path d="{GRL}"/></g>'
                '<path d="M33,21 C33,21 28,28 28,31 a5,5 0 0 0 10,0 C38,28 33,21 33,21 Z" stroke-width="2.6"/>')
    elif k == "gd":     # Greenland + ocean at its margin
        body = (f'<g transform="translate(-4,-1) scale(0.92)"><path d="{GRL}"/></g>'
                '<path d="M24,34 q3,-3.5 6,0 t6,0" stroke-width="2.6"/>'
                '<path d="M27,27 q3,-3.5 6,0 t4,0" stroke-width="2.6"/>')
    else:
        body = ""
    add(g + body + '</g>')

# ---------------- geometry ----------------
BX0, BX1 = 24, 258            # beam
BT, BB = 300, 556             # beam top/bottom
LX0, LX1 = 342, 800           # lane straight section
PITCH, Y0, LH = 70, 166, 52   # lane centres and solid lane height
ADW = 6                       # adopted (dashed) lane stroke width = its slot thickness

# stack (schematic): x positions of 1950, 2025, 2100 and the baseline
SX0, SX1, SX2 = 910, 1036, 1210
BASE = 560

# lanes top->bottom: key, name, model/record, class, colours
lanes = [
    ("wais", "West Antarctica", "slow/fast storyline, independent of emissions", "storyline", EARTH, "#fff"),
    ("eais", "East Antarctica", "snowfall sensitivity from literature", "adopted", B400, NAVY),
    ("ap", "Antarctic Peninsula", "linear in warming · IMBIE-3", "calibrated", B400, "#fff"),
    ("smb", "Greenland SMB", "MAR emulator, 7 CMIP6 models", "emulated", None, NAVY),
    ("gd", "Greenland discharge", "ocean-forced, lagged · Mouginot 2019", "calibrated", B300, NAVY),
    ("gl", "Glaciers", "linear in warming · GlaMBIE", "calibrated", B200, NAVY),
    ("tws", "Terrestrial water storage", "IPCC AR6", "adopted", TWSG, NAVY),
    ("th", "Thermosteric", "two-layer ocean · NOAA 0–2000 m", "calibrated", GREEN, "#fff"),
]
# slice thickness in the beam
slice_in = {"eais": 6, "ap": 26, "smb": 24, "gl": 26, "tws": 6, "th": 28}
STRIP = (420, 472)            # Greenland ocean temperature strip inside the beam (= discharge lane)
# schematic layer thicknesses (px) at 1950, 2025, 2100 (2100 = lowest warming path).
# Shapes only; no data.
thick = {
    "gd":   (14, 22, 38), "th": (18, 30, 58), "gl": (16, 26, 36), "ap": (10, 14, 18),
    "smb":  (12, 18, 28), "eais": (ADW, 7, 7), "tws": (ADW, 7, 6), "wais": (10, 14, 16),
}
INC = 32                      # extra height of each warmer path at 2100 (3 C, 4 C)
TAIL_TOP = 190                # top of the West Antarctica tail at 2100

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'''<defs>
<pattern id="hatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(45)">
  <rect width="12" height="12" fill="{SMB_BG}"/><rect width="5" height="12" fill="{SMB_ST}"/>
</pattern>
<pattern id="hatchs" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">
  <rect width="7" height="7" fill="{SMB_BG}"/><rect width="3" height="7" fill="{SMB_ST}"/>
</pattern>
<linearGradient id="tail" x1="0" y1="1" x2="0" y2="0">
  <stop offset="0" stop-color="{EARTH}" stop-opacity="0.75"/><stop offset="1" stop-color="{EARTH}" stop-opacity="0.06"/>
</linearGradient>
<style>
 text{{font-family:Outfit,sans-serif;font-size:20px;fill:{NAVY}}}
 .h{{font-family:Rubik,sans-serif;font-weight:900}}
 .b{{font-weight:600}}
 .halo{{paint-order:stroke;stroke:#fff;stroke-width:7px;stroke-linejoin:round}}
 .tag{{font-style:italic;font-weight:500}}
</style>
</defs>''')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')

# ---------------- stage headers ----------------
heads = [(24, "1", "Forcing"), (342, "2", "Component models"),
         (842, "3", "Sum and test"), (1062, "4", "Forecast to 2100")]
for x, n, t in heads:
    add(f'<circle cx="{x+15}" cy="96" r="15" fill="{NAVY}"/>'
        f'<text x="{x+15}" y="103" text-anchor="middle" class="h" style="fill:#fff;font-size:20px">{n}</text>'
        f'<text x="{x+40}" y="104" class="h" style="font-size:23px">{t}</text>')
add(f'<line x1="24" x2="{W-24}" y1="124" y2="124" stroke="{OVER}" stroke-width="2"/>')

# title
add(f'<text x="24" y="52" class="h" style="font-size:44px;letter-spacing:0.5px">SPECTRUM</text>')
add(f'<text x="300" y="50" style="font-size:22px;fill:{B400}">Sea-level Projections from Empirical Calibration of '
    f'Thermodynamical, Reduced-physics Models</text>')

# ---------------- stack geometry (schematic) ----------------
order_bottom_up = [k for k, *_r in reversed(lanes)]   # th at the base ... wais on top

def grow(a, b, s, p):
    return a + (b - a) * (s ** p)

def layer_thick(k, x):
    t0, t1, t2 = thick[k]
    if x <= SX1:
        return grow(t0, t1, (x - SX0) / (SX1 - SX0), 1.5)
    return grow(t1, t2, (x - SX1) / (SX2 - SX1), 1.6)

def inc(x):
    """Extra height of one warmer path above the previous one (0 before 2025)."""
    if x <= SX1:
        return 0.0
    return INC * ((x - SX1) / (SX2 - SX1)) ** 1.7

xs = list(range(SX0, SX2 + 1, 4))
if xs[-1] != SX2:
    xs.append(SX2)
# cumulative edges: edges[k] = (list of tops, list of bottoms)
edges = {}
cum = [BASE] * len(xs)
for k in order_bottom_up:
    new = [c - layer_thick(k, x) for c, x in zip(cum, xs)]
    edges[k] = (new, cum)
    cum = new
# scenario totals: 2 C = full stack top; warmer paths add equal increments (forecast only)
L2 = cum[:]
L3 = [c - inc(x) for c, x in zip(L2, xs)]
L4 = [c - inc(x) for c, x in zip(L3, xs)]
g_slot = {k: (edges[k][0][0], edges[k][1][0]) for k in order_bottom_up}

def poly(top, bot):
    pts = [f"{x},{y:.1f}" for x, y in zip(xs, top)] + [f"{x},{y:.1f}" for x, y in zip(reversed(xs), reversed(bot))]
    return "M" + " L".join(pts) + " Z"

# hindcast backdrop (the test window)
add(f'<rect x="{SX0}" y="280" width="{SX1-SX0}" height="{BASE-280}" fill="{B100}" fill-opacity="0.55"/>')

# ---------------- lanes ----------------
lane_y = {k: Y0 + PITCH * i for i, (k, *_r) in enumerate(lanes)}
beam_slot = {}
def _place(keys, lo, hi):
    g = ((hi - lo) - sum(slice_in[k] for k in keys)) / (len(keys) + 1)
    c = lo + g
    for k in keys:
        beam_slot[k] = (c, c + slice_in[k]); c += slice_in[k] + g
_place(["eais", "ap", "smb"], BT, STRIP[0])
_place(["gl", "tws", "th"], STRIP[1], BB)
beam_slot["gd"] = STRIP
assert abs((STRIP[0] + STRIP[1]) / 2 - lane_y["gd"]) < 0.5

# No-crossing check: every left (beam->lane) and right (lane->stack) edge uses one Bezier
# shape, so edges ordered at both ends stay ordered throughout.
def _ends(k, side):
    y = lane_y[k]; cls = dict((l[0], l[3]) for l in lanes)[k]
    a = beam_slot[k] if side == "L" else None
    t, b = (y - 3, y + 3) if cls == "adopted" else (y - LH / 2, y + LH / 2)
    if side == "L":
        if cls == "adopted":
            m = (a[0] + a[1]) / 2; a = (m - 3, m + 3)
        return [(a[0], t), (a[1], b)]
    gt, gb = g_slot[k]
    if cls == "adopted":
        m = (gt + gb) / 2; gt, gb = m - 3, m + 3
    return [(t, gt), (b, gb)]
for side in "LR":
    ks = [l[0] for l in lanes if not (side == "L" and l[0] == "wais")]
    E = [e for k in ks for e in _ends(k, side)]
    for (a0, a1), (b0, b1) in zip(E, E[1:]):
        assert b0 >= a0 and b1 >= a1, ("crossing", side, a0, a1, b0, b1)

def lane_fill(k, col):
    return "url(#hatch)" if k == "smb" else col

for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    t, b = y - LH / 2, y + LH / 2
    gt, gb = g_slot[k]
    if cls == "adopted":
        # dashed ribbon: centred on its stack slot, stroke width equal to slot thickness
        st, sb = beam_slot[k]; ys = (st + sb) / 2
        go = (gt + gb) / 2
        d = f"M{BX1},{ys:.1f} {curve(BX1, ys, LX0, y)} L{LX1},{y} {curve(LX1, y, SX0, go)}"
        add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{ADW}" stroke-dasharray="16 9"/>')
        continue
    fill = lane_fill(k, col)
    if k == "wais":
        xv, w = 300, LH
        r_out, r_in = LH + 10, 10
        d = (f"M{xv-w/2:.1f},128 L{xv-w/2:.1f},{b-r_out:.1f} Q{xv-w/2:.1f},{b:.1f} {xv-w/2+r_out:.1f},{b:.1f} "
             f"L{LX0},{b} L{LX0},{t} L{xv+w/2+r_in:.1f},{t} Q{xv+w/2:.1f},{t:.1f} {xv+w/2:.1f},{t-r_in:.1f} L{xv+w/2:.1f},128 Z")
        add(f'<path d="{d}" fill="{fill}"/>')
    else:
        st, sb = beam_slot[k]
        ribbon(BX1 - 2, st, sb, LX0 + 1, t, b, fill)
    add(f'<rect x="{LX0}" y="{t}" width="{LX1-LX0}" height="{LH}" fill="{fill}"/>')
    ribbon(LX1 - 1, t, b, SX0 + 1, gt, gb, fill)

# beam
add(f'<rect x="{BX0}" y="{BT}" width="{BX1-BX0}" height="{BB-BT}" fill="{NAVY}"/>')
# Greenland ocean temperature: a strip of the beam in the discharge colour; the discharge lane
# leaves the beam straight from it
add(f'<rect x="{BX0+10}" y="{STRIP[0]}" width="{BX1-BX0-10}" height="{STRIP[1]-STRIP[0]}" fill="{B300}"/>')
add(f'<rect x="{BX1-3}" y="{STRIP[0]}" width="{LX0-BX1+4}" height="{STRIP[1]-STRIP[0]}" fill="{B300}"/>')
add(f'<text x="{BX0+20}" y="{lane_y["gd"]-4}" class="b">Greenland ocean T</text>')
add(f'<text x="{BX0+20}" y="{lane_y["gd"]+19}">via Arctic amplification</text>')
tx_, ty_ = BX1 - 26, BT + 12
add(f'<g fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round">'
    f'<rect x="{tx_-5}" y="{ty_}" width="10" height="32" rx="5"/>'
    f'<circle cx="{tx_}" cy="{ty_+40}" r="8" fill="#fff"/>'
    f'<line x1="{tx_}" y1="{ty_+12}" x2="{tx_}" y2="{ty_+34}" stroke-width="4"/>'
    f'<line x1="{tx_+9}" y1="{ty_+8}" x2="{tx_+13}" y2="{ty_+8}"/><line x1="{tx_+9}" y1="{ty_+16}" x2="{tx_+13}" y2="{ty_+16}"/>'
    f'<line x1="{tx_+9}" y1="{ty_+24}" x2="{tx_+13}" y2="{ty_+24}"/></g>')
add(f'<text x="{BX0+16}" y="{BT+44}" class="h" style="fill:#fff;font-size:34px">GMST</text>')
add(f'<text x="{BX0+16}" y="{BT+76}" style="fill:#fff">observed to 2025</text>')
add(f'<text x="{BX0+16}" y="{BT+102}" style="fill:#fff">AR6: 1.8, 2.7, 3.6 °C</text>')
add(f'<text x="{BX0}" y="{BT-42}" style="fill:{B400}">Global mean</text>')
add(f'<text x="{BX0}" y="{BT-16}" style="fill:{B400}">surface temperature</text>')

# lane labels
PCOL = {"gl": NAVY, "smb": B300}   # very light lanes: darker stroke for contrast
TX = LX0 + 64
for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    picto(k, LX0 + 30, y, PCOL.get(k, col), ring=(cls == "adopted"))
    if cls == "adopted":
        add(f'<text x="{TX}" y="{y-8}" class="b halo">{name}</text>')
        add(f'<text x="{TX}" y="{y+23}" class="halo">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-8}" text-anchor="end" class="tag halo" style="fill:{B400}">adopted</text>')
        continue
    add(f'<text x="{TX}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
    add(f'<text x="{TX}" y="{y+19}" style="fill:{tc}">{sub}</text>')
    add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">{cls}</text>')

# ---------------- stacked area: hindcast + forecast (schematic) ----------------
layer_fill = {k: (col if k != "smb" else "url(#hatchs)") for k, n, sb, c, col, t in lanes}
for k in order_bottom_up:
    top, bot = edges[k]
    add(f'<path d="{poly(top, bot)}" fill="{layer_fill[k]}"/>')
CLS = {l[0]: l[3] for l in lanes}
for j, k in enumerate(order_bottom_up[:-1]):   # white hairlines between layers; dashed around adopted layers
    top = edges[k][0]
    dash = CLS[k] == "adopted" or CLS[order_bottom_up[j + 1]] == "adopted"
    sty = 'stroke-width="1.5" stroke-dasharray="4 3"' if dash else 'stroke-width="0.8"'
    add('<path d="M' + " L".join(f"{x},{y:.1f}" for x, y in zip(xs, top)) + f'" fill="none" stroke="#fff" {sty}/>')
fx = [i for i, x in enumerate(xs) if x >= SX1]
# West Antarctica tail: one band above the whole scenario envelope, the same for every path
tail_top = [L2[i] - (L2[i] - TAIL_TOP) * ((xs[i] - SX1) / (SX2 - SX1)) ** 1.6 for i in fx]
pts = [f"{xs[i]},{yy:.1f}" for i, yy in zip(fx, tail_top)] + [f"{xs[i]},{L2[i]:.1f}" for i in reversed(fx)]
add(f'<path d="M{" L".join(pts)} Z" fill="url(#tail)"/>')
# scenario paths (1.8, 2.7, 3.6 C) drawn through the tail
for name, yl in (("1.8 °C", L2), ("2.7 °C", L3), ("3.6 °C", L4)):
    d = "M" + " L".join(f"{xs[i]},{yl[i]:.1f}" for i in fx)
    add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>')
    add(f'<path d="{d}" fill="none" stroke="{B500}" stroke-width="3" stroke-linecap="round"/>')
    yv = yl[-1]
    add(f'<line x1="{SX2}" x2="{SX2+10}" y1="{yv:.1f}" y2="{yv:.1f}" stroke="{B500}" stroke-width="2"/>')
    add(f'<text x="{SX2+16}" y="{yv+7:.1f}" class="b">{name}</text>')
for j, (t, c) in enumerate((("West", "b"), ("Antarctica", "b"), ("tail, all", "tag"), ("scenarios", "tag"))):
    add(f'<text x="{SX2+16}" y="{TAIL_TOP+2+24*j}" class="{c}" style="fill:{EARTH}">{t}</text>')
# modelled sum label
add(f'<text x="{SX1+10}" y="{BASE-9}" class="h" style="fill:#fff;font-size:20px">GMSL</text>')

# observed GMSL (withheld): dotted line over the hindcast stack top only
st = edges["wais"][0]
import math
hx = [i for i, x in enumerate(xs) if x <= SX1]
d = "M" + " L".join(f"{xs[i]},{st[i] - 3 + 2.2*math.sin(xs[i]/7.0) + 1.4*math.sin(xs[i]/3.1):.1f}" for i in hx)
add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" stroke-dasharray="0.1 9"/>')
add(f'<path d="{d}" fill="none" stroke="{NAVY}" stroke-width="4.2" stroke-linecap="round" stroke-dasharray="0.1 9"/>')
lx, ly = 924, 314
add(f'<text x="{lx}" y="{ly}" class="b">observed</text>')
add(f'<text x="{lx}" y="{ly+24}" class="b">GMSL</text>')
add(f'<text x="{lx}" y="{ly+48}" class="tag" style="fill:{B400}">withheld</text>')
ii = min(range(len(xs)), key=lambda i: abs(xs[i] - 990)); ix = xs[ii]; iy = st[ii]
add(f'<line x1="{ix-8}" y1="{ly+60}" x2="{ix}" y2="{iy-12:.1f}" stroke="{NAVY}" stroke-width="2"/>')

# time axis
add(f'<line x1="{SX0}" x2="{SX2}" y1="{BASE}" y2="{BASE}" stroke="{NAVY}" stroke-width="2"/>')
for x, yr in ((SX0, "1950"), (SX1, "2025"), (SX2, "2100")):
    add(f'<line x1="{x}" x2="{x}" y1="{BASE}" y2="{BASE+8}" stroke="{NAVY}" stroke-width="2"/>')
    add(f'<text x="{x}" y="{BASE+32}" text-anchor="middle" class="b">{yr}</text>')
add(f'<line x1="{SX1}" x2="{SX1}" y1="280" y2="{BASE}" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="4 4"/>')
add(f'<line x1="{SX2}" x2="{SX2}" y1="{TAIL_TOP}" y2="{BASE}" stroke="{OVER}" stroke-width="2" stroke-dasharray="4 4"/>')
add(f'<text x="{(SX0+SX2)/2}" y="{BASE+56}" text-anchor="middle" class="tag" style="fill:{GREY}">schematic stack and fan</text>')

# test block below the stack
tx, ty = 890, BASE + 98
add(f'<text x="{tx}" y="{ty}" class="b" style="font-size:22px">Test against observed GMSL,</text>')
add(f'<text x="{tx}" y="{ty+26}" class="b" style="font-size:22px">withheld from every calibration</text>')
for i, s in enumerate(["closes the altimetry budget, 1993–2025", "hindcasts GMSL back to 1950"]):
    yy = ty + 58 + 28 * i
    add(f'<path d="M{tx+2},{yy-7} l6,7 l12,-14" fill="none" stroke="{B400}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    add(f'<text x="{tx+28}" y="{yy}">{s}</text>')

# ---------------- bottom: two dimensions of risk ----------------
ry = 812
add(f'<line x1="24" x2="{W-24}" y1="{ry-38}" y2="{ry-38}" stroke="{OVER}" stroke-width="2"/>')
add(f'<text x="24" y="{ry}" class="h" style="font-size:22px">Two dimensions of risk</text>')
for j, c in enumerate((B200, B300, B400, GREEN)):   # the warming-driven layers, top to bottom
    add(f'<rect x="330" y="{ry-18+5*j}" width="36" height="5" fill="{c}"/>')
add(f'<text x="376" y="{ry}">Emissions set the warming-driven components</text>')
add(f'<rect x="980" y="{ry-18}" width="36" height="20" fill="{EARTH}"/>')
add(f'<text x="1026" y="{ry}">West Antarctica sets the tail</text>')
add(f'<text x="24" y="{ry+44}" style="fill:{GREY}">Lane style: solid = calibrated by Bayesian inference against independent multi-decade records · hatched = emulated · dashed = adopted</text>')

add('</svg>')

html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>SPECTRUM Figure 1</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Rubik:wght@900&display=swap" rel="stylesheet">
<style>body{{margin:0;background:#fff}}</style></head>
<body><div style="position:relative;width:{W}px;height:{H}px;overflow:hidden;background:#fff">
{chr(10).join(out)}
</div></body></html>'''
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "figure.html"), "w").write(html)
print("ok", {k: tuple(round(v) for v in g_slot[k]) for k in g_slot})
