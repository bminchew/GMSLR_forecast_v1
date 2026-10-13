#!/usr/bin/env python3
"""Generate figure.html: SPECTRUM Figure 1, flowing lanes that stack into a schematic
stacked-area hindcast and then fan out to 2100 (variant C)."""
import os

W, H = 1344, 880
NAVY = "#031D3A"; B500 = "#07456C"; B400 = "#036C9A"; B300 = "#72A3C3"
B200 = "#B5D5E9"; B100 = "#D8E7F1"; EARTH = "#AB6638"; OVER = "#D9D9D9"
MID = "#3A88AF"; GREY = "#6b7a88"

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

# ---------------- geometry ----------------
BX0, BX1 = 24, 250            # beam
BT, BB = 330, 480             # beam top/bottom
LX0, LX1 = 352, 790           # lane straight section
PITCH, Y0, LH = 70, 166, 52   # lane centres and solid lane height
ADW = 6                       # adopted (dashed) lane stroke width = its slot thickness

# stack (schematic): x positions of 1950, 2025, 2100 and the baseline
SX0, SX1, SX2 = 910, 1036, 1210
BASE = 560

# lanes top->bottom: key, name, model/record, class, colours
lanes = [
    ("wais", "West Antarctica", "slow/fast storyline, independent of emissions", "storyline", EARTH, "#fff"),
    ("tws", "Terrestrial water storage", "IPCC AR6", "adopted", B300, NAVY),
    ("eais", "East Antarctica", "snowfall sensitivity from literature", "adopted", MID, NAVY),
    ("smb", "Greenland SMB", "MAR emulator, 7 CMIP6 models", "emulated", None, NAVY),
    ("ap", "Antarctic Peninsula", "linear in warming · IMBIE-3", "calibrated", B300, NAVY),
    ("gl", "Glaciers", "linear in warming · GlaMBIE", "calibrated", B400, "#fff"),
    ("th", "Thermosteric", "two-layer ocean · NOAA 0–2000 m", "calibrated", B500, "#fff"),
    ("gd", "Greenland discharge", "ocean-forced, lagged · Mouginot 2019", "calibrated", NAVY, "#fff"),
]
# slice thickness in the beam
slice_in = {"tws": 8, "eais": 8, "smb": 24, "ap": 26, "gl": 26, "th": 28, "gd": 0}
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
  <rect width="12" height="12" fill="{B100}"/><rect width="5" height="12" fill="{B200}"/>
</pattern>
<pattern id="hatchs" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">
  <rect width="7" height="7" fill="{B100}"/><rect width="3" height="7" fill="{B200}"/>
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
heads = [(24, "1", "Forcing"), (352, "2", "One model per contributor"),
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
order_bottom_up = ["gd", "th", "gl", "ap", "smb", "eais", "tws", "wais"]

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
beam_keys = ["tws", "eais", "smb", "ap", "gl", "th"]
gap = ((BB - BT) - sum(slice_in[k] for k in beam_keys) - 26) / (len(beam_keys))
cur = BT + gap / 2
beam_slot = {}
for k in beam_keys:
    beam_slot[k] = (cur, cur + slice_in[k]); cur += slice_in[k] + gap

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
    elif k == "gd":
        pass  # drawn with ocean branch below
    else:
        st, sb = beam_slot[k]
        ribbon(BX1 - 2, st, sb, LX0 + 1, t, b, fill)
    add(f'<rect x="{LX0}" y="{t}" width="{LX1-LX0}" height="{LH}" fill="{fill}"/>')
    ribbon(LX1 - 1, t, b, SX0 + 1, gt, gb, fill)

# ocean temperature branch: peels off the underside of the beam, feeds discharge
yd = lane_y["gd"]
bx = 170
add(f'<path d="M{bx-60},{BB-1} L{bx},{BB-1} C{bx+40},{BB} {bx+20},{yd-9} {bx+80},{yd-9} L{LX0-60},{yd-9} '
    f'L{LX0-60},{yd+9} L{bx+60},{yd+9} C{bx-10},{yd+9} {bx-10},{BB} {bx-60},{BB-1} Z" fill="{B500}"/>')
ribbon(LX0 - 62, yd - 9, yd + 9, LX0 + 1, yd - LH / 2, yd + LH / 2, NAVY)
add(f'<text x="{bx-130}" y="{yd+50}" class="b">Greenland ocean temperature</text>')
add(f'<text x="{bx-130}" y="{yd+74}">via Arctic amplification</text>')

# beam
add(f'<rect x="{BX0}" y="{BT}" width="{BX1-BX0}" height="{BB-BT}" fill="{NAVY}"/>')
add(f'<text x="{BX0+16}" y="{BT+44}" class="h" style="fill:#fff;font-size:34px">GMST</text>')
add(f'<text x="{BX0+16}" y="{BT+76}" style="fill:#fff">observed to 2025</text>')
add(f'<text x="{BX0+16}" y="{BT+102}" style="fill:#fff">AR6 paths: 2, 3, 4 °C</text>')
add(f'<text x="{BX0}" y="{BT-42}" style="fill:{B400}">Global mean</text>')
add(f'<text x="{BX0}" y="{BT-16}" style="fill:{B400}">surface temperature</text>')

# lane labels
for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    if cls == "adopted":
        add(f'<text x="{LX0+14}" y="{y-8}" class="b halo">{name}</text>')
        add(f'<text x="{LX0+14}" y="{y+23}" class="halo">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-8}" text-anchor="end" class="tag halo" style="fill:{B400}">adopted</text>')
        continue
    add(f'<text x="{LX0+14}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
    add(f'<text x="{LX0+14}" y="{y+19}" style="fill:{tc}">{sub}</text>')
    add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">{cls}</text>')

# ---------------- stacked area: hindcast + forecast (schematic) ----------------
layer_fill = {"gd": NAVY, "th": B500, "gl": B400, "ap": B300, "smb": "url(#hatchs)",
              "eais": MID, "tws": B300, "wais": EARTH}
for k in order_bottom_up:
    top, bot = edges[k]
    add(f'<path d="{poly(top, bot)}" fill="{layer_fill[k]}"/>')
for k in order_bottom_up[:-1]:   # white hairlines between layers (no vertical edge at 1950)
    top = edges[k][0]
    add('<path d="M' + " L".join(f"{x},{y:.1f}" for x, y in zip(xs, top)) + '" fill="none" stroke="#fff" stroke-width="0.8"/>')
fx = [i for i, x in enumerate(xs) if x >= SX1]
# West Antarctica tail: one band above the whole scenario envelope, the same for every path
tail_top = [L2[i] - (L2[i] - TAIL_TOP) * ((xs[i] - SX1) / (SX2 - SX1)) ** 1.6 for i in fx]
pts = [f"{xs[i]},{yy:.1f}" for i, yy in zip(fx, tail_top)] + [f"{xs[i]},{L2[i]:.1f}" for i in reversed(fx)]
add(f'<path d="M{" L".join(pts)} Z" fill="url(#tail)"/>')
# scenario paths (2, 3, 4 C) drawn through the tail
for name, yl in (("2 °C", L2), ("3 °C", L3), ("4 °C", L4)):
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
add(f'<text x="1062" y="156">2000 members under 2, 3, 4 °C</text>')

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
add(f'<rect x="330" y="{ry-18}" width="36" height="20" fill="{B400}"/>')
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
