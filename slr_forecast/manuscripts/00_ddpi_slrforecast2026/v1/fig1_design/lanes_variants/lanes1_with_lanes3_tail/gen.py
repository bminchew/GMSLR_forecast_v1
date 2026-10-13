#!/usr/bin/env python3
"""Generate figure.html: SPECTRUM Figure 1, 'prism split' flowing-lanes design."""
import os

W, H = 1344, 880
NAVY = "#031D3A"; B500 = "#07456C"; B400 = "#036C9A"; B300 = "#72A3C3"
B200 = "#B5D5E9"; B100 = "#D8E7F1"; EARTH = "#AB6638"; OVER = "#D9D9D9"
MID = "#3A88AF"

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
BX0, BX1 = 24, 236            # beam
BT, BB = 330, 480             # beam top/bottom
LX0, LX1 = 336, 816           # lane straight section
GX = 896                      # GMSL ribbon starts
GT = 352                      # GMSL ribbon top
PITCH, Y0, LH = 70, 166, 52   # lane centres and solid lane height

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
# slice thickness in the beam and in the GMSL ribbon
slice_in = {"tws": 8, "eais": 8, "smb": 24, "ap": 26, "gl": 26, "th": 28, "gd": 0}
slice_out = {"wais": 26, "tws": 12, "eais": 12, "smb": 14, "ap": 14, "gl": 14, "th": 18, "gd": 14}
GB = GT + sum(slice_out.values())

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'''<defs>
<pattern id="hatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(45)">
  <rect width="12" height="12" fill="{B100}"/><rect width="5" height="12" fill="{B200}"/>
</pattern>
<linearGradient id="tail" x1="0" y1="1" x2="0" y2="0">
  <stop offset="0" stop-color="{EARTH}" stop-opacity="0.85"/><stop offset="1" stop-color="{EARTH}" stop-opacity="0.08"/>
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
heads = [(24, "1", "Forcing"), (336, "2", "One model per contributor"),
         (830, "3", "Sum and test"), (1060, "4", "Forecast to 2100")]
for x, n, t in heads:
    add(f'<circle cx="{x+15}" cy="96" r="15" fill="{NAVY}"/>'
        f'<text x="{x+15}" y="103" text-anchor="middle" class="h" style="fill:#fff;font-size:20px">{n}</text>'
        f'<text x="{x+40}" y="104" class="h" style="font-size:23px">{t}</text>')
add(f'<line x1="24" x2="{W-24}" y1="124" y2="124" stroke="{OVER}" stroke-width="2"/>')

# title
add(f'<text x="24" y="52" class="h" style="font-size:44px;letter-spacing:0.5px">SPECTRUM</text>')
add(f'<text x="300" y="50" style="font-size:22px;fill:{B400}">Sea-level Projections from Empirical Calibration of '
    f'Thermodynamical, Reduced-physics Models</text>')

# ---------------- lanes ----------------
lane_y = {k: Y0 + PITCH * i for i, (k, *_r) in enumerate(lanes)}
# beam slice positions (top->bottom) for beam-fed lanes
beam_keys = ["tws", "eais", "smb", "ap", "gl", "th"]
gap = ((BB - BT) - sum(slice_in[k] for k in beam_keys) - 26) / (len(beam_keys))  # 26 reserved for ocean branch
cur = BT + gap / 2
beam_slot = {}
for k in beam_keys:
    beam_slot[k] = (cur, cur + slice_in[k]); cur += slice_in[k] + gap
# GMSL slots
cur = GT; g_slot = {}
for k, *_r in lanes:
    g_slot[k] = (cur, cur + slice_out[k]); cur += slice_out[k]

# No-crossing check: all convergence edges share one Bezier shape, so edges ordered at both
# ends stay ordered throughout. Dashed strokes (6 px) need >= 3 + 2 px clearance at both ends.
def _edges(k):
    y, (gt, gb) = lane_y[k], g_slot[k]
    if dict((l[0], l[3]) for l in lanes)[k] == "adopted":
        c = (gt + gb) / 2
        return [(y - 5, c - 5), (y + 5, c + 5)]
    return [(y - LH / 2, gt), (y + LH / 2, gb)]
_all = [e for k, *_r in lanes for e in _edges(k)]
for (a0, a1), (b0, b1) in zip(_all, _all[1:]):
    assert b0 >= a0 and b1 >= a1, ("convergence edges cross", a0, a1, b0, b1)

def lane_fill(k, col):
    return "url(#hatch)" if k == "smb" else col

for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    t, b = y - LH / 2, y + LH / 2
    if cls == "adopted":
        # thin dashed ribbon: one continuous dashed stroke from beam to GMSL ribbon
        st, sb = beam_slot[k]; ys = (st + sb) / 2
        go = (g_slot[k][0] + g_slot[k][1]) / 2
        d = f"M{BX1},{ys:.1f} {curve(BX1, ys, LX0, y)} L{LX1},{y} {curve(LX1, y, GX, go)}"
        add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="16 9"/>')
        continue
    fill = lane_fill(k, col)
    # entry
    if k == "wais":
        # enters from above, independent of the beam
        xv, w = 300, LH
        d = (f"M{xv-w/2:.1f},128 L{xv-w/2:.1f},{t+ w*0.0:.1f} C{xv-w/2:.1f},{t+0:.1f} {xv-w/2:.1f},{b:.1f} {xv+w/2+20:.1f},{b:.1f} "
             f"L{LX0},{b} L{LX0},{t} L{xv+w/2+20:.1f},{t} C{xv+w/2:.1f},{t:.1f} {xv+w/2:.1f},{t:.1f} {xv+w/2:.1f},128 Z")
        # simpler: quarter-turn band
        r_out, r_in = LH + 10, 10
        cx, cy = xv - w / 2 + r_out, t + r_out - 0  # unused
        d = (f"M{xv-w/2:.1f},128 L{xv-w/2:.1f},{b-r_out:.1f} Q{xv-w/2:.1f},{b:.1f} {xv-w/2+r_out:.1f},{b:.1f} "
             f"L{LX0},{b} L{LX0},{t} L{xv+w/2+r_in:.1f},{t} Q{xv+w/2:.1f},{t:.1f} {xv+w/2:.1f},{t-r_in:.1f} L{xv+w/2:.1f},128 Z")
        add(f'<path d="{d}" fill="{fill}"/>')
    elif k == "gd":
        pass  # drawn with ocean branch below
    else:
        st, sb = beam_slot[k]
        ribbon(BX1 - 2, st, sb, LX0 + 1, t, b, fill)
    # straight lane
    add(f'<rect x="{LX0}" y="{t}" width="{LX1-LX0}" height="{LH}" fill="{fill}"/>')
    # convergence
    gt, gb = g_slot[k]
    ribbon(LX1 - 1, t, b, GX + 1, gt, gb, fill)

# ocean temperature branch: peels off the underside of the beam, feeds discharge
yd = lane_y["gd"]
ob_t0, ob_b0 = BB - 26, BB
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
tx_, ty_ = BX1 - 26, BT + 12
add(f'<g fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round">'
    f'<rect x="{tx_-5}" y="{ty_}" width="10" height="32" rx="5"/>'
    f'<circle cx="{tx_}" cy="{ty_+40}" r="8" fill="#fff"/>'
    f'<line x1="{tx_}" y1="{ty_+12}" x2="{tx_}" y2="{ty_+34}" stroke-width="4"/>'
    f'<line x1="{tx_+9}" y1="{ty_+8}" x2="{tx_+13}" y2="{ty_+8}"/><line x1="{tx_+9}" y1="{ty_+16}" x2="{tx_+13}" y2="{ty_+16}"/>'
    f'<line x1="{tx_+9}" y1="{ty_+24}" x2="{tx_+13}" y2="{ty_+24}"/></g>')
add(f'<text x="{BX0}" y="{BT-42}" style="fill:{B400}">Global mean</text>')
add(f'<text x="{BX0}" y="{BT-16}" style="fill:{B400}">surface temperature</text>')

# lane labels
PCOL = {"smb": B300}  # hatched lane: one step darker than its hatch
TX = LX0 + 64
for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    picto(k, LX0 + 30, y, PCOL.get(k, col), ring=(cls == "adopted"))
    if cls == "adopted":
        add(f'<text x="{TX}" y="{y-8}" class="b halo">{name}</text>')
        add(f'<text x="{TX}" y="{y+23}" class="halo">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-8}" text-anchor="end" class="tag halo" style="fill:{B400}">adopted</text>')
        continue
    if k == "wais":
        add(f'<text x="{TX}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
        add(f'<text x="{TX}" y="{y+19}" style="fill:{tc}">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">storyline</text>')
        continue
    add(f'<text x="{TX}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
    add(f'<text x="{TX}" y="{y+19}" style="fill:{tc}">{sub}</text>')
    add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">{cls}</text>')

# ---------------- GMSL ribbon + gate ----------------
RX1 = 1050
add(f'<rect x="{GX}" y="{GT}" width="{RX1-GX}" height="{GB-GT}" fill="{NAVY}"/>')
add(f'<text x="{GX+10}" y="{(GT+GB)/2+10}" class="h" style="fill:#fff;font-size:28px">GMSL</text>')
# test gate: two posts and a lintel carrying a check mark
gx0, gx1, pw = 1000, 1036, 10          # post left edges, post width
gtop, gbot = GT - 64, GB + 26
add(f'<rect x="{gx0-6}" y="{GT}" width="{gx1+pw-gx0+12}" height="{GB-GT}" fill="{B400}" fill-opacity="0.35"/>')
for px in (gx0, gx1):
    add(f'<rect x="{px}" y="{gtop}" width="{pw}" height="{gbot-gtop}" rx="2" fill="{B400}" stroke="#fff" stroke-width="3"/>')
lx0, lx1 = gx0 - 14, gx1 + pw + 14
add(f'<rect x="{lx0}" y="{gtop}" width="{lx1-lx0}" height="34" rx="4" fill="{B400}" stroke="#fff" stroke-width="3"/>')
cxg = (lx0 + lx1) / 2
add(f'<path d="M{cxg-11},{gtop+17} l7,8 l15,-16" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
# test block below
tx, ty = 872, GB + 92
add(f'<line x1="{cxg}" y1="{gbot+2}" x2="{cxg}" y2="{ty-28}" stroke="{B400}" stroke-width="3"/>')
add(f'<text x="{tx}" y="{ty}" class="b" style="font-size:22px">Test against observed GMSL,</text>')
add(f'<text x="{tx}" y="{ty+26}" class="b" style="font-size:22px">withheld from every calibration</text>')
for i, s in enumerate(["closes the altimetry budget, 1993–2025", "hindcasts GMSL back to 1950"]):
    yy = ty + 64 + 30 * i
    add(f'<path d="M{tx+2},{yy-7} l6,7 l12,-14" fill="none" stroke="{B400}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    add(f'<text x="{tx+28}" y="{yy}">{s}</text>')

# ---------------- forecast fan (schematic) ----------------
fx0, fx1 = RX1, 1216
fy = (GT + GB) / 2
r0 = (GB - GT) / 2 - 6                 # root half-height, close to the ribbon
def fanband(rt, rb, top_end, bot_end, col, op):
    k1, k2 = 60, 70
    add(f'<path d="M{fx0},{rt:.1f} C{fx0+k1},{rt:.1f} {fx1-k2},{top_end:.1f} {fx1},{top_end:.1f} '
        f'L{fx1},{bot_end:.1f} C{fx1-k2},{bot_end:.1f} {fx0+k1},{rb:.1f} {fx0},{rb:.1f} Z" '
        f'fill="{col}" fill-opacity="{op}"/>')
def fancurve(y_root, y_end):
    k1, k2 = 60, 70
    return f"C{fx0+k1},{y_root:.1f} {fx1-k2},{y_end:.1f} {fx1},{y_end:.1f}"
# Lanes-3 style: the 2 C envelope is filled; one West Antarctica tail sits above it for
# every scenario, and the 3 C and 4 C paths are drawn through the tail as lines.
TAIL_TOP, Y2, Y3, Y4, YBOT = 168, 314, 282, 250, 430
rt, rb = fy - r0, fy + r0
add(f'<path d="M{fx0},{rt:.1f} {fancurve(rt, Y2)} L{fx1},{YBOT} '
    f'C{fx1-70},{YBOT} {fx0+60},{rb:.1f} {fx0},{rb:.1f} Z" fill="{B400}" fill-opacity="0.85"/>')
add(f'<path d="M{fx0},{rt:.1f} {fancurve(rt, TAIL_TOP)} L{fx1},{Y2} '
    f'C{fx1-70},{Y2} {fx0+60},{rt:.1f} {fx0},{rt:.1f} Z" fill="url(#tail)"/>')
for lab, ye in (("2 °C", Y2), ("3 °C", Y3), ("4 °C", Y4)):
    d = f"M{fx0},{rt:.1f} {fancurve(rt, ye)}"
    add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>')
    add(f'<path d="{d}" fill="none" stroke="{B500}" stroke-width="3" stroke-linecap="round"/>')
    add(f'<text x="{fx1+10}" y="{ye+7}" class="b">{lab}</text>')
add(f'<text x="{fx0+8}" y="200" class="b halo" style="fill:{EARTH}">West Antarctica tail,</text>')
add(f'<text x="{fx0+8}" y="224" class="tag halo" style="fill:{EARTH}">all scenarios</text>')
add(f'<line x1="{fx1}" x2="{fx1}" y1="166" y2="{GB+8}" stroke="{OVER}" stroke-width="2" stroke-dasharray="4 4"/>')
add(f'<text x="{fx1}" y="{GB+32}" text-anchor="middle" class="b">2100</text>')
add(f'<text x="{fx0+12}" y="{GB+32}" class="tag" style="fill:#6b7a88">schematic</text>')
add(f'<text x="1052" y="156">2000 members under 2, 3, 4 °C</text>')

# ---------------- bottom: two dimensions of risk ----------------
ry = 806
add(f'<line x1="24" x2="{W-24}" y1="{ry-38}" y2="{ry-38}" stroke="{OVER}" stroke-width="2"/>')
add(f'<text x="24" y="{ry}" class="h" style="font-size:22px">Two dimensions of risk</text>')
add(f'<rect x="330" y="{ry-18}" width="36" height="20" fill="{B400}"/>')
add(f'<text x="376" y="{ry}">Emissions set the warming-driven components</text>')
add(f'<rect x="980" y="{ry-18}" width="36" height="20" fill="{EARTH}"/>')
add(f'<text x="1026" y="{ry}">West Antarctica sets the tail</text>')
add(f'<text x="24" y="{ry+44}" style="fill:#6b7a88">Lane style: solid = calibrated by Bayesian inference against independent multi-decade records · hatched = emulated · dashed = adopted</text>')

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
print("ok", GT, GB)
