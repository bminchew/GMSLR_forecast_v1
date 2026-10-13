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

# ---------------- geometry ----------------
BX0, BX1 = 24, 240            # beam
BT, BB = 330, 480             # beam top/bottom
LX0, LX1 = 330, 680           # lane straight section
GX = 810                      # GMSL ribbon starts
GT = 392                      # GMSL ribbon top
PITCH, Y0, LH = 70, 166, 52   # lane centres and solid lane height

# lanes top->bottom: key, name, model/record, class, colours
lanes = [
    ("wais", "West Antarctica", "slow/fast, independent of emissions", "storyline", EARTH, "#fff"),
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
slice_out = {"wais": 26, "tws": 10, "eais": 10, "smb": 14, "ap": 14, "gl": 14, "th": 18, "gd": 14}
GB = GT + sum(slice_out.values()) + 16

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
def mix(c, f):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(v + (255 - v) * f) for v in (r, g, b))
GRADS = {}
def lane_grad(col):
    if col not in GRADS:
        GRADS[col] = f"lg{len(GRADS)}"
    return f"url(#{GRADS[col]})"
for _c in (EARTH, B300, B400, B500, NAVY):
    lane_grad(_c)
lane_grads = "".join(
    f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{BX1}" y1="0" x2="{GX}" y2="0">'
    f'<stop offset="0" stop-color="{mix(c, 0.22)}"/><stop offset="1" stop-color="{c}"/></linearGradient>'
    for c, gid in GRADS.items())
add(f'''<defs>
<pattern id="hatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(45)">
  <rect width="12" height="12" fill="{B100}"/><rect width="5" height="12" fill="{B200}"/>
</pattern>
<linearGradient id="bg" gradientUnits="userSpaceOnUse" x1="0" y1="300" x2="0" y2="{H}">
  <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#EAF2F8"/>
</linearGradient>
{lane_grads}
<pattern id="wstripe" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(-45)">
  <rect width="10" height="10" fill="{EARTH}" fill-opacity="0.16"/><rect width="4" height="10" fill="{EARTH}" fill-opacity="0.9"/>
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
add(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')

# ---------------- stage headers ----------------
heads = [(24, "1", "Forcing"), (352, "2", "One model per contributor"),
         (780, "3", "Sum and test"), (1036, "4", "Forecast to 2100")]
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
    if k in ("tws", "eais"):
        cur += 5
    if k == "smb":
        cur += 6
    g_slot[k] = (cur, cur + slice_out[k]); cur += slice_out[k]

def lane_fill(k, col):
    return "url(#hatch)" if k == "smb" else lane_grad(col)

for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    t, b = y - LH / 2, y + LH / 2
    if cls == "adopted":
        # thin dashed ribbon: one continuous dashed stroke from beam to GMSL ribbon
        st, sb = beam_slot[k]; ys = (st + sb) / 2
        go = (g_slot[k][0] + g_slot[k][1]) / 2
        d = f"M{BX1},{ys:.1f} {curve(BX1, ys, LX0, y)} L{LX1},{y} {curve(LX1, y, GX, go)}"
        add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="5" stroke-dasharray="14 8"/>')
        continue
    fill = lane_fill(k, col)
    # entry
    if k == "wais":
        # enters from above, independent of the beam
        xv, w = 290, LH
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
    f'L{LX0-60},{yd+9} L{bx+60},{yd+9} C{bx-10},{yd+9} {bx-10},{BB} {bx-60},{BB-1} Z" fill="{mix(B500, 0.1)}"/>')
ribbon(LX0 - 62, yd - 9, yd + 9, LX0 + 1, yd - LH / 2, yd + LH / 2, lane_grad(NAVY))
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
    if k == "wais":
        add(f'<text x="{LX0+14}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
        add(f'<text x="{LX0+14}" y="{y+19}" style="fill:{tc}">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">storyline</text>')
        continue
    add(f'<text x="{LX0+14}" y="{y-4}" class="b" style="fill:{tc}">{name}</text>')
    add(f'<text x="{LX0+14}" y="{y+19}" style="fill:{tc}">{sub}</text>')
    add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag" style="fill:{tc}">{cls}</text>')

# ---------------- GMSL ribbon + tide-gauge staff ----------------
RX1 = 940
add(f'<rect x="{GX}" y="{GT}" width="{RX1-GX}" height="{GB-GT}" fill="{NAVY}"/>')
add(f'<text x="{GX+10}" y="{(GT+GB)/2+10}" class="h" style="fill:#fff;font-size:28px">GMSL</text>')
sx, sw, st, sb = 908, 22, GT - 54, GB + 46
add(f'<rect x="{sx}" y="{st}" width="{sw}" height="{sb-st}" rx="3" fill="#fff" stroke="{B500}" stroke-width="2"/>')
yy, i = st + 6, 0
while yy < sb - 6:
    ln = sw - 6 if i % 5 == 0 else (sw - 12 if i % 5 in (1, 3) else sw - 15)
    add(f'<rect x="{sx+3}" y="{yy:.1f}" width="{ln}" height="3" fill="{B500}"/>')
    yy += 7.5; i += 1
# test block below, leader from staff foot
tx, ty = 790, GB + 96
add(f'<path d="M{sx+sw/2},{sb} L{sx+sw/2},{ty-30}" stroke="{B500}" stroke-width="1.5" fill="none"/>')
add(f'<circle cx="{sx+sw/2}" cy="{ty-30}" r="3" fill="{B500}"/>')
add(f'<text x="{tx}" y="{ty}" class="b" style="font-size:22px">Test against observed GMSL,</text>')
add(f'<text x="{tx}" y="{ty+26}" class="b" style="font-size:22px">withheld from every calibration</text>')
for i, s_ in enumerate(["closes the altimetry budget, 1993–2025", "hindcasts GMSL back to 1950"]):
    yy = ty + 64 + 30 * i
    add(f'<path d="M{tx+2},{yy-7} l6,7 l12,-14" fill="none" stroke="{B400}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    add(f'<text x="{tx+28}" y="{yy}">{s_}</text>')

# ---------------- forecast fan (schematic, nested bands) ----------------
fx0, fx1 = RX1, 1196
fy = (GT + GB) / 2
def band(h0, top_end, bot_end, col, op=1):
    add(f'<path d="M{fx0},{fy-h0:.1f} C{fx0+100},{fy-h0:.1f} {fx1-90},{top_end:.1f} {fx1},{top_end:.1f} '
        f'L{fx1},{bot_end:.1f} C{fx1-90},{bot_end:.1f} {fx0+100},{fy+h0:.1f} {fx0},{fy+h0:.1f} Z" '
        f'fill="{col}" fill-opacity="{op}"/>')
# 2100 marker
add(f'<line x1="{fx1}" x2="{fx1}" y1="160" y2="{GB+6}" stroke="{B300}" stroke-width="1.5" stroke-dasharray="4 4"/>')
bands = [("4 °C", 60, 240, 462, B300), ("3 °C", 46, 302, 450, B400), ("2 °C", 32, 362, 438, B500)]
for lab, h0, tp, bt, col in bands:
    band(h0, tp, bt, col)
# Earth tail (West Antarctica): one band from the top of the 2 C band up past 4 C, over all scenarios
tail_top = (f"M{fx0},{fy-60:.1f} C{fx0+130},{fy-62:.1f} {fx1-80},{172} {fx1},{168}")
add(f'<path d="{tail_top} L{fx1},362 C{fx1-90},362 {fx0+100},{fy-32:.1f} {fx0},{fy-32:.1f} Z" '
    f'fill="url(#wstripe)"/>')
add(f'<path d="{tail_top}" fill="none" stroke="{EARTH}" stroke-width="2.5"/>')
# faint 3/4 C edges kept legible through the tint
for h0, tp in ((60, 240), (46, 302)):
    add(f'<path d="M{fx0},{fy-h0:.1f} C{fx0+100},{fy-h0:.1f} {fx1-90},{tp} {fx1},{tp}" fill="none" stroke="#fff" stroke-width="1.5" stroke-opacity="0.9"/>')
mids = {"4 °C": 271, "3 °C": 332, "2 °C": 398}
for lab, h0, tp, bt, col in bands:
    y = mids[lab]
    add(f'<line x1="{fx1-14}" x2="{fx1+20}" y1="{y}" y2="{y}" stroke="{NAVY}" stroke-width="1.5"/>')
    add(f'<text x="{fx1+26}" y="{y+7}" class="b">{lab}</text>')
# tail label in the open space above the fan, leader into the band
lx, ly = 1080, 200
add(f'<text x="{lx}" y="{ly}" text-anchor="end" class="b" style="fill:{EARTH}">West Antarctica tail,</text>')
add(f'<text x="{lx}" y="{ly+24}" text-anchor="end" class="tag" style="fill:{EARTH}">all scenarios</text>')
add(f'<path d="M{lx+6},{ly-6} L{lx+60},{ly-6} L{lx+86},{ly+44}" fill="none" stroke="{EARTH}" stroke-width="1.5"/>')
add(f'<circle cx="{lx+86}" cy="{ly+44}" r="3" fill="{EARTH}"/>')
add(f'<text x="{fx1}" y="{GB+32}" text-anchor="middle" class="b">2100</text>')
add(f'<text x="{fx0+16}" y="{GB+32}" class="tag" style="fill:#6b7a88">schematic</text>')
add(f'<text x="{W-24}" y="154" text-anchor="end">2000 members under 2, 3, 4 °C</text>')

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
