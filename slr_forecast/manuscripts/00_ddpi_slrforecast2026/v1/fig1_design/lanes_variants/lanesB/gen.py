#!/usr/bin/env python3
"""Generate figure.html: SPECTRUM Figure 1, flowing lanes, restyled to match the
Iceland talk deck (minchew_iceland_globalRiskStab_2026.pptx).

Deck typography (theme1.xml, slide master, slide 05 runs): Rubik Black headings,
Outfit body. Component colours sampled from slide 05's stacked-area image:
ocean #C9544B, glaciers #B5D5E9, Greenland #72A3C3, Antarctica #036C9A,
land water #AB6638, total edge #031D3A. West Antarctica uses #E0A27A, the colour
the deck gives West Antarctic collapse on slide 13.
"""
import os

W, H = 1344, 880
NAVY = "#031D3A"; B500 = "#07456C"; B400 = "#036C9A"; B300 = "#72A3C3"
B200 = "#B5D5E9"; B100 = "#D8E7F1"; OVER = "#D9D9D9"
# slide-05 component colours
OCEAN = "#C9544B"; GLAC = "#B5D5E9"; GRIS = "#72A3C3"; ANT = "#036C9A"; LANDW = "#AB6638"
WAIS = "#E0A27A"; WAIS_D = "#9A5A2E"   # deck slide 13 'Collapsed'; darker tone for text
GREY = "#5E6B78"

out = []
def add(s): out.append(s)

def ribbon(x0, t0, b0, x1, t1, b1, fill, extra=""):
    xm = (x0 + x1) / 2
    d = (f"M{x0:.1f},{t0:.1f} C{xm:.1f},{t0:.1f} {xm:.1f},{t1:.1f} {x1:.1f},{t1:.1f} "
         f"L{x1:.1f},{b1:.1f} C{xm:.1f},{b1:.1f} {xm:.1f},{b0:.1f} {x0:.1f},{b0:.1f} Z")
    add(f'<path d="{d}" fill="{fill}" {extra}/>')

def curve(x0, y0, x1, y1):
    xm = (x0 + x1) / 2
    return f"C{xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"

# ---------------- geometry ----------------
BX0, BX1 = 24, 236            # beam
BT, BB = 330, 480
LX0, LX1 = 330, 734           # straight lane section
GX = 858                      # GMSL ribbon start
RX1 = 1000                    # GMSL ribbon end
GT = 380                      # GMSL ribbon top
SLOTGAP = 4
PITCH, Y0, LH = 70, 166, 52

# top->bottom, grouped as slide 05 stacks them: Antarctica, land water, glaciers, ocean, Greenland
lanes = [
    ("wais", "West Antarctica", "slow/fast, independent of emissions", "storyline", WAIS, NAVY),
    ("eais", "East Antarctica", "snowfall sensitivity from literature", "adopted", ANT, NAVY),
    ("ap", "Antarctic Peninsula", "linear in warming · IMBIE-3", "calibrated", ANT, "#fff"),
    ("tws", "Terrestrial water storage", "IPCC AR6", "adopted", LANDW, NAVY),
    ("gl", "Glaciers", "linear in warming · GlaMBIE", "calibrated", GLAC, NAVY),
    ("th", "Thermosteric", "two-layer ocean · NOAA 0–2000 m", "calibrated", OCEAN, "#fff"),
    ("smb", "Greenland SMB", "MAR emulator, 7 CMIP6 models", "emulated", GRIS, NAVY),
    ("gd", "Greenland discharge", "ocean-forced, lagged · Mouginot 2019", "calibrated", GRIS, NAVY),
]
slice_in = {"eais": 8, "ap": 24, "tws": 8, "gl": 24, "th": 26, "smb": 24}
slice_out = {"wais": 26, "eais": 10, "ap": 14, "tws": 10, "gl": 14, "th": 18, "smb": 14, "gd": 14}
GB = GT + sum(slice_out.values()) + SLOTGAP * (len(lanes) - 1)

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'''<defs>
<pattern id="hatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(45)">
  <rect width="12" height="12" fill="{B100}"/><rect width="5" height="12" fill="{GRIS}"/>
</pattern>
<linearGradient id="tail" x1="0" y1="1" x2="0" y2="0">
  <stop offset="0" stop-color="{WAIS}" stop-opacity="1"/><stop offset="1" stop-color="{WAIS}" stop-opacity="0.6"/>
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

# ---------------- title + stage headers ----------------
add(f'<text x="24" y="52" class="h" style="font-size:44px;letter-spacing:0.5px">SPECTRUM</text>')
add(f'<text x="300" y="50" style="font-size:22px;fill:{B500}">Sea-level Projections from Empirical Calibration of '
    f'Thermodynamical, Reduced-physics Models</text>')
heads = [(24, "1", "Forcing"), (330, "2", "One model per contributor"),
         (800, "3", "Sum and test"), (1040, "4", "Forecast to 2100")]
for x, n, t in heads:
    add(f'<circle cx="{x+15}" cy="96" r="15" fill="{NAVY}"/>'
        f'<text x="{x+15}" y="103" text-anchor="middle" class="h" style="fill:#fff;font-size:20px">{n}</text>'
        f'<text x="{x+40}" y="104" class="h" style="font-size:23px">{t}</text>')
add(f'<line x1="24" x2="{W-24}" y1="124" y2="124" stroke="{OVER}" stroke-width="2"/>')

# ---------------- slots ----------------
lane_y = {k: Y0 + PITCH * i for i, (k, *_r) in enumerate(lanes)}
beam_keys = [k for k, *_r in lanes if k in slice_in]
gap = ((BB - BT) - sum(slice_in.values()) - 22) / len(beam_keys)   # 22 reserved for ocean branch
cur = BT + gap / 2; beam_slot = {}
for k in beam_keys:
    beam_slot[k] = (cur, cur + slice_in[k]); cur += slice_in[k] + gap
cur = GT; g_slot = {}
for k, *_r in lanes:
    g_slot[k] = (cur, cur + slice_out[k]); cur += slice_out[k] + SLOTGAP

# ---------------- lanes ----------------
for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]; t, b = y - LH / 2, y + LH / 2
    if cls == "adopted":
        st, sb = beam_slot[k]; ys = (st + sb) / 2
        go = (g_slot[k][0] + g_slot[k][1]) / 2
        d = f"M{BX1},{ys:.1f} {curve(BX1, ys, LX0, y)} L{LX1},{y} {curve(LX1, y, GX, go)}"
        add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="15 8"/>')
        continue
    fill = "url(#hatch)" if k == "smb" else col
    if k == "wais":
        xv, w = 282, LH; r_out, r_in = LH + 10, 10
        d = (f"M{xv-w/2:.1f},128 L{xv-w/2:.1f},{b-r_out:.1f} Q{xv-w/2:.1f},{b:.1f} {xv-w/2+r_out:.1f},{b:.1f} "
             f"L{LX0},{b} L{LX0},{t} L{xv+w/2+r_in:.1f},{t} Q{xv+w/2:.1f},{t:.1f} {xv+w/2:.1f},{t-r_in:.1f} L{xv+w/2:.1f},128 Z")
        add(f'<path d="{d}" fill="{fill}"/>')
    elif k != "gd":
        st, sb = beam_slot[k]
        ribbon(BX1 - 2, st, sb, LX0 + 1, t, b, fill)
    add(f'<rect x="{LX0}" y="{t}" width="{LX1-LX0}" height="{LH}" fill="{fill}"/>')
    gt, gb = g_slot[k]
    ribbon(LX1 - 1, t, b, GX + 1, gt, gb, fill)

# ocean temperature branch: peels off the underside of the beam, feeds discharge
yd = lane_y["gd"]; bx = 170
add(f'<path d="M{bx-60},{BB-1} L{bx},{BB-1} C{bx+40},{BB} {bx+20},{yd-9} {bx+80},{yd-9} L{LX0-60},{yd-9} '
    f'L{LX0-60},{yd+9} L{bx+60},{yd+9} C{bx-10},{yd+9} {bx-10},{BB} {bx-60},{BB-1} Z" fill="{B500}"/>')
ribbon(LX0 - 62, yd - 9, yd + 9, LX0 + 1, yd - LH / 2, yd + LH / 2, GRIS)
add(f'<text x="40" y="{yd+50}" class="b">Greenland ocean temperature</text>')
add(f'<text x="40" y="{yd+74}">via Arctic amplification</text>')

# beam
add(f'<rect x="{BX0}" y="{BT}" width="{BX1-BX0}" height="{BB-BT}" fill="{NAVY}"/>')
add(f'<text x="{BX0+16}" y="{BT+44}" class="h" style="fill:#fff;font-size:34px">GMST</text>')
add(f'<text x="{BX0+16}" y="{BT+76}" style="fill:#fff">observed to 2025</text>')
add(f'<text x="{BX0+16}" y="{BT+102}" style="fill:#fff">AR6 paths: 2, 3, 4 °C</text>')
add(f'<text x="{BX0}" y="{BT-42}" style="fill:{B500}">Global mean</text>')
add(f'<text x="{BX0}" y="{BT-16}" style="fill:{B500}">surface temperature</text>')

# lane labels
for k, name, sub, cls, col, tc in lanes:
    y = lane_y[k]
    if cls == "adopted":
        add(f'<text x="{LX0+14}" y="{y-8}" class="b halo">{name}</text>')
        add(f'<text x="{LX0+14}" y="{y+23}" class="halo">{sub}</text>')
        add(f'<text x="{LX1-10}" y="{y-8}" text-anchor="end" class="tag halo" style="fill:{B500}">adopted</text>')
        continue
    halo = ' halo' if k == "smb" else ''
    add(f'<text x="{LX0+14}" y="{y-4}" class="b{halo}" style="fill:{tc}">{name}</text>')
    add(f'<text x="{LX0+14}" y="{y+19}" class="{halo.strip()}" style="fill:{tc}">{sub}</text>')
    add(f'<text x="{LX1-10}" y="{y-4}" text-anchor="end" class="tag{halo}" style="fill:{tc}">{cls}</text>')

# ---------------- GMSL ribbon + test gate ----------------
add(f'<rect x="{GX}" y="{GT}" width="{RX1-GX}" height="{GB-GT}" fill="{NAVY}"/>')
add(f'<text x="{GX+12}" y="{(GT+GB)/2+10}" class="h" style="fill:#fff;font-size:28px">GMSL</text>')
# gate: a frame the ribbon passes through, with a check badge
gx0, gx1 = 960, 996
gt0, gb0 = GT - 40, GB + 40
add(f'<rect x="{gx0}" y="{gt0}" width="{gx1-gx0}" height="{gb0-gt0}" rx="6" fill="none" stroke="{ANT}" stroke-width="8"/>')
add(f'<rect x="{gx0+4}" y="{GT}" width="{gx1-gx0-8}" height="{GB-GT}" fill="{B500}"/>')
bcx, bcy = (gx0 + gx1) / 2, gt0 - 4
add(f'<circle cx="{bcx}" cy="{bcy}" r="21" fill="{ANT}" stroke="#fff" stroke-width="3"/>')
add(f'<path d="M{bcx-9},{bcy} l6,7 l12,-14" fill="none" stroke="#fff" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>')
# test text block
tx, ty = 800, gb0 + 44
add(f'<line x1="{bcx}" y1="{gb0+4}" x2="{bcx}" y2="{ty-24}" stroke="{ANT}" stroke-width="3"/>')
add(f'<text x="{tx}" y="{ty}" class="b" style="font-size:22px">Test against observed GMSL,</text>')
add(f'<text x="{tx}" y="{ty+26}" class="b" style="font-size:22px">withheld from every calibration</text>')
for i, s in enumerate(["closes the altimetry budget, 1993–2025", "hindcasts GMSL back to 1950"]):
    yy = ty + 62 + 29 * i
    add(f'<path d="M{tx+2},{yy-7} l6,7 l12,-14" fill="none" stroke="{ANT}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    add(f'<text x="{tx+28}" y="{yy}">{s}</text>')

# ---------------- forecast fan (schematic) ----------------
fx0, fx1 = RX1, 1222
fy = (GT + GB) / 2
# (label, start top/bottom at fx0, end top/bottom at fx1, colour); stacked, non-overlapping
bands = [("2 °C", fy + 6, fy + 30, 404, 452, "#9AA7B4"),
         ("3 °C", fy - 18, fy + 6, 348, 400, "#4F6378"),
         ("4 °C", fy - 42, fy - 18, 286, 344, NAVY)]
def fanband(s_t, s_b, e_t, e_b, fill):
    add(f'<path d="M{fx0},{s_t:.1f} C{fx0+90},{s_t:.1f} {fx1-90},{e_t:.1f} {fx1},{e_t:.1f} '
        f'L{fx1},{e_b:.1f} C{fx1-90},{e_b:.1f} {fx0+90},{s_b:.1f} {fx0},{s_b:.1f} Z" fill="{fill}"/>')
# West Antarctica tail rides on top of the warming bands
tail_s = fy - 42
add(f'<path d="M{fx0+20},{tail_s:.1f} C{fx0+110},{tail_s-2:.1f} {fx1-90},{168} {fx1},{160} '
    f'L{fx1},{282} C{fx1-90},{282} {fx0+90},{tail_s:.1f} {fx0+20},{tail_s:.1f} Z" fill="url(#tail)"/>')
add(f'<path d="M{fx0+20},{tail_s:.1f} C{fx0+110},{tail_s-2:.1f} {fx1-90},{168} {fx1},{160}" fill="none" stroke="{WAIS_D}" stroke-width="2.5"/>')
for lab, st, sb, et, eb, col in bands:
    fanband(st, sb, et, eb, col)
for lab, st, sb, et, eb, col in bands:
    add(f'<text x="{fx1+10}" y="{(et+eb)/2+7}" class="b">{lab}</text>')
add(f'<text x="{fx1+10}" y="196" class="b" style="fill:{NAVY}">West</text>')
add(f'<text x="{fx1+10}" y="220" class="b" style="fill:{NAVY}">Antarctica</text>')
add(f'<text x="{fx1+10}" y="244" class="tag" style="fill:{NAVY}">tail</text>')
add(f'<line x1="{fx1}" x2="{fx1}" y1="164" y2="{GB+16}" stroke="{OVER}" stroke-width="2" stroke-dasharray="4 4"/>')
add(f'<text x="{fx1}" y="{GB+42}" text-anchor="middle" class="b">2100</text>')
add(f'<text x="{fx0+18}" y="{GB+42}" class="tag" style="fill:{GREY}">schematic</text>')
add(f'<text x="1040" y="152">2000 members under 2, 3, 4 °C</text>')

# ---------------- bottom: two dimensions of risk ----------------
ry = 806
add(f'<line x1="24" x2="{W-24}" y1="{ry-38}" y2="{ry-38}" stroke="{OVER}" stroke-width="2"/>')
add(f'<text x="24" y="{ry}" class="h" style="font-size:22px">Two dimensions of risk</text>')
sw = [OCEAN, GLAC, GRIS, ANT]
for i, c in enumerate(sw):
    add(f'<rect x="{318+i*14}" y="{ry-18}" width="14" height="20" fill="{c}"/>')
add(f'<text x="384" y="{ry}">Emissions set the warming-driven components</text>')
add(f'<rect x="980" y="{ry-18}" width="36" height="20" fill="{WAIS}"/>')
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
print("ok", GT, GB)
