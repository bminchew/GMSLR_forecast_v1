"""SPECTRUM wordmark 2A revision: the rising waterline becomes a forecast fan
that spreads across the letters. Builds on ../wordmark/gen.py (glyph layout)."""
import os, sys
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_wordmark_base.py")
code = open(SRC).read().split("# Concept 1")[0]
code = code.rsplit("\n# ----", 1)[0]
ns = {"__file__": SRC}
exec(code, ns)
items, XMIN, XMAX, CAP, PAD, GB, EARTH, DARK, ALL_D = (ns[k] for k in
    ["items", "XMIN", "XMAX", "CAP", "PAD", "GB", "EARTH", "DARK", "ALL_D"])
glyph_path, svg = ns["glyph_path"], ns["svg"]
OUT = os.path.dirname(os.path.abspath(__file__))

N = 120


def curves(x0, x1, hinge_frac, y_obs0, y_h, y_med1, half_in, half_out, tail):
    """Return fan boundary polylines (top->bottom) sampled across [x0, x1].
    Before the hinge every curve sits on the observed line; after it they spread."""
    xh = x0 + hinge_frac * (x1 - x0)
    out = {k: [] for k in ["tail", "out_up", "in_up", "med", "in_lo", "out_lo"]}
    for i in range(N + 1):
        x = x0 + (x1 - x0) * i / N
        if x <= xh:
            s = (x - x0) / (xh - x0)
            m = y_obs0 + (y_h - y_obs0) * (s ** 1.4)
            spread = 0.0
        else:
            t = (x - xh) / (x1 - xh)
            m = y_h + (y_med1 - y_h) * (t ** 1.4)
            spread = t ** 1.1
        out["med"].append((x, m))
        out["in_up"].append((x, m - half_in * spread))
        out["in_lo"].append((x, m + half_in * spread))
        out["out_up"].append((x, m - half_out * spread))
        out["out_lo"].append((x, m + 0.8 * half_out * spread))
        out["tail"].append((x, m - (half_out + tail * spread) * spread))
    return out


def band(upper, lower):
    pts = upper + lower[::-1]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"


def above(upper, ytop):
    x0, x1 = upper[0][0], upper[-1][0]
    return band([(x0, ytop), (x1, ytop)], [(x0, upper[0][1]), (x1, upper[-1][1])][:0] or upper) if False else \
        "M" + f"{x0:.1f},{ytop} L{x1:.1f},{ytop} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in upper[::-1]) + " Z"


def below(lower, ybot):
    x0, x1 = lower[0][0], lower[-1][0]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in lower) + f" L{x1:.1f},{ybot} L{x0:.1f},{ybot} Z"


def polyline(c):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in c)


def build(d_glyphs, x0, x1, dark, p, geom, gaps=True, single=False):
    c = curves(x0, x1, **geom)
    if dark:
        col = dict(top=GB[200], tail=EARTH, outer=GB[100], inner=GB[300], deep=(GB[300], GB[400]))
        gapc = DARK
    else:
        col = dict(top=GB[300], tail=EARTH, outer=GB[200], inner=GB[400], deep=(GB[500], GB[600]))
        gapc = "#FFFFFF"
    if single:
        col["inner"] = col["outer"]
    ytop, ybot = -200, CAP + 200
    regions = [
        (above(c["tail"], ytop), col["top"]),
        (band(c["tail"], c["out_up"]), col["tail"]),
        (band(c["out_up"], c["in_up"]), col["outer"]),
        (band(c["in_up"], c["in_lo"]), col["inner"]),
        (band(c["in_lo"], c["out_lo"]), col["outer"]),
        (below(c["out_lo"], ybot), f"url(#{p}-g)"),
    ]
    if single:  # one fan band, no seams between same-colour pieces
        regions = regions[:2] + [(band(c["out_up"], c["out_lo"]), col["outer"])] + regions[-1:]
    defs = (f'<clipPath id="{p}-w"><path d="{d_glyphs}"/></clipPath>'
            f'<linearGradient id="{p}-g" gradientUnits="userSpaceOnUse" x1="0" y1="300" x2="0" y2="{CAP}">'
            f'<stop offset="0" stop-color="{col["deep"][0]}"/><stop offset="1" stop-color="{col["deep"][1]}"/></linearGradient>')
    body = f'<g clip-path="url(#{p}-w)">'
    body += "".join(f'<path d="{d}" fill="{f}"/>' for d, f in regions)
    if gaps:
        # knockout lines: one at the observed line, splitting into the fan edges
        for k, wdt in [("med", 22)]:
            body += f'<path d="{polyline(c[k])}" fill="none" stroke="{gapc}" stroke-width="{wdt}" stroke-linejoin="round"/>'
    body += "</g>"
    return defs, body


GEOM = dict(hinge_frac=0.26, y_obs0=500, y_h=470, y_med1=290, half_in=110, half_out=250, tail=170)


def wordmark(dark, gaps=True, single=False):
    p = ("fwd" if dark else "fw") + ("s" if single else "")
    defs, body = build(ALL_D, XMIN - PAD, XMAX + PAD, dark, p, GEOM, gaps, single)
    vb = (XMIN - PAD, -PAD, XMAX - XMIN + 2 * PAD, CAP + 2 * PAD)
    return svg(vb, body, defs, DARK if dark else None)


def mono(dark, gaps=True, single=False):
    p = ("fmd" if dark else "fm") + ("s" if single else "")
    d, b, w = glyph_path("S", 0)
    g = dict(GEOM, hinge_frac=0.15, y_obs0=500, y_h=470)
    defs, body = build(d, b[0] - 40, b[2] + 40, dark, p, g, gaps, single)
    s = max(b[2] - b[0], b[3] - b[1]) + 160
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return svg((cx - s / 2, cy - s / 2, s, s), body, defs, DARK if dark else None, "SPECTRUM monogram")


for name, s in {
    "fan_wordmark.svg": wordmark(False), "fan_wordmark_dark.svg": wordmark(True),
    "fan_mono.svg": mono(False), "fan_mono_dark.svg": mono(True),
    "fan1_wordmark.svg": wordmark(False, single=True), "fan1_wordmark_dark.svg": wordmark(True, single=True),
    "fan1_mono.svg": mono(False, single=True), "fan1_mono_dark.svg": mono(True, single=True),
}.items():
    open(os.path.join(OUT, name), "w").write(s)
print("ok")
