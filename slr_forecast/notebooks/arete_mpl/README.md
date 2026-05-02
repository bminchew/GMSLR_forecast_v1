# arete_mpl — Matplotlib Styles & Colormaps for Arête Glacier Initiative

## Installation

Copy the `arete_mpl/` directory into your project or anywhere on `PYTHONPATH`:

```
your_project/
├── arete_mpl/
│   ├── arete_mpl.py        # module: colormaps + style registration
│   └── styles/
│       ├── arete-base.mplstyle
│       ├── arete-paper.mplstyle
│       ├── arete-notebook.mplstyle
│       ├── arete-talk.mplstyle
│       └── arete-poster.mplstyle
├── demo.py                  # visual showcase (optional)
└── your_script.py
```

No dependencies beyond `numpy`, `scipy`, and `matplotlib`.

## Quick Start

```python
import sys
sys.path.insert(0, "/path/to/arete_mpl")   # if not on PYTHONPATH
import arete_mpl                             # registers everything on import
import matplotlib.pyplot as plt

arete_mpl.use("notebook")                   # or "paper", "talk", "poster"

fig, ax = plt.subplots()
ax.plot(x, y)                                # uses Arête color cycle automatically
plt.show()
```

## Styles

Each style combines `arete-base` (colours, fonts, tick direction, grid, cycle)
with a sizing context that mirrors the seaborn `paper / notebook / talk / poster`
scaling.

| Style | Figure size | Label size | Use case |
|---|---|---|---|
| `arete-paper` | 6.4 × 4.4 in | 8.8 pt | Journal figures |
| `arete-notebook` | 8.0 × 5.5 in | 11 pt | Jupyter / screen |
| `arete-talk` | 10.4 × 7.15 in | 14.3 pt | Slide projections |
| `arete-poster` | 12.8 × 8.8 in | 17.6 pt | Conference posters |

You can also apply them manually:

```python
plt.style.use(["arete-base", "arete-talk"])
```

### Color cycle (6 colours)

| Swatch | Hex | Origin |
|---|---|---|
| Glacier Blue 400 | `#036C9A` | Arête brand |
| Earth | `#AB6638` | Arête brand |
| Teal | `#2A9D8F` | Complement |
| Muted red | `#C44E52` | Complement |
| Glacier Blue 300 | `#72A3C3` | Arête brand |
| Amber | `#D4A24E` | Complement |

## Colormaps

All colormaps are registered with matplotlib on `import arete_mpl` and can be
referenced by name string. Reversed variants (`_r` suffix) are also registered.

### Sequential (single-hue)

| Name | Description |
|---|---|
| `arete_blues` | Glacier Blue 100 → 600, multi-stop CIELAB |
| `arete_reds` | Light blush → deep crimson |
| `arete_greens` | Pale sage → deep teal |
| `arete_purples` | Lavender → aubergine |
| `arete_oranges` | Pale sand → deep earth |

### Perceptually-uniform blends

All five start from Arête navy (GB600, L*≈11) and ramp to a light target colour.
L* increases linearly; hue transitions via a sigmoid centred in the upper third
of the map so blues dominate the lower ~60%.

| Name | End colour | L* range | Suggested use |
|---|---|---|---|
| `arete_blue2earth` | Light tan | 11 → 86 | Bed topography, bathymetry-to-land |
| `arete_blue2green` | Light sage | 11 → 80 | Ice-to-vegetation, grounding-line |
| `arete_blue2purple` | Light lavender | 11 → 76 | Geothermal flux, anomalies |
| `arete_blue2red` | Blush | 11 → 74 | Cold-to-warm (non-diverging) |
| `arete_blue2white` | White | 11 → 100 | Ice thickness, data density |

### Usage

```python
import arete_mpl
import matplotlib.pyplot as plt

plt.imshow(velocity, cmap="arete_blues")
plt.pcolormesh(X, Y, bed, cmap="arete_blue2earth", shading="auto")

# Reversed
plt.contourf(X, Y, Z, cmap="arete_blues_r")
```

### Programmatic access

```python
cmap = arete_mpl.get_cmap("arete_blue2red")
print(arete_mpl.list_cmaps())
```

## Fonts

The style files specify `Outfit` and `Rubik` (Arête brand fonts, available on
Google Fonts) with fallbacks to Montserrat → Helvetica Neue → Arial → DejaVu Sans.
Install the brand fonts system-wide or into matplotlib's font directory for
full fidelity:

```bash
# Example on Linux
cp Outfit-*.ttf Rubik-*.ttf ~/.local/share/fonts/
fc-cache -fv
python -c "import matplotlib; matplotlib.font_manager._load_fontmanager(try_read_cache=False)"
```

## Files

```
styles/arete-base.mplstyle      Base aesthetic (colours, fonts, ticks, cycle)
styles/arete-paper.mplstyle     Paper sizing context
styles/arete-notebook.mplstyle  Notebook sizing context
styles/arete-talk.mplstyle      Talk sizing context
styles/arete-poster.mplstyle    Poster sizing context
arete_mpl.py                    Python module (colormaps + registration)
demo.py                         Visual showcase generator
```
